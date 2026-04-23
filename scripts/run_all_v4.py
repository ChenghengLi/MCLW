#!/usr/bin/env python
"""
v4 full-protocol runner: budget-matched gate equivalence study.

Differences from v3:
  - Sweeps temperature T ∈ {0.0, 0.7, 1.0}  (v3's T=0 trivially collapses Δ and H)
  - Sweeps ρ via 3 quantile thresholds Q25 / Q50 / Q75 of pilot entropy / delta
  - Two prompt domains: Wikipedia concepts + factual-closed (corner-probe)
  - Paired BCa bootstrap 95% CIs on per-prompt deltas between gates
  - Pre-registered equivalence margins: ±0.05 log-PPL, ±0.5 z, ±0.05 TPR

Usage:
    uv run python scripts/run_all_v4.py --n-prompts 15 --max-tokens 60
    uv run python scripts/run_all_v4.py --model meta-llama/Llama-3.2-3B-Instruct --n-prompts 50
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltw_watermark import (
    GatedMCLGenerator,
    GateAll,
    GateNone,
    GateEntropyHigh,
    GateEntropyLow,
    GateDelta,
    detect,
    load_prompts,
)
from ltw_watermark.factual_prompts import FACTUAL_CLOSED_PROMPTS
from ltw_watermark.attacks import (
    random_substitution_tokens,
    synonym_substitution,
)


EQ_MARGIN_LOGPPL = 0.05       # log-PPL ratio (~5% PPL ratio)
EQ_MARGIN_Z = 0.5             # z-score
EQ_MARGIN_TPR = 0.05          # 5 percentage points

# Target budgets (fraction of positions gated).  Per-gate tau is chosen in
# pilot so each gate's EMPIRICAL rho matches the target, giving a true
# matched-budget comparison.
TARGET_RHOS = (0.25, 0.50, 0.75)


def paired_bca_bootstrap(
    deltas: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05, rng=None
) -> Tuple[float, float, float]:
    """
    Bias-corrected-and-accelerated paired bootstrap CI on a per-sample delta.

    Returns: (mean, lo, hi) at (1 - alpha) coverage.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(deltas)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(deltas))
    # Resample
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.mean(deltas[idx], axis=1)
    # Bias correction: z0 = Phi^{-1}( P(boot < obs) )
    from scipy.stats import norm
    p_lt = float(np.mean(boots < mean))
    p_lt = min(max(p_lt, 1 / (n_boot + 1)), 1 - 1 / (n_boot + 1))
    z0 = norm.ppf(p_lt)
    # Jackknife acceleration
    jack = np.array([np.mean(np.delete(deltas, i)) for i in range(n)])
    jack_mean = np.mean(jack)
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5) + 1e-12
    accel = num / den
    # BCa quantiles
    za_lo = norm.ppf(alpha / 2)
    za_hi = norm.ppf(1 - alpha / 2)
    alpha_lo = norm.cdf(z0 + (z0 + za_lo) / (1 - accel * (z0 + za_lo)))
    alpha_hi = norm.cdf(z0 + (z0 + za_hi) / (1 - accel * (z0 + za_hi)))
    lo = float(np.quantile(boots, alpha_lo))
    hi = float(np.quantile(boots, alpha_hi))
    return mean, lo, hi


def _pilot(gen, prompts, max_new_tokens, temperature):
    """Return raw per-token H and Delta arrays; callers pick quantiles per-gate."""
    gen.gate = GateNone()
    H_all, D_all = [], []
    for p in prompts:
        res = gen.generate(p, max_new_tokens=max_new_tokens, greedy=(temperature == 0.0), temperature=max(temperature, 1e-6))
        H_all.extend([pos.entropy for pos in res.positions])
        D_all.extend([pos.delta for pos in res.positions])
    return np.array(H_all), np.array(D_all)


def tau_for_target_rho(H_arr, D_arr, gate_name: str, target_rho: float) -> float:
    """
    Pick tau so that the gate fires on ~target_rho fraction of pilot positions.

      H_high fires when H > tau  ->  rho = P(H > tau)         -> tau = Q_{1-rho}(H)
      Delta  fires when D < tau  ->  rho = P(D < tau)         -> tau = Q_{rho}(D)
      H_low  fires when H < tau  ->  rho = P(H < tau)         -> tau = Q_{rho}(H)
    """
    if gate_name == "H_high":
        return float(np.quantile(H_arr, 1 - target_rho))
    if gate_name == "H_low":
        return float(np.quantile(H_arr, target_rho))
    if gate_name == "Delta":
        return float(np.quantile(D_arr, target_rho))
    raise ValueError(f"Unknown gate for tau selection: {gate_name}")


def run_cell(gen, prompts, max_new_tokens, temperature, seed) -> List[Dict]:
    """Generate + detect for the currently-configured gate."""
    records = []
    for i, prompt in enumerate(prompts):
        torch.manual_seed(seed + i)
        res = gen.generate(prompt, max_new_tokens=max_new_tokens, greedy=(temperature == 0.0), temperature=max(temperature, 1e-6))
        det = detect(
            res.token_ids,
            num_states=gen.num_states,
            chain_key=gen.chain_key,
            secret_key=gen.secret_key,
            overlap_ratio=gen.overlap_ratio,
            alpha=0.01,
            exact=True,
        )
        records.append({
            "prompt": prompt,
            "text": res.text,
            "token_ids": res.token_ids,
            "rho": res.rho_empirical,
            "ppl": res.perplexity,
            "phi": det.phi,
            "z": det.z,
            "p_value": det.p_value,
            "detected": det.is_watermarked,
        })
    return records


def run_attack(records, gen, attack_fn, seed) -> Dict:
    """Apply attack, re-detect, summarize."""
    post_z, n_det = [], 0
    for idx, r in enumerate(records):
        _, new_ids = attack_fn(r["text"], r["token_ids"], seed + idx)
        det = detect(
            new_ids,
            num_states=gen.num_states,
            chain_key=gen.chain_key,
            secret_key=gen.secret_key,
            overlap_ratio=gen.overlap_ratio,
            alpha=0.01,
            exact=True,
        )
        post_z.append(det.z)
        if det.is_watermarked:
            n_det += 1
    return {
        "tpr_post": n_det / max(1, len(records)),
        "z_post_mean": float(np.mean(post_z)) if post_z else 0.0,
    }


def equiv_verdict(lo, hi, margin) -> str:
    if hi <= margin and lo >= -margin:
        return "EQUIVALENT"
    if lo > margin:
        return "A > B"
    if hi < -margin:
        return "B > A"
    return "inconclusive"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-270m-it")
    parser.add_argument("--secret-key", default="mclw_v4_2026")
    parser.add_argument("--num-states", type=int, default=5)
    parser.add_argument("--chain-key", default="clockwork")
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--n-prompts", type=int, default=15)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--n-pilot", type=int, default=5)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.0, 0.7, 1.0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/v4_all_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[v4] model={args.model}  out={out_dir}")

    wiki = load_prompts(args.n_prompts)
    fact = FACTUAL_CLOSED_PROMPTS[: args.n_prompts]
    domains = {"wiki": wiki, "factual": fact}

    gen = GatedMCLGenerator(
        model_name=args.model,
        secret_key=args.secret_key,
        num_states=args.num_states,
        chain_key=args.chain_key,
        overlap_ratio=args.overlap,
        gate=GateNone(),
    )

    summary = {
        "run": f"v4_all_{stamp}",
        "model": args.model,
        "n_prompts": args.n_prompts,
        "max_tokens": args.max_tokens,
        "temperatures": args.temperatures,
        "margins": {"log_ppl": EQ_MARGIN_LOGPPL, "z": EQ_MARGIN_Z, "tpr": EQ_MARGIN_TPR},
        "cells": [],
        "equivalence_tests": [],
    }

    # -------- Main nested loop: domain x temperature x quantile --------
    t0 = time.time()
    for dname, prompts in domains.items():
        for T in args.temperatures:
            print(f"\n[v4] === domain={dname}  T={T} ===", flush=True)
            # 1) pilot returns raw H and Delta arrays
            H_arr, D_arr = _pilot(gen, prompts[: args.n_pilot], args.max_tokens, T)
            print(f"[v4] pilot H q25/q50/q75 = {np.quantile(H_arr,[.25,.5,.75])}   "
                  f"D q25/q50/q75 = {np.quantile(D_arr,[.25,.5,.75])}", flush=True)

            # 2) For each target rho, pick tau per gate to hit that rho
            for target_rho in TARGET_RHOS:
                rho_label = f"rho{int(target_rho*100)}"
                results = {}
                gate_defs = [
                    ("H_high", GateEntropyHigh, tau_for_target_rho(H_arr, D_arr, "H_high", target_rho)),
                    ("Delta",  GateDelta,       tau_for_target_rho(H_arr, D_arr, "Delta",  target_rho)),
                    ("H_low",  GateEntropyLow,  tau_for_target_rho(H_arr, D_arr, "H_low",  target_rho)),
                ]
                print(f"  [target rho={target_rho}] taus -> "
                      f"H_high={gate_defs[0][2]:.3f}  Delta={gate_defs[1][2]:.3f}  H_low={gate_defs[2][2]:.3f}", flush=True)
                for gname, gate_cls, tau in gate_defs:
                    gen.gate = gate_cls(tau=tau)
                    records = run_cell(gen, prompts, args.max_tokens, T, args.seed)
                    results[gname] = records
                    mean_rho = float(np.mean([r["rho"] for r in records]))
                    mean_ppl = float(np.mean([r["ppl"] for r in records]))
                    mean_z = float(np.mean([r["z"] for r in records]))
                    tpr = float(np.mean([r["detected"] for r in records]))
                    print(f"    [{gname:>6s}  tau={tau:.3f}] rho={mean_rho:.2f} PPL={mean_ppl:.2f} z={mean_z:5.2f} TPR={tpr:.2f}", flush=True)

                    # Attack once per gate (random_sub20 + synonym20)
                    from ltw_watermark.attacks import random_substitution_tokens, synonym_substitution
                    def rs20(text, ids, seed):
                        new = random_substitution_tokens(ids, gen.vocab_size, 0.20, seed=seed)
                        return gen.tokenizer.decode(new, skip_special_tokens=True), new
                    def syn20(text, ids, seed):
                        new_text = synonym_substitution(text, 0.20, seed=seed)
                        return new_text, gen.tokenizer.encode(new_text, add_special_tokens=False)
                    atk_rs = run_attack(records, gen, rs20, args.seed)
                    atk_syn = run_attack(records, gen, syn20, args.seed + 1)

                    summary["cells"].append({
                        "domain": dname, "temperature": T, "target_rho": target_rho,
                        "gate": gname, "tau": tau,
                        "mean_rho": mean_rho, "mean_ppl": mean_ppl,
                        "mean_z": mean_z, "tpr": tpr,
                        "random_sub20": atk_rs, "synonym20": atk_syn,
                    })

                # 3) Paired BCa bootstrap: H_high vs Delta at this quantile
                if "H_high" in results and "Delta" in results:
                    H = results["H_high"]; D = results["Delta"]
                    # Paired on prompt index
                    d_logppl = np.log(np.array([r["ppl"] for r in H])) - np.log(np.array([r["ppl"] for r in D]))
                    d_z = np.array([r["z"] for r in H]) - np.array([r["z"] for r in D])
                    d_detected = np.array([int(r["detected"]) for r in H]) - np.array([int(r["detected"]) for r in D])
                    m_p, lo_p, hi_p = paired_bca_bootstrap(d_logppl)
                    m_z, lo_z, hi_z = paired_bca_bootstrap(d_z)
                    m_t, lo_t, hi_t = paired_bca_bootstrap(d_detected.astype(float))
                    v_p = equiv_verdict(lo_p, hi_p, EQ_MARGIN_LOGPPL)
                    v_z = equiv_verdict(lo_z, hi_z, EQ_MARGIN_Z)
                    v_t = equiv_verdict(lo_t, hi_t, EQ_MARGIN_TPR)
                    summary["equivalence_tests"].append({
                        "domain": dname, "temperature": T, "target_rho": target_rho,
                        "contrast": "H_high - Delta",
                        "logPPL": {"mean": m_p, "lo": lo_p, "hi": hi_p, "verdict": v_p, "margin": EQ_MARGIN_LOGPPL},
                        "z": {"mean": m_z, "lo": lo_z, "hi": hi_z, "verdict": v_z, "margin": EQ_MARGIN_Z},
                        "tpr": {"mean": m_t, "lo": lo_t, "hi": hi_t, "verdict": v_t, "margin": EQ_MARGIN_TPR},
                    })
                    print(
                        f"    [equivalence rho={target_rho}] "
                        f"dlogPPL={m_p:+.3f} [{lo_p:+.3f},{hi_p:+.3f}] {v_p};  "
                        f"dz={m_z:+.2f} [{lo_z:+.2f},{hi_z:+.2f}] {v_z};  "
                        f"dTPR={m_t:+.2f} [{lo_t:+.2f},{hi_t:+.2f}] {v_t}",
                        flush=True,
                    )

    summary["elapsed_seconds"] = time.time() - t0
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---- Human-readable final table ----
    print("\n" + "=" * 120)
    print(f"v4 PROTOCOL COMPLETE  model={args.model}  N={args.n_prompts}  elapsed={summary['elapsed_seconds']:.0f}s")
    print("=" * 120)
    print("\nEquivalence tests (H_high vs Delta at matched target rho):")
    print(f"{'domain':>10s}  {'T':>4s}  {'rho':>5s}  {'log-PPL 95CI':>22s}  {'z 95CI':>22s}  {'TPR 95CI':>22s}  verdict")
    for t in summary["equivalence_tests"]:
        p = t["logPPL"]; z = t["z"]; tp = t["tpr"]
        vs = [p["verdict"], z["verdict"], tp["verdict"]]
        overall = "EQUIVALENT" if all(v == "EQUIVALENT" for v in vs) else (
            "inconclusive" if any(v == "inconclusive" for v in vs) else "split"
        )
        print(
            f"{t['domain']:>10s}  {t['temperature']:>4.1f}  {t['target_rho']:>5.2f}  "
            f"{p['mean']:+.3f}[{p['lo']:+.3f},{p['hi']:+.3f}]  "
            f"{z['mean']:+.2f}[{z['lo']:+.2f},{z['hi']:+.2f}]      "
            f"{tp['mean']:+.2f}[{tp['lo']:+.2f},{tp['hi']:+.2f}]      "
            f"{overall}"
        )
    print(f"\n[v4] Summary: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
