#!/usr/bin/env python
"""
Run the full v3 gate-comparison protocol in ONE script.

Covers Experiments 1 (calibration), 2 (gate comparison), 3 (robustness),
and 4 (quality/PPL) from research_plan_v3.md -- all sharing the same
generations so compute is O(N_prompts * N_gates) instead of O(E * N).

For each gate in {all, H_high, H_low, Delta, none}:
  1. Generate N texts (greedy decoding, same prompts).
  2. Detect each text (corrected z-score, p-value).
  3. For each attack in {random_sub10, random_sub20, synonym20}:
       attack -> realign with tokenizer -> re-detect -> per-position survival.
  4. Compute per-gate summary: rho, H_mean, Delta_mean, PPL (generator
     self-perplexity), z, TPR, post-attack TPR, per-entropy-bin survival.

Cheap: Gemma 3 270M finishes N=20 prompts x 5 gates x 60 tokens in ~1 min on CPU.

Outputs: data/v3_all_<timestamp>/{summary.json, generations/, attacks/, per_gate_table.csv}

Usage:
    # Quick smoke run (Gemma 3 270M, 20 prompts, 60 tokens)
    uv run python scripts/run_all_v3.py

    # Full run on Llama-3.2-3B
    uv run python scripts/run_all_v3.py --model meta-llama/Llama-3.2-3B-Instruct \\
        --n-prompts 100 --max-tokens 150
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
from ltw_watermark.alignment import align_token_sequences
from ltw_watermark.attacks import (
    random_substitution_tokens,
    synonym_substitution,
)


# ============================================================
# Attacks (light set; DIPPER/SIRA skipped for speed)
# ============================================================
def make_attacks(tokenizer, vocab_size: int):
    def rs10(text, ids, seed):
        new = random_substitution_tokens(ids, vocab_size, 0.10, seed=seed)
        return tokenizer.decode(new, skip_special_tokens=True), new

    def rs20(text, ids, seed):
        new = random_substitution_tokens(ids, vocab_size, 0.20, seed=seed)
        return tokenizer.decode(new, skip_special_tokens=True), new

    def syn20(text, ids, seed):
        new_text = synonym_substitution(text, 0.20, seed=seed)
        new_ids = tokenizer.encode(new_text, add_special_tokens=False)
        return new_text, new_ids

    return {
        "random_sub10": rs10,
        "random_sub20": rs20,
        "synonym20": syn20,
    }


# ============================================================
# Entropy binning for Prop-8-style per-bin survival analysis
# ============================================================
BIN_EDGES = [0.0, 0.5, 1.0, 2.0, 3.0, np.inf]


def entropy_binned_survival(H, survived) -> List[Dict]:
    H = np.asarray(H, dtype=np.float64)
    S = np.asarray(survived, dtype=np.int32)
    out = []
    for lo, hi in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
        mask = (H >= lo) & (H < hi)
        if mask.sum() == 0:
            continue
        out.append({
            "H_lo": lo,
            "H_hi": None if np.isinf(hi) else hi,
            "n": int(mask.sum()),
            "survival": float(S[mask].mean()),
        })
    return out


# ============================================================
# Pilot: measure entropy + Delta percentiles to pick tau values
# ============================================================
def pilot_percentiles(
    gen: GatedMCLGenerator, prompts: List[str], max_new_tokens: int
) -> Dict[str, Dict[str, float]]:
    gen.gate = GateNone()
    H_all, D_all = [], []
    for p in prompts:
        res = gen.generate(p, max_new_tokens=max_new_tokens, greedy=True)
        H_all.extend([pos.entropy for pos in res.positions])
        D_all.extend([pos.delta for pos in res.positions])
    H_arr = np.array(H_all)
    D_arr = np.array(D_all)
    qs = [0.10, 0.25, 0.50, 0.75, 0.90]
    return {
        "entropy": {f"Q{int(q*100)}": float(np.quantile(H_arr, q)) for q in qs},
        "delta": {f"Q{int(q*100)}": float(np.quantile(D_arr, q)) for q in qs},
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-270m-it",
                        help="HF model ID. Small default for fast runs.")
    parser.add_argument("--secret-key", default="mclw_v3_all_2026")
    parser.add_argument("--num-states", type=int, default=5)
    parser.add_argument("--chain-key", default="clockwork")
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=60)
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--n-pilot", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--skip-attacks", action="store_true",
                        help="E1+E2 only, no E3 robustness attacks")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/v3_all_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "generations").mkdir(exist_ok=True)
    (out_dir / "attacks").mkdir(exist_ok=True)
    print(f"[run_all_v3] model={args.model}")
    print(f"[run_all_v3] writing to {out_dir}")

    prompts = load_prompts(args.n_prompts)
    print(f"[run_all_v3] N prompts = {len(prompts)}")

    # ---------- 1. Load generator ----------
    gen = GatedMCLGenerator(
        model_name=args.model,
        secret_key=args.secret_key,
        num_states=args.num_states,
        chain_key=args.chain_key,
        overlap_ratio=args.overlap,
        gate=GateNone(),
    )

    # ---------- 2. Pilot for tau percentiles ----------
    t0 = time.time()
    print(f"[run_all_v3] Pilot ({args.n_pilot} prompts)...")
    pcs = pilot_percentiles(gen, prompts[: args.n_pilot], args.max_tokens)
    tau_H = pcs["entropy"]["Q50"]
    tau_D = pcs["delta"]["Q50"]
    print(f"[run_all_v3] Percentiles: {pcs}")
    print(f"[run_all_v3] Using tau_H={tau_H:.3f}, tau_Delta={tau_D:.3f}")

    # ---------- 3. For each gate: generate + detect + quality ----------
    gates = [
        ("none",   GateNone()),
        ("all",    GateAll()),
        ("H_high", GateEntropyHigh(tau=tau_H)),
        ("H_low",  GateEntropyLow(tau=tau_H)),
        ("Delta",  GateDelta(tau=tau_D)),
    ]

    per_gate: Dict[str, Dict] = {}
    per_gate_records: Dict[str, List] = {}
    baseline_ppl: Optional[float] = None

    for gname, gobj in gates:
        print(f"\n[run_all_v3] === Generating under gate '{gname}' ===")
        gen.gate = gobj
        records = []
        for i, prompt in enumerate(prompts):
            res = gen.generate(prompt, max_new_tokens=args.max_tokens, greedy=True)
            det = detect(
                res.token_ids,
                num_states=args.num_states,
                chain_key=args.chain_key,
                secret_key=args.secret_key,
                overlap_ratio=args.overlap,
                alpha=args.alpha,
                exact=True,
            )
            records.append({
                "prompt": prompt,
                "text": res.text,
                "token_ids": res.token_ids,
                "positions": [p.__dict__ for p in res.positions],
                "rho_empirical": res.rho_empirical,
                "perplexity": res.perplexity,
                "phi": det.phi,
                "z": det.z,
                "p_value": det.p_value,
                "detected": det.is_watermarked,
            })
        per_gate_records[gname] = records
        # persist
        with (out_dir / "generations" / f"{gname}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        # Aggregate stats
        ppl_mean = float(np.mean([r["perplexity"] for r in records]))
        if gname == "none":
            baseline_ppl = ppl_mean
        phi_mean = float(np.mean([r["phi"] for r in records]))
        z_mean = float(np.mean([r["z"] for r in records]))
        rho_mean = float(np.mean([r["rho_empirical"] for r in records]))
        H_mean = float(np.mean([pos["entropy"] for r in records for pos in r["positions"]]))
        D_mean = float(np.mean([pos["delta"] for r in records for pos in r["positions"]]))
        tpr = float(np.mean([r["detected"] for r in records]))
        per_gate[gname] = {
            "rho": rho_mean,
            "H_mean": H_mean,
            "Delta_mean": D_mean,
            "phi": phi_mean,
            "z": z_mean,
            "tpr": tpr,
            "ppl": ppl_mean,
            "tau_H_used": tau_H,
            "tau_Delta_used": tau_D,
        }
        print(
            f"[{gname:>6s}] rho={rho_mean:.2f} H={H_mean:.2f} D={D_mean:.2f} "
            f"PPL={ppl_mean:.2f} z={z_mean:5.2f} TPR={tpr:.2f}"
        )

    # ---------- 4. Attacks (E3) ----------
    tokenizer = gen.tokenizer
    vocab_size = gen.vocab_size
    # Free model memory BEFORE running attacks (attacks may load nltk/wordnet)
    # We still need tokenizer for decode/encode; keep it alive.

    attack_results: Dict[str, Dict[str, Dict]] = {}
    if not args.skip_attacks:
        attacks = make_attacks(tokenizer, vocab_size)
        for gname, records in per_gate_records.items():
            attack_results[gname] = {}
            for aname, afn in attacks.items():
                pooled_H, pooled_W, pooled_surv = [], [], []
                n_detected = 0
                z_post, phi_post = [], []
                for idx, r in enumerate(records):
                    new_text, new_ids = afn(r["text"], r["token_ids"], seed=args.seed + idx)
                    try:
                        align = align_token_sequences(r["token_ids"], new_ids)
                    except RuntimeError:
                        print(f"[run_all_v3] alignment unavailable (python-Levenshtein missing); "
                              f"skip survival-binned analysis.")
                        align = None
                    det = detect(
                        new_ids,
                        num_states=args.num_states,
                        chain_key=args.chain_key,
                        secret_key=args.secret_key,
                        overlap_ratio=args.overlap,
                        alpha=args.alpha,
                        exact=True,
                    )
                    z_post.append(det.z)
                    phi_post.append(det.phi)
                    if det.is_watermarked:
                        n_detected += 1
                    if align is not None:
                        for pos, lab in zip(r["positions"], align.labels):
                            pooled_H.append(pos["entropy"])
                            pooled_W.append(int(pos["watermarked"]))
                            pooled_surv.append(int(lab == "survived"))

                tpr_post = n_detected / max(1, len(records))
                bins = entropy_binned_survival(pooled_H, pooled_surv) if pooled_H else []
                survival_all = float(np.mean(pooled_surv)) if pooled_surv else None
                survival_wm = None
                survival_free = None
                if pooled_W:
                    wm_mask = np.array(pooled_W) == 1
                    free_mask = np.array(pooled_W) == 0
                    if wm_mask.any():
                        survival_wm = float(np.mean(np.array(pooled_surv)[wm_mask]))
                    if free_mask.any():
                        survival_free = float(np.mean(np.array(pooled_surv)[free_mask]))

                attack_results[gname][aname] = {
                    "tpr_post_attack": tpr_post,
                    "mean_z_post": float(np.mean(z_post)) if z_post else 0.0,
                    "mean_phi_post": float(np.mean(phi_post)) if phi_post else 0.0,
                    "survival_all": survival_all,
                    "survival_watermarked": survival_wm,
                    "survival_free_positions": survival_free,
                    "entropy_bins": bins,
                }
                print(
                    f"  [{gname:>6s} / {aname:>12s}] TPR_post={tpr_post:.2f} "
                    f"z_post={np.mean(z_post):5.2f} "
                    f"surv_wm={survival_wm} surv_free={survival_free}"
                )

    # ---------- 5. Write summary.json and CSV ----------
    summary = {
        "run": f"v3_all_{stamp}",
        "model": args.model,
        "num_states": args.num_states,
        "chain_key": args.chain_key,
        "overlap": args.overlap,
        "n_prompts": args.n_prompts,
        "max_tokens": args.max_tokens,
        "alpha": args.alpha,
        "tau_H": tau_H,
        "tau_Delta": tau_D,
        "baseline_ppl": baseline_ppl,
        "per_gate": per_gate,
        "attacks": attack_results,
        "elapsed_seconds": time.time() - t0,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # CSV
    csv_lines = ["gate,rho,H_mean,Delta_mean,phi,z,tpr,ppl,ppl_ratio_vs_baseline"]
    for gname in ["none", "all", "H_high", "H_low", "Delta"]:
        if gname not in per_gate:
            continue
        g = per_gate[gname]
        ratio = g["ppl"] / baseline_ppl if baseline_ppl else 1.0
        csv_lines.append(
            f"{gname},{g['rho']:.3f},{g['H_mean']:.3f},{g['Delta_mean']:.3f},"
            f"{g['phi']:.3f},{g['z']:.3f},{g['tpr']:.3f},{g['ppl']:.3f},{ratio:.3f}"
        )
    (out_dir / "per_gate_table.csv").write_text("\n".join(csv_lines) + "\n")

    # ---------- 6. Human-readable table ----------
    print("\n" + "=" * 100)
    print(f"v3 FULL PROTOCOL: model={args.model}  N={args.n_prompts}  max_tokens={args.max_tokens}")
    print("=" * 100)
    print(f"{'gate':>6s}  {'rho':>4s}  {'H':>4s}  {'D':>5s}  {'z':>6s}  {'TPR':>4s}  {'PPL':>6s}  {'PPL/base':>8s}")
    for gname in ["none", "all", "H_high", "H_low", "Delta"]:
        if gname not in per_gate:
            continue
        g = per_gate[gname]
        ratio = g["ppl"] / baseline_ppl if baseline_ppl else 1.0
        print(
            f"{gname:>6s}  {g['rho']:>4.2f}  {g['H_mean']:>4.2f}  {g['Delta_mean']:>5.2f}  "
            f"{g['z']:>6.2f}  {g['tpr']:>4.2f}  {g['ppl']:>6.2f}  {ratio:>8.2f}"
        )
    if attack_results:
        print("\nPost-attack TPR:")
        print(f"{'gate':>6s} | " + " | ".join(f"{a:>12s}" for a in ["random_sub10", "random_sub20", "synonym20"]))
        for gname in ["none", "all", "H_high", "H_low", "Delta"]:
            if gname not in attack_results:
                continue
            cells = []
            for aname in ["random_sub10", "random_sub20", "synonym20"]:
                if aname in attack_results[gname]:
                    cells.append(f"{attack_results[gname][aname]['tpr_post_attack']:>12.2f}")
                else:
                    cells.append(f"{'-':>12s}")
            print(f"{gname:>6s} | " + " | ".join(cells))
    print(f"\n[run_all_v3] Done. elapsed={summary['elapsed_seconds']:.1f}s")
    print(f"[run_all_v3] Summary: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
