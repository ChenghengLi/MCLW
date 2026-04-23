#!/usr/bin/env python
"""
Experiment 1 -- Detectability baseline & calibration.

Sweeps (S, tau_H) and for each config:
  - Generates N watermarked texts with entropy-gated MCL
  - Measures empirical rho (= fraction of gated positions)
  - Runs MCL detection on each text
  - Also runs detection on N non-watermarked baseline texts (FPR check)
  - Reports observed z vs predicted z = rho * sqrt((S-1)(n-1)); TPR@FPR=1%

Outputs: data/exp1_<timestamp>/{config}.jsonl + summary.json

Usage:
    uv run python scripts/exp1_calibration.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --n-prompts 200 \
        --max-tokens 200 \
        --states 3 5 7 \
        --tau-quantiles Q25 Q50 Q75
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltw_watermark.entropy_gated_mcl import (
    EntropyGatedMCLGenerator,
    pilot_measure_entropy_quantiles,
)
from ltw_watermark.detection_stats import (
    detect,
    expected_z_H1,
    s_min_calibration,
)


def load_prompts(n: int) -> List[str]:
    # Reuse the curated concept list from the main dataset script.
    from scripts.generate_curated_dataset import WIKIPEDIA_CONCEPTS  # type: ignore
    tmpl = "Explain {c} in a comprehensive way."
    concepts = WIKIPEDIA_CONCEPTS[:n]
    return [tmpl.format(c=c.replace("_", " ")) for c in concepts]


def run_config(
    gen: EntropyGatedMCLGenerator,
    tau_H: float,
    prompts: List[str],
    max_tokens: int,
    alpha: float,
    out_path: Path,
) -> Dict:
    gen.tau_H = tau_H
    records = []
    rhos, z_obs, detected = [], [], 0
    for p in prompts:
        res = gen.generate(p, max_new_tokens=max_tokens, greedy=True)
        n_tok = len(res.token_ids)
        det = detect(
            res.token_ids,
            gen.num_states,
            gen.chain_key,
            gen.secret_key,
            overlap_ratio=gen.overlap_ratio,
            alpha=alpha,
            exact=n_tok < 200,
        )
        records.append({
            "prompt": p,
            "n_tokens": n_tok,
            "rho_empirical": res.rho_empirical,
            "perplexity": res.perplexity,
            "phi": det.phi,
            "z": det.z,
            "p_value": det.p_value,
            "is_watermarked": det.is_watermarked,
        })
        rhos.append(res.rho_empirical)
        z_obs.append(det.z)
        if det.is_watermarked:
            detected += 1

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    mean_rho = float(np.mean(rhos)) if rhos else 0.0
    mean_z_obs = float(np.mean(z_obs)) if z_obs else 0.0
    z_pred = expected_z_H1(mean_rho, gen.num_states, max_tokens)
    s_pred = s_min_calibration(
        z_alpha=abs(float(__import__("scipy.stats", fromlist=["norm"]).norm.ppf(1 - alpha))),
        rho=mean_rho or 1e-6,
        n=max_tokens,
    )

    return {
        "num_states": gen.num_states,
        "chain_key": gen.chain_key,
        "tau_H": tau_H,
        "alpha": alpha,
        "n_prompts": len(prompts),
        "max_tokens": max_tokens,
        "mean_rho": mean_rho,
        "mean_phi": float(np.mean([r["phi"] for r in records])),
        "mean_z_observed": mean_z_obs,
        "z_predicted_corrected": z_pred,
        "s_min_predicted": s_pred,
        "tpr": detected / max(1, len(records)),
        "mean_ppl": float(np.mean([r["perplexity"] for r in records])),
    }


def run_non_watermarked_baseline(
    gen: EntropyGatedMCLGenerator,
    prompts: List[str],
    max_tokens: int,
    alpha: float,
    out_path: Path,
) -> Dict:
    """tau_H = -inf so we never watermark; measures FPR under H_0."""
    return run_config(gen, tau_H=float("-inf"), prompts=prompts, max_tokens=max_tokens, alpha=alpha, out_path=out_path)


def main():
    parser = argparse.ArgumentParser(description="E1: calibration and detectability sweep")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--secret-key", default="mclw_entropy_gated_2026")
    parser.add_argument("--chain-key", default="clockwork")
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--states", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--tau-quantiles", nargs="+", default=["Q25", "Q50", "Q75"])
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--n-pilot", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"data/exp1_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[E1] Writing to {out_dir}")

    prompts = load_prompts(args.n_prompts)
    pilot_prompts = prompts[: args.n_pilot]

    summary = {
        "experiment": "E1_calibration",
        "timestamp": stamp,
        "model": args.model,
        "n_prompts": args.n_prompts,
        "max_tokens": args.max_tokens,
        "alpha": args.alpha,
        "chain_key": args.chain_key,
        "overlap": args.overlap,
        "configs": [],
    }

    # Pilot once per S to pick tau_H values from percentiles of entropy distribution.
    for S in args.states:
        gen = EntropyGatedMCLGenerator(
            model_name=args.model,
            secret_key=args.secret_key,
            num_states=S,
            chain_key=args.chain_key,
            overlap_ratio=args.overlap,
            tau_H=float("inf"),
        )

        print(f"\n[E1][S={S}] Pilot: measuring entropy percentiles on {args.n_pilot} prompts...")
        qs = pilot_measure_entropy_quantiles(gen, pilot_prompts, max_new_tokens=args.max_tokens)
        print(f"[E1][S={S}] Percentiles (nats): {qs}")
        summary_entry_percentiles = qs

        # Non-watermarked baseline: use tau=-inf so MCL never fires; measures H_0 FPR.
        nonwm_path = out_dir / f"nonwm_S{S}.jsonl"
        base = run_non_watermarked_baseline(gen, prompts, args.max_tokens, args.alpha, nonwm_path)
        base["role"] = "non_watermarked"
        base["percentiles"] = summary_entry_percentiles
        summary["configs"].append(base)
        print(f"[E1][S={S}] non-watermarked FPR={base['tpr']:.3f}  mean-z={base['mean_z_observed']:.2f}")

        for qname in args.tau_quantiles:
            tau_H = qs[qname]
            tag = f"S{S}_{qname}_tau{tau_H:.2f}"
            path = out_dir / f"{tag}.jsonl"
            res = run_config(gen, tau_H, prompts, args.max_tokens, args.alpha, path)
            res["role"] = "watermarked"
            res["quantile"] = qname
            res["percentiles"] = summary_entry_percentiles
            summary["configs"].append(res)
            print(
                f"[E1][{tag}] rho={res['mean_rho']:.2f} phi={res['mean_phi']:.3f} "
                f"z_obs={res['mean_z_observed']:.2f} z_pred={res['z_predicted_corrected']:.2f} "
                f"TPR={res['tpr']:.3f} PPL={res['mean_ppl']:.2f}"
            )

        # free VRAM between S values
        del gen
        import gc; import torch
        gc.collect(); torch.cuda.empty_cache() if __import__("torch").cuda.is_available() else None

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[E1] Done. Summary: {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()
