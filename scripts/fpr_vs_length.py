#!/usr/bin/env python
"""
FPR vs sequence length: empirical validation of Theorem 1's Hoeffding bound.

For each target length n in {25, 50, 100, 200, 400}, we:

  1. Take all non-watermarked samples from the dataset.
  2. Truncate each to n tokens (drop samples shorter than n).
  3. Compute the chain_score and check whether it exceeds threshold tau.
  4. Report empirical FPR and the Hoeffding upper bound
        FPR <= exp(-2 * (n-1) * (tau - k/S)^2)

This is the empirical-vs-theory plot the paper needs to defend against
the i.i.d. assumption critique. Even though tokens are not strictly
i.i.d., if the empirical FPR stays at or below the bound across n, the
bound is a useful operating guarantee.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcl_watermark.enhanced_mcl import EnhancedMCLDetector


DEFAULT_TOKENIZER = os.environ.get("MCL_TOKENIZER", "meta-llama/Llama-3.1-8B-Instruct")


def hoeffding_fpr_bound(n: int, tau: float, k: int, S: int) -> float:
    if n <= 1:
        return 1.0
    margin = tau - k / S
    if margin <= 0:
        return 1.0
    return math.exp(-2 * (n - 1) * margin * margin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--model", default=DEFAULT_TOKENIZER)
    ap.add_argument("--secret-key", default="curated_wiki_dataset_2024")
    ap.add_argument("--lengths", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--states", type=int, nargs="+", default=[5, 7, 9, 11])
    ap.add_argument("--chain-key", default="soft_cycle")
    ap.add_argument("--k", type=int, default=2, help="Valid successors per state (2 for soft_cycle)")
    args = ap.parse_args()

    non_wm_path = args.data_dir / "non_watermarked.jsonl"
    if not non_wm_path.exists():
        sys.exit(f"non_watermarked.jsonl not found in {args.data_dir}")

    samples = []
    with open(non_wm_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"Loaded {len(samples)} non-watermarked samples from {non_wm_path}")

    # Pre-tokenize once with one detector to get token_ids per sample
    base_det = EnhancedMCLDetector(
        tokenizer_name=args.model,
        secret_key=args.secret_key,
        num_states=args.states[0],
        chain_key=args.chain_key,
        overlap_ratio=0.0,
        detection_threshold=args.tau,
    )
    tokenizer = base_det.tokenizer
    token_id_lists = [tokenizer.encode(s["text"]) for s in samples]
    sample_lengths = [len(t) for t in token_id_lists]
    print(f"Token-length stats: min={min(sample_lengths)} median={int(np.median(sample_lengths))} max={max(sample_lengths)}")

    results = []
    for S in args.states:
        det = EnhancedMCLDetector(
            tokenizer_name=args.model,
            secret_key=args.secret_key,
            num_states=S,
            chain_key=args.chain_key,
            overlap_ratio=0.0,
            detection_threshold=args.tau,
        )
        for n in args.lengths:
            usable_token_lists = [tl[:n] for tl in token_id_lists if len(tl) >= n]
            if not usable_token_lists:
                print(f"  S={S} n={n}: no samples of sufficient length, skipping")
                continue
            scores = []
            fp = 0
            for tl in usable_token_lists:
                text = tokenizer.decode(tl, skip_special_tokens=True)
                r = det.detect(text)
                scores.append(r.chain_score)
                if r.chain_score >= args.tau:
                    fp += 1
            emp_fpr = fp / len(usable_token_lists)
            bound = hoeffding_fpr_bound(n=n, tau=args.tau, k=args.k, S=S)
            avg = float(np.mean(scores))
            results.append({
                "num_states": S,
                "n_tokens": n,
                "n_samples": len(usable_token_lists),
                "avg_score": avg,
                "empirical_fpr": emp_fpr,
                "hoeffding_bound": bound,
                "bound_holds": emp_fpr <= bound + 1e-9,
            })
            print(f"  S={S} n={n:>3} : avg={avg:.4f}  emp FPR={emp_fpr:.4f}  bound={bound:.4e}  "
                  f"{'OK' if emp_fpr <= bound + 1e-9 else 'BOUND VIOLATED'}")

    out_path = args.data_dir / "fpr_vs_length.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "secret_key": args.secret_key,
            "tau": args.tau,
            "chain_key": args.chain_key,
            "k": args.k,
            "results": results,
        }, f, indent=2)
    print(f"\n[ok] Results saved to: {out_path}")

    print("\n" + "=" * 80)
    print("SUMMARY: Empirical FPR vs Hoeffding bound")
    print("=" * 80)
    print(f"{'S':>3}  {'n':>4}  {'n_samples':>9}  {'avg':>8}  {'emp FPR':>10}  {'Hoeffding':>14}  status")
    for r in results:
        print(f"{r['num_states']:>3}  {r['n_tokens']:>4}  {r['n_samples']:>9}  "
              f"{r['avg_score']:>8.4f}  {r['empirical_fpr']:>10.4f}  {r['hoeffding_bound']:>14.4e}  "
              f"{'OK' if r['bound_holds'] else 'VIOLATED'}")


if __name__ == "__main__":
    main()
