#!/usr/bin/env python
"""
Tokenizer-drift detection (T1.3 from EXPERIMENT_PLAN.md).

Re-detects existing watermarked text using a *different* tokenizer than the
one that generated it. The state assignment SHA-256(secret || token_id) is
keyed on the integer token id, so any tokenizer that yields a different id
sequence for the same UTF-8 string should produce a φ score that collapses
to the random baseline k/S = 2/7 ≈ 0.286 (for soft-cycle, S=7).

Why this matters:
  The paper's Limitations enumerate now says "tokenizer-only detection";
  reviewers will ask whether *any* tokenizer suffices or only the one used
  at generation. This experiment makes the answer numeric.

Usage:
    uv run python scripts/tokenizer_drift.py \\
        --data-dir data/curated_wiki_dataset_<stamp> \\
        --config states7_overlap0pct \\
        --tokenizers \\
            meta-llama/Llama-3.1-8B-Instruct \\
            mistralai/Mistral-7B-Instruct-v0.3 \\
            gpt2-xl

Output: reports/tokenizer_drift.{csv,md} plus per-tokenizer JSON next to
the dataset.
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mcl_watermark.enhanced_mcl import EnhancedMCLDetector


def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _read_jsonl(p: Path) -> List[dict]:
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _parse_config_name(cfg: str):
    parts = cfg.split("_")
    return {
        "num_states": int(parts[0].replace("states", "")),
        "overlap": int(parts[1].replace("overlap", "").replace("pct", "")) / 100.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Path to a curated_wiki_dataset_* directory.")
    ap.add_argument("--config", default="states7_overlap0pct")
    ap.add_argument("--tokenizers", nargs="+",
                    default=["meta-llama/Llama-3.1-8B-Instruct"],
                    help="Tokenizer model ids to evaluate. The first one is "
                         "expected to be the same family as the generator and "
                         "should give the same score as the original run.")
    ap.add_argument("--secret-key", default="curated_wiki_dataset_2024")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out-csv", type=Path,
                    default=Path(__file__).parent.parent / "reports" / "tokenizer_drift.csv")
    ap.add_argument("--out-md", type=Path,
                    default=Path(__file__).parent.parent / "reports" / "tokenizer_drift.md")
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    cfg = _parse_config_name(args.config)
    wm_path = args.data_dir / f"{args.config}.jsonl"
    if not wm_path.exists():
        sys.exit(f"Watermarked file not found: {wm_path}")
    wm_records = _read_jsonl(wm_path)
    print(f"Loaded {len(wm_records)} watermarked samples from {wm_path}")

    nwm_path = args.data_dir / "non_watermarked.jsonl"
    nwm_records = _read_jsonl(nwm_path) if nwm_path.exists() else []
    if nwm_records:
        print(f"Loaded {len(nwm_records)} non-watermarked samples from {nwm_path}")

    rows = []
    for tok_id in args.tokenizers:
        print(f"\n=== Tokenizer: {tok_id} ===")
        try:
            det = EnhancedMCLDetector(
                tokenizer_name=tok_id,
                secret_key=args.secret_key,
                num_states=cfg["num_states"],
                chain_key="soft_cycle",
                overlap_ratio=cfg["overlap"],
                detection_threshold=args.threshold,
            )
        except Exception as e:
            print(f"  [skip] could not load tokenizer: {e}")
            continue

        wm_scores, wm_above = [], 0
        for r in wm_records:
            res = det.detect(r["text"])
            wm_scores.append(res.chain_score)
            if res.is_watermarked:
                wm_above += 1
        nwm_scores, nwm_above = [], 0
        for r in nwm_records:
            res = det.detect(r["text"])
            nwm_scores.append(res.chain_score)
            if res.is_watermarked:
                nwm_above += 1

        wm_n = len(wm_records)
        nwm_n = len(nwm_records)
        tpr = wm_above / wm_n if wm_n else None
        fpr = nwm_above / nwm_n if nwm_n else None
        wci = wilson_ci(wm_above, wm_n)
        nci = wilson_ci(nwm_above, nwm_n) if nwm_n else (None, None)

        row = {
            "tokenizer": tok_id,
            "S": cfg["num_states"],
            "rho": cfg["overlap"],
            "n_wm": wm_n,
            "n_nwm": nwm_n,
            "tpr": tpr,
            "tpr_lo": wci[0],
            "tpr_hi": wci[1],
            "fpr": fpr,
            "fpr_lo": nci[0],
            "fpr_hi": nci[1],
            "wm_score_mean": float(np.mean(wm_scores)),
            "nwm_score_mean": float(np.mean(nwm_scores)) if nwm_scores else None,
            "expected_random_baseline": det.expected_random,
        }
        rows.append(row)
        print(f"  TPR={tpr*100:.1f}% [{wci[0]*100:.1f}, {wci[1]*100:.1f}]   "
              f"avg phi_wm={row['wm_score_mean']:.4f}   "
              f"baseline k/S={det.expected_random:.4f}")
        if nwm_n:
            print(f"  FPR={fpr*100:.1f}% [{nci[0]*100:.1f}, {nci[1]*100:.1f}]   "
                  f"avg phi_nwm={row['nwm_score_mean']:.4f}")
        del det

    if not rows:
        sys.exit("No tokenizers succeeded.")

    # CSV
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] CSV: {args.out_csv}")

    # Markdown
    md = ["# Tokenizer drift — detection under non-generator tokenizers", "",
          f"Source dataset: `{args.data_dir.name}`, config `{args.config}`, "
          f"S={cfg['num_states']} ρ={cfg['overlap']*100:.0f}%, threshold τ="
          f"{args.threshold:.2f}.", "",
          "| Tokenizer | n_wm | n_nwm | avg φ_wm | TPR (95% CI) | avg φ_nwm | FPR (95% CI) | k/S baseline |",
          "|-----------|-----:|------:|---------:|---------------|----------:|---------------|-------------:|"]
    for r in rows:
        tpr_s = f"{r['tpr']*100:.1f}% [{r['tpr_lo']*100:.1f}, {r['tpr_hi']*100:.1f}]"
        if r["fpr"] is not None:
            fpr_s = f"{r['fpr']*100:.1f}% [{r['fpr_lo']*100:.1f}, {r['fpr_hi']*100:.1f}]"
            nwm_avg_s = f"{r['nwm_score_mean']:.4f}"
        else:
            fpr_s = "—"
            nwm_avg_s = "—"
        md.append(
            f"| `{r['tokenizer']}` | {r['n_wm']} | {r['n_nwm']} | "
            f"{r['wm_score_mean']:.4f} | {tpr_s} | {nwm_avg_s} | {fpr_s} | "
            f"{r['expected_random_baseline']:.4f} |"
        )
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[ok] MD : {args.out_md}")


if __name__ == "__main__":
    main()
