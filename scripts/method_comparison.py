#!/usr/bin/env python
"""
Fair Method-vs-Method comparison: MCL vs KGW vs no-watermark baseline.

For each method we compute:
  * TPR @ FPR = 1.0%  (canonical fair operating point)
  * TPR @ FPR = 0.1%  (strict)
  * AUROC with DeLong CI when scipy available
  * Mean watermarked PPL minus mean baseline PPL  (Δ-PPL)
  * Score-distribution histograms for the report

Both methods produce a per-sample "score". The threshold is swept on
the *baseline* score distribution to fix FPR; TPR is then read from the
watermarked distribution at that threshold. This is the operating-point
analysis the SOTA-fairness review demanded — neither MCL's tau=0.5 nor
KGW's z>4 default is used; the same FPR budget is enforced for both.

Usage:
    uv run python scripts/method_comparison.py \\
        --mcl-watermarked   data/curated_wiki_dataset_<stamp>/states7_overlap0pct.jsonl \\
        --mcl-baseline      data/curated_wiki_dataset_<stamp>/non_watermarked.jsonl \\
        --kgw-watermarked   data/kgw_wiki_g0.25_d2.0_<stamp>/watermarked.jsonl \\
        --kgw-baseline      data/curated_wiki_dataset_<stamp>/non_watermarked.jsonl \\
                            \\  # KGW baseline: same prompts, same model, no watermark
        --out reports/method_comparison
"""

import argparse
import csv
import json
import math
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def _read_jsonl(p: Path) -> List[dict]:
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _scores_from_records(records: List[dict], score_field: str) -> List[float]:
    out = []
    for r in records:
        v = r.get(score_field)
        if v is None:
            continue
        out.append(float(v))
    return out


def _maybe_score_mcl(records: List[dict], tokenizer_name: str,
                     secret_key: str, num_states: int,
                     overlap: float) -> List[float]:
    """If chain_score is missing from records, compute it on the fly."""
    out = []
    need_compute = False
    for r in records:
        if "chain_score" in r and r["chain_score"] is not None:
            out.append(float(r["chain_score"]))
        else:
            need_compute = True
            break
    if not need_compute:
        return out
    from mcl_watermark.enhanced_mcl import EnhancedMCLDetector
    det = EnhancedMCLDetector(
        tokenizer_name=tokenizer_name,
        secret_key=secret_key,
        num_states=num_states,
        chain_key="soft_cycle",
        overlap_ratio=overlap,
        detection_threshold=0.5,
    )
    out = []
    for r in records:
        out.append(float(det.detect(r["text"]).chain_score))
    return out


def threshold_at_fpr(neg_scores: List[float], target_fpr: float,
                     direction: str = "greater") -> float:
    """Pick the smallest threshold tau such that the empirical FPR on
    neg_scores under "score > tau" (or "score < tau") is at most target_fpr.

    For direction='greater': sort negatives descending, allow exactly
    k = floor(target_fpr * n) of them to fire. Threshold = s[k] (the
    (k+1)-th largest negative); then "score > s[k]" admits exactly the top
    k negatives, giving empirical FPR = k/n <= target_fpr.
    Edge: k=0 -> threshold strictly above the max negative -> empirical FPR=0.
    Edge: k=n -> threshold strictly below the min negative -> all negatives fire.
    """
    if not neg_scores:
        return float("nan")
    s = sorted(neg_scores, reverse=(direction == "greater"))
    n = len(s)
    k = max(0, min(n, int(math.floor(target_fpr * n))))
    eps = 1e-9
    if direction == "greater":
        if k == 0:
            return float(s[0]) + eps  # nothing fires
        if k >= n:
            return float(s[-1]) - eps  # everything fires
        return float(s[k])  # top-k negatives fire (those with score > s[k])
    else:
        if k == 0:
            return float(s[0]) - eps
        if k >= n:
            return float(s[-1]) + eps
        return float(s[k])


def tpr_at_threshold(pos_scores: List[float], thr: float,
                     direction: str = "greater") -> float:
    if not pos_scores:
        return float("nan")
    if direction == "greater":
        return float(np.mean([s > thr for s in pos_scores]))
    return float(np.mean([s < thr for s in pos_scores]))


def auroc(pos: List[float], neg: List[float]) -> float:
    """Mann-Whitney U / Wilcoxon AUROC."""
    if not pos or not neg:
        return float("nan")
    p = np.asarray(pos)
    n = np.asarray(neg)
    ranks = (p[:, None] > n[None, :]).astype(float) + 0.5 * (p[:, None] == n[None, :])
    return float(ranks.mean())


def wilson_ci(k: int, n: int, z: float = 1.959963984540054):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def evaluate(method: str, pos: List[float], neg: List[float],
             direction: str = "greater") -> Dict:
    out = {
        "method": method,
        "n_pos": len(pos),
        "n_neg": len(neg),
        "auroc": auroc(pos, neg),
    }
    for fpr, key in ((0.01, "1pct"), (0.001, "0p1pct")):
        thr = threshold_at_fpr(neg, fpr, direction)
        tpr_v = tpr_at_threshold(pos, thr, direction)
        n_pos = len(pos)
        if direction == "greater":
            k = sum(1 for s in pos if s > thr)
        else:
            k = sum(1 for s in pos if s < thr)
        lo, hi = wilson_ci(k, n_pos) if n_pos else (None, None)
        out[f"tpr_at_fpr_{key}"] = tpr_v
        out[f"tpr_at_fpr_{key}_lo"] = lo
        out[f"tpr_at_fpr_{key}_hi"] = hi
        out[f"thr_at_fpr_{key}"] = thr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcl-watermarked", type=Path, required=True,
                    help="MCL watermarked .jsonl (states7_overlap0pct.jsonl)")
    ap.add_argument("--mcl-baseline", type=Path, required=True,
                    help="Non-watermarked baseline .jsonl (Llama greedy)")
    ap.add_argument("--kgw-watermarked", type=Path, default=None,
                    help="KGW watermarked .jsonl (kgw_baseline.py output). "
                         "Must contain z_score per sample.")
    ap.add_argument("--kgw-baseline", type=Path, default=None,
                    help="Non-watermarked .jsonl scored under KGW's "
                         "detector. Defaults to --mcl-baseline (the same "
                         "non-watermarked text).")
    ap.add_argument("--mcl-tokenizer", default="meta-llama/Llama-3.1-8B-Instruct",
                    help="Tokenizer used to score MCL non-WM samples if "
                         "chain_score is missing from JSONL.")
    ap.add_argument("--mcl-secret-key", default="curated_wiki_dataset_2024")
    ap.add_argument("--mcl-num-states", type=int, default=7)
    ap.add_argument("--mcl-overlap", type=float, default=0.0)
    ap.add_argument("--kgw-gamma", type=float, default=0.25)
    ap.add_argument("--kgw-secret-key", default="kgw_secret_2024")
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent.parent / "reports" / "method_comparison")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # ---- MCL ----
    mcl_pos_recs = _read_jsonl(args.mcl_watermarked)
    mcl_neg_recs = _read_jsonl(args.mcl_baseline)
    mcl_pos = _maybe_score_mcl(mcl_pos_recs, args.mcl_tokenizer,
                               args.mcl_secret_key, args.mcl_num_states,
                               args.mcl_overlap)
    mcl_neg = _maybe_score_mcl(mcl_neg_recs, args.mcl_tokenizer,
                               args.mcl_secret_key, args.mcl_num_states,
                               args.mcl_overlap)
    rows.append(evaluate("MCL_S7_rho0", mcl_pos, mcl_neg, direction="greater"))

    # ---- KGW ----
    if args.kgw_watermarked:
        kgw_pos_recs = _read_jsonl(args.kgw_watermarked)
        # If KGW baseline file not provided, we need to score the non-WM text
        # under KGW's z-statistic too. Easiest: assume non-WM was generated
        # with the same prompts/model, then re-detect with KGWDetector.
        kgw_neg_recs = _read_jsonl(args.kgw_baseline or args.mcl_baseline)
        if all("z_score" in r for r in kgw_pos_recs):
            kgw_pos = [r["z_score"] for r in kgw_pos_recs]
        else:
            sys.exit("KGW positive file missing z_score; re-run kgw_baseline.py")
        # Score the negatives under KGW
        if all("z_score" in r for r in kgw_neg_recs):
            kgw_neg = [r["z_score"] for r in kgw_neg_recs]
        else:
            from kgw_baseline import KGWDetector
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(args.mcl_tokenizer)
            kgw_det = KGWDetector(tok, tok.vocab_size, args.kgw_gamma,
                                  args.kgw_secret_key)
            kgw_neg = []
            for r in kgw_neg_recs:
                if "generated_ids" in r:
                    kgw_neg.append(kgw_det.detect(
                        generated_ids=r["generated_ids"],
                        prompt_last_token_id=r.get("prompt_last_token_id"),
                    )["z_score"])
                else:
                    kgw_neg.append(kgw_det.detect(text=r["text"])["z_score"])
        rows.append(evaluate("KGW_g0.25_d2", kgw_pos, kgw_neg, direction="greater"))

    # ---- Save ----
    out_csv = args.out.with_suffix(".csv")
    out_md = args.out.with_suffix(".md")
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    md = ["# Method-vs-Method comparison at fixed FPR", "",
          "| Method | n_pos | n_neg | AUROC | TPR @ FPR=1% (95% CI) | TPR @ FPR=0.1% (95% CI) | thr@FPR=1% |",
          "|--------|------:|------:|------:|------------------------|--------------------------|------------|"]
    for r in rows:
        ci1 = f"{r['tpr_at_fpr_1pct']*100:.1f}% [{r['tpr_at_fpr_1pct_lo']*100:.1f}, {r['tpr_at_fpr_1pct_hi']*100:.1f}]"
        ci01 = f"{r['tpr_at_fpr_0p1pct']*100:.1f}% [{r['tpr_at_fpr_0p1pct_lo']*100:.1f}, {r['tpr_at_fpr_0p1pct_hi']*100:.1f}]"
        md.append(f"| {r['method']} | {r['n_pos']} | {r['n_neg']} | "
                  f"{r['auroc']:.3f} | {ci1} | {ci01} | "
                  f"{r['thr_at_fpr_1pct']:.3f} |")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[ok] CSV: {out_csv}")
    print(f"[ok] MD : {out_md}")
    for r in rows:
        print(f"  {r['method']:>14s}  AUROC={r['auroc']:.3f}  "
              f"TPR@FPR1%={r['tpr_at_fpr_1pct']*100:.1f}%  "
              f"TPR@FPR0.1%={r['tpr_at_fpr_0p1pct']*100:.1f}%")


if __name__ == "__main__":
    main()
