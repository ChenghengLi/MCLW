#!/usr/bin/env python
"""
Statistical-rigor report for MCL Watermarking results (T1.1 + T1.2).

For each `data/curated_wiki_dataset_*` directory present, computes:

  * Wilson 95% CI for TPR (watermarked detection rate)
  * Wilson 95% CI for FPR (non-watermarked false-positive rate)
  * Clopper-Pearson exact CI for both
  * Rule-of-three FNR upper bound when TPR == 100%
  * Paired log-PPL test (paired t + Wilcoxon signed-rank + bootstrap CI)
    between watermarked and non-watermarked samples that share a `concept`
    or `prompt`. Only runs when both files exist for the same dataset.

Output: one Markdown table per dataset + a flat CSV under reports/.

CPU-only. No transformers loaded — uses the per-sample `chain_score` and
`perplexity` already cached inside the .jsonl files (or recomputes
detection scores via EnhancedMCLDetector if they are absent).
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

try:
    from scipy import stats as _scipy_stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ---------- Confidence intervals -------------------------------------------------

def wilson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    z = 1.959963984540054 if alpha == 0.05 else -math.sqrt(2) * _erfcinv(alpha)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _erfcinv(x: float) -> float:
    return -_scipy_stats.norm.ppf(x / 2) if HAVE_SCIPY else 0.0


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact Clopper-Pearson 95% CI."""
    if not HAVE_SCIPY:
        return wilson_ci(k, n, alpha)
    if k == 0:
        lower = 0.0
    else:
        lower = _scipy_stats.beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        upper = 1.0
    else:
        upper = _scipy_stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lower), float(upper))


def rule_of_three_fnr(n: int) -> float:
    """Upper 95% bound on the failure rate when 0 failures observed in n trials."""
    return 3.0 / n if n > 0 else 1.0


# ---------- Loading helpers ------------------------------------------------------

def _read_jsonl(p: Path) -> List[dict]:
    out = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _scores_and_decisions(records: List[dict], threshold: float = 0.5):
    """Pull cached scores; return (scores, decisions, count_above_threshold)."""
    scores = []
    above = 0
    for r in records:
        s = r.get("chain_score", r.get("score"))
        if s is None:
            return None, None, None
        scores.append(float(s))
        if s >= threshold:
            above += 1
    return scores, [s >= threshold for s in scores], above


# ---------- Paired log-PPL -------------------------------------------------------

def _index_by_concept(records: List[dict]) -> Dict[str, dict]:
    out = {}
    for r in records:
        key = r.get("concept") or r.get("prompt") or r.get("text", "")[:64]
        if key and key not in out:
            out[key] = r
    return out


def paired_log_ppl(wm: List[dict], nwm: List[dict]) -> Optional[Dict]:
    if not wm or not nwm:
        return None
    w_idx = _index_by_concept(wm)
    n_idx = _index_by_concept(nwm)
    keys = sorted(set(w_idx) & set(n_idx))
    if not keys:
        return None
    log_w, log_n = [], []
    for k in keys:
        pw = w_idx[k].get("perplexity")
        pn = n_idx[k].get("perplexity")
        if pw is None or pn is None or pw <= 0 or pn <= 0:
            continue
        log_w.append(math.log(pw))
        log_n.append(math.log(pn))
    if len(log_w) < 5:
        return None
    log_w = np.array(log_w)
    log_n = np.array(log_n)
    diff = log_w - log_n  # positive = watermark inflates PPL
    mean_diff = float(np.mean(diff))
    n = len(diff)
    out = {
        "n_paired": n,
        "mean_log_ppl_wm": float(np.mean(log_w)),
        "mean_log_ppl_nwm": float(np.mean(log_n)),
        "mean_delta_log_ppl": mean_diff,
        "median_delta_log_ppl": float(np.median(diff)),
        "geomean_relative_inflation_pct": float((math.exp(mean_diff) - 1) * 100),
    }
    if HAVE_SCIPY:
        t, p = _scipy_stats.ttest_rel(log_w, log_n)
        wstat, wpval = _scipy_stats.wilcoxon(log_w, log_n) if np.any(diff != 0) else (np.nan, np.nan)
        out["paired_t"] = float(t)
        out["paired_t_pvalue"] = float(p)
        out["wilcoxon_stat"] = float(wstat) if not math.isnan(wstat) else None
        out["wilcoxon_pvalue"] = float(wpval) if not math.isnan(wpval) else None
        out["cohen_dz"] = float(mean_diff / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else None
    # Bootstrap 95% CI on mean Δlog-PPL
    rng = np.random.default_rng(42)
    boots = []
    for _ in range(10_000):
        sample = rng.choice(diff, size=n, replace=True)
        boots.append(float(np.mean(sample)))
    boots.sort()
    out["bootstrap_ci_lo"] = boots[int(0.025 * len(boots))]
    out["bootstrap_ci_hi"] = boots[int(0.975 * len(boots))]
    return out


# ---------- Per-dataset summary --------------------------------------------------

def summarise_dataset(d: Path, configs: List[str], threshold: float = 0.5) -> List[Dict]:
    rows = []
    nwm_path = d / "non_watermarked.jsonl"
    nwm_records = _read_jsonl(nwm_path) if nwm_path.exists() else []

    for cfg in configs:
        wm_path = d / f"{cfg}.jsonl"
        if not wm_path.exists():
            continue
        wm_records = _read_jsonl(wm_path)
        n_wm = len(wm_records)
        n_nwm = len(nwm_records)

        ws, _, w_above = _scores_and_decisions(wm_records, threshold) if n_wm else (None, None, 0)
        ns, _, n_above = _scores_and_decisions(nwm_records, threshold) if n_nwm else (None, None, 0)
        if ws is None:
            print(f"  [skip] {d.name}/{cfg}: missing chain_score in jsonl")
            continue

        tpr = w_above / n_wm if n_wm else None
        fpr = (n_above / n_nwm) if (n_nwm and n_above is not None) else None
        wci = wilson_ci(w_above, n_wm) if n_wm else (None, None)
        cci = clopper_pearson_ci(w_above, n_wm) if n_wm else (None, None)
        wci_n = wilson_ci(n_above, n_nwm) if (n_nwm and n_above is not None) else (None, None)
        cci_n = clopper_pearson_ci(n_above, n_nwm) if (n_nwm and n_above is not None) else (None, None)
        fnr_ub = rule_of_three_fnr(n_wm) if (tpr == 1.0 and n_wm) else None
        fpr_ub = rule_of_three_fnr(n_nwm) if (fpr == 0.0 and n_nwm) else None

        ppl_paired = paired_log_ppl(wm_records, nwm_records)

        row = {
            "dataset": d.name,
            "config": cfg,
            "n_wm": n_wm,
            "n_nwm": n_nwm,
            "tpr": tpr,
            "tpr_wilson_lo": wci[0],
            "tpr_wilson_hi": wci[1],
            "tpr_clopper_lo": cci[0],
            "tpr_clopper_hi": cci[1],
            "fnr_upper_95": fnr_ub,
            "fpr": fpr,
            "fpr_wilson_lo": wci_n[0],
            "fpr_wilson_hi": wci_n[1],
            "fpr_clopper_lo": cci_n[0],
            "fpr_clopper_hi": cci_n[1],
            "fpr_upper_95": fpr_ub,
            "wm_score_mean": float(np.mean(ws)) if ws else None,
            "nwm_score_mean": float(np.mean(ns)) if ns else None,
        }
        if ppl_paired:
            for k, v in ppl_paired.items():
                row[f"ppl_{k}"] = v
        rows.append(row)
    return rows


def _fmt_pct_ci(p: float, lo: float, hi: float) -> str:
    if p is None or lo is None or hi is None:
        return "—"
    return f"{p*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"


def _markdown(rows: List[Dict]) -> str:
    if not rows:
        return "_(no rows produced)_"
    lines = ["| Dataset | Config | n_wm | n_nwm | TPR (95% CI) | FPR (95% CI) | Δlog-PPL | rel ΔPPL | paired t p |",
             "|---------|--------|-----:|------:|---------------|---------------|----------|----------|------------|"]
    for r in rows:
        delta = r.get("ppl_mean_delta_log_ppl")
        ci_lo = r.get("ppl_bootstrap_ci_lo")
        ci_hi = r.get("ppl_bootstrap_ci_hi")
        rel = r.get("ppl_geomean_relative_inflation_pct")
        p = r.get("ppl_paired_t_pvalue")
        delta_s = (f"{delta:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}]"
                   if delta is not None else "—")
        rel_s = f"{rel:+.1f}%" if rel is not None else "—"
        p_s = f"{p:.2e}" if p is not None else "—"
        lines.append(
            f"| {r['dataset']} | {r['config']} | {r['n_wm']} | {r['n_nwm']} | "
            f"{_fmt_pct_ci(r['tpr'], r['tpr_wilson_lo'], r['tpr_wilson_hi'])} | "
            f"{_fmt_pct_ci(r['fpr'], r['fpr_wilson_lo'], r['fpr_wilson_hi'])} | "
            f"{delta_s} | {rel_s} | {p_s} |"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=Path(__file__).parent.parent / "data")
    ap.add_argument("--out-csv", type=Path,
                    default=Path(__file__).parent.parent / "reports" / "stats_report.csv")
    ap.add_argument("--out-md", type=Path,
                    default=Path(__file__).parent.parent / "reports" / "stats_report.md")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--configs", nargs="+",
                    default=["states7_overlap0pct"],
                    help="Watermark configs to summarise. Default: just S=7 ρ=0%.")
    ap.add_argument("--all-configs", action="store_true",
                    help="Override --configs and use every states*_overlap*pct file found.")
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    if not HAVE_SCIPY:
        print("[warn] scipy not available; falling back to Wilson-only CIs and "
              "skipping paired t-test / Wilcoxon. Install scipy for the full report.")

    datasets = sorted(p for p in args.data_root.iterdir()
                      if p.is_dir() and any(p.glob("*.jsonl")))
    print(f"Found {len(datasets)} dataset directories under {args.data_root}")

    all_rows = []
    for d in datasets:
        if args.all_configs:
            cfgs = sorted({p.stem for p in d.glob("states*_overlap*pct.jsonl")})
        else:
            cfgs = args.configs
        rows = summarise_dataset(d, cfgs, threshold=args.threshold)
        if rows:
            print(f"\n{d.name}:")
            for r in rows:
                print(f"  {r['config']:<26}  n_wm={r['n_wm']:<4}  "
                      f"TPR={_fmt_pct_ci(r['tpr'], r['tpr_wilson_lo'], r['tpr_wilson_hi'])}  "
                      f"FPR={_fmt_pct_ci(r['fpr'], r['fpr_wilson_lo'], r['fpr_wilson_hi'])}")
                if r.get("ppl_mean_delta_log_ppl") is not None:
                    print(f"      Δlog-PPL = {r['ppl_mean_delta_log_ppl']:+.3f} "
                          f"[{r['ppl_bootstrap_ci_lo']:+.3f}, {r['ppl_bootstrap_ci_hi']:+.3f}]"
                          f" (paired t p={r.get('ppl_paired_t_pvalue', float('nan')):.2e})")
        all_rows.extend(rows)

    if not all_rows:
        print("No rows produced.")
        return

    # CSV
    fieldnames = sorted({k for r in all_rows for k in r.keys()})
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    # Markdown
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# MCL Watermarking — statistical rigour report\n\n")
        f.write("Wilson 95% CIs on TPR/FPR; paired log-PPL t-test + bootstrap CI on "
                "Δlog-PPL between watermarked and non-watermarked samples that share "
                "a `concept`. Generated by `scripts/stats_report.py`.\n\n")
        f.write(_markdown(all_rows))
        f.write("\n")
    print(f"\n[ok] CSV: {args.out_csv}")
    print(f"[ok] MD : {args.out_md}")


if __name__ == "__main__":
    main()
