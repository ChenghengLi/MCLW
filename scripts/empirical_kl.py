#!/usr/bin/env python
"""
Aggregate per-step empirical D_KL(P_MCL || P_M) emitted by EnhancedMCLGenerator
and compare against Theorem 4's log(S) upper bound.

Usage:
    uv run python scripts/empirical_kl.py [--data-root data]

For each `data/curated_wiki_dataset_*` directory, walks all watermarked
JSONL files, extracts `empirical_mean_kl_nats` and `log_S_bound_nats`
from each sample, and emits a CSV + Markdown table grouped by (S, ρ,
domain).
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np


def _read_jsonl(p: Path) -> List[dict]:
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _infer_domain(samples: List[dict], dirname: str) -> str:
    for s in samples[:5]:
        if "domain" in s:
            return s["domain"]
        if "concept" in s and "_" in str(s["concept"]):
            return "wiki"
        prompt = s.get("prompt", "")
        if "Reddit" in prompt or "X (Twitter)" in prompt:
            return "social"
        if "abstract" in prompt.lower() and "academic" in prompt.lower():
            return "abstract"
        if "BBC" in prompt or "wire-style" in prompt:
            return "news"
    name = dirname.lower()
    for d in ("news", "social", "abstract"):
        if d in name:
            return d
    return "wiki"


def _parse_config_name(cfg: str):
    parts = cfg.split("_")
    return {
        "S": int(parts[0].replace("states", "")),
        "rho": int(parts[1].replace("overlap", "").replace("pct", "")) / 100.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=Path(__file__).parent.parent / "data")
    ap.add_argument("--out-csv", type=Path,
                    default=Path(__file__).parent.parent / "reports" / "empirical_kl.csv")
    ap.add_argument("--out-md", type=Path,
                    default=Path(__file__).parent.parent / "reports" / "empirical_kl.md")
    args = ap.parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in sorted(p for p in args.data_root.iterdir() if p.is_dir()):
        nwm_path = d / "non_watermarked.jsonl"
        nwm = _read_jsonl(nwm_path) if nwm_path.exists() else []
        domain = _infer_domain(nwm, d.name)
        for jp in sorted(d.glob("states*_overlap*pct.jsonl")):
            recs = _read_jsonl(jp)
            kls = [r.get("empirical_mean_kl_nats") for r in recs
                   if r.get("empirical_mean_kl_nats") is not None]
            log_s = [r.get("log_S_bound_nats") for r in recs
                     if r.get("log_S_bound_nats") is not None]
            if not kls:
                continue
            cfg = _parse_config_name(jp.stem)
            arr = np.asarray(kls)
            row = {
                "dataset": d.name,
                "domain": domain,
                "config": jp.stem,
                "S": cfg["S"],
                "rho": cfg["rho"],
                "n": len(kls),
                "mean_kl_nats": float(arr.mean()),
                "median_kl_nats": float(np.median(arr)),
                "p95_kl_nats": float(np.quantile(arr, 0.95)),
                "max_kl_nats": float(arr.max()),
                "log_S_bound_nats": float(log_s[0]) if log_s else math.log(cfg["S"]),
                "tightness": float(arr.mean() / (math.log(cfg["S"]) + 1e-9)),
            }
            rows.append(row)

    if not rows:
        print("No rows produced — re-generate datasets with the updated "
              "EnhancedMCLGenerator that logs empirical_mean_kl_nats.")
        return

    fieldnames = list(rows[0].keys())
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    md = ["# Empirical KL vs Theorem 4 bound (D_KL ≤ log S)", "",
          "| Dataset | Domain | S | ρ | n | mean D_KL (nats) | p95 | log S (bound) | mean / bound |",
          "|---------|--------|---|---|---|------------------|-----|---------------|--------------|"]
    for r in sorted(rows, key=lambda x: (x["S"], x["rho"], x["domain"])):
        md.append(
            f"| {r['dataset']} | {r['domain']} | {r['S']} | {r['rho']*100:.0f}% | "
            f"{r['n']} | {r['mean_kl_nats']:.3f} | {r['p95_kl_nats']:.3f} | "
            f"{r['log_S_bound_nats']:.3f} | {r['tightness']:.2f}x |"
        )
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"[ok] CSV: {args.out_csv}")
    print(f"[ok] MD : {args.out_md}")


if __name__ == "__main__":
    main()
