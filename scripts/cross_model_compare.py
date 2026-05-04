#!/usr/bin/env python
"""
Cross-model, cross-domain comparison report.

Walks every dataset directory under data/ produced by
generate_curated_dataset.py, runs detection at the configured (S, ρ),
and emits a single comparison table covering:

  - Model (Llama-3.2-3B, gemma-3-1b, etc.)
  - Domain (wiki / news / social / abstract)
  - Decoding (greedy / sampling)
  - Sequence length (median tokens)
  - Watermarked detection rate (TPR), avg score, avg PPL
  - Non-watermarked false-positive rate (FPR), avg score
  - Theory baseline (k/S)

Each row is a (dataset_dir, config) pair. Output is a CSV plus a
human-readable Markdown table suitable for pasting into the paper
appendix.

Usage:
    uv run python scripts/cross_model_compare.py \\
        [--data-root data] [--out reports/cross_model.{csv,md}]
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mcl_watermark.enhanced_mcl import EnhancedMCLDetector


def _read_jsonl(p: Path) -> List[dict]:
    out: List[dict] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _infer_domain(samples: List[dict], dirname: str) -> str:
    """Best-effort domain inference. We look at metadata first, then fall
    back to the dataset directory name. Cross-domain runs should record
    the domain in the per-sample metadata; older wiki-only runs do not.
    """
    for s in samples[:5]:
        if "domain" in s:
            return s["domain"]
        if "concept" in s and "_" in str(s["concept"]):
            return "wiki"
        prompt = s.get("prompt", "")
        if "Reddit" in prompt or "X (Twitter)" in prompt or "Mastodon" in prompt:
            return "social"
        if "abstract" in prompt.lower() and "academic" in prompt.lower():
            return "abstract"
        if "wire-style" in prompt or "news brief" in prompt or "BBC-style" in prompt:
            return "news"
    name = dirname.lower()
    for d in ("news", "social", "abstract"):
        if d in name:
            return d
    return "wiki"


def _parse_config_name(cfg: str) -> Optional[Dict[str, float]]:
    if not cfg.startswith("states"):
        return None
    try:
        parts = cfg.split("_")
        S = int(parts[0].replace("states", ""))
        rho = int(parts[1].replace("overlap", "").replace("pct", "")) / 100.0
        return {"num_states": S, "overlap": rho}
    except Exception:
        return None


def _collect_datasets(root: Path) -> List[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir() and any(p.glob("*.jsonl")))


def _summary_meta(d: Path) -> Dict[str, str]:
    s = d / "summary.json"
    if not s.exists():
        return {}
    try:
        with open(s) as f:
            j = json.load(f)
        return {
            "model": j.get("model", ""),
            "secret_key": j.get("secret_key", "curated_wiki_dataset_2024"),
            "chain_key": j.get("configs", [{}])[0].get("chain_key", "soft_cycle")
                          if j.get("configs") else "soft_cycle",
        }
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path(__file__).parent.parent / "data")
    ap.add_argument("--out-csv", type=Path, default=Path(__file__).parent.parent / "reports" / "cross_model.csv")
    ap.add_argument("--out-md",  type=Path, default=Path(__file__).parent.parent / "reports" / "cross_model.md")
    ap.add_argument("--tokenizer", default=os.environ.get("MCL_TOKENIZER", "meta-llama/Llama-3.1-8B-Instruct"),
                    help="Detection tokenizer fallback when summary.json doesn't list a model.")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--configs", nargs="+", default=["states7_overlap0pct"],
                    help="Watermark configs to detect. The non-watermarked file is always read.")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip dataset dirs that already have detection JSON cached.")
    args = ap.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    datasets = _collect_datasets(args.data_root)
    print(f"Found {len(datasets)} dataset directories under {args.data_root}")

    detector_cache: Dict[tuple, EnhancedMCLDetector] = {}

    for d in datasets:
        meta = _summary_meta(d)
        model_id = meta.get("model") or args.tokenizer
        chain_key = meta.get("chain_key", "soft_cycle")
        secret_key = meta.get("secret_key", "curated_wiki_dataset_2024")

        non_wm_path = d / "non_watermarked.jsonl"
        non_wm = _read_jsonl(non_wm_path) if non_wm_path.exists() else []
        domain = _infer_domain(non_wm or [], d.name)
        decoding = (non_wm[0].get("decoding") if non_wm else None) or "unknown"

        for cfg in args.configs:
            wm_path = d / f"{cfg}.jsonl"
            if not wm_path.exists():
                continue
            params = _parse_config_name(cfg)
            if not params:
                continue

            cache_key = (model_id, chain_key, secret_key, params["num_states"], params["overlap"])
            det = detector_cache.get(cache_key)
            if det is None:
                print(f"  loading detector for {model_id} S={params['num_states']} ρ={params['overlap']}")
                det = EnhancedMCLDetector(
                    tokenizer_name=model_id,
                    secret_key=secret_key,
                    num_states=int(params["num_states"]),
                    chain_key=chain_key,
                    overlap_ratio=params["overlap"],
                    detection_threshold=args.threshold,
                )
                detector_cache[cache_key] = det

            wm = _read_jsonl(wm_path)

            wm_scores, wm_detected, wm_ppls, wm_lens = [], 0, [], []
            for s in wm:
                r = det.detect(s["text"])
                wm_scores.append(r.chain_score)
                if r.is_watermarked:
                    wm_detected += 1
                if "perplexity" in s:
                    wm_ppls.append(s["perplexity"])
                wm_lens.append(len(det.tokenizer.encode(s["text"])))

            nwm_scores, nwm_detected = [], 0
            for s in non_wm:
                r = det.detect(s["text"])
                nwm_scores.append(r.chain_score)
                if r.is_watermarked:
                    nwm_detected += 1

            row = {
                "dataset": d.name,
                "model": model_id,
                "domain": domain,
                "decoding": decoding,
                "config": cfg,
                "S": params["num_states"],
                "rho": params["overlap"],
                "n_wm": len(wm),
                "n_nwm": len(non_wm),
                "wm_avg_score": float(np.mean(wm_scores)) if wm_scores else None,
                "wm_tpr": wm_detected / len(wm) if wm else None,
                "wm_avg_ppl": float(np.mean(wm_ppls)) if wm_ppls else None,
                "wm_med_tokens": int(np.median(wm_lens)) if wm_lens else None,
                "nwm_avg_score": float(np.mean(nwm_scores)) if nwm_scores else None,
                "nwm_fpr": nwm_detected / len(non_wm) if non_wm else None,
                "theory_baseline": det.expected_random,
            }
            rows.append(row)
            print(f"    {d.name} | {domain:>8} | {cfg} | "
                  f"wm={row['wm_avg_score']:.3f} ({row['wm_tpr']*100:.0f}%)  "
                  f"nwm={row['nwm_avg_score']:.3f} ({row['nwm_fpr']*100:.1f}% FPR)  "
                  f"med_tok={row['wm_med_tokens']}")

    if not rows:
        print("No rows produced; nothing to write.")
        return

    field_order = [
        "dataset", "model", "domain", "decoding", "config", "S", "rho",
        "n_wm", "n_nwm", "wm_avg_score", "wm_tpr", "wm_avg_ppl",
        "wm_med_tokens", "nwm_avg_score", "nwm_fpr", "theory_baseline",
    ]
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=field_order)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[ok] CSV: {args.out_csv}")

    md_lines = ["# Cross-model / cross-domain MCL detection summary",
                "",
                f"Generated by `scripts/cross_model_compare.py`. "
                f"Detection threshold = {args.threshold}.",
                ""]
    md_lines.append("| Model | Domain | Decoding | S | ρ | n | med tok | WM avg | TPR | NWM avg | FPR | PPL |")
    md_lines.append("|-------|--------|----------|---|---|---|---------|--------|-----|---------|-----|-----|")
    for r in sorted(rows, key=lambda x: (x["model"], x["domain"], x["S"], x["rho"])):
        md_lines.append(
            f"| {r['model'].split('/')[-1]} | {r['domain']} | {r['decoding']} | "
            f"{int(r['S'])} | {r['rho']*100:.0f}% | {r['n_wm']} | "
            f"{r['wm_med_tokens']} | "
            f"{r['wm_avg_score']:.3f} | {r['wm_tpr']*100:.1f}% | "
            f"{r['nwm_avg_score']:.3f} | {r['nwm_fpr']*100:.1f}% | "
            f"{r['wm_avg_ppl']:.2f} |" if r['wm_avg_ppl'] is not None else
            f"| {r['model'].split('/')[-1]} | {r['domain']} | {r['decoding']} | "
            f"{int(r['S'])} | {r['rho']*100:.0f}% | {r['n_wm']} | "
            f"{r['wm_med_tokens']} | "
            f"{r['wm_avg_score']:.3f} | {r['wm_tpr']*100:.1f}% | "
            f"{r['nwm_avg_score']:.3f} | {r['nwm_fpr']*100:.1f}% | n/a |"
        )
    with open(args.out_md, "w") as f:
        f.write("\n".join(md_lines))
    print(f"[ok] MD : {args.out_md}")


if __name__ == "__main__":
    main()
