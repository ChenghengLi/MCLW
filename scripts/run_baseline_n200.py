#!/usr/bin/env python3
"""Non-watermarked baseline at N=200 prompts to match v5_big_N200's
watermarked design.

For each of 173 wiki and 20 factual prompts, generate up to 100 tokens
from google/gemma-3-270m-it with GateNone (no watermarking), at the
same sampling regime used at ungated positions in the watermarked run
(temperature 0.7, no greedy override). Then run the model-free MCL
detector (Algorithm 2 of the paper) on each generation and aggregate.

Outputs data/v5_baseline_N200/summary.json with mean PPL, phi, z,
and FPR at alpha=0.01.

Usage:
    python3 scripts/run_baseline_n200.py
    python3 scripts/run_baseline_n200.py --max-tokens 100 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from ltw_watermark import (
    GatedMCLGenerator,
    GateNone,
    detect,
    load_prompts,
)
from ltw_watermark.factual_prompts import FACTUAL_CLOSED_PROMPTS


def aggregate(records: list[dict], alpha: float) -> dict:
    arr = np.array
    ppl  = arr([r["ppl"]  for r in records])
    phi  = arr([r["phi"]  for r in records])
    z    = arr([r["z"]    for r in records])
    pval = arr([r["pval"] for r in records])
    detected = pval < alpha
    n_tok = arr([r["n_tokens"] for r in records])
    return {
        "n":             int(len(records)),
        "mean_ppl":      float(ppl.mean()),
        "std_ppl":       float(ppl.std(ddof=1)) if len(records) > 1 else 0.0,
        "mean_phi":      float(phi.mean()),
        "mean_z":        float(z.mean()),
        "fpr":           float(detected.mean()),  # = TPR-against-WM = FPR for non-WM
        "mean_n_tokens": float(n_tok.mean()),
        "min_ppl":       float(ppl.min()),
        "max_ppl":       float(ppl.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-270m-it")
    ap.add_argument("--secret-key", default="mclw_v4_2026")
    ap.add_argument("--num-states", type=int, default=5)
    ap.add_argument("--chain-key",  default="clockwork")
    ap.add_argument("--n-wiki",     type=int, default=173)
    ap.add_argument("--n-factual",  type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--alpha",      type=float, default=0.01)
    ap.add_argument("--seed",       type=int,   default=42)
    ap.add_argument("--out-dir",    default=None)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else REPO / "data" / "v5_baseline_N200"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pick best available device on this Mac (MPS) else CPU. Use fp32 on
    # MPS because fp16 logits + multinomial sampling on Apple Silicon can
    # produce inf/nan (the model is only 270M params, easily fits at fp32).
    if torch.backends.mps.is_available():
        device = "mps"; dtype = torch.float32
    elif torch.cuda.is_available():
        device = "cuda"; dtype = torch.float16
    else:
        device = "cpu"; dtype = torch.float32

    print(f"[baseline] device={device} dtype={dtype}")
    gen = GatedMCLGenerator(
        model_name=args.model,
        secret_key=args.secret_key,
        num_states=args.num_states,
        chain_key=args.chain_key,
        overlap_ratio=0.0,
        gate=GateNone(),       # no watermarking
        device=device,
        dtype=dtype,
    )

    wiki = load_prompts(args.n_wiki)
    fact = FACTUAL_CLOSED_PROMPTS[: args.n_factual]
    domains = {"wiki": wiki, "factual": fact}

    domain_records: dict[str, list[dict]] = {"wiki": [], "factual": []}
    t0 = time.time()
    for dom, prompts in domains.items():
        print(f"\n[baseline] domain={dom}  N={len(prompts)}")
        for i, p in enumerate(prompts):
            res = gen.generate(
                p,
                max_new_tokens=args.max_tokens,
                greedy=False,                  # nucleus-style sampling
                temperature=args.temperature,
            )
            det = detect(
                res.token_ids,
                num_states=args.num_states,
                chain_key=args.chain_key,
                secret_key=args.secret_key,
                alpha=args.alpha,
            )
            domain_records[dom].append({
                "ppl":      res.perplexity,
                "phi":      det.phi,
                "z":        det.z,
                "pval":     det.p_value,
                "n_tokens": res.meta["n_tokens"],
            })
            if (i + 1) % 25 == 0 or i == len(prompts) - 1:
                print(f"  [{dom} {i+1}/{len(prompts)}]  "
                      f"ppl={res.perplexity:.2f}  phi={det.phi:.3f}  z={det.z:+.2f}",
                      flush=True)

    elapsed = time.time() - t0
    summary = {
        "run":          "v5_baseline_N200",
        "model":        args.model,
        "n_wiki":       args.n_wiki,
        "n_factual":    args.n_factual,
        "max_tokens":   args.max_tokens,
        "temperature":  args.temperature,
        "num_states":   args.num_states,
        "chain_key":    args.chain_key,
        "secret_key":   args.secret_key,
        "alpha":        args.alpha,
        "seed":         args.seed,
        "elapsed_s":    round(elapsed, 2),
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
        "wiki":         aggregate(domain_records["wiki"], args.alpha),
        "factual":      aggregate(domain_records["factual"], args.alpha),
        "combined":     aggregate(domain_records["wiki"] + domain_records["factual"], args.alpha),
    }

    out = out_dir / "summary.json"
    with out.open("w") as f:
        json.dump(summary, f, indent=2)
    try:
        rel = out.relative_to(REPO)
    except ValueError:
        rel = out
    print(f"\n[baseline] wrote {rel}")
    print("\n=== HEADLINE ===")
    print(f"  combined N={summary['combined']['n']}")
    print(f"  mean PPL  = {summary['combined']['mean_ppl']:.3f}")
    print(f"  mean phi  = {summary['combined']['mean_phi']:.3f} (expected 1/S = {1/args.num_states:.3f})")
    print(f"  mean z    = {summary['combined']['mean_z']:+.3f}")
    print(f"  FPR @α={args.alpha} = {summary['combined']['fpr']:.4f}")
    print(f"  elapsed   = {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
