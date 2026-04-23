#!/usr/bin/env python
"""
End-to-end smoke test with Gemma 3 270M (https://developers.googleblog.com/en/introducing-gemma-3-270m/).

Runs one short generation with each gate (all / H_high / H_low / Delta) and
reports rho, mean H, mean Delta, PPL proxy, and detection z-score.

Tiny model (~270M params) so the whole thing runs in ~a minute even on CPU.

Usage:
    uv run python scripts/demo_gemma3_270m.py
    uv run python scripts/demo_gemma3_270m.py --model google/gemma-3-270m
    uv run python scripts/demo_gemma3_270m.py --prompt "Explain entropy in one paragraph."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltw_watermark import (
    GatedMCLGenerator,
    GateAll,
    GateEntropyHigh,
    GateEntropyLow,
    GateDelta,
    detect,
)


def summarize(res, det) -> str:
    H = np.array([p.entropy for p in res.positions])
    D = np.array([p.delta for p in res.positions])
    rho = res.rho_empirical
    return (
        f"gate={res.gate_name:>7s}  n={len(res.positions):3d}  "
        f"rho={rho:.2f}  mean_H={H.mean():.2f}  mean_Delta={D.mean():.2f}  "
        f"PPL={res.perplexity:6.2f}  "
        f"phi={det.phi:.3f}  z={det.z:5.2f}  p={det.p_value:.1e}  "
        f"detected={det.is_watermarked}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-270m-it",
                        help="HF model ID. Try google/gemma-3-270m for base.")
    parser.add_argument("--num-states", type=int, default=5)
    parser.add_argument("--chain-key", default="clockwork")
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--tau-H", type=float, default=1.5,
                        help="Entropy threshold (nats).")
    parser.add_argument("--tau-delta", type=float, default=0.3,
                        help="Surprisal-gap threshold.")
    parser.add_argument("--prompt", default="Explain photosynthesis in simple terms.")
    parser.add_argument("--secret-key", default="mclw_gemma3_smoke_test")
    args = parser.parse_args()

    print(f"[demo] model={args.model}  device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"[demo] prompt: {args.prompt!r}")
    print(f"[demo] MCL: S={args.num_states} chain={args.chain_key} overlap={args.overlap}")
    print()

    gates = [
        ("all",    GateAll()),
        ("H_high", GateEntropyHigh(tau=args.tau_H)),
        ("H_low",  GateEntropyLow(tau=args.tau_H)),
        ("Delta",  GateDelta(tau=args.tau_delta)),
    ]

    # Load the generator ONCE; swap the gate each time (avoids re-loading model).
    gen = GatedMCLGenerator(
        model_name=args.model,
        secret_key=args.secret_key,
        num_states=args.num_states,
        chain_key=args.chain_key,
        overlap_ratio=args.overlap,
        gate=gates[0][1],
    )

    print("-" * 100)
    for name, gate in gates:
        gen.gate = gate
        res = gen.generate(args.prompt, max_new_tokens=args.max_tokens, greedy=True)
        det = detect(
            res.token_ids,
            num_states=args.num_states,
            chain_key=args.chain_key,
            secret_key=args.secret_key,
            overlap_ratio=args.overlap,
            alpha=0.01,
            exact=(len(res.token_ids) < 200),
        )
        print(summarize(res, det))
        print(f"  sample: {res.text[:140]!r}")
        print()

    # ---- Control: run detection on a non-watermarked generation ----
    print("-" * 100)
    print("[control] Same model, no watermark (GateNone):")
    from ltw_watermark import GateNone
    gen.gate = GateNone()
    res = gen.generate(args.prompt, max_new_tokens=args.max_tokens, greedy=True)
    det = detect(
        res.token_ids,
        num_states=args.num_states,
        chain_key=args.chain_key,
        secret_key=args.secret_key,
        overlap_ratio=args.overlap,
        alpha=0.01,
        exact=True,
    )
    print(summarize(res, det))
    print(f"  sample: {res.text[:140]!r}")


if __name__ == "__main__":
    main()
