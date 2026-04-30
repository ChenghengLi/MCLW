#!/usr/bin/env python
"""
SHA-256 per-token overhead micro-benchmark.

The MCL pipeline calls SHA-256 once per vocabulary token at startup
(state assignment is precomputed) and zero times per token at generation
or detection (lookups go through the precomputed mask). This script
measures both:

  (a) Wall-clock cost of one SHA-256(secret || token_id) operation, in
      isolation, on the host CPU.
  (b) Wall-clock cost of building the full state-assignment table for a
      Llama-3-class vocabulary (128k tokens) at startup.

Output is reported as ns/token and as a percentage of typical 3B-class
LLM token generation latency (~10-30 ms/token on a single A100 in fp16,
much higher on CPU). Use --gen-latency-ms to override the latency
denominator with a measured value from your hardware.
"""

import argparse
import hashlib
import time


def bench_one(secret: bytes, n_iters: int = 200_000) -> float:
    """Return ns per single SHA-256(secret || token_id_bytes) call."""
    t0 = time.perf_counter_ns()
    for i in range(n_iters):
        hashlib.sha256(secret + i.to_bytes(4, "little")).digest()
    t1 = time.perf_counter_ns()
    return (t1 - t0) / n_iters


def bench_table(secret: bytes, vocab_size: int) -> float:
    """Return wall-clock seconds to build the full state table once."""
    t0 = time.perf_counter()
    table = [int.from_bytes(hashlib.sha256(secret + i.to_bytes(4, "little")).digest()[:4], "little")
             for i in range(vocab_size)]
    t1 = time.perf_counter()
    assert len(table) == vocab_size
    return t1 - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--secret", default="curated_wiki_dataset_2024")
    parser.add_argument("--vocab-size", type=int, default=128_256,
                        help="Llama-3 tokenizer default vocab size")
    parser.add_argument("--iters", type=int, default=200_000,
                        help="Iterations for the per-call micro-benchmark")
    parser.add_argument("--gen-latency-ms", type=float, default=20.0,
                        help="Reference per-token LLM generation latency (ms) "
                             "for the % overhead estimate. Override with measured "
                             "value from your hardware.")
    args = parser.parse_args()

    secret = args.secret.encode()

    print("SHA-256 per-token cost")
    print("-" * 40)
    ns_per_call = bench_one(secret, args.iters)
    print(f"per call (n={args.iters}): {ns_per_call:8.1f} ns")

    print("\nFull state-table build (one-time cost)")
    print("-" * 40)
    secs = bench_table(secret, args.vocab_size)
    print(f"vocab = {args.vocab_size}: {secs*1000:.2f} ms total "
          f"({secs/args.vocab_size*1e9:.1f} ns/token amortised)")

    print("\nRelative cost vs LLM generation")
    print("-" * 40)
    gen_ns = args.gen_latency_ms * 1e6
    pct_per_call = ns_per_call / gen_ns * 100
    print(f"reference per-token LLM latency: {args.gen_latency_ms:.1f} ms")
    print(f"hash cost per generated token (worst case, 1 call): "
          f"{pct_per_call:.4f}% of LLM step time")
    print(f"hash cost per generated token (amortised: 1 table build of "
          f"{args.vocab_size} tokens, reused for the entire run): negligible")
    print()
    print("Headline: SHA-256 is not on the critical path of MCL. The state "
          "table is built once at startup and reused; per-token cost is "
          "dominated by the language-model forward pass.")


if __name__ == "__main__":
    main()
