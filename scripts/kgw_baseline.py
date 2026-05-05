#!/usr/bin/env python
"""
Kirchenbauer-2023 (KGW) "green-list" watermark — head-to-head baseline.

Self-contained reimplementation (~100 LoC) of the hard green-list scheme from
"A Watermark for Large Language Models" (ICML 2023). Avoids depending on an
external repo so we can run on the same protocol as MCL.

  At each step, partition the vocabulary into a "green" subset of fraction γ
  using a hash seeded by the previous token id and a secret key. Add a bias δ
  to the logits of green tokens before sampling/argmax. Detection: count
  fraction of green tokens; compute z = (count - γ T) / sqrt(γ (1-γ) T).

Usage (single-config head-to-head with MCL):

    uv run python scripts/kgw_baseline.py \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --domain wiki \
        --gamma 0.25 --delta 2.0 \
        --temperature 1.0 --do-sample \
        --max-tokens 512 \
        --output-dir data/kgw_run_<stamp>

  Then re-detect to compute z-statistics:

    uv run python scripts/kgw_baseline.py --detect-only \
        --model meta-llama/Llama-3.2-3B-Instruct \
        --gamma 0.25 \
        --input-dir data/kgw_run_<stamp>

The companion script `scripts/cross_method_compare.py` builds the
"Method × Metric" table mixing MCL, KGW, and the unwatermarked baseline.
"""

import argparse
import json
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))


# Reuse the same prompt pools as generate_curated_dataset.py so the KGW row
# is a true head-to-head with MCL.
from generate_curated_dataset import (  # noqa: E402
    WIKIPEDIA_CONCEPTS,
    NEWS_PROMPTS,
    SOCIAL_PROMPTS,
    ABSTRACT_PROMPTS,
    generate_prompt as _wiki_prompt,
)


def get_prompts(domain: str):
    if domain == "wiki":
        return WIKIPEDIA_CONCEPTS, _wiki_prompt
    if domain == "news":
        return NEWS_PROMPTS, lambda p: p
    if domain == "social":
        return SOCIAL_PROMPTS, lambda p: p
    if domain == "abstract":
        return ABSTRACT_PROMPTS, lambda p: p
    raise ValueError(f"unknown domain {domain}")


# -----------------------------------------------------------------------------
# Green-list hashing
# -----------------------------------------------------------------------------

def green_mask(prev_token_id: int, vocab_size: int, gamma: float,
               secret_key: str) -> torch.Tensor:
    """
    Deterministic green-list selection for the next token, seeded by the
    previous token id and the secret key. Returns a bool tensor of shape
    [vocab_size] with `int(gamma * vocab_size)` True entries.

    SHA-256(secret_key || prev_token_id) -> 4-byte int -> torch.Generator seed.
    """
    h = hashlib.sha256(f"{secret_key}|{prev_token_id}".encode()).digest()
    seed = int.from_bytes(h[:8], "little") % (2**63 - 1)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    perm = torch.randperm(vocab_size, generator=g)
    n_green = int(gamma * vocab_size)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[perm[:n_green]] = True
    return mask


# -----------------------------------------------------------------------------
# KGW logits processor for HuggingFace generate()
# -----------------------------------------------------------------------------

class KGWLogitsProcessor:
    """Adds delta to green-list logits at each step."""

    def __init__(self, vocab_size: int, gamma: float, delta: float,
                 secret_key: str, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.secret_key = secret_key
        self.device = device
        self._cache = {}

    def _mask_for(self, prev_id: int) -> torch.Tensor:
        if prev_id not in self._cache:
            self._cache[prev_id] = green_mask(
                prev_id, self.vocab_size, self.gamma, self.secret_key
            ).to(self.device)
        return self._cache[prev_id]

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: [batch, seq_so_far]; scores: [batch, vocab]
        for b in range(input_ids.shape[0]):
            prev = int(input_ids[b, -1].item())
            mask = self._mask_for(prev)
            scores[b] = scores[b] + self.delta * mask.float()
        return scores


# -----------------------------------------------------------------------------
# KGW detector
# -----------------------------------------------------------------------------

class KGWDetector:
    def __init__(self, tokenizer, vocab_size: int, gamma: float,
                 secret_key: str):
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.secret_key = secret_key

    def detect(self, text: str):
        ids = self.tokenizer.encode(text)
        if len(ids) < 2:
            return {"n_tokens": len(ids), "green_count": 0,
                    "green_fraction": 0.0, "z_score": 0.0}
        green = 0
        T = 0
        for i in range(1, len(ids)):
            mask = green_mask(ids[i - 1], self.vocab_size, self.gamma,
                              self.secret_key)
            if bool(mask[ids[i]]):
                green += 1
            T += 1
        gamma = self.gamma
        z = (green - gamma * T) / np.sqrt(gamma * (1 - gamma) * T)
        return {
            "n_tokens": len(ids),
            "green_count": green,
            "green_fraction": green / T,
            "z_score": float(z),
        }


# -----------------------------------------------------------------------------
# Generation driver
# -----------------------------------------------------------------------------

def generate_with_kgw(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {args.model} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    model.eval()

    vocab_size = model.config.vocab_size
    proc = KGWLogitsProcessor(vocab_size, args.gamma, args.delta,
                              args.secret_key, device)

    concepts, prompt_fn = get_prompts(args.domain)
    if args.limit:
        concepts = concepts[: args.limit]
    print(f"Domain={args.domain}, n_prompts={len(concepts)}, "
          f"gamma={args.gamma}, delta={args.delta}, "
          f"sample={args.do_sample}, T={args.temperature}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "watermarked.jsonl"

    bs = max(1, args.batch_size)
    samples = []
    detector = KGWDetector(tokenizer, vocab_size, args.gamma, args.secret_key)
    t_start = time.time()
    n_tokens_total = 0

    for start in tqdm(range(0, len(concepts), bs), desc="KGW gen"):
        chunk = concepts[start : start + bs]
        prompts = [prompt_fn(c) for c in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True).to(device)

        gen_kwargs = dict(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=args.max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=[proc],
        )
        if args.do_sample:
            gen_kwargs.update(do_sample=True, temperature=args.temperature,
                              top_p=args.top_p)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            out = model.generate(**gen_kwargs)

        in_len = enc["input_ids"].shape[1]
        for i, (concept, prompt) in enumerate(zip(chunk, prompts)):
            gen_ids = out[i, in_len:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            det = detector.detect(text)
            n_tokens_total += det["n_tokens"]
            samples.append({
                "concept": concept,
                "prompt": prompt,
                "text": text,
                "type": "kgw_watermarked",
                "gamma": args.gamma,
                "delta": args.delta,
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "n_tokens": det["n_tokens"],
                "green_count": det["green_count"],
                "green_fraction": det["green_fraction"],
                "z_score": det["z_score"],
            })

    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")

    elapsed = time.time() - t_start
    summary = {
        "model": args.model,
        "domain": args.domain,
        "n_samples": len(samples),
        "gamma": args.gamma,
        "delta": args.delta,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "max_new_tokens": args.max_tokens,
        "secret_key": args.secret_key,
        "vocab_size": vocab_size,
        "elapsed_sec": elapsed,
        "tokens_per_sec": n_tokens_total / elapsed if elapsed > 0 else 0,
        "mean_green_fraction": float(np.mean([s["green_fraction"] for s in samples])),
        "mean_z_score": float(np.mean([s["z_score"] for s in samples])),
        "median_z_score": float(np.median([s["z_score"] for s in samples])),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\n[ok] {len(samples)} watermarked samples saved to {out_path}")


# -----------------------------------------------------------------------------
# Detect-only mode (re-detect existing samples; lightweight CPU pass)
# -----------------------------------------------------------------------------

def detect_only(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    detector = KGWDetector(tokenizer, args.vocab_size or tokenizer.vocab_size,
                           args.gamma, args.secret_key)

    in_path = Path(args.input_dir)
    files = sorted(in_path.glob("*.jsonl"))
    if not files:
        sys.exit(f"no .jsonl in {in_path}")

    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                det = detector.detect(rec["text"])
                rows.append({"file": f.name, "z": det["z_score"],
                             "green_frac": det["green_fraction"],
                             "n_tokens": det["n_tokens"]})
    arr = np.array([r["z"] for r in rows])
    print(f"n={len(rows)}  mean_z={arr.mean():.3f}  median_z={np.median(arr):.3f}  "
          f"frac_above_z=4: {float((arr > 4).mean()):.3f}")
    out = Path(args.output_dir or in_path) / "kgw_redetect.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"[ok] {out}")


# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct")
    ap.add_argument("--domain", default="wiki",
                    choices=["wiki", "news", "social", "abstract"])
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--delta", type=float, default=2.0)
    ap.add_argument("--do-sample", action="store_true",
                    help="KGW canonical config uses sampling, T=1.0. "
                         "Pass --do-sample for the (γ=.25,δ=2,T=1) operating point.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--secret-key", default="kgw_secret_2024")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of prompts (debug).")
    ap.add_argument("--output-dir", default=None,
                    help="Where to write watermarked.jsonl + summary.json. "
                         "Default: data/kgw_<domain>_g{γ}_d{δ}_<stamp>")
    ap.add_argument("--detect-only", action="store_true",
                    help="Skip generation; re-detect every .jsonl in --input-dir.")
    ap.add_argument("--input-dir", default=None)
    ap.add_argument("--vocab-size", type=int, default=None)
    args = ap.parse_args()

    if args.detect_only:
        if not args.input_dir:
            sys.exit("--detect-only requires --input-dir")
        detect_only(args)
        return

    if args.output_dir is None:
        from datetime import datetime
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = (
            f"data/kgw_{args.domain}_g{args.gamma}_d{args.delta}_{stamp}"
        )
    generate_with_kgw(args)


if __name__ == "__main__":
    main()
