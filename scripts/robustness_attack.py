#!/usr/bin/env python
"""
Robustness Attack Experiment

Tests watermark detection under word-level and token-level attacks.

Word-level attack: replace X% of words with "masked"
Token-level attack: replace X% of tokens with a random token (id=0)

Usage:
    # Run on a specific dataset directory
    uv run python scripts/robustness_attack.py --data-dir data/curated_wiki_dataset_XXXXXXXX_XXXXXX

    # Specify config to attack (default: states7_overlap0pct)
    uv run python scripts/robustness_attack.py --data-dir data/curated_wiki_dataset_XXXXXXXX_XXXXXX --config states7_overlap0pct

    # Custom model tokenizer
    uv run python scripts/robustness_attack.py --data-dir ... --model mistralai/Mistral-7B-v0.3
"""

import sys
import json
import argparse
import random
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltw_watermark.enhanced_mcl import EnhancedMCLDetector


def word_level_attack(text: str, modification_rate: float, seed: int = 42) -> str:
    """Replace a fraction of words with 'masked'."""
    rng = random.Random(seed)
    words = text.split()
    if not words:
        return text
    n_modify = max(0, int(len(words) * modification_rate))
    indices = rng.sample(range(len(words)), min(n_modify, len(words)))
    for i in indices:
        words[i] = "masked"
    return " ".join(words)


def token_level_attack(token_ids: List[int], modification_rate: float, replacement_id: int = 0, seed: int = 42) -> List[int]:
    """Replace a fraction of token IDs with a fixed replacement token."""
    rng = random.Random(seed)
    ids = list(token_ids)
    if not ids:
        return ids
    n_modify = max(0, int(len(ids) * modification_rate))
    indices = rng.sample(range(len(ids)), min(n_modify, len(ids)))
    for i in indices:
        ids[i] = replacement_id
    return ids


def main():
    parser = argparse.ArgumentParser(description="Robustness attack experiment")
    parser.add_argument("--data-dir", required=True, help="Path to curated dataset directory")
    parser.add_argument("--config", default="states7_overlap0pct", help="Which watermark config to attack")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-3B-Instruct", help="Tokenizer model name")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config_file = data_dir / f"{args.config}.jsonl"

    if not config_file.exists():
        print(f"Error: {config_file} not found")
        print(f"Available configs: {[f.stem for f in data_dir.glob('states*.jsonl')]}")
        sys.exit(1)

    # Load samples
    samples = []
    with open(config_file) as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples from {args.config}")

    # Parse config params from filename
    # e.g. states7_overlap0pct -> num_states=7, overlap=0.0
    parts = args.config.split("_")
    num_states = int(parts[0].replace("states", ""))
    overlap_pct = int(parts[1].replace("overlap", "").replace("pct", ""))
    overlap = overlap_pct / 100.0

    # Read the summary to get the chain_key/topology used
    summary_file = data_dir / "summary.json"
    chain_key = "soft_cycle"  # default
    secret_key = "curated_wiki_dataset_2024"
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        if "configs" in summary and summary["configs"]:
            chain_key = summary["configs"][0].get("chain_key", "soft_cycle")

    modification_rates = [0.0, 0.10, 0.20, 0.30, 0.40]

    # =========================================================================
    # WORD-LEVEL ATTACK
    # =========================================================================
    print("\n" + "=" * 80)
    print("WORD-LEVEL ATTACK")
    print(f"Replace X% of words with 'masked'")
    print(f"Config: {args.config} (states={num_states}, overlap={overlap}, chain={chain_key})")
    print("=" * 80)

    word_results = []
    for rate in modification_rates:
        detector = EnhancedMCLDetector(
            tokenizer_name=args.model,
            secret_key=secret_key,
            num_states=num_states,
            chain_key=chain_key,
            overlap_ratio=overlap,
            detection_threshold=0.5,
        )

        scores = []
        detected = 0
        for i, sample in enumerate(samples):
            attacked_text = word_level_attack(sample["text"], rate, seed=args.seed + i)
            result = detector.detect(attacked_text)
            scores.append(result.chain_score)
            if result.is_watermarked:
                detected += 1

        avg_score = np.mean(scores)
        detection_rate = detected / len(samples)
        word_results.append({
            "attack": "word_level",
            "modification_rate": rate,
            "avg_score": float(avg_score),
            "detection_rate": float(detection_rate),
            "n_samples": len(samples),
            "n_detected": detected,
        })
        print(f"  {rate*100:5.0f}% modified: score={avg_score:.4f}, detection={detection_rate*100:.1f}% ({detected}/{len(samples)})")

        del detector

    # =========================================================================
    # TOKEN-LEVEL ATTACK
    # =========================================================================
    print("\n" + "=" * 80)
    print("TOKEN-LEVEL ATTACK")
    print(f"Replace X% of tokens with token_id=0")
    print(f"Config: {args.config} (states={num_states}, overlap={overlap}, chain={chain_key})")
    print("=" * 80)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    token_results = []
    for rate in modification_rates:
        detector = EnhancedMCLDetector(
            tokenizer_name=args.model,
            secret_key=secret_key,
            num_states=num_states,
            chain_key=chain_key,
            overlap_ratio=overlap,
            detection_threshold=0.5,
        )

        scores = []
        detected = 0
        for i, sample in enumerate(samples):
            token_ids = tokenizer.encode(sample["text"])
            attacked_ids = token_level_attack(token_ids, rate, replacement_id=0, seed=args.seed + i)
            attacked_text = tokenizer.decode(attacked_ids, skip_special_tokens=True)
            result = detector.detect(attacked_text)
            scores.append(result.chain_score)
            if result.is_watermarked:
                detected += 1

        avg_score = np.mean(scores)
        detection_rate = detected / len(samples)
        token_results.append({
            "attack": "token_level",
            "modification_rate": rate,
            "avg_score": float(avg_score),
            "detection_rate": float(detection_rate),
            "n_samples": len(samples),
            "n_detected": detected,
        })
        print(f"  {rate*100:5.0f}% modified: score={avg_score:.4f}, detection={detection_rate*100:.1f}% ({detected}/{len(samples)})")

        del detector

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output = {
        "config": args.config,
        "num_states": num_states,
        "overlap": overlap,
        "chain_key": chain_key,
        "model": args.model,
        "n_samples": len(samples),
        "modification_rates": modification_rates,
        "word_level_results": word_results,
        "token_level_results": token_results,
    }

    output_file = data_dir / f"robustness_{args.config}.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Attack':<15} {'Rate':>6} {'Score':>8} {'Detect':>8}")
    print("-" * 40)
    for r in word_results:
        print(f"{'word':<15} {r['modification_rate']*100:>5.0f}% {r['avg_score']:>8.4f} {r['detection_rate']*100:>7.1f}%")
    print("-" * 40)
    for r in token_results:
        print(f"{'token':<15} {r['modification_rate']*100:>5.0f}% {r['avg_score']:>8.4f} {r['detection_rate']*100:>7.1f}%")


if __name__ == "__main__":
    main()
