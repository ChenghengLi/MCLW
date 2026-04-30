#!/usr/bin/env python
"""
Evaluate non-watermarked text against all MCL configurations for curated dataset.
"""

import sys
import json
import argparse
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcl_watermark.enhanced_mcl import EnhancedMCLDetector

DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "curated_wiki_dataset_20260201_112721"
DEFAULT_MODEL_NAME = os.environ.get("MCL_TOKENIZER", "meta-llama/Llama-3.2-3B-Instruct")
SECRET_KEY = "curated_wiki_dataset_2024"

# MCL configurations: states 2, 4, 5, 7, 9, 11, 15 × overlaps 0%, 5%, 10%, 15%
MCL_CONFIGS = []
for num_states in [2, 4, 5, 7, 9, 11, 15]:
    for overlap_pct in [0, 5, 10, 15]:
        overlap = overlap_pct / 100.0
        MCL_CONFIGS.append({
            "name": f"states{num_states}_overlap{overlap_pct}pct",
            "num_states": num_states,
            "chain_key": "soft_cycle",
            "overlap": overlap,
        })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME,
                        help="Tokenizer model name. Llama-3.x family share the same tokenizer, "
                             "so the cached Llama-3.1-8B-Instruct works as a drop-in for 3.2-3B.")
    args = parser.parse_args()
    global DATA_DIR, MODEL_NAME
    DATA_DIR = args.data_dir
    MODEL_NAME = args.model
    print("=" * 80)
    print("EVALUATING NON-WATERMARKED TEXT AGAINST ALL CONFIGURATIONS")
    print(f"Tokenizer: {MODEL_NAME}")
    print(f"Data dir : {DATA_DIR}")
    print("=" * 80)
    
    # Load non-watermarked samples
    non_wm_file = DATA_DIR / "non_watermarked.jsonl"
    samples = []
    with open(non_wm_file) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    print(f"\nLoaded {len(samples)} non-watermarked samples")
    print()
    
    # Results storage
    results = []
    
    # For each configuration, run detection on all non-watermarked texts
    for config in MCL_CONFIGS:
        config_name = config["name"]
        num_states = config["num_states"]
        chain_key = config["chain_key"]
        overlap = config["overlap"]
        
        print(f"Testing {config_name}...", end=" ", flush=True)
        
        detector = EnhancedMCLDetector(
            tokenizer_name=MODEL_NAME,
            secret_key=SECRET_KEY,
            num_states=num_states,
            chain_key=chain_key,
            overlap_ratio=overlap,
            detection_threshold=0.5
        )
        
        scores = []
        detected = 0
        
        for sample in samples:
            text = sample["text"]
            result = detector.detect(text)
            scores.append(result.chain_score)
            if result.is_watermarked:
                detected += 1
        
        avg_score = np.mean(scores)
        max_score = np.max(scores)
        min_score = np.min(scores)
        std_score = np.std(scores)
        fpr = detected / len(samples)  # False Positive Rate
        
        results.append({
            "config": config_name,
            "num_states": num_states,
            "overlap": overlap,
            "avg_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "std_score": std_score,
            "false_positive_rate": fpr,
            "baseline": detector.expected_random,
        })
        
        print(f"Avg={avg_score:.4f}, Max={max_score:.4f}, FPR={fpr*100:.1f}%")
        
        del detector
    
    # Print summary table
    print("\n" + "=" * 100)
    print("NON-WATERMARKED TEXT DETECTION RESULTS")
    print("=" * 100)
    print(f"{'Config':<25} {'States':>6} {'Overlap':>8} {'Avg Score':>10} {'Max Score':>10} {'FPR':>8} {'Baseline':>10}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['config']:<25} {r['num_states']:>6} {r['overlap']*100:>7.0f}% {r['avg_score']:>10.4f} {r['max_score']:>10.4f} {r['false_positive_rate']*100:>7.1f}% {r['baseline']:>10.4f}")
    
    # Save results
    output = {
        "per_config_results": results,
        "num_samples": len(samples),
    }
    
    output_file = DATA_DIR / "non_watermarked_detection_results.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[ok] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
