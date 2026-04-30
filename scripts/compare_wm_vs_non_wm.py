#!/usr/bin/env python
"""
Compare watermarked vs non-watermarked text for each configuration.
Produces a detailed side-by-side comparison.
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

# MCL configurations
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
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    args = parser.parse_args()
    global DATA_DIR, MODEL_NAME
    DATA_DIR = args.data_dir
    MODEL_NAME = args.model
    print("=" * 120)
    print("DETAILED COMPARISON: WATERMARKED vs NON-WATERMARKED FOR EACH CONFIGURATION")
    print(f"Tokenizer: {MODEL_NAME}")
    print(f"Data dir : {DATA_DIR}")
    print("=" * 120)
    
    # Load non-watermarked samples
    non_wm_file = DATA_DIR / "non_watermarked.jsonl"
    non_wm_samples = []
    with open(non_wm_file) as f:
        for line in f:
            if line.strip():
                non_wm_samples.append(json.loads(line))
    
    print(f"\nLoaded {len(non_wm_samples)} non-watermarked samples")
    
    all_results = []
    
    # For each configuration
    for config in MCL_CONFIGS:
        config_name = config["name"]
        num_states = config["num_states"]
        chain_key = config["chain_key"]
        overlap = config["overlap"]
        
        print(f"\n{'='*80}")
        print(f"Config: {config_name}")
        print(f"{'='*80}")
        
        # Load watermarked samples for this config
        wm_file = DATA_DIR / f"{config_name}.jsonl"
        if not wm_file.exists():
            print(f"  [SKIP] File not found: {wm_file}")
            continue
            
        wm_samples = []
        with open(wm_file) as f:
            for line in f:
                if line.strip():
                    wm_samples.append(json.loads(line))
        
        # Create detector
        detector = EnhancedMCLDetector(
            tokenizer_name=MODEL_NAME,
            secret_key=SECRET_KEY,
            num_states=num_states,
            chain_key=chain_key,
            overlap_ratio=overlap,
            detection_threshold=0.5
        )
        
        # Detect on non-watermarked
        non_wm_scores = []
        non_wm_detected = 0
        for sample in non_wm_samples:
            result = detector.detect(sample["text"])
            non_wm_scores.append(result.chain_score)
            if result.is_watermarked:
                non_wm_detected += 1
        
        # Detect on watermarked
        wm_scores = []
        wm_ppls = []
        wm_detected = 0
        for sample in wm_samples:
            result = detector.detect(sample["text"])
            wm_scores.append(result.chain_score)
            wm_ppls.append(sample.get("perplexity", 0))
            if result.is_watermarked:
                wm_detected += 1
        
        # Calculate statistics
        non_wm_avg = np.mean(non_wm_scores)
        non_wm_std = np.std(non_wm_scores)
        non_wm_min = np.min(non_wm_scores)
        non_wm_max = np.max(non_wm_scores)
        non_wm_fpr = non_wm_detected / len(non_wm_samples)
        
        wm_avg = np.mean(wm_scores)
        wm_std = np.std(wm_scores)
        wm_min = np.min(wm_scores)
        wm_max = np.max(wm_scores)
        wm_tpr = wm_detected / len(wm_samples)
        wm_ppl_avg = np.mean(wm_ppls)
        
        gap = wm_avg - non_wm_avg
        
        # Print comparison
        print(f"\n  {'Metric':<20} {'Non-Watermarked':>18} {'Watermarked':>18} {'Difference':>15}")
        print(f"  {'-'*73}")
        print(f"  {'Avg Score':<20} {non_wm_avg:>18.4f} {wm_avg:>18.4f} {gap:>+15.4f}")
        print(f"  {'Std Score':<20} {non_wm_std:>18.4f} {wm_std:>18.4f}")
        print(f"  {'Min Score':<20} {non_wm_min:>18.4f} {wm_min:>18.4f}")
        print(f"  {'Max Score':<20} {non_wm_max:>18.4f} {wm_max:>18.4f}")
        print(f"  {'Detection Rate':<20} {non_wm_fpr*100:>17.1f}% {wm_tpr*100:>17.1f}%")
        print(f"  {'Avg PPL':<20} {'N/A':>18} {wm_ppl_avg:>18.2f}")
        print(f"  {'Baseline':<20} {detector.expected_random:>18.4f}")
        
        # Separability analysis
        print(f"\n  Separability Analysis:")
        # Count how many non-wm scores are above threshold
        # Count how many wm scores are below non-wm max
        overlap_count = sum(1 for s in wm_scores if s <= non_wm_max)
        print(f"    - Non-WM max score: {non_wm_max:.4f}")
        print(f"    - WM min score: {wm_min:.4f}")
        print(f"    - WM samples below non-WM max: {overlap_count}/{len(wm_samples)} ({overlap_count/len(wm_samples)*100:.1f}%)")
        print(f"    - Score gap (WM avg - Non-WM avg): {gap:.4f}")
        
        if gap > 0.5:
            verdict = "[EXCELLENT] Large separation, easy to distinguish"
        elif gap > 0.3:
            verdict = "[GOOD] Clear separation"
        elif gap > 0.15:
            verdict = "[MARGINAL] Some overlap, harder to distinguish"
        else:
            verdict = "[POOR] Too much overlap, unreliable detection"
        print(f"    - Verdict: {verdict}")
        
        all_results.append({
            "config": config_name,
            "num_states": num_states,
            "overlap": overlap,
            "non_wm_avg": non_wm_avg,
            "non_wm_std": non_wm_std,
            "non_wm_min": non_wm_min,
            "non_wm_max": non_wm_max,
            "non_wm_fpr": non_wm_fpr,
            "wm_avg": wm_avg,
            "wm_std": wm_std,
            "wm_min": wm_min,
            "wm_max": wm_max,
            "wm_tpr": wm_tpr,
            "wm_ppl_avg": wm_ppl_avg,
            "gap": gap,
            "baseline": detector.expected_random,
        })
        
        del detector
    
    # Print summary table
    print("\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    print(f"{'Config':<22} {'States':>6} {'Ovrlp':>6} {'Non-WM':>9} {'WM Avg':>9} {'Gap':>8} {'WM Det%':>8} {'Non-WM FP%':>11} {'PPL':>7}")
    print("-" * 120)
    
    for r in all_results:
        print(f"{r['config']:<22} {r['num_states']:>6} {r['overlap']*100:>5.0f}% {r['non_wm_avg']:>9.4f} {r['wm_avg']:>9.4f} {r['gap']:>+8.4f} {r['wm_tpr']*100:>7.1f}% {r['non_wm_fpr']*100:>10.1f}% {r['wm_ppl_avg']:>7.2f}")
    
    # Save results
    output_file = DATA_DIR / "detailed_comparison.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n[ok] Results saved to: {output_file}")


if __name__ == "__main__":
    main()
