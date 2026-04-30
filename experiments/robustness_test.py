#!/usr/bin/env python
"""
Robustness Test - Word Replacement with 'masked'

Uses the CORRECT configuration: soft_cycle (not clockwork!)
Tests how detection score degrades when words are replaced.

Usage: python robustness_test.py
"""

import os
import sys
import json
import random
from pathlib import Path

# Set HF_TOKEN environment variable before running: export HF_TOKEN="your-token"
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mcl_watermark.enhanced_mcl import EnhancedMCLDetector


def load_data(filepath):
    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    config = {
        'num_states': samples[0]['num_states'],
        'overlap': samples[0]['overlap']
    }
    return samples, config


def replace_words_with_masked(text: str, replace_rate: float) -> str:
    words = text.split()
    n_replace = int(len(words) * replace_rate)
    if n_replace > 0:
        positions = random.sample(range(len(words)), min(n_replace, len(words)))
        for pos in positions:
            words[pos] = "masked"
    return " ".join(words)


def run_experiment():
    print("=" * 70)
    print("ROBUSTNESS TEST: Word Replacement with 'masked'")
    print("=" * 70)
    
    data_path = "/Users/chenghengli/Desktop/Hackathon/LTW/data/curated_wiki_dataset_20260201_112721/states7_overlap0pct.jsonl"
    secret_key = "curated_wiki_dataset_2024"
    
    samples, config = load_data(data_path)
    num_states = config['num_states']
    
    print(f"\nLoaded {len(samples)} samples")
    print(f"Config: num_states={num_states}, overlap={config['overlap']}")
    print(f"Chain type: soft_cycle (allows i→i+1 OR i→i+2)")
    
    # Create detector with CORRECT settings: soft_cycle!
    print("\nCreating detector...")
    detector = EnhancedMCLDetector(
        tokenizer_name="meta-llama/Llama-3.2-3B-Instruct",
        secret_key=secret_key,
        num_states=num_states,
        chain_key="soft_cycle",  
        overlap_ratio=config['overlap'],
        detection_threshold=0.5
    )
    
    # Verify on unmodified text
    original_scores = [s['chain_score'] for s in samples]
    avg_original = sum(original_scores) / len(original_scores)
    
    sample_result = detector.detect(samples[0]['text'])
    print(f"\nOriginal stored avg: {avg_original:.4f}")
    print(f"Re-detection (0%):   {sample_result.chain_score:.4f}")
    
    # Test replacement rates
    replace_rates = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    results = []
    
    for rate in replace_rates:
        scores = []
        for sample in samples:
            modified_text = replace_words_with_masked(sample['text'], rate)
            result = detector.detect(modified_text)
            scores.append(result.chain_score)
        
        avg_score = sum(scores) / len(scores)
        std_score = (sum((s - avg_score)**2 for s in scores) / len(scores)) ** 0.5
        detected = sum(1 for s in scores if s > 0.5)
        detection_rate = 100 * detected / len(samples)
        
        # Theoretical prediction
        delta = rate
        theoretical = (1 - delta)**2 + delta * (2 - delta) * 2 / num_states
        
        results.append({
            'rate': rate,
            'observed': avg_score,
            'std': std_score,
            'theoretical': theoretical,
            'detection_rate': detection_rate
        })
        
        status = "✓" if detection_rate >= 50 else "✗"
        print(f"\n{rate*100:.0f}% words replaced:")
        print(f"  Observed:    φ = {avg_score:.4f} ± {std_score:.4f}")
        print(f"  Theoretical: φ = {theoretical:.4f}")
        print(f"  Detection:   {detection_rate:.1f}% {status}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Rate':>8} | {'Observed φ':>12} | {'Theoretical':>11} | {'Detected':>10}")
    print("-" * 55)
    for r in results:
        print(f"{r['rate']*100:>7.0f}% | {r['observed']:>12.4f} | {r['theoretical']:>11.4f} | {r['detection_rate']:>9.1f}%")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    random.seed(42)
    run_experiment()
