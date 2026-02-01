#!/usr/bin/env python
"""
Experiment 2: False Positive Rate Analysis

This experiment analyzes the false positive rate of LTW detection
on human-written texts across different thresholds.

Key questions:
- What is the FPR for different detection thresholds?
- How does the anisotropy problem affect false positives?
- How does LTW FPR compare to perplexity-based detection?
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltw_watermark.watermark import LTWWatermarker
from ltw_watermark.perplexity import PerplexityDetector
from ltw_watermark.analysis import plot_roc_comparison


# Extended human text corpus for FPR testing
HUMAN_TEXT_CORPUS = [
    # Formal/Technical (often falsely flagged by perplexity detectors)
    "The mitochondria is the powerhouse of the cell. It produces adenosine triphosphate through oxidative phosphorylation. This process occurs in the inner membrane.",
    "According to the terms of this agreement, the party of the first part shall hereby transfer all rights and obligations to the party of the second part.",
    "The HTTP protocol operates on a request-response model. Clients send requests to servers. Servers process requests and return responses.",
    
    # Casual/Informal
    "Hey! Wanna grab lunch? I'm thinking maybe that new Thai place? Heard their pad thai is pretty good... lmk!",
    "OMG I can't believe what happened yesterday!! You won't believe it—ran into Jake at the coffee shop. So awkward lol.",
    "So tired. Been up since 5am trying to finish this paper. Why did I leave it to the last minute AGAIN? Never learn smh.",
    
    # Creative/Literary
    "The autumn leaves danced through the crisp morning air, painting the sidewalk in shades of amber and rust. She watched them fall, each one a tiny goodbye.",
    "He stood at the crossroads, literally and figuratively. The left path wound through familiar territory. The right... well, that was the adventure.",
    
    # Mixed style
    "OK so here's the thing about quantum mechanics—and bear with me here—it's WEIRD. Like, really weird. Cats being alive and dead? Come on!",
    "Dear Hiring Manager, I am writing to express my strong interest in the Software Engineer position. My experience includes...",
    "The results of the study indicate a statistically significant correlation (p < 0.05) between the variables. However, causation cannot be inferred.",
    "My grandma's recipe is simple: two cups flour, one cup sugar, a pinch of love. Mix until it feels right. Bake until it smells like home.",
    
    # News-like
    "Local authorities reported a significant increase in traffic violations last month. Officials attribute the spike to construction on Main Street.",
    "The company announced quarterly earnings that exceeded analyst expectations. Stock prices rose 5% in after-hours trading.",
    
    # Academic
    "Foucault's concept of power differs fundamentally from traditional notions. Power, he argues, is not merely repressive but productive.",
    "The Krebs cycle, also known as the citric acid cycle, is a series of chemical reactions used by all aerobic organisms.",
    
    # Conversational
    "Wait, so you're telling me... he just LEFT? Without saying anything? Wow. That's... yeah, that's rough buddy.",
    "Remember that time we got lost in Barcelona? And you insisted you knew the way? Good times, good times.",
    
    # Technical documentation style
    "To install the package, run pip install ltw-watermark. Configure settings in config.yaml. Run experiments with python experiments/exp1.py.",
    "Error handling follows conventional patterns. Exceptions are logged and re-raised. Users should implement appropriate try-catch blocks.",
]


def generate_synthetic_texts(n_samples: int, style: str = "mixed") -> list:
    """Generate additional synthetic human-like texts."""
    templates = {
        "formal": [
            "The {noun} demonstrates significant {adj} properties. Research indicates that {result}.",
            "According to established {field} principles, the {phenomenon} exhibits {characteristic}.",
        ],
        "casual": [
            "Can't believe {thing} happened! So {emotion} right now haha",
            "Just {action} and honestly? {reaction}. Wild times.",
        ]
    }
    # Use the corpus as-is for now, expanded with repetition
    return (HUMAN_TEXT_CORPUS * (n_samples // len(HUMAN_TEXT_CORPUS) + 1))[:n_samples]


def run_fpr_analysis(config: dict, results_dir: Path, quick: bool = False):
    """Run false positive rate analysis."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: False Positive Rate Analysis")
    print("="*60)
    
    # Parameters
    thresholds = config.get('experiments', {}).get('exp2', {}).get(
        'thresholds', [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    )
    n_samples = 20 if quick else config.get('experiments', {}).get('exp2', {}).get('n_samples', 100)
    
    # Generate test corpus
    print(f"\n[1/4] Preparing corpus of {n_samples} human texts...")
    human_texts = generate_synthetic_texts(n_samples)
    
    # Initialize detector
    print("[2/4] Initializing LTW detector...")
    detector = LTWWatermarker(
        secret_key=config['watermark']['secret_key'],
        rotation_strength=config['watermark']['rotation_strength'],
        detection_threshold=0.1,  # Will vary
        use_differential_scoring=config['watermark']['use_differential_scoring'],
        unit=config['watermark']['unit']
    )
    
    # Collect scores for all texts
    print("[3/4] Analyzing texts...")
    all_scores = []
    for text in tqdm(human_texts, desc="Processing"):
        result = detector.detect(text)
        all_scores.append({
            'text_preview': text[:50],
            'differential_score': result.differential_score,
            'alignment_score': result.alignment_score,
            'n_units': result.details['n_units'],
        })
    
    diff_scores = [s['differential_score'] for s in all_scores]
    
    # Calculate FPR at each threshold
    print("\n[4/4] Calculating FPR at various thresholds...")
    fpr_results = []
    
    print("\n" + "-"*50)
    print(f"{'Threshold':<12} {'FPR':<10} {'False Positives':<15} {'Total':<10}")
    print("-"*50)
    
    for threshold in thresholds:
        false_positives = sum(1 for s in diff_scores if s > threshold)
        fpr = false_positives / len(diff_scores)
        fpr_results.append({
            'threshold': threshold,
            'fpr': fpr,
            'false_positives': false_positives,
            'total': len(diff_scores)
        })
        print(f"{threshold:<12.3f} {fpr:<10.4f} {false_positives:<15} {len(diff_scores):<10}")
    
    print("-"*50)
    
    # Score statistics
    print(f"\nScore Statistics:")
    print(f"  Mean:   {np.mean(diff_scores):.4f}")
    print(f"  Std:    {np.std(diff_scores):.4f}")
    print(f"  Min:    {np.min(diff_scores):.4f}")
    print(f"  Max:    {np.max(diff_scores):.4f}")
    print(f"  Median: {np.median(diff_scores):.4f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # FPR curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot([r['threshold'] for r in fpr_results], 
            [r['fpr'] for r in fpr_results], 
            'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Detection Threshold', fontsize=12)
    ax.set_ylabel('False Positive Rate', fontsize=12)
    ax.set_title('LTW False Positive Rate vs Detection Threshold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add target FPR line
    ax.axhline(y=0.05, color='green', linestyle='--', label='5% Target FPR')
    ax.axhline(y=0.01, color='red', linestyle='--', label='1% Target FPR')
    ax.legend()
    
    fig.savefig(results_dir / 'exp2_fpr_curve.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {results_dir / 'exp2_fpr_curve.png'}")
    
    # Score histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(diff_scores, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='Default threshold (0.1)')
    ax.set_xlabel('Differential Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Distribution of Differential Scores on Human Text', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.savefig(results_dir / 'exp2_score_histogram.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {results_dir / 'exp2_score_histogram.png'}")
    
    # Save detailed results
    results_path = results_dir / 'exp2_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'fpr_by_threshold': fpr_results,
            'score_statistics': {
                'mean': float(np.mean(diff_scores)),
                'std': float(np.std(diff_scores)),
                'min': float(np.min(diff_scores)),
                'max': float(np.max(diff_scores)),
                'median': float(np.median(diff_scores)),
            },
            'config': config['watermark'],
        }, f, indent=2)
    print(f"  Saved: {results_path}")
    
    # Find recommended threshold for < 5% FPR
    recommended = None
    for r in fpr_results:
        if r['fpr'] <= 0.05:
            recommended = r['threshold']
            break
    
    if recommended:
        print(f"\n✓ Recommended threshold for <5% FPR: {recommended}")
    else:
        print(f"\n✗ Warning: Could not achieve <5% FPR at tested thresholds")
        print(f"  Consider increasing rotation_strength or adjusting the method")
    
    return {
        'fpr_results': fpr_results,
        'recommended_threshold': recommended,
        'mean_score': np.mean(diff_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: False Positive Rate Analysis")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer samples")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'watermark': {
                'secret_key': 'test-secret-key',
                'rotation_strength': 0.3,
                'detection_threshold': 0.1,
                'use_differential_scoring': True,
                'unit': 'sentence'
            },
            'experiments': {'exp2': {'n_samples': 50, 'thresholds': [0.05, 0.1, 0.15, 0.2]}}
        }
    
    results_dir = Path(__file__).parent.parent / "results" / "exp2"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_fpr_analysis(config, results_dir, args.quick)
    
    print("\n" + "="*60)
    print("EXPERIMENT 2 COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
