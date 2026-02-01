#!/usr/bin/env python
"""
Experiment 5: Anisotropy Analysis

This experiment investigates the anisotropy problem in embedding spaces
and its impact on LTW watermarking. Key questions:

1. How anisotropic are typical embedding spaces?
2. How does rotation strength affect detection and FPR?
3. Can orthogonal injection reduce false positives?

This is crucial for understanding when LTW will work and when it will fail.
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

from ltw_watermark.embeddings import EmbeddingExtractor
from ltw_watermark.rotation import OrthogonalRotation, generate_rotation_matrix
from ltw_watermark.watermark import LTWWatermarker
from ltw_watermark.analysis import plot_anisotropy, plot_trajectory_2d


# Large corpus for anisotropy analysis
ANALYSIS_CORPUS = [
    # Technical
    "The algorithm complexity is O(n log n) for the average case.",
    "Neural networks use backpropagation for training.",
    "The database query was optimized using indexing.",
    "Parallel processing improves computational efficiency.",
    "Machine learning models require extensive training data.",
    
    # Casual
    "I'm going to the store to buy some groceries.",
    "The movie last night was absolutely fantastic!",
    "Do you want to grab coffee tomorrow morning?",
    "My phone battery dies way too quickly these days.",
    "Traffic was terrible on my way to work today.",
    
    # News
    "The president announced new economic policies today.",
    "Scientists discovered a new species in the Amazon.",
    "Stock markets closed higher on positive earnings.",
    "The weather forecast predicts rain this weekend.",
    "Local schools announced extended summer breaks.",
    
    # Academic  
    "The hypothesis was supported by experimental evidence.",
    "Statistical analysis revealed significant correlations.",
    "The research methodology followed established protocols.",
    "Peer review is essential for scientific validity.",
    "The findings have important theoretical implications.",
    
    # Creative
    "The sunset painted the sky in shades of gold.",
    "She walked through the forest, lost in thought.",
    "Music filled the empty room with memories.",
    "The old book smelled of dust and adventure.",
    "Stars emerged one by one as night fell.",
    
    # Mixed
    "So basically, the quantum state collapses upon observation.",
    "I tried that new restaurant—the pasta was amazing!",
    "According to sources, the deal may close next week.",
    "The cat knocked over my coffee. Again. Typical.",
    "Mathematical proofs require rigorous logical steps.",
]


def analyze_embedding_anisotropy(extractor: EmbeddingExtractor, texts: list, results_dir: Path):
    """Analyze anisotropy of the embedding space."""
    print("\n--- Analyzing Embedding Space Anisotropy ---")
    
    # Get embeddings
    print(f"  Embedding {len(texts)} texts...")
    embeddings = extractor.embed_texts(texts)
    
    # Analyze anisotropy
    metrics, fig = plot_anisotropy(
        embeddings,
        title="Embedding Space Anisotropy Analysis",
        save_path=str(results_dir / 'exp5_anisotropy.png')
    )
    
    print(f"\n  Anisotropy Metrics:")
    print(f"    IsoScore (1=isotropic): {metrics['iso_score']:.4f}")
    print(f"    Top PC variance:        {metrics['top_1_variance']:.4f}")
    print(f"    Top 10 cumulative:      {metrics['top_10_cumulative']:.4f}")
    print(f"    Effective dimensions:   {metrics['effective_dimensions']:.1f}")
    
    return embeddings, metrics


def test_rotation_strengths(config: dict, embeddings: np.ndarray, results_dir: Path):
    """Test different rotation strengths and their effects."""
    print("\n--- Testing Rotation Strengths ---")
    
    strengths = config.get('experiments', {}).get('exp5', {}).get(
        'rotation_strengths', [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    )
    
    dim = embeddings.shape[1]
    results = []
    
    for strength in strengths:
        print(f"\n  Rotation strength: {strength}")
        
        rotation = OrthogonalRotation(
            secret_key=config['watermark']['secret_key'],
            dim=dim,
            rotation_strength=strength
        )
        
        # Measure rotation magnitude
        angle = rotation.get_rotation_angle()
        
        # Calculate alignment distribution for random pairs
        n_pairs = min(100, len(embeddings) - 1)
        alignments = []
        for i in range(n_pairs):
            j = (i + 1) % len(embeddings)
            alignment = rotation.compute_alignment_score(embeddings[i], embeddings[j])
            alignments.append(alignment)
        
        # Calculate differential scores
        differentials = []
        for i in range(n_pairs):
            j = (i + 1) % len(embeddings)
            diff = rotation.compute_differential_score(embeddings[i], embeddings[j])
            differentials.append(diff)
        
        result = {
            'strength': strength,
            'rotation_angle_rad': angle,
            'rotation_angle_deg': np.degrees(angle),
            'mean_alignment': np.mean(alignments),
            'std_alignment': np.std(alignments),
            'mean_differential': np.mean(differentials),
            'std_differential': np.std(differentials),
        }
        results.append(result)
        
        print(f"    Rotation angle: {np.degrees(angle):.1f}°")
        print(f"    Mean alignment: {np.mean(alignments):.4f} ± {np.std(alignments):.4f}")
        print(f"    Mean differential: {np.mean(differentials):.4f} ± {np.std(differentials):.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Rotation angle vs strength
    ax = axes[0]
    ax.plot([r['strength'] for r in results], 
            [r['rotation_angle_deg'] for r in results], 
            'b-o', linewidth=2, markersize=8)
    ax.set_xlabel('Rotation Strength', fontsize=12)
    ax.set_ylabel('Rotation Angle (degrees)', fontsize=12)
    ax.set_title('Rotation Magnitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Mean differential vs strength
    ax = axes[1]
    means = [r['mean_differential'] for r in results]
    stds = [r['std_differential'] for r in results]
    ax.errorbar([r['strength'] for r in results], means, yerr=stds, 
                fmt='g-o', linewidth=2, markersize=8, capsize=5)
    ax.set_xlabel('Rotation Strength', fontsize=12)
    ax.set_ylabel('Mean Differential Score', fontsize=12)
    ax.set_title('Detection Signal Strength', fontsize=12)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    # Alignment distribution for different strengths
    ax = axes[2]
    for strength in [0.1, 0.3, 0.7]:
        rotation = OrthogonalRotation(config['watermark']['secret_key'], dim, strength)
        alignments = [rotation.compute_alignment_score(embeddings[i], embeddings[(i+1)%len(embeddings)]) 
                     for i in range(50)]
        ax.hist(alignments, bins=20, alpha=0.5, label=f'strength={strength}', density=True)
    ax.set_xlabel('Alignment Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Alignment Distribution by Strength', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('Effect of Rotation Strength on LTW Detection', fontsize=14)
    plt.tight_layout()
    
    fig.savefig(results_dir / 'exp5_rotation_strength.png', dpi=150, bbox_inches='tight')
    print(f"\n  Saved: {results_dir / 'exp5_rotation_strength.png'}")
    
    return results


def test_orthogonal_vs_natural(config: dict, embeddings: np.ndarray, results_dir: Path):
    """Compare orthogonal rotations to natural semantic flow."""
    print("\n--- Orthogonal vs Natural Semantic Flow ---")
    
    dim = embeddings.shape[1]
    
    # Natural coherence (no rotation)
    natural_similarities = []
    for i in range(len(embeddings) - 1):
        emb1 = embeddings[i] / np.linalg.norm(embeddings[i])
        emb2 = embeddings[i+1] / np.linalg.norm(embeddings[i+1])
        natural_similarities.append(np.dot(emb1, emb2))
    
    print(f"  Natural coherence: {np.mean(natural_similarities):.4f} ± {np.std(natural_similarities):.4f}")
    
    # Rotated alignment (with watermark)
    rotation = OrthogonalRotation(
        config['watermark']['secret_key'], dim, 
        rotation_strength=config['watermark']['rotation_strength']
    )
    
    rotated_alignments = []
    for i in range(len(embeddings) - 1):
        alignment = rotation.compute_alignment_score(embeddings[i], embeddings[i+1])
        rotated_alignments.append(alignment)
    
    print(f"  Rotated alignment: {np.mean(rotated_alignments):.4f} ± {np.std(rotated_alignments):.4f}")
    
    # The key insight: differential score
    differential_scores = [r - n for r, n in zip(rotated_alignments, natural_similarities)]
    print(f"  Differential:      {np.mean(differential_scores):.4f} ± {np.std(differential_scores):.4f}")
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distribution comparison
    ax = axes[0]
    ax.hist(natural_similarities, bins=30, alpha=0.6, label='Natural Coherence', color='blue', density=True)
    ax.hist(rotated_alignments, bins=30, alpha=0.6, label='Rotated Alignment', color='red', density=True)
    ax.set_xlabel('Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Natural vs Rotated Score Distributions', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Differential scores
    ax = axes[1]
    ax.hist(differential_scores, bins=30, alpha=0.7, color='green', density=True)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
    ax.axvline(x=np.mean(differential_scores), color='black', linestyle='-', 
               linewidth=2, label=f'Mean = {np.mean(differential_scores):.3f}')
    ax.set_xlabel('Differential Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Differential Score Distribution (Human Text)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(results_dir / 'exp5_orthogonal_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  Saved: {results_dir / 'exp5_orthogonal_analysis.png'}")
    
    return {
        'natural_mean': np.mean(natural_similarities),
        'natural_std': np.std(natural_similarities),
        'rotated_mean': np.mean(rotated_alignments),
        'rotated_std': np.std(rotated_alignments),
        'differential_mean': np.mean(differential_scores),
        'differential_std': np.std(differential_scores),
    }


def run_anisotropy_experiment(config: dict, results_dir: Path, quick: bool = False):
    """Run full anisotropy analysis experiment."""
    print("\n" + "="*60)
    print("EXPERIMENT 5: Anisotropy Analysis")
    print("="*60)
    
    # Initialize
    print("\n[1/4] Initializing embedding extractor...")
    extractor = EmbeddingExtractor()
    
    # Analyze anisotropy
    print("\n[2/4] Analyzing embedding space anisotropy...")
    embeddings, anisotropy_metrics = analyze_embedding_anisotropy(
        extractor, ANALYSIS_CORPUS, results_dir
    )
    
    # Test rotation strengths
    print("\n[3/4] Testing rotation strengths...")
    rotation_results = test_rotation_strengths(config, embeddings, results_dir)
    
    # Orthogonal analysis
    print("\n[4/4] Comparing orthogonal rotation to natural flow...")
    orthogonal_results = test_orthogonal_vs_natural(config, embeddings, results_dir)
    
    # Summary and recommendations
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\n1. ANISOTROPY:")
    print(f"   IsoScore: {anisotropy_metrics['iso_score']:.3f}")
    if anisotropy_metrics['iso_score'] < 0.3:
        print("   ⚠️  HIGH ANISOTROPY: Embedding space is highly concentrated.")
        print("   → This may cause false positives if rotations stay within the 'cone'.")
    else:
        print("   ✓ Moderate isotropy: Good for watermarking.")
    
    print(f"\n2. ROTATION STRENGTH:")
    best_strength = None
    for r in rotation_results:
        if abs(r['mean_differential']) < 0.05 and r['std_differential'] < 0.1:
            best_strength = r['strength']
            break
    if best_strength:
        print(f"   Recommended strength: {best_strength}")
    else:
        print(f"   Current strength ({config['watermark']['rotation_strength']}) may need tuning")
    
    print(f"\n3. DIFFERENTIAL SCORING:")
    print(f"   Mean differential on human text: {orthogonal_results['differential_mean']:.4f}")
    if orthogonal_results['differential_mean'] < 0.05:
        print("   ✓ Good: Human text has low differential scores")
    else:
        print("   ⚠️  Warning: Human text may trigger false positives")
    
    # Save results
    all_results = {
        'anisotropy': anisotropy_metrics,
        'rotation_strengths': rotation_results,
        'orthogonal_analysis': orthogonal_results,
        'recommendations': {
            'suggested_threshold': max(0.1, orthogonal_results['differential_mean'] + 2*orthogonal_results['differential_std']),
            'anisotropy_level': 'high' if anisotropy_metrics['iso_score'] < 0.3 else 'moderate',
        }
    }
    
    with open(results_dir / 'exp5_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\nResults saved to: {results_dir / 'exp5_results.json'}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Anisotropy Analysis")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    
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
            }
        }
    
    results_dir = Path(__file__).parent.parent / "results" / "exp5"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = run_anisotropy_experiment(config, results_dir, args.quick)
    
    print("\n" + "="*60)
    print("EXPERIMENT 5 COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
