#!/usr/bin/env python
"""
Experiment 1: Basic Watermark Detection

This experiment demonstrates the core LTW watermarking concept:
1. Generate sample texts (simulating watermarked AI output)
2. Apply watermark detection with the correct secret key
3. Show detection fails with incorrect key
4. Visualize the embedding trajectories
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import yaml
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltw_watermark.embeddings import EmbeddingExtractor
from ltw_watermark.rotation import OrthogonalRotation
from ltw_watermark.watermark import LTWWatermarker
from ltw_watermark.analysis import plot_trajectory_2d, plot_score_distribution


# Sample texts for testing (simulating AI-generated watermarked text)
SAMPLE_WATERMARKED_TEXTS = [
    "The quantum computer operates by utilizing qubits. These qubits can exist in superposition states. This allows parallel computation of multiple values. The result is exponentially faster processing for certain problems.",
    "Machine learning models learn patterns from data. Neural networks use layers of neurons. Each layer transforms the input representation. Deep networks can learn hierarchical features automatically.",
    "Climate change affects global weather patterns significantly. Rising temperatures cause ice caps to melt. Sea levels increase threatening coastal communities. Urgent action is needed to reduce emissions.",
    "The stock market reflects economic sentiment. Investors analyze trends and patterns carefully. Algorithms now dominate trading decisions. Volatility has increased in recent years.",
    "Artificial intelligence continues to advance rapidly. Natural language processing enables chatbots. Computer vision powers autonomous vehicles. These technologies transform industries worldwide.",
]

# Human-written texts for comparison
SAMPLE_HUMAN_TEXTS = [
    "I went to the store yesterday and, well, you know how it is—couldn't find parking anywhere! Ended up walking two blocks in the rain. Worth it though, they had those cookies I love.",
    "So my friend Sarah, she's absolutely crazy about birds. Last week she saw this rare warbler or something? I don't really get it but she was over the moon. Made my day just seeing her so happy!",
    "The thing about cooking is... you gotta feel it, right? Recipes are just suggestions. A pinch here, a dash there. My grandmother never measured anything and her food was amazing.",
    "Been reading this book—can't remember the title now—but it's about some detective in 1920s Chicago. Kinda slow at first, picked up around chapter 5 though. Would recommend maybe?",
    "Traffic was absolutely INSANE this morning!! Three accidents on the highway. Took me an hour and a half to get to work. Coffee was cold by the time I arrived. Mondays, am I right?",
]


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_basic_detection_demo(config: dict, results_dir: Path):
    """Run the basic detection demonstration."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Basic Watermark Detection Demo")
    print("="*60)
    
    secret_key = config['watermark']['secret_key']
    wrong_key = "wrong-secret-key-12345"
    
    # Initialize detectors
    print("\n[1/4] Initializing LTW watermarker...")
    correct_detector = LTWWatermarker(
        secret_key=secret_key,
        rotation_strength=config['watermark']['rotation_strength'],
        detection_threshold=config['watermark']['detection_threshold'],
        use_differential_scoring=config['watermark']['use_differential_scoring'],
        unit=config['watermark']['unit']
    )
    
    wrong_detector = LTWWatermarker(
        secret_key=wrong_key,
        rotation_strength=config['watermark']['rotation_strength'],
        detection_threshold=config['watermark']['detection_threshold'],
        use_differential_scoring=config['watermark']['use_differential_scoring'],
        unit=config['watermark']['unit']
    )
    
    # Test detection on "watermarked" texts with correct key
    print("\n[2/4] Testing detection on watermarked texts...")
    print("-" * 40)
    
    watermarked_scores = []
    human_scores = []
    
    print("\n--- Watermarked Texts (Simulated AI) ---")
    for i, text in enumerate(SAMPLE_WATERMARKED_TEXTS):
        result_correct = correct_detector.detect(text)
        result_wrong = wrong_detector.detect(text)
        
        watermarked_scores.append(result_correct.differential_score)
        
        print(f"\nText {i+1} (first 50 chars): \"{text[:50]}...\"")
        print(f"  Correct Key - Diff Score: {result_correct.differential_score:.4f}, "
              f"Detected: {result_correct.is_watermarked}")
        print(f"  Wrong Key   - Diff Score: {result_wrong.differential_score:.4f}, "
              f"Detected: {result_wrong.is_watermarked}")
    
    # Test on human texts
    print("\n\n--- Human Texts ---")
    for i, text in enumerate(SAMPLE_HUMAN_TEXTS):
        result = correct_detector.detect(text)
        human_scores.append(result.differential_score)
        
        print(f"\nText {i+1} (first 50 chars): \"{text[:50]}...\"")
        print(f"  Diff Score: {result.differential_score:.4f}, "
              f"Detected as Watermarked: {result.is_watermarked}")
    
    # Summary statistics
    print("\n\n[3/4] Summary Statistics")
    print("-" * 40)
    print(f"Watermarked texts - Mean diff score: {np.mean(watermarked_scores):.4f} "
          f"(std: {np.std(watermarked_scores):.4f})")
    print(f"Human texts       - Mean diff score: {np.mean(human_scores):.4f} "
          f"(std: {np.std(human_scores):.4f})")
    print(f"Detection threshold: {config['watermark']['detection_threshold']}")
    
    # Visualization
    print("\n[4/4] Generating visualizations...")
    
    # Score distribution
    fig = plot_score_distribution(
        watermarked_scores, human_scores,
        score_name="Differential Score",
        threshold=config['watermark']['detection_threshold'],
        title="Differential Score Distribution: Watermarked vs Human",
        save_path=str(results_dir / "exp1_score_distribution.png")
    )
    print(f"  Saved: {results_dir / 'exp1_score_distribution.png'}")
    
    # Trajectory visualization for first text
    text = SAMPLE_WATERMARKED_TEXTS[0]
    units, embeddings, align_scores, diff_scores = correct_detector.analyze_trajectory(text)
    
    if len(embeddings) > 1:
        # Calculate expected rotated positions
        rotated = np.array([
            correct_detector.rotation.rotate(embeddings[i])
            for i in range(len(embeddings) - 1)
        ])
        
        fig = plot_trajectory_2d(
            embeddings, units,
            watermarked_rotations=rotated,
            title="Embedding Trajectory: Actual vs Expected Rotation",
            save_path=str(results_dir / "exp1_trajectory.png")
        )
        print(f"  Saved: {results_dir / 'exp1_trajectory.png'}")
    
    # Return summary
    return {
        "watermarked_mean_score": np.mean(watermarked_scores),
        "watermarked_std_score": np.std(watermarked_scores),
        "human_mean_score": np.mean(human_scores),
        "human_std_score": np.std(human_scores),
        "threshold": config['watermark']['detection_threshold'],
        "separation": np.mean(watermarked_scores) - np.mean(human_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: Basic Watermark Detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if config_path.exists():
        config = load_config(str(config_path))
    else:
        print(f"Warning: Config not found at {config_path}, using defaults")
        config = {
            'watermark': {
                'secret_key': 'test-secret-key',
                'rotation_strength': 0.3,
                'detection_threshold': 0.1,
                'use_differential_scoring': True,
                'unit': 'sentence'
            }
        }
    
    # Setup results directory
    results_dir = Path(__file__).parent.parent / "results" / "exp1"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    results = run_basic_detection_demo(config, results_dir)
    
    # Print final summary
    print("\n" + "="*60)
    print("EXPERIMENT 1 COMPLETE")
    print("="*60)
    print(f"Score separation: {results['separation']:.4f}")
    print(f"Results saved to: {results_dir}")
    
    # Analysis insight
    if results['separation'] > 0:
        print("\n✓ POSITIVE: Watermarked texts have higher differential scores on average.")
    else:
        print("\n✗ CONCERN: Watermarked texts NOT distinguishable. Consider adjusting parameters.")


if __name__ == "__main__":
    main()
