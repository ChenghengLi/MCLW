#!/usr/bin/env python
"""
Run all experiments using data from the data/ folder.

Usage:
    uv run python scripts/run_experiments.py
    uv run python scripts/run_experiments.py --exp 1 2 3
"""

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENTS = {
    1: ("Basic Detection", "experiments/exp1_basic_detection.py"),
    2: ("FPR Analysis", "experiments/exp2_false_positive_rate.py"),
    3: ("Paraphrase Robustness", "experiments/exp3_paraphrase_robustness.py"),
    4: ("Perplexity Comparison", "experiments/exp4_perplexity_comparison.py"),
    5: ("Anisotropy Analysis", "experiments/exp5_anisotropy_analysis.py"),
}


def run_experiment(exp_num: int, quick: bool = False):
    """Run a single experiment."""
    name, script = EXPERIMENTS[exp_num]
    print(f"\n{'='*60}")
    print(f"Running Experiment {exp_num}: {name}")
    print(f"{'='*60}\n")
    
    cmd = [sys.executable, script]
    if quick:
        cmd.append("--quick")
    
    result = subprocess.run(
        cmd,
        cwd=Path(__file__).parent.parent,
    )
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run LTW experiments")
    parser.add_argument("--exp", type=int, nargs="+", default=list(EXPERIMENTS.keys()),
                       help="Experiment numbers to run (default: all)")
    parser.add_argument("--quick", action="store_true", help="Quick mode for testing")
    args = parser.parse_args()
    
    print("LTW Watermark Experiments")
    print("=" * 60)
    print(f"Experiments to run: {args.exp}")
    print(f"Quick mode: {args.quick}")
    
    results = {}
    for exp_num in args.exp:
        if exp_num not in EXPERIMENTS:
            print(f"Warning: Unknown experiment {exp_num}, skipping")
            continue
        
        success = run_experiment(exp_num, args.quick)
        results[exp_num] = "✓ PASSED" if success else "✗ FAILED"
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for exp_num, status in results.items():
        name, _ = EXPERIMENTS[exp_num]
        print(f"  Exp {exp_num}: {name:<25} {status}")


if __name__ == "__main__":
    main()
