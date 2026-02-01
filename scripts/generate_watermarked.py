#!/usr/bin/env python
"""
Generate watermarked text using FULL LTW (Latent Trajectory Watermarking).

This script uses logit manipulation during generation to embed a secret
rotation pattern in the text's embedding trajectory.

Usage:
    uv run python scripts/generate_watermarked.py --n 10
    uv run python scripts/generate_watermarked.py --prompt "Explain AI" --n 5
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltw_watermark.generator import LTWGenerator


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate watermarked text (Full LTW)")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--n", type=int, default=5, help="Number of samples per prompt")
    parser.add_argument("--output", type=str, default="data/watermarked/")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (overrides config)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--watermark-strength", type=float, default=None, help="Override watermark strength")
    args = parser.parse_args()
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    config = load_config(str(config_path))
    
    # Determine watermark strength
    watermark_strength = args.watermark_strength or config['watermark'].get('watermark_strength', 2.0)
    
    # Initialize generator with FULL LTW
    print("\n" + "=" * 60)
    print("LTW Watermarked Text Generator (Full Implementation)")
    print("=" * 60)
    
    generator = LTWGenerator(
        model_name=config['model']['generator']['name'],
        secret_key=config['watermark']['secret_key'],
        rotation_strength=config['watermark']['rotation_strength'],
        watermark_strength=watermark_strength,
        embedding_model=config['model']['embedder']['name'],
        device=config['model']['generator'].get('device'),
    )
    
    # Get prompts
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = config['generation']['prompts']
    
    # Generate
    print(f"\nGenerating {args.n} samples per prompt ({len(prompts)} prompts)...")
    print(f"Watermark strength: {watermark_strength}")
    print(f"Max tokens: {args.max_tokens}")
    
    results = generator.generate_batch(
        prompts,
        n_per_prompt=args.n,
        max_new_tokens=args.max_tokens,
        temperature=config['model']['generator'].get('temperature', 0.7),
        top_p=config['model']['generator'].get('top_p', 0.9),
    )
    
    # Save
    output_dir = Path(__file__).parent.parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"watermarked_{timestamp}.jsonl"
    
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\n✓ Generated {len(results)} watermarked samples")
    print(f"  Saved to: {output_file}")
    
    # Show sample
    if results:
        print(f"\nSample output:")
        print(f"  Prompt: {results[0]['prompt'][:50]}...")
        print(f"  Text: {results[0]['text'][:100]}...")
        print(f"  Mean alignment: {results[0].get('mean_alignment', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
