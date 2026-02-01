#!/usr/bin/env python
"""
Generate non-watermarked text (control group).

This generates standard AI text without any watermarking,
for comparison with LTW watermarked text.

Usage:
    uv run python scripts/generate_non_watermarked.py --n 10
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ltw_watermark.generator import NonWatermarkedGenerator


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Generate non-watermarked text")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--n", type=int, default=5, help="Number of samples per prompt")
    parser.add_argument("--output", type=str, default="data/non_watermarked/")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    
    config_path = Path(__file__).parent.parent / args.config
    config = load_config(str(config_path))
    
    print("\n" + "=" * 60)
    print("Non-Watermarked Text Generator (Control Group)")
    print("=" * 60)
    
    generator = NonWatermarkedGenerator(
        model_name=config['model']['generator']['name'],
        device=config['model']['generator'].get('device'),
    )
    
    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = config['generation']['prompts']
    
    print(f"\nGenerating {args.n} samples per prompt ({len(prompts)} prompts)...")
    
    results = generator.generate_batch(
        prompts,
        n_per_prompt=args.n,
        max_new_tokens=args.max_tokens,
        temperature=config['model']['generator'].get('temperature', 0.7),
        top_p=config['model']['generator'].get('top_p', 0.9),
    )
    
    output_dir = Path(__file__).parent.parent / args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"non_watermarked_{timestamp}.jsonl"
    
    with open(output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    
    print(f"\n✓ Generated {len(results)} non-watermarked samples")
    print(f"  Saved to: {output_file}")


if __name__ == "__main__":
    main()
