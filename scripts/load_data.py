#!/usr/bin/env python
"""
Load data from the data folders for experiments.

Provides utilities to load watermarked, non-watermarked, and human text
from the data/ directory structure.

Usage:
    from scripts.load_data import load_all_data, load_watermarked, load_human
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml


def load_config(config_path: Optional[str] = None) -> dict:
    """Load configuration."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_txt_files(directory: Path) -> List[Dict]:
    """Load data from individual .txt files."""
    data = []
    for txt_file in sorted(directory.glob("*.txt")):
        with open(txt_file, 'r') as f:
            text = f.read().strip()
            data.append({
                "id": txt_file.stem,
                "text": text,
            })
    return data


def load_from_directory(directory: Path) -> List[Dict]:
    """Load all data from a directory (supports .jsonl and .txt)."""
    data = []
    
    # Load JSONL files
    for jsonl_file in directory.glob("*.jsonl"):
        data.extend(load_jsonl(jsonl_file))
    
    # Load TXT files
    txt_data = load_txt_files(directory)
    data.extend(txt_data)
    
    return data


def load_watermarked(config: Optional[dict] = None) -> List[Dict]:
    """Load watermarked text data."""
    if config is None:
        config = load_config()
    
    directory = Path(__file__).parent.parent / config['data']['watermarked_dir']
    data = load_from_directory(directory)
    
    # Ensure watermarked flag is set
    for item in data:
        item['watermarked'] = True
        item['label'] = 'watermarked'
    
    return data


def load_non_watermarked(config: Optional[dict] = None) -> List[Dict]:
    """Load non-watermarked AI text data."""
    if config is None:
        config = load_config()
    
    directory = Path(__file__).parent.parent / config['data']['non_watermarked_dir']
    data = load_from_directory(directory)
    
    for item in data:
        item['watermarked'] = False
        item['label'] = 'non_watermarked'
    
    return data


def load_human(config: Optional[dict] = None) -> List[Dict]:
    """Load human-written text data."""
    if config is None:
        config = load_config()
    
    directory = Path(__file__).parent.parent / config['data']['human_dir']
    data = load_from_directory(directory)
    
    for item in data:
        item['watermarked'] = False
        item['label'] = 'human'
    
    return data


def load_all_data(config: Optional[dict] = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load all data from all three categories.
    
    Returns:
        Tuple of (watermarked_data, non_watermarked_data, human_data)
    """
    watermarked = load_watermarked(config)
    non_watermarked = load_non_watermarked(config)
    human = load_human(config)
    
    return watermarked, non_watermarked, human


def get_texts_and_labels(
    include_watermarked: bool = True,
    include_non_watermarked: bool = True,
    include_human: bool = True,
    config: Optional[dict] = None
) -> Tuple[List[str], List[bool], List[str]]:
    """
    Get texts and labels for experiments.
    
    Returns:
        Tuple of (texts, is_watermarked_labels, category_labels)
    """
    texts = []
    is_watermarked = []
    categories = []
    
    if include_watermarked:
        for item in load_watermarked(config):
            texts.append(item['text'])
            is_watermarked.append(True)
            categories.append('watermarked')
    
    if include_non_watermarked:
        for item in load_non_watermarked(config):
            texts.append(item['text'])
            is_watermarked.append(False)
            categories.append('non_watermarked')
    
    if include_human:
        for item in load_human(config):
            texts.append(item['text'])
            is_watermarked.append(False)
            categories.append('human')
    
    return texts, is_watermarked, categories


def print_data_summary():
    """Print summary of available data."""
    watermarked, non_watermarked, human = load_all_data()
    
    print("Data Summary")
    print("=" * 40)
    print(f"Watermarked:     {len(watermarked):5d} samples")
    print(f"Non-watermarked: {len(non_watermarked):5d} samples")
    print(f"Human:           {len(human):5d} samples")
    print("=" * 40)
    print(f"Total:           {len(watermarked) + len(non_watermarked) + len(human):5d} samples")


if __name__ == "__main__":
    print_data_summary()
