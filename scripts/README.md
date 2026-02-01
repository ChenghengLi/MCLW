# Scripts

This folder contains utility scripts for data generation and processing.

## Available Scripts

### `generate_watermarked.py`
Generate AI text WITH LTW watermark embedded.

```bash
# Generate 10 samples per prompt (uses prompts from config.yaml)
uv run python scripts/generate_watermarked.py --n 10

# Generate with custom prompt
uv run python scripts/generate_watermarked.py --n 5 --prompt "Explain quantum computing"

# Custom output location
uv run python scripts/generate_watermarked.py --n 10 --output data/watermarked/
```

### `generate_non_watermarked.py`
Generate AI text WITHOUT any watermark (control group).

```bash
uv run python scripts/generate_non_watermarked.py --n 10
```

### `load_data.py`
Utility to load data from all folders for experiments.

```python
from scripts.load_data import load_all_data, get_texts_and_labels

# Load all data
watermarked, non_watermarked, human = load_all_data()

# Get texts and labels for training/evaluation
texts, labels, categories = get_texts_and_labels()
```

## Data Flow

```
prompts (config.yaml)
        │
        ▼
┌───────────────────┐     ┌─────────────────────┐
│ generate_watermarked │     │ generate_non_watermarked │
└─────────┬─────────┘     └──────────┬──────────┘
          │                          │
          ▼                          ▼
   data/watermarked/          data/non_watermarked/
          │                          │
          └──────────┬───────────────┘
                     │
                     ▼
              load_data.py
                     │
                     ▼
              experiments/
```
