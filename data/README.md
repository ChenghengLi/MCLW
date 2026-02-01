# Data Directory

This folder contains datasets for LTW watermarking experiments.

## Structure

```
data/
├── watermarked/       # AI-generated text WITH watermark
├── non_watermarked/   # AI-generated text WITHOUT watermark  
├── human/             # Human-written text (you provide this)
└── README.md
```

## File Format

Each subfolder should contain `.txt` files with one sample per file, or a single `.jsonl` file with format:
```json
{"id": "001", "text": "Your text here..."}
{"id": "002", "text": "Another sample..."}
```

## Usage

The experiments will automatically load data from these folders using `scripts/load_data.py`.

## Generating Data

Use the scripts in `scripts/` folder:
- `generate_watermarked.py` - Generate text with LTW watermark
- `generate_non_watermarked.py` - Generate text without watermark
