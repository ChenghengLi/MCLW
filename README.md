# Markov Chain-Lock (MCL) Watermarking

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A cryptographically secure framework for embedding verifiable fingerprints in LLM-generated text with **provable detection guarantees**.

---

## 🎯 How MCL Watermarking Works

MCL watermarking embeds a hidden "fingerprint" by forcing the language model to generate tokens that follow a secret **Markov chain state pattern**.

### Step 1: Vocabulary Partitioning

Every token in the vocabulary is assigned to a state (0 to S-1) using SHA-256:

```
state = SHA256(secret_key + token_id) mod S
```

For S=4 states with a 128K vocabulary:
- State 0: ~32,000 tokens
- State 1: ~32,000 tokens
- State 2: ~32,000 tokens
- State 3: ~32,000 tokens

### Step 2: Constrained Generation (Embedding)

During text generation, only tokens from valid **successor states** are allowed:

```
Soft Cycle (k=2): State s → next state must be (s+1) mod S or (s+2) mod S

Example with S=4:
  Token 1 → State 0 → Allow states {1, 2}
  Token 2 → State 1 → Allow states {2, 3}
  Token 3 → State 2 → Allow states {3, 0}
  ...
```

All other tokens are masked to `-∞` logits before sampling.

### Step 3: Detection (No Model Needed!)

Detection is **model-free** and runs in O(n) time:

1. Tokenize the text
2. Map each token to its state using the secret key
3. Count valid transitions (consecutive tokens following the chain)
4. `Score = valid_transitions / total_transitions`
5. If `Score > threshold`: **Watermarked**

| Text Type | Expected Score (S=7, Soft Cycle) |
|-----------|----------------------------------|
| Watermarked | ~0.99 (all transitions valid) |
| Random | ~0.29 (2/7 by chance) |

---

## 🚀 Quick Start

```bash
# Clone and install
git clone https://github.com/ChenghengLi/MCLW.git
cd MCLW
uv sync  # or: pip install -e .

# Set your HuggingFace token (for Llama access)
export HF_TOKEN="your-huggingface-token"

# Run the main experiment
uv run python scripts/generate_curated_dataset.py
```

---

## 📁 Project Structure

```
MCLW/
├── src/mcl_watermark/           # Core library
│   ├── __init__.py              # Package exports
│   ├── mcl_watermark.py         # Basic MCL watermarking (clockwork)
│   └── enhanced_mcl.py          # Soft cycle, overlaps, custom transitions
│
├── scripts/                     # Executable scripts
│   ├── generate_curated_dataset.py        # Main experiment (28 configs)
│   ├── generate_large_wikipedia_dataset.py # Large-scale generation
│   ├── compare_wm_vs_non_wm.py            # Comparison analysis
│   ├── evaluate_curated_non_watermarked.py # Baseline evaluation
│   ├── robustness_attack.py               # Adversarial attack tests
│   └── load_data.py                       # Data loading utilities
│
├── experiments/                 # Additional experiments
│   └── robustness_test.py       # Word replacement robustness
│
├── data/                        # Generated datasets (auto-created)
├── docs/                        # Paper (main.tex)
├── config.yaml                  # Configuration
└── LICENSE                      # MIT License
```

---

## ⚙️ Configuration Guide

Edit `config.yaml` to customize the watermarking system:

```yaml
# =============================================
# MODEL SETTINGS
# =============================================
model:
  generator:
    name: "meta-llama/Llama-3.2-3B-Instruct"  # HuggingFace model
    device: "cuda"                             # "cuda" or "cpu"
    max_length: 256                            # Max tokens per generation

# =============================================
# WATERMARK SETTINGS
# =============================================
mcl:
  secret_key: "your-secret-key"   # CHANGE THIS! Determines state assignment
  num_states: 7                   # Number of states (5-11 recommended)
  chain_key: "soft_cycle"         # Transition topology (see below)
  overlap_ratio: 0.0              # Soft partition overlap (0.0 = hard)
  detection_threshold: 0.5        # Score threshold for detection
```

### Transition Topologies (`chain_key`)

| Topology | Valid Successors | Random Baseline | Use Case |
|----------|------------------|-----------------|----------|
| `clockwork` | s → (s+1) mod S | 1/S | Maximum security |
| `soft_cycle` | s → {(s+1), (s+2)} mod S | 2/S | **Recommended** |

### State Count (`num_states`)

| States (S) | Vocab per State | Detection Power | Quality |
|:----------:|:---------------:|:---------------:|:-------:|
| 2 | 50% | ❌ None (all valid for soft cycle) | Best |
| 4 | 25% | ⚠️ Weak (33% FPR for soft cycle) | Good |
| **7** | 14.3% | ✅ **Perfect** | **Optimal** |
| 11 | 9.1% | ✅ Perfect | Lower |

### Overlap Ratio (`overlap_ratio`)

| Overlap (ρ) | Vocab per State | Detection | Quality |
|:-----------:|:---------------:|:---------:|:-------:|
| **0%** | 14.3% | ✅ 100% | Standard |
| 5% | ~19% | ⚠️ 92% | Better |
| 10% | ~24% | ❌ 35% | Good |

**Recommendation**: Use `num_states: 7`, `chain_key: soft_cycle`, `overlap_ratio: 0.0`

---

## 💻 Python API

### Generate Watermarked Text

```python
from mcl_watermark import EnhancedMCLGenerator

generator = EnhancedMCLGenerator(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    secret_key="my-secret-key-2024",
    num_states=7,
    chain_key="soft_cycle",
    overlap_ratio=0.0
)

text, metadata = generator.generate("Explain quantum computing")
print(f"Score: {metadata['chain_score']:.3f}")  # ~0.99
```

### Detect Watermark

```python
from mcl_watermark import EnhancedMCLDetector

detector = EnhancedMCLDetector(
    secret_key="my-secret-key-2024",  # Must match!
    num_states=7,
    chain_key="soft_cycle",
    detection_threshold=0.5
)

result = detector.detect(text)
print(f"Watermarked: {result.is_watermarked}")  # True
print(f"Score: {result.chain_score:.2%}")        # ~99%
```

---

## 🔧 How to Run

### Generate Watermarked Dataset

```bash
uv run python scripts/generate_curated_dataset.py
```

### Run Robustness Tests

```bash
uv run python experiments/robustness_test.py
```

### Compare Watermarked vs Non-Watermarked

```bash
uv run python scripts/compare_wm_vs_non_wm.py
```

---

## 📊 Key Results

| States | Overlap | Detection | FPR | PPL |
|:------:|:-------:|:---------:|:---:|:---:|
| **7** | **0%** | **100%** | **0%** | **4.20** |
| 9 | 0% | 100% | 0% | 5.37 |
| 11 | 0% | 100% | 0% | 4.61 |

**Robustness**: Detection remains >96% even with 30% word replacement.

---

## 📄 Paper

See [`docs/Report.pdf`](docs/Report.pdf) for the full paper:

> **Markov Chain Lock Watermarking: Provably Secure Authentication for LLM Outputs**

```bash
cd docs && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## 📜 License

[MIT License](LICENSE) - Chengheng Li & Kyuhee Kim, 2026
