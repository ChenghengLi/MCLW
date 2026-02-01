# Markov Chain-Lock (MCL) Watermarking

A research implementation of **Markov Chain-Lock Watermarking** for embedding verifiable fingerprints in AI-generated text. This method provides provable detection guarantees with exponentially small false positive rates.

---

## Overview

**MCL Watermarking** embeds a cryptographic fingerprint by forcing the language model to generate tokens that follow a secret **Markov chain state pattern**. Unlike passive detection methods (perplexity-based), this is an *active* watermarking approach where the signal is deliberately embedded during generation.

### Key Concept: The Fingerprint

A **fingerprint** is the hidden sequence of Markov chain states that tokens traverse:

```
Tokens:     t₁  →  t₂  →  t₃  →  t₄  →  t₅  → ...
              ↓      ↓      ↓      ↓      ↓
States:      0  →   1  →   2  →   3  →   0  → ...  (clockwork cycle)
```

Each token is secretly assigned to a state via `state = SHA256(secret_key || token_id) mod S`. The watermark forces consecutive tokens to follow the transition pattern (e.g., 0→1→2→3→0→...).

### Comparison with Other Methods

| Method | Type | Detection | False Positive Rate | Robustness |
|--------|------|-----------|---------------------|------------|
| **Perplexity** | Passive | Model-based | High on formal text | Weak |
| **Green-Red Lists** | Active | Statistical | Low | Weak to paraphrase |
| **MCL (This Project)** | Active | Hash-based | **Exponentially small** | **Strong** |

---

## How MCL Watermarking Works

### 1. Vocabulary Partitioning

The vocabulary is partitioned into `S` disjoint sets using a cryptographic hash:

```python
def get_token_state(token_id: int, secret_key: str, num_states: int) -> int:
    """Assign each token to a state (0 to S-1) using SHA-256."""
    data = f"{secret_key}-{token_id}".encode()
    hash_val = int(hashlib.sha256(data).hexdigest(), 16)
    return hash_val % num_states
```

For a 128,000-token vocabulary with S=4 states:
- State 0: ~32,000 tokens
- State 1: ~32,000 tokens  
- State 2: ~32,000 tokens
- State 3: ~32,000 tokens

### 2. Constrained Generation (Embedding)

During text generation, we constrain the model to only select tokens from the **required next state**:

```
Algorithm: MCL Embedding

1. Get current state from previous token: s_current = σ(last_token)
2. Determine required next state: s_next = (s_current + 1) mod S
3. Mask all tokens NOT in state s_next to -∞ logits
4. Sample/argmax from remaining valid tokens
5. Repeat
```

This produces text where every consecutive token pair follows the state transition pattern.

### 3. Detection (Verifying the Fingerprint)

Detection is simple and **does not require the language model**:

```
Algorithm: MCL Detection

1. Tokenize the text
2. Map each token to its state using the secret key
3. Count how many transitions follow the expected pattern
4. Score = valid_transitions / total_transitions
5. If score > threshold: watermarked
```

**Example**:
- Watermarked text: Score ≈ 1.00 (100% valid transitions)
- Random text: Score ≈ 0.25 (for S=4, by chance)

---

## Mathematical Guarantees

### Theorem 1: Exponential Detection Power

For random (non-watermarked) text of length `n` tokens with `S` states:

```
P(false positive) ≤ exp(-2(n-1)(τ - 1/S)²)
```

| Text Length | S=4, τ=0.5 | False Positive Rate |
|:-----------:|:----------:|:-------------------:|
| 50 tokens | | 1.8 × 10⁻³ |
| 100 tokens | | 3.2 × 10⁻⁶ |
| 200 tokens | | 1.0 × 10⁻¹¹ |
| 500 tokens | | 1.0 × 10⁻²⁷ |

### Theorem 2: Robustness to Modifications

If an adversary modifies fraction `δ` of tokens:

```
Expected score = (1-δ)² + δ(2-δ)/S
```

With S=4:
- 10% modifications → Expected score ≈ 0.86 (still detectable)
- 20% modifications → Expected score ≈ 0.70 (still detectable)
- 30% modifications → Expected score ≈ 0.54 (marginally detectable)

---

## Transition Topologies

We support multiple Markov chain transition patterns:

### Clockwork (Default)
Strict cycle: 0 → 1 → 2 → 3 → 0 → ...
```
Transition Matrix:
    0  1  2  3
0 [ 0  1  0  0 ]
1 [ 0  0  1  0 ]
2 [ 0  0  0  1 ]
3 [ 1  0  0  0 ]
```
- Random baseline: 25%
- Strongest detection, but most restrictive

### Soft Cycle
Each state can transition to next OR skip one:
- Random baseline: 50%
- Better text quality, weaker detection

### Binary Alternation
Two states only: 0 → 1 → 0 → 1 → ...
- Random baseline: 50%
- 50% of vocabulary available per token

---

## Soft Partitions (Overlap)

To improve text quality, tokens can belong to **multiple states** with configurable overlap:

| Overlap ρ | Tokens per State | Quality | Detection |
|:---------:|:----------------:|:-------:|:---------:|
| 0% | 25% of vocab | Lower | Strongest |
| 5% | 29% of vocab | OK | Strong |
| 10% | 33% of vocab | Good | Good |
| 15% | 36% of vocab | Better | Moderate |

---

## Installation

```bash
# Clone and install
cd LTW
uv sync

# Or with pip
pip install -e .
```

## Project Structure

```
LTW/
├── src/ltw_watermark/
│   ├── mcl_watermark.py      # Basic MCL implementation
│   ├── enhanced_mcl.py       # Soft partitions & custom transitions
│   └── analysis.py           # Visualization utilities
├── scripts/
│   ├── generate_curated_dataset.py      # Generate watermarked datasets
│   ├── generate_large_wikipedia_dataset.py
│   └── load_data.py          # Data loading utilities
├── data/                     # Generated datasets
├── config.yaml               # Configuration
└── pyproject.toml            # Dependencies
```

## Usage

### Generate Watermarked Text

```python
from ltw_watermark import EnhancedMCLGenerator

generator = EnhancedMCLGenerator(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    secret_key="your-secret-key",
    num_states=4,
    chain_key="clockwork",
    overlap_ratio=0.10  # 10% soft overlap
)

text, metadata = generator.generate("Explain quantum computing")
print(text)
print(f"Tokens generated: {metadata['tokens_generated']}")
```

### Detect Watermark

```python
from ltw_watermark import EnhancedMCLDetector

detector = EnhancedMCLDetector(
    secret_key="your-secret-key",  # Must match!
    num_states=4,
    chain_key="clockwork",
    detection_threshold=0.5
)

result = detector.detect(text)
print(f"Watermarked: {result.is_watermarked}")
print(f"Score: {result.chain_score:.2%}")  # ~100% for watermarked
print(f"Random baseline: {result.expected_random:.2%}")  # ~25%
```

### Generate Dataset with Multiple Configurations

```bash
# Generate watermarked samples across different state/overlap combinations
uv run python scripts/generate_curated_dataset.py
```

This generates files like:
- `states4_overlap0pct.jsonl`
- `states4_overlap10pct.jsonl`
- `states5_overlap5pct.jsonl`
- etc.

---

## Configuration

Edit `config.yaml`:

```yaml
model:
  generator:
    name: "meta-llama/Llama-3.2-3B-Instruct"
    device: "cuda"
    
watermark:
  secret_key: "your-secret-key-change-me"
  detection_threshold: 0.5
```

---

## References

1. Kirchenbauer et al. (2023) - "A Watermark for Large Language Models" (ICML)
2. SemStamp (NAACL 2024) - "A Semantic Watermark with Paraphrastic Robustness"
3. Aaronson (2022) - Cryptographic watermarking concepts

---

## License

MIT License
