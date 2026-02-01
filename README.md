# Latent Trajectory Watermarking (LTW)

A research implementation of **Latent Trajectory Watermarking** for detecting AI-generated text. This project compares LTW with traditional perplexity-based detection methods.

## Overview

**LTW** (Latent Trajectory Watermarking) is an *active* watermarking method that embeds a secret mathematical signature in AI-generated text by controlling the trajectory of word embeddings in semantic space.

### Key Differences from Perplexity-Based Detection

| Feature | Perplexity (GPTZero-style) | LTW (This Project) |
|---------|---------------------------|-------------------|
| Type | Passive scanning | Active watermarking |
| Mechanism | "Is this text predictable?" | "Does this follow my secret rotation?" |
| False Positives | High on formal text | Low (uses differential scoring) |
| Robustness | Weak to prompting tricks | Robust to paraphrasing |

## Installation

```bash
# Clone the repository
cd apart-tech-gov

# Install with UV
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Basic Detection Demo
```bash
uv run python experiments/exp1_basic_detection.py
```

### Run All Experiments
```bash
# Experiment 1: Basic watermark detection
uv run python experiments/exp1_basic_detection.py

# Experiment 2: False positive rate analysis
uv run python experiments/exp2_false_positive_rate.py

# Experiment 3: Paraphrase robustness
uv run python experiments/exp3_paraphrase_robustness.py

# Experiment 4: Comparison with perplexity
uv run python experiments/exp4_perplexity_comparison.py

# Experiment 5: Anisotropy analysis
uv run python experiments/exp5_anisotropy_analysis.py
```

### Run Tests
```bash
uv run pytest tests/ -v
```

## Project Structure

```
apart-tech-gov/
├── src/ltw_watermark/        # Core library
│   ├── embeddings.py         # Embedding extraction
│   ├── rotation.py           # Rotation matrix generation
│   ├── watermark.py          # LTW detector
│   ├── perplexity.py         # Perplexity baseline
│   └── analysis.py           # Visualization utilities
├── experiments/              # Research experiments
│   ├── exp1_basic_detection.py
│   ├── exp2_false_positive_rate.py
│   ├── exp3_paraphrase_robustness.py
│   ├── exp4_perplexity_comparison.py
│   └── exp5_anisotropy_analysis.py
├── tests/                    # Unit tests
├── results/                  # Experiment outputs
├── config.yaml              # Configuration
└── pyproject.toml           # Dependencies
```

## How LTW Works

### The Secret Rotation Pattern

LTW embeds a watermark by making consecutive text segments follow a **secret rotation pattern** in embedding space:

```
Text:    [Word/Sentence 1] → [Word/Sentence 2] → [Word/Sentence 3] → ...
Embeddings:     E₁ → E₂ → E₃ → ...
                 ↓
Watermark:  E₂ ≈ R·E₁  (E₂ aligns with rotated E₁)
            E₃ ≈ R·E₂  (E₃ aligns with rotated E₂)
            ...
            where R is the SECRET rotation matrix
```

### 1. Watermark Injection (during generation)

```python
from ltw_watermark import LTWGenerator

generator = LTWGenerator(
    model_name="gpt2",
    secret_key="your-secret-key",
    rotation_strength=0.3,      # Rotation angle
    watermark_strength=2.0,     # Logit bias strength
)

text, metadata = generator.generate("Explain machine learning")
# text now contains the watermark pattern!
```

**How it works internally:**
1. At each token, get logits from the language model
2. Compute embeddings for top-k candidate tokens
3. Score each by alignment with `R · previous_embedding`
4. Add bias to logits: high-alignment tokens get bonus
5. Sample token (now biased towards watermark pattern)

### 2. Watermark Detection

```python
from ltw_watermark import LTWWatermarker

detector = LTWWatermarker(
    secret_key="your-secret-key",  # Must match!
    detection_threshold=0.1
)

result = detector.detect(text)
print(f"Watermarked: {result.is_watermarked}")
print(f"Score: {result.differential_score:.4f}")
```

### 3. Differential Scoring (reduces false positives)

```
differential_score = watermark_alignment - natural_coherence
```

- **Watermark alignment**: How well does the trajectory follow R?
- **Natural coherence**: How semantically similar are consecutive parts?

Human text has high natural coherence → low differential score.
Watermarked AI text has high watermark alignment → high differential score.

## Configuration

Edit `config.yaml` to adjust parameters:

```yaml
watermark:
  secret_key: "your-secret-key"
  rotation_strength: 0.3      # 0-1, higher = stronger watermark
  detection_threshold: 0.1    # Score threshold for detection
  use_differential_scoring: true
  unit: "sentence"            # "word" or "sentence"
```

## Research Analysis

### Will This Method Work?

**Short answer:** Conditionally yes, with caveats.

#### Strengths:
- **Low false positives** when using differential scoring
- **Robust to paraphrasing** (operates in semantic space)
- **Mathematically verifiable** (not guessing based on statistics)

#### Challenges:
1. **Anisotropy Problem**: Embedding spaces are not uniformly distributed. If rotations stay within the natural "cone" of embeddings, false positives increase.

2. **Rotation Strength Trade-off**: 
   - Too weak → Hard to detect
   - Too strong → Affects text quality

3. **Comparison to State of Art**:
   | Method | FPR | Notes |
   |--------|-----|-------|
   | Green-Red List (Kirchenbauer) | <0.01% | Token-level, breaks under paraphrasing |
   | SemStamp | 1-5% | Sentence-level, semantic |
   | LTW (this) | Variable | Depends on rotation optimization |

### Key Experiment Results

Run the experiments to see:
- **Exp 2**: FPR at different thresholds
- **Exp 3**: Survival rate under paraphrasing
- **Exp 5**: Anisotropy analysis and recommendations

## Literature Context

This work relates to:
- Kirchenbauer et al. (2023) - "A Watermark for Large Language Models"
- SemStamp (NAACL 2024) - "A Semantic Watermark with Paraphrastic Robustness"
- Ethayarajh (2019) - "How Contextual are Contextualized Word Representations?"

## License

MIT License - See LICENSE file for details.
