# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Markov Chain-Lock (MCL) watermarking for AI-generated text. The package name `ltw_watermark` is historical — the original "Latent Trajectory Watermarking" (rotation in embedding space) was replaced by MCL, which partitions the vocabulary into `S` states via SHA-256 and forces generation to follow a secret Markov chain transition pattern. Detection re-derives the state sequence from the same secret key and checks how many transitions are valid; no LM needed at detection time.

Research code: paper draft lives in `docs/main.tex`; experimental results accumulate as dated dataset folders under `data/curated_wiki_dataset_*/`.

## Commands

```bash
# Install dependencies (uv)
uv sync

# Generate a full curated dataset: 176 Wikipedia concepts × non-watermarked baseline
# + watermarked for all (states × overlaps) combinations. Writes to
# data/curated_wiki_dataset_<timestamp>/.
uv run python scripts/generate_curated_dataset.py

# Narrow a run to a subset of configurations (used for parallel GPU runs across
# machines — pick disjoint state/overlap slices, one topology per invocation)
uv run python scripts/generate_curated_dataset.py \
  --topology soft_cycle --states 2 4 --overlaps 0 5 10 15 \
  --model meta-llama/Llama-3.2-3B-Instruct

# Resume a partially completed run (skips configs until the named one)
uv run python scripts/generate_curated_dataset.py --resume-from-config states7_overlap10pct

# Evaluate an existing non-watermarked baseline under every MCL config
uv run python scripts/evaluate_curated_non_watermarked.py

# Compare watermarked vs non-watermarked score/PPL distributions
uv run python scripts/compare_wm_vs_non_wm.py

# Robustness: apply paraphrase/edit attacks to watermarked samples and re-detect
uv run python scripts/robustness_attack.py
```

### Tests and demo are stale

`tests/test_watermark.py`, `tests/test_rotation.py`, `tests/test_embeddings.py`, and `src/ltw_watermark/demo.py` still import the deleted rotation-based API (`LTWWatermarker`, `ltw_watermark.watermark`, `ltw_watermark.rotation`, `ltw_watermark.embeddings`). They will fail to import. Don't cite them as the current API, and don't run `uv run pytest` expecting green — either port them to `MCLDetector`/`EnhancedMCLDetector` or delete them before trusting CI signal.

## Architecture

### The MCL algorithm

1. **Token → state**: `state(t) = SHA256(secret_key || token_id) mod S`. Deterministic, key-dependent partition of the vocabulary.
2. **Generation**: at each step, the transition matrix `T` says which next states are allowed from the current state. A precomputed per-state mask sets logits of all disallowed-state tokens to `-inf`; the model picks argmax (or samples) from what's left. Update the state, repeat.
3. **Detection**: tokenize, map every token to its state with the same key, count pairs `(s_i, s_{i+1})` where `T[s_i][s_{i+1}] > 0`. Score = valid / total; compare to the random baseline (mean of `T > 0`, which is `1/S` for clockwork, `2/S` for soft_cycle).

### Module layout (`src/ltw_watermark/`)

- **`mcl_watermark.py`** — `MCLGenerator` / `MCLDetector`: the original hard-partition clockwork version (`0→1→…→S-1→0`). Kept as the simplest reference implementation.
- **`enhanced_mcl.py`** — `EnhancedMCLGenerator` / `EnhancedMCLDetector`: the version used in current experiments. Adds (a) configurable transition topology via `chain_key` — built-ins are `"clockwork"`, `"binary"`, `"soft_cycle"` (two allowed successors with probs 0.67 / 0.33, random baseline `2/S`); any other string seeds a deterministic random topology. (b) Soft partitions via `overlap_ratio`: tokens can belong to multiple states, making generation smoother at the cost of a higher random baseline. `generate_transition_matrix` and `get_token_state_soft` / `precompute_soft_masks` are the central primitives.
- **`analysis.py`** — plotting utilities for ROC curves, score distributions, state trajectories.
- **`__init__.py`** — public exports: `MCLGenerator`, `MCLDetector`, `EnhancedMCLGenerator`, `EnhancedMCLDetector`.

### Experiment structure

Everything runs through `scripts/generate_curated_dataset.py`. Each invocation produces a timestamped folder `data/curated_wiki_dataset_<YYYYMMDD_HHMMSS>/` containing:
- `non_watermarked.jsonl` — the baseline (unless `--skip-non-watermarked`)
- one `<config_name>.jsonl` per (state, overlap) combination, where `config_name` looks like `states4_overlap10pct`
- `summary.json` — per-config aggregate stats (avg score, avg perplexity, detection rate, random baseline)

The curated concept list (176 Wikipedia titles spanning people, places, events, science, pop culture, etc.) is hardcoded in `WIKIPEDIA_CONCEPTS` at the top of the script. The prompt template is literally `f"Explain {concept} in a comprehensive way."`.

Recent runs have swept **models** (Llama-3.2-3B, Mistral, GPT2-XL) × **topologies** (clockwork, soft_cycle) × **states** (2, 4, 5, 7, 9, 11, 15) × **overlaps** (0, 5, 10, 15%). Runs are parallelized across GPUs by slicing the `--states` / `--overlaps` arg ranges — that's why commit history shows many partial dataset folders. The `--resume-from-config` flag recovers from crashes.

Notebooks under `notebook/` (`read_results.ipynb`, `results_table.ipynb`) consume those dataset folders to build cross-model comparison tables.

## Configuration

`config.yaml` is the default-values file; the scripts override it via CLI flags, so CLI args are authoritative for experiments. The keys that matter (all under `mcl:`):

- `secret_key` — shared secret; detector must use the same value
- `num_states` — 2–15 typical; larger S ⇒ lower random baseline ⇒ stronger detection but fewer tokens per state (worse fluency)
- `chain_key` — `"clockwork"` (baseline `1/S`), `"soft_cycle"` (baseline `2/S`, better fluency), `"binary"`, or any custom string for a deterministic random topology
- `overlap_ratio` — 0 = hard partition; higher values raise per-state token counts and smooth generation at the cost of detection power
- `detection_threshold` — score cutoff above which `is_watermarked=True`; with `clockwork` + `S=4` the random baseline is 0.25, so 0.5 is the standard threshold

## Gotchas

- The model's `config.vocab_size` can differ from the tokenizer's vocab size; the generators use the **model's** vocab size for masks, so if you swap tokenizers the mask indexing must follow the model. This is handled in `MCLGenerator.__init__`.
- `EnhancedMCLDetector` uses `get_token_state_soft(...)[0]` — the primary state — when decoding a sequence. Soft overlap makes generation easier but detection still treats each token as living in a single state.
- Mask precomputation with non-zero overlap is `O(vocab × num_states)` with a SHA-256 per cell; for large vocabs this is the slow part of startup, not model loading.
- `soft_cycle`'s transition matrix is stochastic with non-binary entries (0.67 / 0.33). The detector only checks whether an entry is `> 0`, so the actual weights don't affect detection — they're there so `get_allowed_next_states` returns both successors during generation.
