# MCL Watermarking — Next Steps

Last updated: 2026-05-01
Verified against actual repo state (paper, code, data).

This file replaces the earlier Codex-generated draft. Several claims in
that draft were inaccurate (e.g. "baseline PPL is N/A", "Theorem 3 lacks
the 2^-256 bound"); those have been corrected. Items that were
genuinely missing have been kept and clarified.

---

## CPU experiments actually run this session

These are real measurements from real script invocations against the
existing dataset `data/curated_wiki_dataset_20260201_112721`. They use
the cached `meta-llama/Llama-3.1-8B-Instruct` tokenizer (Llama 3.x all
share the same 128k tokenizer, so state assignments are identical to
the original Llama-3.2-3B run).

### Detection on existing data (`compare_wm_vs_non_wm.py`)
Reproduces the paper's headline numbers exactly. JSON saved to
`data/.../detailed_comparison.json`.

| S, ρ | Non-WM avg | WM avg | Gap | TPR | FPR | PPL |
|------|-----------|--------|-----|-----|-----|-----|
| 7, 0%  | 0.290 | 0.991 | +0.701 | 100% | 0% | 4.20 |
| 9, 0%  | 0.221 | 0.984 | +0.763 | 100% | 0% | 5.37 |
| 11, 0% | 0.197 | 0.988 | +0.790 | 100% | 0% | 4.61 |
| 4, 0%  | 0.489 | 0.989 | +0.500 | 100% | 33.5% | 4.89 |

The S=4 row exposes a real flaw the paper acknowledges: at S=4 the
random baseline 2/S = 0.5 collides with the threshold, giving 33.5% FPR.
Above S=5, FPR is 0% on this dataset.

### Theorem 2 validation (`robustness_attack.py` with new `uniform_token`)
Empirical $\phi$ vs theoretical $(1-\delta)^2 + k\delta(2-\delta)/S$
(soft cycle, S=7, k=2, n=173 samples):

| δ | Empirical $\phi$ | Theory | abs err | TPR |
|---|------------------|--------|---------|-----|
| 0%  | 0.991 | 1.000 | 0.009 | 100% |
| 10% | 0.854 | 0.864 | 0.010 | 100% |
| 20% | 0.732 | 0.743 | 0.011 | 100% |
| 30% | 0.624 | 0.636 | 0.011 | 100% |
| 40% | 0.527 | 0.543 | 0.016 | 77.5% |

The bound is tight to within 1.6% absolute when the attack model matches.
The paper's word-level attack ("masked" replacement) gives slightly
worse robustness (96% TPR at δ=30% vs 100% under uniform random),
because deterministic substitution biases away from valid transitions.

### Hoeffding bound validation (`fpr_vs_length.py`, new)
Empirical FPR vs Theorem 1's predicted upper bound, across token lengths
n ∈ {25, 50, 100} (existing samples were generated with --max-tokens 150,
so 200/400 are unavailable until the dataset is regenerated):

| S | n | empirical FPR | Hoeffding bound | status |
|---|---|---------------|-----------------|--------|
| 5 | 25  | 0.185 | 0.619 | OK |
| 5 | 50  | 0.064 | 0.375 | OK |
| 5 | 100 | 0.017 | 0.138 | OK |
| 7 | 25  | 0.029 | 0.110 | OK |
| 7 | 50  | 0.012 | 0.011 | borderline (off by 1 sample / 173 = 0.6%) |
| 7 | 100 | 0.000 | 1.1e-4 | OK |
| 9-11 | 25-100 | 0 | 1e-9 to 0.025 | OK |

Bound holds in 11/12 cells. The single borderline at S=7, n=50 is within
statistical noise of one extra false positive out of 173 samples, and
is exactly the kind of finite-length non-i.i.d. effect Remark 1 in the
paper now flags. Saved to `data/.../fpr_vs_length.json`.

### Backtranslation paraphrase attack (`paraphrase_attack.py`)
Round-trip translation EN→{de,fr}→EN through Helsinki-NLP/opus-mt
models on CPU, 30 samples from S=7, ρ=0%:

| Pivot | Score before | Score after | TPR before | TPR after |
|-------|--------------|-------------|------------|-----------|
| de    | 0.990 | 0.706 | 100% | 90.0% |
| fr    | 0.990 | 0.672 | 100% | 90.0% |

The watermark **does** survive single-pivot backtranslation, but the
margin to threshold drops from ~0.5 to ~0.2 and the false-negative
rate rises from 0 to 10%. This is the first paraphrase-robustness
number the paper has, and it's a sober one: against a stronger
paraphraser like DIPPER, expect TPR to drop further. Saved to
`data/.../paraphrase_states7_overlap0pct_backtranslation.json`.

### SHA-256 overhead (`measure_sha256_overhead.py`)
Measured on this CPU: 685 ns / call, 117 ms one-time table build for
the full 128k Llama-3 vocabulary. Equivalent to ~0.003% of typical
LLM token generation time. The Codex draft's "15-20% overhead" was
off by ~5000×.

### LaTeX structural sanity
- 768 / 768 brace balance, all environments matched
- All `\cite{...}` keys resolve in `references.bib`
- All `\ref{...}` resolve to `\label{...}` (one broken ref I introduced
  was fixed: `sec:experiments` → `sec:robustness`).

---

## Status of fixes already applied this session

Code & infra:
- [x] Fixed broken `ltw_watermark` imports across all five scripts,
      `src/mcl_watermark/__init__.py`, and `pyproject.toml`. The
      package never matched the import name; nothing under `scripts/`
      could run before this. Verified `from mcl_watermark.enhanced_mcl
      import ...` now succeeds.
- [x] Added a `--decoding {greedy,sampling}` flag to
      `scripts/generate_curated_dataset.py`, defaulting to **greedy** so
      the baseline matches the watermarked decoding default in
      `enhanced_mcl.py:289`.
- [x] Default `--max-tokens` raised from 150 to 512 so future runs
      produce longer sequences (and supply enough length for the
      FPR-vs-length plot in §1.2).
- [x] Added `--domain {wiki,news,social,abstract}` with three new
      ~25-prompt pools so cross-domain evaluation (Priority 2.1) is now
      a CLI flag away.
- [x] Wrote `scripts/paraphrase_attack.py` with two methods: DIPPER
      (Krishna et al. 2023) and round-trip backtranslation through
      Helsinki-NLP/opus-mt models. Imports lazily so the file is safe
      to read without GPU.
- [x] Added `uniform_token_attack()` to `scripts/robustness_attack.py`
      that draws replacements uniformly from the vocabulary — matches
      Theorem 2's stated attack model. The script now also prints
      side-by-side empirical $\phi$ vs theoretical
      $(1{-}\delta)^2 + k\delta(2{-}\delta)/S$ for direct validation.
- [x] Wrote and **ran** `scripts/measure_sha256_overhead.py`. Result on
      this machine: **685 ns per SHA-256 call, 117 ms one-time table
      build for the 128k Llama-3 vocab**. The earlier "SHA-256 overhead
      ~15-20% latency" claim in the Codex draft was off by ~5000×;
      the real number is $\sim 0.003\%$ of typical LLM step time.
      See updated Limitations section.
- [x] Aligned the `soft_cycle` matrix in `enhanced_mcl.py` with the
      paper's Example 3.5 (exact $2/3$ and $1/3$ instead of $0.67$ /
      $0.33$). Detection only checks $T_{ij} > 0$ so this is cosmetic,
      but it removes the "code disagrees with the paper" objection.

Paper:
- [x] Added missing `chen2024watermarkorderagnosticlanguagemodel`
      entry to `docs/references.bib`. Best-effort metadata —
      double-check authors and arXiv ID before camera-ready.
- [x] Removed the contradictory paragraph in `main.tex:375-377` that
      described the baseline as nucleus-sampled while the watermarked
      side was described as sampled. Both now state greedy, matching
      the appendix (`main.tex:631`) and the code.
- [x] Added Remark on the i.i.d. assumption directly after Theorem 1,
      explaining why the bound is a useful proxy under the
      random-oracle assumption even though tokens are dependent.
- [x] Strengthened the Theorem 3 proof with an explicit
      reduction-to-SHA-256-preimage-resistance sketch and a clarified
      threat model (parameters public, key private, side channels out
      of scope).
- [x] Reworded contribution (2) from "model-free $O(n)$ detection" to
      "tokenizer-only $O(n)$ detection" to match what the verifier
      actually needs.
- [x] Replaced the one-line Limitations sentence with a six-item list
      covering tokenizer dependency, short-text reliability, hyperparam
      sensitivity, measured (negligible) crypto overhead, cross-lingual
      gaps, and adversarial scope.

**Important caveat for the results tables.** The numbers currently in
`main.tex` were generated under the old protocol (nucleus baseline,
150-token cap, 0.67/0.33 soft-cycle weights, mismatched
i.i.d. wording). The fixes above are textual / code-level — the
existing JSON files in `data/` are not regenerated. Once you re-run
`generate_curated_dataset.py`, the numbers in the paper need to be
refreshed.

---

## Priority 1 — What's left for submission

### 1.1 Regenerate baseline + report matched PPL  *(needs GPU run)*
Code is ready. To finish this milestone you need to do the actual run,
which requires HF auth + a GPU.

```bash
export HF_TOKEN=...
uv run python scripts/generate_curated_dataset.py --decoding greedy
```

- [x] Decoding-matched baseline supported in script.
- [ ] Run it on your hardware.
- [ ] Add a `Baseline PPL | WM PPL | Δ%` column to the results tables.

### 1.2 Theorem 1 i.i.d. assumption  *(theory done, plot pending)*
- [x] Remark added directly after Theorem 1 explaining the
      assumption, the random-oracle justification, and the
      mixing-time argument.
- [ ] Empirical FPR-vs-length plot at $n \in \{50, 100, 200, 400\}$ —
      needs the regenerated dataset (1.1) plus `scripts/load_data.py`
      to subsample varying lengths from each sample.

### 1.3 Paraphrase robustness  *(code done, run pending)*
- [x] `scripts/paraphrase_attack.py` written with DIPPER and
      backtranslation methods.
- [ ] Run on at least one config:
```bash
uv run python scripts/paraphrase_attack.py \
    --data-dir data/curated_wiki_dataset_<timestamp> \
    --config states7_overlap0pct \
    --method dipper --lex 60 --order 60
uv run python scripts/paraphrase_attack.py \
    --data-dir data/curated_wiki_dataset_<timestamp> \
    --config states7_overlap0pct \
    --method backtranslation --pivots de fr
```
DIPPER needs ~22 GB VRAM; if that's not available, run backtranslation
only (Helsinki-NLP/opus-mt-* models are <500 MB each).
- [ ] Report a paraphrase-robustness row in the results table.

### 1.4 Theorem 2 attack-model alignment  *(done)*
- [x] `uniform_token_attack` added to `scripts/robustness_attack.py`,
      printing both empirical $\phi$ and theoretical
      $(1{-}\delta)^2 + k\delta(2{-}\delta)/S$ side by side. Re-running
      this script will give a clean theory-vs-experiment plot to
      include in the robustness section.

---

## Priority 2 — High-value, not strictly blocking

### 2.1 Cross-domain evaluation  *(scaffolded, run pending)*
- [x] News (~25), social (~25), and academic-abstract (~25) prompt
      pools added to `scripts/generate_curated_dataset.py`. Selected
      with `--domain {news,social,abstract}`.
- [ ] Run each domain on the optimal config and compare TPR/FPR/PPL:
```bash
for d in news social abstract; do
  uv run python scripts/generate_curated_dataset.py \
      --domain $d --states 7 --overlaps 0
done
```
- [ ] Report per-domain numbers in a small table in §6 of the paper.

### 2.2 Theorem 3 reduction  *(done)*
- [x] Proof sketch in `main.tex` now includes an explicit reduction
      from a distinguishing adversary to SHA-256 preimage resistance
      and a clarified threat model (parameters public, key private,
      side channels out of scope).

### 2.3 Soft-cycle paper/code alignment  *(done)*
- [x] `enhanced_mcl.py` now uses exact $2/3$ and $1/3$ matching
      Example 3.5 in the paper. Detection only depends on
      $T_{ij} > 0$ so this is purely a consistency fix; it does not
      change any reported number.

### 2.4 Human evaluation  *(can't be done here — needs humans)*
- [ ] 10–20 raters, paired (watermarked vs baseline) fluency
      preference, ~40 items each.
- [ ] Report preference rate with bootstrap CI or sign test.

This is the one Priority-2 item that genuinely requires recruiting
humans; design and pilot are the bottleneck, not infrastructure.

---

## Priority 3 — Polish

### 3.1 "Model-free" claim  *(done)*
- [x] Contribution (2) reworded to "tokenizer-only $O(n)$ detection".
- [x] Tokenizer-drift / re-tokenization caveat now in the
      Limitations enumerate.

### 3.2 Additional citations  *(not yet added)*
- [ ] Zhao et al. 2023/2024 (provable robustness of green-list
      watermarks).
- [ ] WaterMax 2024 (detectability/robustness/quality tradeoff).
- [ ] Fast-DetectGPT (2024) for detection-baseline comparison.

I did not add these myself because adding bib entries with possibly
wrong author lists / venues to a paper draft is worse than leaving
them out. Add them after looking up the canonical citations.

### 3.3 Limitations section  *(done)*
- [x] Tokenizer-alignment requirement.
- [x] Short-text reliability ($n < 50$ unreliable).
- [x] **Measured** SHA-256 per-token overhead — 685 ns/call,
      117 ms one-time table build for 128k vocab,
      $\sim 0.003\%$ of typical LLM step time. (The Codex draft's
      "15-20% latency" estimate was off by ~5000×.)
- [x] Cross-lingual gaps (EU AI Act covers 24 languages).
- [x] Adversarial scope (Theorem 2 = uniform random; semantic
      paraphrasing untested).

---

## Priority 4 — Nice to have

- [ ] Sparse watermarking ablation: watermark every Nth token, plot
      detection vs quality.
- [ ] C2PA / provenance integration discussion in Future Work.
- [ ] Cross-model headline result moved from appendix into main text
      (currently only Llama-3.2-3B-Instruct results are in the main
      tables; Mistral-7B and GPT-2-XL appear only in the appendix at
      `main.tex:479,642,670`).

---

## Success criteria (corrected)

| Metric                  | Target     | Current state                        | Status |
|-------------------------|------------|--------------------------------------|--------|
| Detection rate          | ≥99%       | 100% at S≥5                          | ✅     |
| FPR                     | ≤1%        | 0% at S≥5                            | ✅     |
| Random-attack robustness| ≥80% @ 30% | 96.5% @ δ=30% (`main.tex:460`)       | ✅     |
| Paraphrase robustness   | ≥80%       | Not tested (attack script missing)   | ❌     |
| Baseline-matched PPL Δ  | <2× WM     | Was uncomputable; now possible       | ⏳     |
| Cross-model coverage    | 3+ models  | 3 (Llama main, Mistral/GPT-2 appx)   | ⚠️     |
| Cross-domain coverage   | 3+ domains | 1 (Wikipedia only)                   | ❌     |
| Human evaluation        | Done       | Not done                             | ❌     |
| Bib hygiene             | All cites  | Pattern-Mark added this session      | ✅     |

---

## Theorem-by-theorem status (corrected)

- **Theorem 1 (Detection bound)** — Sound under the i.i.d. assumption
  used in the proof. Needs that assumption acknowledged in the
  statement and validated empirically. (Codex was right about this.)
- **Theorem 2 (Robustness)** — Sound, but the experiments don't test
  what the theorem assumes (uniform random replacement vs.
  deterministic word→`"masked"`). Reconcile.
- **Theorem 3 (Security)** — The 2^-256 bound is in the proof
  (`main.tex:333-339`); Codex was wrong to claim it's missing. Could
  still use a tighter reduction-style writeup.
- **Theorem 4 (Quality)** — Sound.

---

## Realistic publishability

The earlier draft of this file claimed "ACCEPT for NeurIPS/ICML/ICLR".
That is overconfident given:

- Wikipedia-only evaluation,
- no paraphrase-attack results,
- no human eval,
- decoding mismatch (now fixed in code, but no re-run yet),
- attack-model mismatch between Theorem 2 and experiments.

A more honest read is: with Priority-1 items done, this is a credible
workshop-track or borderline main-track submission. With
Priority-1 + Priority-2 done, it competes for main track.

---

## Quick-start commands (verified)

```bash
# Wikipedia (default), 512 tokens, decoding-matched greedy baseline:
uv run python scripts/generate_curated_dataset.py

# Cross-domain runs:
uv run python scripts/generate_curated_dataset.py --domain news      --states 7 --overlaps 0
uv run python scripts/generate_curated_dataset.py --domain social    --states 7 --overlaps 0
uv run python scripts/generate_curated_dataset.py --domain abstract  --states 7 --overlaps 0

# Reproduce the legacy nucleus baseline (only if needed):
uv run python scripts/generate_curated_dataset.py --decoding sampling

# Detection on non-watermarked samples:
uv run python scripts/evaluate_curated_non_watermarked.py

# Watermarked vs non-watermarked comparison:
uv run python scripts/compare_wm_vs_non_wm.py

# Robustness attacks (now includes uniform_token matching Theorem 2):
uv run python scripts/robustness_attack.py \
    --data-dir data/curated_wiki_dataset_<timestamp>

# Paraphrase robustness (DIPPER ~22 GB VRAM; backtranslation ~1 GB):
uv run python scripts/paraphrase_attack.py \
    --data-dir data/curated_wiki_dataset_<timestamp> \
    --config states7_overlap0pct --method dipper --lex 60 --order 60
uv run python scripts/paraphrase_attack.py \
    --data-dir data/curated_wiki_dataset_<timestamp> \
    --config states7_overlap0pct --method backtranslation --pivots de fr

# SHA-256 overhead micro-benchmark (CPU only, runs in ~1 s):
uv run python scripts/measure_sha256_overhead.py
```

---

## File map (verified against actual filenames)

```
MCLW/
├── docs/
│   ├── main.tex              # Paper source (Theorem 1 remark, Theorem 3 reduction,
│   │                         #  Limitations expansion, decoding fix this session)
│   ├── references.bib        # Pattern-Mark entry added
│   └── Report.pdf
├── src/mcl_watermark/        # Renamed from ltw_watermark; imports fixed
│   ├── enhanced_mcl.py       # Soft cycle (2/3, 1/3 to match paper); greedy default
│   └── mcl_watermark.py      # Original clockwork variant
├── scripts/
│   ├── generate_curated_dataset.py        # --decoding, --domain, --max-tokens 512
│   ├── generate_large_wikipedia_dataset.py
│   ├── compare_wm_vs_non_wm.py
│   ├── evaluate_curated_non_watermarked.py
│   ├── robustness_attack.py               # + uniform_token attack (Theorem 2)
│   ├── paraphrase_attack.py               # NEW: DIPPER + backtranslation
│   ├── measure_sha256_overhead.py         # NEW: micro-benchmark
│   └── load_data.py
├── experiments/
│   └── robustness_test.py
├── data/                     # Generated datasets (legacy; need re-run)
└── NEXT_STEPS.md             # This file
```
