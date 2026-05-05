# MCL Watermarking — Workshop-Publishable Experiment Plan

Synthesized from 5 parallel design agents (theory-stress, adversarial robustness,
generalization breadth, statistical rigor, SOTA comparison). Selected to maximize
paper credibility per A100 GPU-hour, sequenced for one-job-at-a-time execution
on EPFL RCP (dlab-lichen, preemptible interactive, batch=64 where supported).

Total budget target: ~10 GPU-hours.

---

## Tier 1 — Must-have (≈3 GPU-hours, mostly CPU)

These plug the biggest holes that any reviewer notices in <30 seconds. They
mostly reuse data already on the cluster PVC.

### T1.1 — Wilson 95% CIs on every TPR/FPR cell (CPU-only)
- **Defect:** "100%" / "0%" reported as point estimates with no uncertainty.
  At n=173 the true TPR could legitimately be as low as 97.8%; at n=25 as low
  as 86.7%. (statistical-rigor agent: "the embarrassing hole.")
- **Code:** `scripts/stats_report.py` — re-tabulate every existing run with
  Wilson and Clopper–Pearson CIs, plus rule-of-three FNR upper bound for
  100% cells.
- **Artifact:** Replaces every cell `100%` with `100.0% [97.8, 100]` in
  Tables 2/3/4/Cross-domain.

### T1.2 — Paired log-PPL t-test vs greedy baseline (CPU on existing data)
- **Defect:** Baseline PPL absent from main tables; "PPL=4.20" has no anchor.
- **Code:** Reuse `scripts/compare_wm_vs_non_wm.py` outputs, add paired
  t-test + Wilcoxon + bootstrap CI on Δlog-PPL.
- **Artifact:** Two new columns `PPL_base` and `Δlog-PPL [95% CI], p` in the
  main results table.

### T1.3 — Tokenizer-drift detection (CPU on existing data)
- **Defect:** "Tokenizer-only" claim is hand-wavy. Reviewer will ask whether
  detection survives a *different* tokenizer.
- **Code:** `scripts/tokenizer_drift.py` — re-detect existing Llama-watermarked
  text using {Llama-3.1, Mistral-7B-v0.3, GPT-2-XL} tokenizers; report how
  TPR collapses.
- **Artifact:** New 4-row "Tokenizer-Family Drift" table in §6 + one
  sentence in Limitations.

### T1.4 — Empirical D_KL vs theoretical bound (free side-channel)
- **Defect:** Theorem 4 proves `D_KL ≤ log S` but paper reports zero
  empirical KL.
- **Code:** Modify `enhanced_mcl.py` `generate()` to log per-step
  `log(1/Z_s)`; aggregate over an existing watermarked run.
- **Artifact:** New row `Empirical D_KL (nats)` next to PPL; figure showing
  empirical KL vs `log S` for S ∈ {4,5,7,9,11}.

### T1.5 — Minimum-n curve (CPU on existing data)
- **Defect:** Hoeffding bound stated, but practitioner-facing
  *minimum tokens for target FPR* never computed.
- **Code:** Extend `scripts/fpr_vs_length.py` with theoretical
  `n* = 1 + ln(1/α) / [2(τ - k/S)²]` and overlay on empirical curve.
- **Artifact:** New figure: `n_min` vs target FPR, theory band vs empirical
  points; one paragraph in §6.

---

## Tier 2 — High-impact GPU work (≈6 GPU-hours)

### T2.1 — Kirchenbauer-2023 (KGW) head-to-head reproduction (3 GPU-hours)
- **Defect:** No direct comparison vs any prior watermark. Workshop reviewers
  reject on this alone.
- **Code:** `scripts/kgw_baseline.py` — wraps the canonical
  `WatermarkLogitsProcessor` (γ=0.25, δ=2.0 *and* γ=0.5, δ=4.0) into the
  same `generate_curated_dataset.py` pipeline. Same Llama-3.2-3B, same 173
  wiki prompts, same `max_new_tokens=512`. Report TPR @ FPR=1% (sweep z),
  PPL Δ, ms/token, TPR @ δ=30% uniform-token attack.
- **Runai:** `mcl-kgw-{stamp}`, A100-80GB, batch=64 for non-WM baseline,
  batch=1 for KGW generation, ~1.5h wall-clock.
- **Artifact:** A "Method × Metric" table with rows {MCL S=7,
  KGW γ=.25/δ=2 T=1, KGW γ=.5/δ=4 T=0.7} and 5 columns.

### T2.2 — DIPPER paraphrase kill-shot (1.5 GPU-hours)
- **Defect:** The single most likely failure mode of MCL is paraphrase
  retokenization. Krishna-2023 already showed DIPPER 60/60 reduces KGW
  AUC from 0.97 → 0.57; MCL is BPE-id-keyed and likely worse.
- **Code:** Existing `scripts/paraphrase_attack.py` already supports DIPPER;
  extend with the (lex, order) sweep `{(0,0),(20,20),(40,40),(60,60),(60,0),
  (0,60),(80,80)}` and a length-of-output curve (truncate at
  100/200/300/400/500 tokens).
- **Runai:** `mcl-dipper-{stamp}`, A100-80GB, DIPPER fp16 fits in 24 GB,
  batch=4, ~1.5h for 173 prompts × 7 settings.
- **Artifact:** TPR(lex, order) heatmap + a "TPR vs n_tokens" curve for
  the 60/60 cell. **Honest failure case** — set up the next experiment.

### T2.3 — Multi-pivot backtranslation, full corpus (1.5 GPU-hours)
- **Defect:** Existing pilot used only n=30, 2 Latin pivots, 90% TPR.
  Need n=173, including non-Latin scripts (de, fr, ja, zh) where
  retokenization is more aggressive.
- **Code:** Extend `paraphrase_attack.py` to use `NLLB-200-3.3B` instead of
  Helsinki-Opus (better non-Latin quality). Add double-pivot
  `en→de→fr→en`.
- **Runai:** `mcl-backtrans-{stamp}`, A100-80GB, NLLB fp16 ~7 GB, ~1.5h.
- **Artifact:** Backtranslation table extended from 2 to 5 pivots + one
  double-pivot row.

---

## Tier 3 — Polish (≈2 GPU-hours, only if budget left)

### T3.1 — Structured-edit attack suite (CPU, ~10 min)
- Add synonym swap (WordNet), homoglyph substitution, adjacent-token swap
  to `robustness_attack.py`. Compare each to Theorem 2's predicted φ.

### T3.2 — n=300 baseline regen for FPR ≤ 1% certificate (1.5 GPU-hours)
- Need n ≥ 299 to claim "FPR ≤ 1% (95% Clopper–Pearson UCB)" given 0
  observed false positives.

### T3.3 — Empirical security distinguishers (CPU, ~15 min)
- Bigram-LLR, χ², best-of-S! permutation attack on existing data.
- Convert Theorem 3 from a pure theoretical bound to an empirical
  "best known attack ≤ ε".

### T3.4 — Cross-lingual single-language run (1.5 GPU-hours)
- Qwen2.5-7B-Instruct on French wiki prompts. Proves the partition is
  language-blind (or finds a counterexample).

---

## Sequencing on a single A100 (preemptible)

```
0. (CPU)  T1.1, T1.2, T1.3, T1.5      ← can run on dev machine right now,
                                         no cluster needed
1. GPU    T2.1 KGW (3h)               ← biggest credibility delta
2. GPU    T2.2 DIPPER (1.5h)          ← honest robustness story
3. GPU    T2.3 multi-pivot NLLB (1.5h)
4. (CPU)  T1.4 KL re-aggregation      ← needs T2.x outputs
5. CPU    T3.1, T3.3                  ← only if Tier 1+2 looks good
6. GPU    T3.2 n=300 baseline (1.5h)  ← optional polish
```

Total: ~7-8 GPU-hours sequential, ~1h CPU, fits a single preemptible day.

---

## What goes in the paper after this is done

Three new tables, two new figures, one new subsection:

| Slot                          | Content from                  |
|-------------------------------|-------------------------------|
| Table 2 (extended)            | T1.1 + T1.2 (CIs + ΔPPL)     |
| Table 3 (extended)            | T1.1 + T3.1 (CIs + edits)    |
| Table "Method × Metric"       | T2.1 (KGW comparison)         |
| Table "Tokenizer drift"       | T1.3                          |
| Table "Paraphrase robustness" | T2.2 + T2.3                   |
| Figure "DIPPER lex×order"     | T2.2                          |
| Figure "FPR vs n / n_min"     | T1.5                          |
| Figure "Empirical KL"         | T1.4                          |
| §6.X "Empirical security"     | T3.3                          |
| §6.4 "Limitations" rewrite    | DIPPER findings + tokenizer   |

This converts the paper from "we propose MCL and detect it" to "we propose
MCL, compare it head-to-head against KGW, validate four theorems empirically,
characterize where it fails (DIPPER/cross-tokenizer), and release CIs on
every claim." That's the workshop bar.

---

## Out of scope (acknowledged, not run)

- **Pattern-Mark (Chen 2024) reimplementation** — paper targets order-agnostic
  LMs; faithful autoregressive port is non-trivial and not tractable in
  budget. Cite + future work.
- **Christ-2024 undetectable** — needs full per-token distribution access
  not exposed by HF `generate()`. Skip with one-line disclaimer.
- **Human evaluation** — needs humans, not GPU. Defer.
- **24-language EU coverage** — beyond budget. Single-language proof of
  concept (T3.4) gestures at it.
