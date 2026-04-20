# Entropy-Gated MCL Watermarking: Research Plan

## Extension of "Markov Chain Lock Watermarking" (Li & Kim, 2026)

---

## Part I: The New Approach

### 1.1 What We Propose

We extend MCL watermarking with two new ideas:

**Idea 1: Entropy Gating.** Instead of watermarking every token (current MCL), only watermark tokens where the model is confident (low entropy H < tau_H). At high-entropy positions, the model generates freely. This reduces quality degradation because:
- Low-entropy positions have few viable tokens, so the constraint is mild
- High-entropy positions (where quality matters most) are unconstrained

**Idea 2: Calibrated State Count.** Instead of fixing S=7, treat S as a tunable parameter calibrated to text length and domain:

```
S* = ceil(4 * z_alpha^2 / (rho^2 * (n-1)) + 1)
```

Longer texts need fewer states (S=2 suffices for 500+ tokens). Shorter texts need more states. This gives better quality than fixed S=7 whenever the text is long enough.

**Idea 3 (Claim to Validate): Low-Entropy Robustness.** We hypothesize that watermarks placed at low-entropy positions are more robust to paraphrasing than watermarks at high-entropy positions, because:
- Low-entropy tokens are harder to replace without degrading meaning
- The SIRA attack (ICML 2025) showed high-entropy tokens are the primary attack surface
- This inverts the conventional wisdom (SWEET, EWD, etc. all watermark at HIGH entropy)

This hypothesis is **plausible but unproven** -- our experiments will be the first to test it directly.

### 1.2 How It Works

**Generation (needs model):**
```
For each token position i:
  1. Run model forward pass → get logits
  2. Compute entropy: H_i = -sum(p * log(p))
  3. If H_i < tau_H:  → LOW ENTROPY → WATERMARK
       Required state = (prev_state + 1) mod S        [clockwork]
       Mask all tokens not in required state
       Pick argmax from allowed tokens
  4. If H_i >= tau_H: → HIGH ENTROPY → FREE
       Pick argmax from all tokens (no constraint)
  5. Track actual state for next position
```

**Detection (does NOT need model):**
```
For each consecutive token pair:
  1. Compute states: s_i = SHA256(key || token_i) mod S
  2. Check: does s_i == (s_{i-1} + 1) mod S ?
  3. Score = valid_transitions / total_transitions
  4. z = (score - 1/S) / sqrt((1/S)(1-1/S) / n)
  5. p-value = 1 - Phi(z)
```

The detector does not need entropy, does not need the model, runs in O(n).

### 1.3 Connection to Our Previous MCL Paper

| MCL Paper Component | New Framework |
|---|---|
| State assignment: sigma_k(t) = SHA256(k\|\|t) mod S | Same, but S is calibrated |
| Transition matrix: soft cycle or clockwork | Clockwork (simpler, sufficient with entropy gating) |
| All positions watermarked (rho=1) | Only low-H positions (rho < 1) |
| Fixed S=7 | S = f(n, rho, alpha) |
| Theorems 1-4 | Generalized to arbitrary S and rho |
| Detection: model-free O(n) | Same, unchanged |

---

## Part II: Theoretical Results to Prove

### Theorem 1: Detection Bounds (Generalized)

**Statement.** For entropy-gated MCL with S states, clockwork transitions, watermark fraction rho, text length n, and threshold tau > 1/S:

```
(i)   E[phi | watermarked] = 1/S + rho*(S-1)/S
(ii)  E[phi | random]      = 1/S
(iii) FPR <= exp(-2(n-1)(tau - 1/S)^2)                    [Hoeffding]
(iv)  z_FPR = (rho/2) * sqrt((S-1) * (n-1))               [CLT]
(v)   S_min = ceil(4*z_alpha^2 / (rho^2*(n-1)) + 1)       [Calibration]
```

**Proof strategy:** Direct application of Hoeffding's inequality and CLT to i.i.d. Bernoulli(1/S) variables under H0. The key insight is that the null distribution does NOT depend on entropy (model-free detection is valid regardless of which positions were watermarked).

**Status:** Proven in our audit document. Needs clean write-up.

### Theorem 2: Quality Bound (Entropy-Gated)

**Statement.** The expected per-token KL divergence satisfies:

```
E[D_KL]_total = rho * E[D_KL | S, watermarked positions]

where E[D_KL | S=s, top-token prob q] = (1/2) * ln(4/(1-q^2))  for S=2
      E[D_KL | S=s, general]          <= rho * ln(S)
```

And for all q in (0,1): E[D_KL | S=2] < E[D_KL | S=7 soft cycle].

**Proof strategy:** Direct computation of KL divergence between constrained and unconstrained distributions, conditioning on whether the top token falls in the target partition.

**Status:** Proven in calibration document. Needs formal write-up.

### Theorem 3: Robustness Under Modification (Generalized)

**Statement.** Under fraction delta random modification of tokens:

```
E[phi | attack] = 1/S + rho*(1-delta)^2*(S-1)/S
```

Critical modification fraction (at midpoint threshold):

```
delta* = 1 - 1/sqrt(2) ≈ 29.3%    [INDEPENDENT of S, rho, n]
```

Self-healing rate = 1/S for clockwork.

**Proof strategy:** Case analysis over (modified, unmodified) pairs. The universality of delta* follows from algebraic cancellation of S and rho terms.

**Status:** Proven. The universality of delta* = 29.3% is a new result not in the original paper.

### Theorem 4: Information-Theoretic Security

**Statement.** Under the random oracle model, without key k:

```
Adv_A = |P[A(t^w) = 1] - P[A(t^r) = 1]| <= negl(256)
```

Additionally, the adversary cannot determine which positions are watermarked.

**Proof strategy:** Identical to MCL Theorem 4.4 (SHA-256 random oracle argument). The entropy gating adds no observable artifact because it only affects the generation process, not the final token distribution's relationship to the hash.

**Status:** Trivially follows from original proof. Needs brief write-up.

### Theorem 5: HDD Optimality

**Statement.** The entropy-weighted log-likelihood ratio test:

```
Lambda = sum_i [X_i * ln(1 + pi_i*(S-1)) + (1-X_i) * ln(1 - pi_i*(S-1)/(S-1+epsilon))]
```

is Neyman-Pearson optimal among all size-alpha tests when the entropy profile (pi_i values) is known.

**Proof strategy:** Direct application of the Neyman-Pearson lemma to the product likelihood.

**Status:** Proven in HDD framework document. Needs clean write-up.

### Theorem 6: Self-Healing (New Formalization)

**Statement.** For any modification of a watermarked MCL sequence:

```
(i)   Self-healing rate = 1/S per modified token (strategy-independent)
(ii)  Self-healing surplus: Sigma(delta) = (1/S) * delta * (2-delta)
(iii) At the critical threshold, self-healing contributes
      Sigma(delta*) / tau fraction of the detection signal
(iv)  S=2 maximizes self-healing (rate = 50%) among all detectable schemes
```

**Proof strategy:** Hash uniformity under random oracle gives strategy-independence. Part (iv) follows from k/S being maximized at k=1, S=2 subject to the constraint that k < S (needed for detection).

**Status:** Formalized in feasibility report. New contribution -- not in any prior paper.

### Proposition 7: Entropy-Conditioned Robustness (New, to validate empirically)

**Hypothesis.** At positions with top-token probability q, the probability that a meaning-preserving rewrite keeps the same token is >= q. Therefore:

```
P(watermark survives at position i) >= q_i^2
```

For low-entropy positions (q > 0.8): P(survive) >= 0.64.
For high-entropy positions (q < 0.3): P(survive) >= 0.09.

**This gives low-entropy watermarks ~7x higher survival probability per position.**

**Status:** The mathematical bound P >= q^2 is trivially correct (if the rewriter independently picks the same token with probability q at both sides of a transition). But the assumption that LLM rewriters preserve tokens with probability >= q is **UNPROVEN and requires empirical validation.** This is our key experimental contribution.

**Supporting evidence (indirect):**
- SIRA (Cheng et al., ICML 2025): high-entropy tokens are the primary attack surface
- Wang et al. (2025): ~80% of LLM tokens are low-entropy and near-deterministic
- Christ et al. (COLT 2024): low-entropy outputs fundamentally resist undetectable modification

**Counter-evidence:**
- Bhagat & Hovy (2013): function words change during syntactic restructuring
- Active-passive transformation changes low-entropy function words

---

## Part III: Experiments to Run

### Experiment 1: Calibration Validation

**Goal:** Verify that the calibration formula S* = f(n, rho, alpha) matches empirical detection rates.

**Setup:**
- Model: Llama-3.2-3B-Instruct (same as original paper)
- Dataset: 173 Wikipedia prompts (same as original paper)
- Configurations: S in {2, 3, 4, 5, 7, 9}, tau_H in {1.5, 2.0, 2.5, 3.0, 3.5}
- Max tokens: 200, 500
- Metrics: TPR, FPR, phi score, z-score, perplexity

**Expected results:**
- The formula predicts which (S, n, rho) combinations achieve target FPR
- Configurations matching S >= S_min achieve 100% TPR and ~0% FPR
- Configurations with S < S_min show elevated FPR

**Comparison:** Side-by-side with original MCL S=7 soft cycle results from the paper.

### Experiment 2: Quality Improvement

**Goal:** Demonstrate that entropy-gated MCL achieves lower perplexity than standard MCL at equivalent detection power.

**Setup:**
- Generate matched pairs: entropy-gated (calibrated S) vs standard MCL (S=7 soft cycle)
- Same prompts, same model, same detection threshold
- Measure: perplexity, MAUVE score (if feasible), human evaluation (if feasible)

**Key comparison table to produce:**

| Method | S | rho | PPL | TPR | FPR | z-score |
|---|---|---|---|---|---|---|
| MCL S=7 soft cycle (baseline) | 7 | 1.0 | ~4.20 | 100% | 0% | ~14 |
| Entropy-gated, n=200 | 3 | ~0.5 | ? | ? | ? | ? |
| Entropy-gated, n=500 | 2 | ~0.5 | ? | ? | ? | ? |
| Unwatermarked | - | - | ~3.0 | - | - | - |

**Expected:** Entropy-gated achieves PPL significantly closer to unwatermarked baseline while maintaining TPR >= 95%.

### Experiment 3: Robustness Comparison

**Goal:** Compare robustness of entropy-gated vs standard MCL under multiple attack types.

**Attacks to test:**
1. Random word replacement (delta = 10%, 20%, 30%, 40%, 50%)
2. Synonym substitution (using WordNet or LLM-based)
3. Sentence-level paraphrasing (using DIPPER or GPT-based rewriter)
4. Back-translation (English → French → English, English → Chinese → English)
5. Character-level perturbation (homoglyph substitution at 2%, 5%, 10%)
6. Word insertion (10%, 20%, 30%)
7. Word deletion (10%, 20%, 30%)

**For each attack, measure:**
- Detection rate (TPR at p < 0.01)
- Score degradation (phi before and after)
- z-score after attack

**Key comparison:** At matched detection power (same z-score on clean text), which scheme degrades less under each attack?

### Experiment 4: The Critical Experiment -- Entropy vs Survival Rate

**Goal:** PROVE OR DISPROVE Proposition 7 (the fixed-point hypothesis). This is the novel empirical contribution.

**Setup:**
1. Generate 500 watermarked texts (200 tokens each) with Llama-3.2-3B
2. Record entropy H_i at each position during generation
3. Paraphrase each text using 3 methods:
   - DIPPER (controllable paraphraser, lexical diversity = 40, 60, 80)
   - GPT-4 rewrite ("Rewrite the following text while preserving its meaning")
   - Back-translation (English → French → English via NLLB or MarianMT)
4. Align original and paraphrased tokens (using edit distance alignment)
5. For each position: record whether the token survived (same token ID) or changed

**Analysis:**
- Plot: token survival rate vs entropy H_i (binned into H < 1, 1-2, 2-3, 3-4, 4+)
- Statistical test: is survival rate significantly higher at low H?
- Compute: correlation between H_i and P(token survives)

**Expected result (if hypothesis holds):**
```
H < 1.0:  survival rate ~80-95%  (very confident → kept by rewriter)
H 1-2:   survival rate ~50-70%
H 2-3:   survival rate ~30-50%
H 3-4:   survival rate ~15-30%
H > 4:   survival rate ~5-15%   (uncertain → easy to rewrite differently)
```

**If this pattern holds:** We have the first empirical evidence that low-entropy watermarks survive paraphrasing better. This is a publishable result on its own.

**If this pattern does NOT hold:** The fixed-point defense is weaker than claimed, and we should revise the robustness claims. Still publishable (negative result + honest analysis).

### Experiment 5: Self-Healing Validation

**Goal:** Empirically validate the self-healing formalization (Theorem 6).

**Setup:**
- Generate watermarked texts with S in {2, 3, 5, 7}
- Apply random word replacement at delta = 10%, 20%, 30%, 40%, 50%
- Measure observed phi vs theoretical E[phi] = 1/S + rho*(1-delta)^2*(S-1)/S

**Key result to show:**
- Self-healing surplus Sigma(delta) = (1/S)*delta*(2-delta) matches empirically
- S=2 has measurably higher self-healing than S=7
- The universal critical threshold delta* ≈ 29.3% holds across all S values

### Experiment 6: HDD vs Simple Detection

**Goal:** Show that entropy-weighted HDD detection outperforms simple phi test when model access is available.

**Setup:**
- Generate watermarked texts with entropy gating
- Detect using:
  (a) Simple phi test (model-free)
  (b) HDD with entropy weights from the same model
  (c) HDD with entropy weights from a proxy model (e.g., GPT-2 as proxy for Llama)
- Compare: ROC curves, AUC, TPR at fixed FPR

**Expected:** HDD > proxy HDD > simple phi in detection power, especially for short texts.

### Experiment 7: Multi-Model Validation

**Goal:** Confirm that entropy-gated MCL works across different LLMs.

**Models:** Llama-3.2-3B, Mistral-7B, GPT-2-XL (1.5B)
**Prompts:** Same 173 Wikipedia concepts
**Configurations:** Calibrated S for each model (S may differ due to different entropy profiles)

---

## Part IV: Paper Outline

### Title
"Entropy-Gated Markov Chain Lock: Calibrated Watermarking with Provable Quality-Detection Tradeoffs"

### Abstract (~150 words)
MCL watermarking achieves perfect detection but at a fixed quality cost (PPL 4.20). We introduce entropy-gated MCL, which constrains only low-entropy positions and calibrates the state count S to text length. We derive a one-formula calibration S* = f(n, rho, alpha) and prove that detection z-score = (rho/2)*sqrt((S-1)(n-1)). We formalize the self-healing property (rate = 1/S, universal delta* = 29.3%), derive a Neyman-Pearson optimal detection test (HDD), and demonstrate that S=2 with entropy gating achieves 3x lower perplexity than S=7 for texts >= 300 tokens. In the first controlled study of entropy-conditioned watermark survival under paraphrasing, we show that [results of Experiment 4]. Code at github.com/ChenghengLi/MCLW.

### Sections

**1. Introduction**
- MCL recap (brief)
- Limitation: fixed quality cost
- Our contribution: entropy gating + calibration + self-healing formalization
- Key results preview

**2. Background and Related Work**
- MCL framework (cite our own paper)
- Entropy-adaptive watermarking: SWEET, EWD, Invisible Entropy, Adaptive WM
  - All watermark at HIGH entropy -- we invert this
- Attacks: SIRA (supports our approach), DIPPER, back-translation
- Statistical detection: Li & Huang (2024), Fernandez et al. (2025)

**3. Method: Entropy-Gated MCL**
- 3.1 Entropy-gated embedding algorithm
- 3.2 Model-free detection (unchanged from MCL)
- 3.3 Entropy-weighted HDD (optional, model-assisted)
- 3.4 Calibration formula for S

**4. Theoretical Results**
- Theorem 1: Detection bounds (generalized to arbitrary S, rho)
- Theorem 2: Quality bound
- Theorem 3: Robustness + universal delta*
- Theorem 4: Security
- Theorem 5: HDD optimality
- Theorem 6: Self-healing formalization

**5. Experiments**
- 5.1 Calibration validation (Exp 1)
- 5.2 Quality improvement (Exp 2)
- 5.3 Robustness comparison (Exp 3)
- 5.4 Entropy vs survival rate (Exp 4) -- the key novel experiment
- 5.5 Self-healing validation (Exp 5)
- 5.6 HDD vs simple detection (Exp 6)

**6. Discussion**
- When to use entropy-gated vs standard MCL
- Limitations: short texts, rho uncertainty, bimodal quality
- The fixed-point hypothesis: what we proved, what remains open
- Regulatory implications (EU AI Act)

**7. Conclusion**

---

## Part V: Timeline

| Week | Task |
|---|---|
| 1-2 | Clean up theorems, write proofs formally |
| 2-3 | Implement entropy-gated generation + calibrated S in codebase |
| 3-4 | Experiment 1 (calibration) + Experiment 2 (quality) |
| 4-5 | Experiment 3 (robustness attacks) |
| 5-6 | **Experiment 4 (entropy vs survival -- the key experiment)** |
| 6-7 | Experiment 5 (self-healing) + Experiment 6 (HDD) |
| 7-8 | Experiment 7 (multi-model) + paper writing |
| 8-9 | Revision, figures, submission |

---

## Part VI: Key Claims and Their Evidence Status

| Claim | Status | What We Need |
|---|---|---|
| Calibration formula is correct | Proven (math) | Exp 1 validates empirically |
| Quality improves over MCL S=7 | Proven (math) | Exp 2 measures actual PPL |
| Model-free detection works | Proven (math) | Exp 1 confirms |
| Self-healing rate = 1/S | Proven (math) | Exp 5 validates |
| delta* = 29.3% is universal | Proven (math) | Exp 3 confirms |
| HDD is optimal | Proven (Neyman-Pearson) | Exp 6 shows improvement |
| Security identical to MCL | Proven (same argument) | N/A |
| Low-H tokens survive paraphrasing better | **UNPROVEN** | **Exp 4 is the critical test** |
| Entropy-gated is more robust to LLM rewrite | **UNPROVEN** | **Exp 3 + Exp 4 together** |
| SIRA-style attacks are less effective against low-H watermarks | Supported by SIRA paper | Exp 3 with targeted attacks |
