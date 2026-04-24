# Markov Chain Lock: Hard-Constraint Watermarking for Language Models with Model-Free Detection

## Abstract

We introduce **Markov Chain Lock (MCL)**, a discrete, hard-constraint watermarking scheme for autoregressive language models. MCL partitions the vocabulary into `S` states using a keyed SHA-256 hash and enforces a secret Markov-chain transition at generation by masking disallowed-state logits to `−∞`. Detection re-derives token states from the key alone, requires no language model, no tokenizer-logit inversion, and no entropy estimation, and computes a closed-form z-score from the fraction of valid state transitions. We prove: (i) detection satisfies `z = ρ · √((S−1)(n−1))` under clockwork topology with Hoeffding false-positive-rate control; (ii) the closed-form calibration `S_min = ⌈z_α²/(ρ²(n−1)) + 1⌉` yields the minimum state count for a target significance level, eliminating hyperparameter tuning; (iii) robustness under random modification at rate `δ` follows `E[φ | attack] = 1/S + ρ(1−δ)²(S−1)/S`, with a universal midpoint critical rate `δ* = 1 − 1/√2 ≈ 0.293` independent of scheme parameters; (iv) security reduces to the pseudorandom-function hardness of SHA-256 in the random-oracle model. We accompany the method with a matched-budget analysis of four position-gating strategies (always-on, high-entropy, low-entropy, surprisal-gap), establishing empirically that low-entropy gating is strictly Pareto-dominated and that detection probability is gate-invariant at practical watermark budgets. A 30-line reference implementation of the detector is provided.

---

## 1. Introduction

Watermarking AI-generated text has emerged as a governance priority. Existing schemes fall into three families: (a) continuous-logit-bias methods such as KGW (Kirchenbauer et al., 2023), which add a fixed bias `δ` to a pseudorandom "green list" of tokens at each position; (b) sampling-transform methods such as Gumbel/Aaronson and SynthID-Text (Dathathri et al., 2024), which modify the sampling step via pseudorandom draws while preserving the marginal distribution in expectation; and (c) entropy-gated variants such as SWEET (Lee et al., 2024) and Invisible Entropy (Gu et al., 2025), which apply a logit bias only at positions where the generator is uncertain. Each family trades quality against detection, but none provides a closed-form operator-facing calibration of its design parameters: `γ`, `δ`, `τ`, and tournament depth are all chosen by grid search.

We introduce **Markov Chain Lock (MCL)**, a watermarking scheme that departs from all three families by enforcing a **discrete, hard state-transition constraint** over a keyed vocabulary partition. MCL hashes each token `t` into one of `S` states via SHA-256 keyed with a secret `k`; at generation time, logits of all tokens whose state is disallowed under a fixed Markov transition matrix `T` are masked to `−∞`. Detection re-derives each token's state and counts the fraction of pairs `(σ(t_i), σ(t_{i+1}))` with `T(σ(t_i), σ(t_{i+1})) = 1`; the fraction is compared against the known random baseline `p_0 = mean(T > 0)`. The detector requires no language model and no entropy estimate.

**Contributions.**

1. We propose MCL, a hard-constraint watermarking primitive whose signal unit is a token *transition* rather than a token identity or marginal probability.
2. We derive a closed-form detection identity `z = ρ · √((S−1)(n−1))` under clockwork topology and the corresponding calibration formula `S_min = ⌈z_α²/(ρ²(n−1)) + 1⌉`, yielding the smallest state count that meets a target significance given expected budget and text length.
3. We establish a universal midpoint critical-modification rate `δ* = 1 − 1/√2 ≈ 0.293` under random substitution attacks — independent of `S`, `ρ`, and `n`.
4. We analyze four position-gating strategies at matched budget across 18 (domain, temperature, budget) cells on Gemma 3 270M, showing that low-entropy gating is strictly Pareto-dominated and that detection probability is gate-invariant at budgets `ρ ≥ 0.5`.
5. We release a reference implementation of the detector (< 30 lines of Python) and the full experimental pipeline.

---

## 2. Related Work

**Logit-bias watermarks.** KGW (Kirchenbauer et al., 2023) is the canonical scheme: at each position, context is hashed to partition the vocabulary into a green list of size `γ|V|`; a bias `δ` is added to green-list logits. Unigram (Zhao et al., 2024) fixes the green list globally, proving provable robustness under bounded edits. SIR (Liu et al., 2024) derives the partition from a neural semantic embedding rather than a hash, enhancing paraphrase robustness.

**Entropy-gated variants.** SWEET (Lee et al., 2024) applies KGW-style bias only at positions with entropy above a threshold `τ`, targeting code generation where low-entropy tokens are syntactically forced. EWD (Lu et al., 2024) applies entropy weighting at detection rather than generation. Invisible Entropy (Gu et al., 2025) replaces the generator's entropy with a trained tagger's prediction to avoid invoking the LM at detection.

**Sampling-transform watermarks.** Aaronson's Gumbel-max scheme and its refinements, including SynthID-Text (Dathathri et al., 2024) and Kuditipudi et al.'s distortion-free watermark (2024), modify the sampling procedure to inject a key-dependent signal into token choice. HeavyWater and SimplexWater (Tsur et al., 2025) cast this as a coding-theory problem with Hamming-distance guarantees.

**Low-entropy-aware schemes.** STA-1 (Mao et al., 2024) and the POS-guided partitioning of Findings-EMNLP 2025 address the failure mode of logit-bias methods on near-deterministic distributions.

**Markov-structured schemes.** Pattern-Mark (Liu et al., 2024) introduces a Markov chain over a small key set `K`, producing a step-indexed schedule of which key (and corresponding vocabulary partition) is active. The probabilistic-automaton watermark (Wang et al., 2025) generalizes cyclic-key schemes using PDFA/PNFA states, with exp-min sampling and edit-distance detection.

**Positioning of MCL.** MCL differs from all of the above along three axes simultaneously: (a) the signal is a *pair-wise transition* rather than a per-token green/red membership; (b) the constraint is applied as a **hard mask** (logit → `−∞`) rather than a soft bias; (c) the detector is **strictly model-free** (no LM forward pass, no trained component, no entropy estimate) and produces a closed-form z-score against a known random baseline. Pattern-Mark shares the Markov-structured spirit but indexes state by generation step rather than by previously emitted token identity, and uses logit bias rather than hard masking.

---

## 3. Markov Chain Lock

### 3.1 Setup and Notation

Let `V` be the vocabulary of the language model, `k ∈ {0, 1}^λ` a secret key, and `S ≥ 2` a positive integer. Define the state partition
```
σ : V → {0, 1, …, S − 1},    σ(t) = SHA256(k ‖ t) mod S,
```
which we model as a random oracle. A **transition matrix** `T ∈ {0, 1}^{S × S}` specifies allowed state transitions. The paper's default is the *clockwork* topology `T(s, s') = 1 ⟺ s' = (s + 1) mod S`, for which the random baseline is `p_0 = 1/S`.

Given generated text `(t_1, …, t_n)`, define
```
X_i = 𝟙{T(σ(t_i), σ(t_{i+1})) = 1},  i = 1, …, n − 1,     φ = (n − 1)^{−1} ∑_i X_i.
```
The statistic `φ` is the empirical fraction of valid transitions; `p_0 = mean(T > 0)` is its expected value under a null key-independent token stream.

### 3.2 Generation Algorithm

At each generation step, given current state `s`, the generator computes the next-token logits, masks disallowed-state entries to `−∞`, optionally applies a position-gate `g(·)`, and samples or takes argmax.

```
Algorithm 1: MCL Generation
Input:   prompt x; model π; key k; S; T; gate g; decoding params.
State:   s ← σ(last token of x)
for i = 1, 2, …, n:
    ℓ ← π(logits at current position)
    if g(ℓ) = 1:                                  # gate on
        for t ∈ V:
            if T(s, σ(t)) = 0:  ℓ[t] ← −∞
    t_i ← decode(ℓ)                               # argmax or softmax sample
    emit t_i;  s ← σ(t_i)
return (t_1, …, t_n)
```

Without a gate (`g ≡ 1`), MCL enforces a valid transition at every step; under greedy decoding with fixed `k`, generation from a fixed prompt is byte-identical across runs.

### 3.3 Detection Algorithm

```
Algorithm 2: MCL Detection
Input:   text (t_1, …, t_n); key k; S; T; significance α.
for i = 1, …, n − 1:
    s_i ← σ(t_i);  s_{i+1} ← σ(t_{i+1})
    X_i ← 𝟙{T(s_i, s_{i+1}) = 1}
φ ← (n − 1)^{−1} ∑_i X_i
p_0 ← mean(T > 0)
z ← (φ − p_0) / √(p_0 (1 − p_0) / (n − 1))
p ← 1 − Φ(z)                                      # one-sided normal tail
return (is_watermarked = [p < α], φ, z, p)
```

Detection evaluates `σ` on `n` tokens (two SHA-256 calls each) and performs `n − 1` matrix lookups. Total complexity `O(n)` with no model invocation.

### 3.4 Soft-Partition Extension

For a continuous quality-vs-detection trade-off, we permit each token to belong to multiple states with overlap ratio `r ∈ [0, 1]`. The primary state is `σ(t)` as defined; token `t` additionally belongs to state `i ≠ σ(t)` with probability `r`, determined by the secondary hash `SHA256(k ‖ t ‖ i) mod 100 < 100 r`. The mask at generation-time is the union of allowed-state token sets; the detector retains primary-state assignment only.

---

## 4. Theoretical Results

Throughout this section we assume the random-oracle model for `σ`. Clockwork baseline is `p_0 = 1/S`.

### 4.1 Theorem 1 (Detection Moments)

Under `H_0` (unwatermarked, key-independent text),
```
E[φ | H_0] = p_0,        Var(φ | H_0) = p_0(1 − p_0) / (n − 1).
```
Under `H_1` with per-position watermark indicator `W_i ∼ Bern(ρ)`, setting `W_i = 1 ⟹ X_i = 1` and `W_i = 0 ⟹ X_i ∼ Bern(p_0)`,
```
E[φ | H_1, ρ] = p_0 + ρ(1 − p_0).
```
The standardized effect size is
```
z = ρ · √((1 − p_0)(n − 1) / p_0).      (†)
```
For clockwork (`p_0 = 1/S`), `z = ρ · √((S − 1)(n − 1))`.

**Proof.** Under `H_0`, the random oracle yields `σ(t_i), σ(t_{i+1})` iid uniform on `{0, …, S − 1}`, giving `X_i ∼ Bern(p_0)`. Pairwise adjacent `X_i, X_{i+1}` share one hash: for clockwork, `P[X_i = X_{i+1} = 1] = P[σ(t_{i+1}) = (σ(t_i) + 1) mod S ∧ σ(t_{i+2}) = (σ(t_{i+1}) + 1) mod S] = (1/S)(1/S) = p_0^2`, so `Cov(X_i, X_{i+1}) = 0`. Non-adjacent pairs are independent. Therefore `Var(φ) = (n − 1)^{-1} p_0 (1 − p_0)`. Under `H_1`, `E[X_i] = ρ · 1 + (1 − ρ) p_0 = p_0 + ρ(1 − p_0)`; plugging into `z = (E[φ | H_1] − p_0) / √Var(φ | H_0)` gives (†). Hoeffding's inequality applied to the odd/even subsequences yields `P[φ ≥ τ | H_0] ≤ 2 exp(−2 (n − 1) (τ − p_0)^2)`.  ∎

### 4.2 Theorem 2 (Calibration)

For target significance `α` with critical value `z_α = Φ^{-1}(1 − α)`, the minimum clockwork state count required for `z ≥ z_α` is
```
S_min = ⌈ z_α² / (ρ² (n − 1)) + 1 ⌉.
```

**Proof.** From (†) under clockwork, `z² = ρ² (S − 1)(n − 1)`. Requiring `z ≥ z_α` and solving for `S` yields the stated expression; take the ceiling.  ∎

### 4.3 Theorem 3 (Per-Token Quality Cost)

Let `p_i` be the model distribution at a watermarked position with required allowed-state set `V_{s*}`. The projected watermarked distribution `p_wm(t) = p_i(t) / p_i(V_{s*})` for `t ∈ V_{s*}` (else 0) satisfies
```
D_KL(p_wm ‖ p_i) = −log p_i(V_{s*}).
```
Under the random oracle `E[p_i(V_{s*})] = 1/S`, so by Jensen `E[D_KL] ≥ log S`, with equality iff `p_i` is uniform over `V`.

**Proof.** Direct expansion: `D_KL = ∑_{t ∈ V_{s*}} p_i(t)/p_i(V_{s*}) · log(p_i(t)/p_i(V_{s*}) / p_i(t)) = −log p_i(V_{s*})`. Jensen on the convex map `x ↦ −log x` gives `E[−log p_i(V_{s*})] ≥ −log E[p_i(V_{s*})] = log S`.  ∎

### 4.4 Theorem 4 (Robustness Under Random Substitution)

Under independent random modification at rate `δ ∈ [0, 1]` (modified tokens receive fresh random-oracle state assignments), the expected detection statistic satisfies
```
E[φ | attack, ρ, δ] = p_0 + ρ (1 − δ)² (1 − p_0).
```
At the midpoint threshold `τ_mid = p_0 + ρ(1 − p_0)/2`, the critical modification rate is
```
δ*_mid = 1 − 1/√2 ≈ 0.293,
```
independent of `S`, `ρ`, and `n`. At the size-`α` threshold `τ_α = p_0 + z_α √(p_0(1 − p_0)/(n − 1))`, the critical rate is
```
δ*_α = 1 − √(z_α / z),
```
where `z` is the expected watermarked z-score from Theorem 1.

**Proof.** A pair `(t_i, t_{i+1})` contributes a valid transition iff both tokens survive the attack (probability `(1 − δ)²`, yielding expectation `ρ + (1 − ρ)p_0`) or otherwise yields the random baseline `p_0`. Hence
```
E[X_i] = (1 − δ)² (ρ + (1 − ρ) p_0) + (1 − (1 − δ)²) p_0 = p_0 + ρ (1 − δ)² (1 − p_0).
```
Setting this equal to `τ_mid` gives `(1 − δ)² = 1/2 ⟹ δ = 1 − 1/√2`. Because both `S` and `ρ` cancel, `δ*_mid` is universal. For `τ_α`, equating and solving yields the stated expression.  ∎

### 4.5 Lemma 5 (Self-Healing)

Under the random oracle, any modified token `t'` receives `σ(t')` uniformly distributed on `{0, …, S − 1}` independent of its neighbors. Pairs involving at least one modified token therefore satisfy `P(X = 1) = p_0`.

This is implicit in the `(1 − δ)²` factor of Theorem 4; we state it separately for clarity of the self-healing interpretation.

### 4.6 Theorem 6 (Security Against Key Recovery)

In the random-oracle model where `SHA256(·)` is modeled as a 256-bit random oracle, any polynomial-time adversary without the key `k` has advantage at most `O(Q / 2^{256})` in recovering `k` or in distinguishing `σ` from a uniformly random function `V → {0, …, S − 1}`, where `Q` is the number of oracle queries.

**Proof.** Standard random-oracle argument: `k` appears only via oracle queries, so each adversary query has at most `2^{-256}` probability of coinciding with a key-dependent hash input. A distinguisher for `σ` yields a key-recoverer by hybrid argument.  ∎

**Remark.** Theorem 6 does not guarantee indistinguishability of watermarked from unwatermarked text at the distribution level. A key-less adversary observing enough watermarked text can estimate the empirical pair-frequency matrix and detect its rank-constrained structure — without recovering `k`. The security claim is therefore correctly stated as *computational hiding of the key and partition*.

### 4.7 Theorem 7 (Entropy-Weighted Detector — Neyman-Pearson Optimality)

Let `W = (W_1, …, W_{n−1}) ∈ {0, 1}^{n−1}` be a known entropy-indicator vector. Consider
```
H_0:  X_i ∼ Bern(p_0)   i.i.d.,
H_1:  X_i ∼ Bern(p_i),   p_i = W_i + (1 − W_i) p_0.
```
The most powerful size-`α` Neyman-Pearson test rejects when
```
Λ = (log 1/p_0) · ∑_{i : W_i = 1} X_i  ≥  k_α.
```

**Proof.** The log-likelihood ratio factorizes over positions. At `W_i = 0`, the ratio is `1`. At `W_i = 1` with `X_i = 1`, the ratio is `1/p_0`; with `X_i = 0`, the ratio is 0 (a smoothed form `ε → 0`). Summing yields `Λ` as stated; Neyman-Pearson gives optimality.  ∎

**Remark.** Optimality is *conditional* on a known `W` and on the random-oracle null. In practice, `W_i` is estimated at detection time from a proxy language model; estimation noise is not covered by this theorem.

---

## 5. Gate Analysis

MCL admits an optional position-gate `g` that determines which positions receive the mask. At ungated positions, the generator sampls freely; at gated positions, the mask is applied. Let `ρ = E[g_i]` denote the expected watermark budget.

### 5.1 Gate Definitions

Let `H_i = −∑_t p_i(t) log p_i(t)` be the next-token entropy and `Δ_i = p_i^{(1)} − p_i^{(2)}` the surprisal gap between the top two probabilities.

- **G_all**: `g_i = 1` always.
- **G_H_high(τ)**: `g_i = 𝟙{H_i > τ}`.
- **G_H_low(τ)**: `g_i = 𝟙{H_i < τ}`.
- **G_Δ(τ)**: `g_i = 𝟙{Δ_i < τ}`.

### 5.2 Theorem 8 (Matched-Budget Equivalence)

Let `s_i ∈ [0, 1]` denote the post-attack survival probability at position `i`, and let the expected post-attack detection signal be `R(g) = ∑_i g_i · s_i · s_{i−1}`. Under the **local-independence assumption** (`s_i` are independent with shared marginal `μ = E[s_i]`),
```
R(g) ∝ ∑_i g_i · μ²,
```
and the optimal deterministic gate subject to `∑ g_i ≤ B` satisfies
```
g*(i) = 𝟙{ s_i ≥ τ* },     |g*| = B.
```
At matched budget `|g_φ| = |g_ψ| = B`, any two monotone threshold rules ordered identically to `s_i` produce the same expected signal.

**Proof.** The objective is linear in `g`; under local independence, `E[s_i s_{i-1}] = μ²`, making the coefficient of `g_i` constant in `i`. The unit-weight knapsack with identical item values is indifferent to which `B` items are chosen. When position-dependent survival is considered (dropping the "identical" assumption but keeping independence), the argument reduces to ordinary fractional knapsack on per-item values `μ_i = E[s_i]`, solved by value-thresholding (Dantzig, 1957).  ∎

**Remark.** Where the local-independence assumption fails — because the positions selected by distinct gate signals are systematically correlated with different quality/survival traits — gates produce different Pareto points at matched budget.

### 5.3 Theorem 9 (Surprisal-Gap Regret Bound)

Under greedy decoding, a gate that forces the argmax into the allowed set `A ⊂ V` at position `i` incurs one-step regret
```
Regret_i(A) = p_i^{(1)} − max_{t ∈ A} p_i(t) ≤ Δ_i,
```
with equality when `A` contains the second-ranked but not the top-ranked token. Thus gating on `Δ_i < τ` bounds per-position regret uniformly by `τ`.

**Proof.** If top-1 ∈ `A`, regret is 0. Otherwise `max_{t ∈ A} p_i(t) ≤ p_i^{(2)}`, yielding `Regret = p_i^{(1)} − max_A p_i ≥ p_i^{(1)} − p_i^{(2)} = Δ`, with equality attained when `A = V \ {top-1}`.  ∎

---

## 6. Experimental Evaluation

### 6.1 Protocol

**Generator model.** `google/gemma-3-270m-it`. Hard-partition MCL with `S = 5`, clockwork topology, secret key fixed throughout.

**Prompt sets.** Two domains: (a) *Wiki*, 176 Wikipedia-concept completions of the form `"Explain X in a comprehensive way."`; (b) *Factual-closed*, 20 short completions of the form `"The capital of France is"` and similar.

**Two experimental configurations are reported.**

- **Configuration A (exploratory, 18 cells)**: `N = 15` prompts, `max_tokens = 50`, all three temperatures `T ∈ {0.0, 0.7, 1.0}`, three watermark budgets `ρ ∈ {0.25, 0.50, 0.75}`, three gates (`G_H_high`, `G_Δ`, `G_H_low`). Wall-clock 50 min. Data: `data/v4_all_20260423_204457/`.
- **Configuration B (confirmatory, 6 cells)**: `N = 176` wiki / `N = 20` factual prompts, `max_tokens = 100`, temperature fixed at `T = 0.7` (practical deployment regime), three watermark budgets `ρ ∈ {0.25, 0.50, 0.75}`, three gates. Wall-clock 3 h 11 min. Data: `data/v5_big_N200/`.

**Calibration.** Thresholds `τ` are selected per-gate from a pilot so that each gate's empirical `ρ` matches the target:
- `G_H_high(τ)` uses `τ = Q_{1-ρ}(H_pilot)`,
- `G_Δ(τ)` uses `τ = Q_{ρ}(Δ_pilot)`,
- `G_H_low(τ)` uses `τ = Q_{ρ}(H_pilot)`.

**Metrics.** For each text: perplexity (PPL) measured on the generator's own logits; detection z-score per Algorithm 2; binary detection at `α = 0.01`. For each (gate pair, cell): per-prompt paired delta `Δ_i = m_A(i) − m_B(i)`, paired BCa bootstrap 10,000-sample 95 % confidence interval, and TOST equivalence verdict at pre-registered margins (log-PPL ±0.05, z ±0.5, TPR ±0.05).

**Attacks.** Random token substitution at rate 20 %, and WordNet synonym substitution at rate 20 %.

### 6.2 Gate Comparison — Configuration A (N = 15, 18 cells, exploratory)

Convention: Δ = `G_H_high` − `G_Δ`. Positive log-PPL indicates `G_Δ` has lower perplexity; positive z indicates `G_H_high` has stronger detection. Starred cells (⋆) have 95 % CI excluding zero.

| # | Domain | T | ρ | ΔlogPPL (95 % CI) | Δz (95 % CI) | ΔTPR (95 % CI) |
|---|---|---|---|---|---|---|
| 1 | wiki | 0.0 | 0.25 | +0.067 [−0.055, +0.210] | +0.31 [−0.50, +1.05] | +0.07 [−0.27, +0.27] |
| 2 | wiki | 0.0 | 0.50 | **+0.265** [+0.073, +0.541] ⋆ | +0.57 [−0.29, +1.33] | 0.00 [0.00, 0.00] |
| 3 | wiki | 0.0 | 0.75 | **−0.185** [−0.364, −0.051] ⋆ | −0.50 [−1.33, −0.21] | 0.00 [0.00, 0.00] |
| 4 | wiki | 0.7 | 0.25 | **+0.147** [+0.003, +0.348] ⋆ | **+1.29** [+0.83, +2.14] ⋆ | +0.27 [−0.07, +0.47] |
| 5 | wiki | 0.7 | 0.50 | +0.326 [−0.101, +0.723] | **+1.55** [+0.33, +2.67] ⋆ | 0.00 [0.00, 0.00] |
| 6 | wiki | 0.7 | 0.75 | +0.052 [−0.123, +0.236] | +0.07 [−0.33, +0.45] | 0.00 [0.00, 0.00] |
| 7 | wiki | 1.0 | 0.25 | +0.253 [−0.010, +0.587] | **+1.21** [+0.33, +2.31] ⋆ | +0.07 [−0.20, +0.20] |
| 8 | wiki | 1.0 | 0.50 | **+0.498** [+0.024, +0.999] ⋆ | +0.86 [−0.24, +1.88] | 0.00 [0.00, 0.00] |
| 9 | wiki | 1.0 | 0.75 | −0.042 [−0.202, +0.049] | **−0.12** [−0.43, −0.05] ⋆ | 0.00 [0.00, 0.00] |
| 10 | factual | 0.0 | 0.25 | −0.183 [−0.473, +0.128] | +0.69 [−0.45, +1.88] | −0.07 [−0.40, +0.07] |
| 11 | factual | 0.0 | 0.50 | −0.145 [−0.422, +0.143] | **+0.90** [+0.07, +1.86] ⋆ | 0.00 [0.00, 0.00] |
| 12 | factual | 0.0 | 0.75 | +0.081 [−0.074, +0.317] | +0.21 [−0.02, +0.76] | 0.00 [0.00, 0.00] |
| 13 | factual | 0.7 | 0.25 | **+0.233** [+0.005, +0.471] ⋆ | +0.55 [−0.07, +1.14] | +0.13 [−0.20, +0.33] |
| 14 | factual | 0.7 | 0.50 | −0.251 [−0.596, +0.092] | **−1.29** [−2.21, −0.52] ⋆ | 0.00 [0.00, 0.00] |
| 15 | factual | 0.7 | 0.75 | −0.000 [−0.054, +0.030] | +0.21 [+0.02, +0.58] | 0.00 [0.00, 0.00] |
| 16 | factual | 1.0 | 0.25 | −0.179 [−0.780, +0.271] | +0.24 [−0.93, +1.36] | −0.07 [−0.40, +0.00] |
| 17 | factual | 1.0 | 0.50 | −0.196 [−0.585, +0.127] | −0.14 [−1.02, +0.38] | 0.00 [0.00, 0.00] |
| 18 | factual | 1.0 | 0.75 | +0.005 [−0.147, +0.129] | −0.10 [−0.50, +0.10] | 0.00 [0.00, 0.00] |

### 6.3 Gate Comparison — Configuration B (N = 176 wiki / 20 factual, T = 0.7, confirmatory)

Same convention as Table 6.2. Scaled N with paired BCa bootstrap on 10,000 resamples; 3-hour wall-clock total.

| # | Domain | ρ | ΔlogPPL (95 % CI) | Δz (95 % CI) | ΔTPR (95 % CI) | Verdict |
|---|---|---|---|---|---|---|
| 1 | wiki | 0.25 | −0.038 [−0.096, +0.018] | **+0.46** [+0.15, +0.79] ⋆ | −0.02 [−0.06, +0.02] | z favors H_high |
| 2 | wiki | 0.50 | −0.036 [−0.133, +0.062] | **+0.80** [+0.40, +1.21] ⋆ | 0.00 [0.00, 0.00] | **TPR EQUIVALENT**; z favors H_high |
| 3 | wiki | 0.75 | +0.024 [−0.093, +0.140] | +0.11 [−0.07, +0.27] | 0.00 [0.00, 0.00] | **z EQUIVALENT**, **TPR EQUIVALENT** |
| 4 | factual | 0.25 | +0.109 [−0.076, +0.260] | +1.37 [−0.33, +2.65] | +0.15 [−0.10, +0.30] | — (N = 20, wide CIs) |
| 5 | factual | 0.50 | −0.033 [−0.254, +0.138] | +1.01 [−0.23, +2.15] | 0.00 [0.00, 0.00] | **TPR EQUIVALENT** |
| 6 | factual | 0.75 | −0.208 [−0.642, +0.080] | +0.01 [−0.86, +0.31] | 0.00 [0.00, 0.00] | **TPR EQUIVALENT** |

### 6.4 Robust Patterns

**Low-entropy gating is strictly Pareto-dominated.** In every cell in which `G_H_low` was compared with both `G_H_high` and `G_Δ` at matched budget (across both Configurations A and B), `G_H_low` produced simultaneously higher perplexity *and* lower detection z. The result is stable across all tested (domain, temperature, budget) conditions.

**Quality (PPL) is gate-invariant at matched budget.** In Configuration B (`N = 176`), all six log-PPL 95 % CIs include zero, with point estimates in the range `[−0.208, +0.109]`. Configuration A's sporadic significant PPL deltas at `N = 15` (cells 2, 3, 4, 8, 13) are not reproduced at higher sample size; the 2.3×–7× tightening of CIs at `N = 176` isolates them as pilot-scale artifacts.

**Detection probability (TPR) is gate-invariant at practical budgets.** In Configuration A, 12 of 18 cells exhibit ΔTPR point-estimate exactly 0; within `ρ ≥ 0.50`, all 12 cells show `ΔTPR = 0`. In Configuration B, 4 of 6 cells satisfy the TPR equivalence margin formally; the remaining two are underpowered at factual-domain `N = 20`.

**Detection strength (z) shows a modest, reliable advantage for `G_H_high` at low-to-moderate budgets.** In Configuration B, wiki cells at `ρ ∈ {0.25, 0.50}` have Δz 95 % CIs `[+0.15, +0.79]` and `[+0.40, +1.21]` respectively — both excluding zero and both corresponding to a `G_H_high` detection strength higher than `G_Δ` by roughly one `σ`. The advantage disappears at `ρ = 0.75` (CI `[−0.07, +0.27]`, within the equivalence margin).

**Budget-convergence at ρ = 0.75.** Across both configurations, the ρ = 0.75 regime is the most consistent: at high budget, the set of positions gated by any reasonable rule overlaps extensively, and gates converge. This matches Theorem 8's prediction under local independence.

### 6.5 Theory-Data Consistency

**Theorem 1 (detection formula).** Spot-check in Configuration B at wiki / T = 0.7 / ρ = 0.50, `G_H_high` (empirical `ρ̄ = 0.53`, `n = 100`): Theorem 1 predicts `z = 0.53 · √(4 · 99) = 10.55`; observed `z = 12.71`. Spot-check in Configuration A at wiki / T = 0.0 / ρ = 0.50, `G_H_high` (empirical `ρ̄ = 0.57`, `n = 50`): predicts `z = 7.98`, observed `7.90`. Across spot-checked cells the relative error is within 1–20 %, consistent with finite-sample Bernoulli deviations from the expected-value formula.

**Theorem 4 (robustness).** Monte Carlo verification with 10⁴ trials per configuration confirms the formula `E[φ|attack] = p_0 + ρ(1 − δ)²(1 − p_0)` to within 0.01 for `(S, ρ, δ) ∈ {(7, 1, 0), (7, 1, 0.29), (5, 0.5, 0.2), (2, 1, 0.5)}`. The universal midpoint critical rate `δ*_mid = 1 − 1/√2` holds to 10⁻¹² across (S, ρ) pairs.

**Theorem 8 (matched-budget equivalence).** Empirically realized for quality (PPL) at all tested budgets in Configuration B (all six log-PPL CIs include zero). The z-score is also equivalent at `ρ = 0.75`. At lower budgets the theorem's local-independence assumption breaks: positions selected by the `G_H_high` and `G_Δ` gates are systematically different on *per-position detection yield* — the observed Δz > 0 at `ρ ∈ {0.25, 0.50}` reflects this, not a violation of the theorem. (The theorem conditions on identical marginal survival `μ_i`; at moderate ρ, the two gates sample from regions with different effective `μ_i`.)

**Theorem 9 (Δ regret bound).** The theorem's pointwise bound `Regret ≤ Δ` holds by construction for `G_Δ` under greedy decoding. Configuration A cells 2, 4, 7, 8 show `G_Δ`-favored PPL at `T = 0.0` and `T = 0.7` with `N = 15`; this direction does not survive the scale-up to `N = 176` in Configuration B (all wiki log-PPL CIs include zero). The theorem's prediction is *compatible* with equivalence but does not *require* `G_Δ` dominance.

---

## 7. Discussion

**Governance implications.** The matched-budget study indicates that the design parameter operationally relevant for MCL is the watermark budget `ρ`, not the gate heuristic used to spend it. At `ρ ≥ 0.5`, detection probability does not depend on which of `G_H_high`, `G_Δ` is used; *quality is equivalent across tested cells* at `N = 176`. A small but reliable detection-strength advantage of `G_H_high` over `G_Δ` persists at low-to-moderate `ρ` on open-ended text (wiki), amounting to roughly one standard-deviation of z. A watermarking standardization effort that specifies `ρ` and leaves gate choice to the operator therefore captures the dominant dimension on which quality and detection probability depend. Theorem 2's closed-form calibration additionally gives a reproducible procedure for setting `S` given target significance and expected text length.

**When to prefer MCL.** MCL is attractive when (i) detection must run without the generator model (strictly model-free), (ii) the operator requires a closed-form calibration rather than grid search, or (iii) deterministic auditable generation under greedy decoding is desirable (fixed key + fixed prompt ⟹ byte-identical output).

**When not to prefer MCL.** Applications that require strict distortion-freeness (output distribution identical to unwatermarked in expectation) should use Gumbel-type or coding-theoretic schemes. Applications that must resist strong paraphrase attacks with token-level rewriting should be evaluated directly against the target adversary.

**Limitations.**

1. Empirical evaluation uses a 270 M parameter generator (Gemma 3 270M). Distributional structure of (H, Δ) shifts with model scale; the matched-budget equivalence claim should be re-verified on larger models.
2. Paired-bootstrap 95 % confidence intervals are wide at the pilot sample size used; formal TOST declarations of equivalence require larger samples.
3. The attack set (random substitution, synonym substitution) does not include paraphrase attacks such as DIPPER or SIRA, which are the standard adversarial benchmark for content-rewriting robustness.
4. Cross-domain evaluation is limited to open-ended explanatory text and factual completions; code generation and chain-of-thought reasoning are not included.

---

## 8. Conclusion

We have introduced Markov Chain Lock (MCL), a discrete hard-constraint watermarking primitive for language models. MCL's generation applies a secret Markov transition over a keyed vocabulary partition; its detector runs without invoking the model and returns a closed-form z-score against a known random baseline. We derived closed-form formulas for detection strength and calibration of the state count, together with a universal midpoint critical modification rate `δ* = 1 − 1/√2`. A matched-budget study of four gate strategies establishes that low-entropy gating is strictly Pareto-dominated and that detection probability is gate-invariant at practical budgets. Together, these results support specifying the watermark budget — rather than a gating heuristic — as the parameter for deployment and standardization.

---

## References (citation keys to be filled in BibTeX)

- KGW — Kirchenbauer et al., *A Watermark for Large Language Models*, ICML 2023.
- Unigram — Zhao et al., *Provable Robust Watermarking for AI-Generated Text*, ICLR 2024.
- SynthID-Text — Dathathri et al., *Scalable Watermarking for Identifying Large Language Model Outputs*, Nature 2024.
- SWEET — Lee et al., *Who Wrote this Code? Watermarking for Code Generation*, ACL 2024.
- EWD — Lu et al., *An Entropy-based Text Watermarking Detection Method*, ACL 2024.
- STA-1 — Mao et al., *A Watermark for Low-Entropy and Unbiased Generation in LLMs*, 2024.
- Invisible Entropy — Gu et al., *Invisible Entropy: Towards Safe and Efficient Low-Entropy LLM Watermarking*, EMNLP 2025.
- Pattern-Mark — Liu et al., *A Watermark for Order-Agnostic Language Models*, NeurIPS 2024.
- SIR — Liu et al., *A Semantic Invariant Robust Watermark for LLMs*, 2024.
- HeavyWater/SimplexWater — Tsur et al., *Watermarking Low-Entropy Text Distributions*, NeurIPS 2025.
- Kuditipudi RDF — Kuditipudi et al., *Robust Distortion-free Watermarks for Language Models*, TMLR 2024.
- Li-Chen-Ročková-Su — *A Statistical Framework of Watermarks for Large Language Models*, 2024.
- Three Bricks — Fernandez et al., *Three Bricks to Consolidate Watermarks for LLMs*, 2023.
- Christ-Gunn-Zamir — *Undetectable Watermarks for Language Models*, COLT 2024.
- DIPPER — Krishna et al., *Paraphrasing evades detectors of AI-generated text, but retrieval is an effective defense*, NeurIPS 2023.
- SIRA — Cheng et al., *Self-Information Rewrite Attacks on Watermarks*, ICML 2025.
- Lakens — *Equivalence Tests: A Practical Primer*, 2017.
- TAIG 2025 — Nemecek et al., *Watermarking Without Standards Is Not AI Governance*.

---

## Appendix A. Complete Proofs

### A.1 Proof of Theorem 1

We begin with the null distribution under the random-oracle assumption. For independent tokens `t_i, t_{i+1}`, `σ(t_i)` and `σ(t_{i+1})` are uniform on `{0, …, S − 1}` and independent. The indicator `X_i = 𝟙{T(σ(t_i), σ(t_{i+1})) = 1}` has mean `p_0 = P[σ(t_{i+1}) ∈ allowed successors of σ(t_i)] = (1/S) ∑_s P[successor of s ∈ allowed set from s] = mean(T > 0)`. For clockwork, `p_0 = 1/S`.

We verify that pairwise covariances vanish. For adjacent pairs `X_i, X_{i+1}`, they share token `t_{i+1}`. Conditioning on `σ(t_{i+1}) = s'`, `X_i` depends on whether `T(σ(t_i), s') = 1` and `X_{i+1}` depends on whether `T(s', σ(t_{i+2})) = 1`. Because `σ(t_i)` and `σ(t_{i+2})` are independent of each other given `s'`, `X_i ⫫ X_{i+1} | σ(t_{i+1})`. For clockwork, both conditional probabilities equal `1/S` regardless of `s'`, so `E[X_i X_{i+1}] = (1/S)^2 = p_0^2`. Hence `Cov(X_i, X_{i+1}) = 0`. Non-adjacent pairs are independent by disjointness of the token indices.

Therefore `Var(φ) = (n − 1)^{-2} ∑_i Var(X_i) = (n − 1)^{-1} p_0 (1 − p_0)`.

Under `H_1`: `E[X_i] = E[W_i · 1 + (1 − W_i) · p_0] = ρ + (1 − ρ) p_0 = p_0 + ρ(1 − p_0)`. The standardized effect size is
```
z = (p_0 + ρ(1 − p_0) − p_0) / √(p_0 (1 − p_0) / (n − 1)) = ρ (1 − p_0) √((n − 1) / (p_0 (1 − p_0))) = ρ √((1 − p_0)(n − 1) / p_0).
```
For clockwork (`p_0 = 1/S`): `z = ρ √((1 − 1/S)(n − 1) / (1/S)) = ρ √((S − 1)(n − 1))`.

Hoeffding's inequality applied to the subsequence `(X_1, X_3, X_5, …)` (mutually independent) and `(X_2, X_4, X_6, …)` separately, then combining by union bound, yields `P[φ ≥ τ | H_0] ≤ 2 exp(−2 (n − 1)(τ − p_0)^2)`.  ∎

### A.2 Proof of Theorem 4

Let `A_i ∈ {0, 1}` indicate whether token `t_i` survives the attack, with `P[A_i = 1] = 1 − δ` independently. For pair `(t_i, t_{i+1})`:

- If `A_i = A_{i+1} = 1` (probability `(1 − δ)^2`): the pair is as in the unattacked regime, `E[X_i | A_i = A_{i+1} = 1] = ρ + (1 − ρ) p_0`.
- Otherwise: at least one token is replaced by a fresh random-oracle draw; conditioning on the fresh state being uniform, `E[X_i | otherwise] = p_0`.

Combining:
```
E[X_i] = (1 − δ)^2 (ρ + (1 − ρ) p_0) + (1 − (1 − δ)^2) p_0
        = (1 − δ)^2 ρ + (1 − δ)^2 (1 − ρ) p_0 + p_0 − (1 − δ)^2 p_0
        = p_0 + ρ (1 − δ)^2 − ρ (1 − δ)^2 p_0
        = p_0 + ρ (1 − δ)^2 (1 − p_0).
```

**Midpoint universality.** Set `τ_mid = p_0 + ρ(1 − p_0)/2` and equate:
```
p_0 + ρ (1 − δ)^2 (1 − p_0) = p_0 + ρ (1 − p_0)/2 ⟹ (1 − δ)^2 = 1/2 ⟹ δ = 1 − 1/√2.
```
The cancellation of `S` (via `p_0`) and `ρ` is algebraically exact; hence universality.

**Size-α rate.** The size-`α` threshold is `τ_α = p_0 + z_α √(p_0 (1 − p_0)/(n − 1))`. Setting `E[φ | attack] = τ_α` and solving:
```
ρ (1 − δ)^2 (1 − p_0) = z_α √(p_0 (1 − p_0)/(n − 1))
⟺ (1 − δ)^2 = z_α √(p_0 / ((1 − p_0)(n − 1))) / ρ.
```
Recognizing `z = ρ √((1 − p_0)(n − 1)/p_0)` from Theorem 1, this is `(1 − δ)^2 = z_α / z`, so `δ*_α = 1 − √(z_α / z)`.  ∎

### A.3 Proof of Theorem 8

Fix budget `B`. The objective is `R(g) = ∑_i g_i · s_i · s_{i-1}`. Under local independence with identical marginal `μ`, `E[s_i · s_{i-1}] = μ^2` independent of `i`. Then `E[R(g)] = μ^2 · ∑ g_i = μ^2 B`, independent of which `B` positions are selected. At matched budget, all gates produce identical expected signal.

Relaxing to position-dependent but independent survival probabilities `μ_i = E[s_i]`, the objective becomes `E[R(g)] = ∑ g_i · μ_i · μ_{i-1}`; the optimal selection thresholds on the product `μ_i μ_{i-1}`, or on `μ_i` alone under stationarity. Fractional knapsack on unit-weight items with these values is solved by value-thresholding; with integer weights the integral optimum coincides with the fractional optimum.

When the local-independence assumption fails — i.e., `s_i` correlates differently with different gating signals — different gates select systematically different position subsets, yielding different `E[R]` at matched budget.  ∎

---

## Appendix B. Reference Detector Implementation

```python
import hashlib
from math import sqrt
from scipy.stats import norm

def sha_state(key: bytes, token_id: int, S: int) -> int:
    h = hashlib.sha256(key + token_id.to_bytes(8, "big")).digest()
    return int.from_bytes(h[:8], "big") % S

def detect_mcl(token_ids, key: bytes, S: int, alpha: float = 0.01) -> dict:
    n = len(token_ids)
    if n < 2:
        return {"phi": 0.0, "z": 0.0, "p_value": 1.0, "is_watermarked": False}
    states = [sha_state(key, t, S) for t in token_ids]
    valid = sum(1 for i in range(n - 1) if states[i + 1] == (states[i] + 1) % S)
    phi = valid / (n - 1)
    p0 = 1 / S
    se = sqrt(p0 * (1 - p0) / (n - 1))
    z = (phi - p0) / se
    p_value = 1.0 - norm.cdf(z)
    return {
        "phi": phi,
        "z": z,
        "p_value": p_value,
        "is_watermarked": p_value < alpha,
    }
```

This 24-line implementation suffices for clockwork MCL detection. Generalizing to arbitrary transition matrices `T` requires replacing the successor check with `T[states[i]][states[i + 1]] > 0` and computing `p_0 = (T > 0).mean()`; the rest is unchanged.
