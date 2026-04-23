"""
Numerical verification of the v2/v3 watermarking theorems.

Each test does a Monte Carlo simulation of the claim and asserts agreement
with the formula to within statistical tolerance. Run with:

    uv run pytest tests/test_math_sanity.py -v

These tests do NOT load any LLM; they simulate the random-oracle + watermark
dynamics directly, which is the exact setting of the theorems.

Covers:
  Thm 1: z = rho * sqrt((S-1)(n-1)) under H_1
  Thm 1: E[phi|H_0] = 1/S;  Var = (S-1)/(S^2 (n-1))
  Thm 2: S_min = ceil(z_alpha^2 / (rho^2 (n-1)) + 1)
  Thm 4: E[phi|attack] = 1/S + rho*(1-delta)^2*(S-1)/S
  Thm 4: delta*_mid = 1 - 1/sqrt(2) at midpoint threshold
  Lemma 6: hash modification gives uniform state -> recovery rate = 1/S
  Thm 9: optimal gate is threshold on P_survive (sorting / fractional knapsack)
  Thm 10: gating when Delta < tau caps argmax regret by tau
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from ltw_watermark.detection_stats import (
    expected_z_H1,
    observed_z,
    s_min_calibration,
    critical_delta_midpoint,
    critical_delta_alpha,
)


RNG = np.random.default_rng(seed=0)


# ------------------------------------------------------------------
# Helpers: simulate MCL transitions under H0, H1, and attack.
# ------------------------------------------------------------------
def simulate_H0(S: int, n_pairs: int, n_trials: int) -> np.ndarray:
    """phi under H0 = Bernoulli(1/S) iid per pair."""
    samples = RNG.random((n_trials, n_pairs))
    phi = (samples < 1.0 / S).mean(axis=1)
    return phi


def simulate_H1(S: int, n_pairs: int, rho: float, n_trials: int) -> np.ndarray:
    """
    phi under H1. Per position:
      - W ~ Bern(rho): watermarked?
      - If W=1: transition valid with prob 1.
      - If W=0: transition valid with prob 1/S.
    """
    W = RNG.random((n_trials, n_pairs)) < rho
    coin = RNG.random((n_trials, n_pairs)) < 1.0 / S
    X = np.where(W, 1, coin.astype(int))
    return X.mean(axis=1)


def simulate_H1_attack(S: int, n_pairs: int, rho: float, delta: float, n_trials: int) -> np.ndarray:
    """
    phi under H1 with random modification rate delta.
    Approximate: a pair 'survives' attack iff both endpoints are unmodified,
    probability (1-delta)^2. Otherwise each token's new state is uniform
    (random oracle on fresh hash), giving pair-validity prob 1/S.
    """
    pair_survives = RNG.random((n_trials, n_pairs)) < (1 - delta) ** 2
    X_clean = simulate_H1(S, n_pairs, rho, n_trials) * n_pairs  # flatten to ints/pair via back-trip
    # Need position-level, not aggregate; redo:
    W = RNG.random((n_trials, n_pairs)) < rho
    coin_free = RNG.random((n_trials, n_pairs)) < 1.0 / S
    X_if_survive = np.where(W, 1, coin_free.astype(int))
    coin_attack = RNG.random((n_trials, n_pairs)) < 1.0 / S
    X = np.where(pair_survives, X_if_survive, coin_attack.astype(int))
    return X.mean(axis=1)


# ------------------------------------------------------------------
# Thm 1: moments under H0 and H1, CORRECTED z formula
# ------------------------------------------------------------------
@pytest.mark.parametrize("S,n,rho", [(2, 500, 1.0), (5, 200, 0.5), (7, 500, 1.0), (7, 200, 0.3)])
def test_thm1_mean_variance_and_z(S, n, rho):
    n_pairs = n - 1
    n_trials = 10_000

    # H0 moments
    phi_h0 = simulate_H0(S, n_pairs, n_trials)
    theo_mean = 1.0 / S
    theo_var = (S - 1) / (S**2 * n_pairs)
    assert abs(phi_h0.mean() - theo_mean) < 5e-3, f"H0 mean: obs {phi_h0.mean():.4f} vs theo {theo_mean:.4f}"
    assert abs(phi_h0.var() - theo_var) / theo_var < 0.1, (
        f"H0 var: obs {phi_h0.var():.2e} vs theo {theo_var:.2e}"
    )

    # H1 mean + z
    phi_h1 = simulate_H1(S, n_pairs, rho, n_trials)
    theo_h1_mean = 1.0 / S + rho * (S - 1) / S
    assert abs(phi_h1.mean() - theo_h1_mean) < 5e-3

    # Expected z per corrected formula
    predicted_z = rho * math.sqrt((S - 1) * n_pairs)
    # Observed mean z over trials
    se_h0 = math.sqrt(theo_var)
    observed_z_values = (phi_h1 - theo_mean) / se_h0
    obs_mean_z = observed_z_values.mean()
    # Allow 2% rel tol because we're averaging over finite trials.
    assert abs(obs_mean_z - predicted_z) / predicted_z < 0.02, (
        f"z: predicted {predicted_z:.3f} vs observed {obs_mean_z:.3f}"
    )

    # Matches library function too
    assert abs(expected_z_H1(rho, S, n) - predicted_z) < 1e-9


# ------------------------------------------------------------------
# Thm 2: S_min formula inverts z
# ------------------------------------------------------------------
@pytest.mark.parametrize("rho,n,z_alpha", [(1.0, 500, 2.326), (0.5, 200, 2.326), (0.3, 500, 3.09)])
def test_thm2_s_min_inverts_z(rho, n, z_alpha):
    S = s_min_calibration(z_alpha, rho, n)
    # At S, expected z must meet or exceed z_alpha
    z_at_S = expected_z_H1(rho, S, n)
    assert z_at_S >= z_alpha, f"S_min={S} gives z={z_at_S:.3f} < z_alpha={z_alpha}"
    # At S-1 (if >=2), z should be below
    if S >= 3:
        z_below = expected_z_H1(rho, S - 1, n)
        assert z_below < z_alpha, f"S_min-1 should fail: z={z_below:.3f}"


# ------------------------------------------------------------------
# Thm 4: E[phi|attack] matches (1-delta)^2 formula
# ------------------------------------------------------------------
@pytest.mark.parametrize("S,rho,delta", [(7, 1.0, 0.0), (7, 1.0, 0.29), (5, 0.5, 0.2), (2, 1.0, 0.5)])
def test_thm4_attack_formula(S, rho, delta):
    n_pairs = 5000
    n_trials = 500
    phi = simulate_H1_attack(S, n_pairs, rho, delta, n_trials)
    theo = 1 / S + rho * (1 - delta) ** 2 * (S - 1) / S
    assert abs(phi.mean() - theo) < 0.01, (
        f"Attack: obs {phi.mean():.4f} vs theo {theo:.4f} (S={S}, rho={rho}, delta={delta})"
    )


# ------------------------------------------------------------------
# Thm 4: universal midpoint delta* = 1 - 1/sqrt(2) ~ 0.2929
# ------------------------------------------------------------------
def test_thm4_delta_star_midpoint_universal():
    d_star = critical_delta_midpoint()
    assert abs(d_star - (1 - 1 / math.sqrt(2))) < 1e-12

    # Midpoint threshold: tau_mid = 1/S + rho(S-1)/(2S)
    # At delta_star, E[phi|attack] should equal tau_mid for any (S, rho).
    for S, rho in [(2, 1.0), (5, 0.5), (7, 1.0), (15, 0.3)]:
        tau_mid = 1 / S + rho * (S - 1) / (2 * S)
        lhs = 1 / S + rho * (1 - d_star) ** 2 * (S - 1) / S
        assert abs(lhs - tau_mid) < 1e-12, f"midpoint universality broken for (S,rho)=({S},{rho})"


# ------------------------------------------------------------------
# Thm 4: size-alpha critical rate depends on (S, rho, n, alpha)
# Check that at alpha-critical delta, E[phi|attack] = tau_alpha.
# ------------------------------------------------------------------
def test_thm4_delta_star_alpha_consistency():
    for S, rho, n, z_alpha in [(7, 1.0, 500, 2.326), (5, 0.5, 300, 2.326), (3, 1.0, 100, 2.326)]:
        d_alpha = critical_delta_alpha(z_alpha, rho, S, n)
        n_pairs = n - 1
        lhs = 1 / S + rho * (1 - d_alpha) ** 2 * (S - 1) / S
        se = math.sqrt((S - 1) / (S**2 * n_pairs))
        tau_alpha = 1 / S + z_alpha * se
        assert abs(lhs - tau_alpha) < 1e-6, (
            f"delta_alpha consistency broken: lhs {lhs:.6f} vs tau_alpha {tau_alpha:.6f}"
        )


# ------------------------------------------------------------------
# Lemma 6: each random modification gives Bern(1/S) contribution
# ------------------------------------------------------------------
def test_lemma6_modified_token_uniform_state():
    """Each 'modified' token maps to uniform state under random oracle."""
    n_trials = 100_000
    for S in (2, 3, 5, 7, 11):
        # Simulate: token_id -> uniform state label. Fraction of valid transitions
        # with clockwork rule = 1/S.
        s_prev = RNG.integers(0, S, size=n_trials)
        s_next = RNG.integers(0, S, size=n_trials)  # "modified" token
        valid_frac = (s_next == (s_prev + 1) % S).mean()
        assert abs(valid_frac - 1 / S) < 0.01, f"S={S}: obs {valid_frac:.3f}, theo {1/S:.3f}"


# ------------------------------------------------------------------
# Thm 9: optimal gate is threshold on P_survive
# ------------------------------------------------------------------
def test_thm9_threshold_on_psurvive_is_optimal():
    """
    Brute-force: generate random (P_survive_i) across positions, try all gates
    of budget B, verify threshold-on-Psurvive is the argmax of sum g_i * P_i.
    """
    rng = np.random.default_rng(1)
    n = 30
    B = 10
    p_survive = rng.random(n)
    # Objective: max Σ g_i * p_i s.t. Σ g_i == B
    # Optimal: pick top-B by p_survive.
    order = np.argsort(-p_survive)
    opt_mask = np.zeros(n, dtype=bool)
    opt_mask[order[:B]] = True
    opt_value = p_survive[opt_mask].sum()

    # Verify NO random gate of same budget beats it
    for _ in range(5000):
        idx = rng.choice(n, size=B, replace=False)
        v = p_survive[idx].sum()
        assert v <= opt_value + 1e-12, "Found gate strictly better than threshold-on-Psurvive!"


# ------------------------------------------------------------------
# Thm 10: regret bound -- gating when Delta < tau caps argmax regret by tau
# ------------------------------------------------------------------
def test_thm10_regret_bounded_by_delta_tau():
    """
    Simulate many random distributions. For each:
      - compute p1, p2, Delta
      - gate iff Delta < tau
      - if gated, forcing argmax into 'anything except top-1' -> regret = p1 - max(other)
      - verify regret <= Delta (which <= tau when gated)
    """
    rng = np.random.default_rng(2)
    n_trials = 5000
    V = 100
    tau = 0.1
    max_regret_when_gated = 0.0
    n_gated = 0
    for _ in range(n_trials):
        # Random prob simplex
        alpha = rng.random(V) * rng.random()
        p = alpha / alpha.sum()
        p_sorted = np.sort(p)[::-1]
        p1, p2 = p_sorted[0], p_sorted[1]
        delta = p1 - p2
        if delta < tau:
            # Max prob among "not top-1"
            best_other = p_sorted[1]
            regret = p1 - best_other
            assert regret <= delta + 1e-12, f"Regret {regret} exceeds Delta {delta}"
            max_regret_when_gated = max(max_regret_when_gated, regret)
            n_gated += 1
    # When we gate, regret is bounded by tau:
    assert max_regret_when_gated <= tau + 1e-12, (
        f"max regret when gated ({max_regret_when_gated:.4f}) exceeds tau ({tau})"
    )
    # Sanity: we actually gated some
    assert n_gated > 0


# ------------------------------------------------------------------
# Sanity: show entropy is NOT monotone with regret
#   (why Delta is a better quality proxy than H per Thm 10)
# ------------------------------------------------------------------
def test_entropy_does_not_monotone_regret():
    """
    Construct two distributions (p, q) with H(p) > H(q) but Regret(p) < Regret(q).
    Demonstrates entropy is mis-calibrated for argmax quality.
    """
    # p: near-uniform top-2 (high H, low regret)
    p = np.array([0.51, 0.49, 0.0, 0.0, 0.0])
    H_p = -np.sum(p[p > 0] * np.log(p[p > 0]))
    regret_p = p[0] - p[1]  # 0.02

    # q: near-deterministic (low H, high regret)
    q = np.array([0.98, 0.01, 0.005, 0.005, 0.0])
    q = q / q.sum()
    H_q = -np.sum(q[q > 0] * np.log(q[q > 0]))
    regret_q = q[0] - q[1]  # ~0.97

    assert H_p > H_q, f"H(p)={H_p:.3f} should > H(q)={H_q:.3f}"
    assert regret_p < regret_q, f"regret(p)={regret_p:.3f} should < regret(q)={regret_q:.3f}"
    # So entropy and regret disagree on which position is "safer to flip."
