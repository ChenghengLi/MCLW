"""
Corrected statistical detection utilities for MCL watermarking.

Thm 1 (corrected):
    Under H_0: phi = fraction of valid transitions, E[phi] = 1/S,
               Var(phi) = (S-1) / (S^2 (n-1)).
    z = (phi - 1/S) / sqrt((S-1) / (S^2 (n-1)))
      = (phi - 1/S) * S * sqrt(n-1) / sqrt(S-1)

Thm 2 (corrected): S_min = ceil(z_alpha^2 / (rho^2 (n-1)) + 1).
Thm 4: Under random sub at rate delta,
       E[phi|attack] = 1/S + rho * (1-delta)^2 * (S-1)/S.
       Midpoint critical delta* = 1 - 1/sqrt(2).

Notes on random baselines:
  - clockwork: exactly 1 allowed out of S -> p0 = 1/S
  - soft_cycle: 2 nonzero entries per row -> p0 = 2/S (detector uses `T>0`)
We expose `random_baseline_for_topology` so callers pass `p0` honestly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats  # only for norm.sf and binom.sf (standard in watermarking)

from ltw_watermark.enhanced_mcl import generate_transition_matrix


def random_baseline_for_topology(
    num_states: int, chain_key: str, secret_key: str = "mclw_entropy_gated_2026"
) -> float:
    """Return p0 = mean(T > 0) for the configured topology."""
    T = generate_transition_matrix(num_states, chain_key, secret_key)
    return float(np.mean(T > 0))


def expected_z_H1(rho: float, S: int, n: int, p0: Optional[float] = None) -> float:
    """
    Predicted z-score under H_1 for the CORRECTED formula.

    If p0 = 1/S (clockwork), z = rho * sqrt((S-1)(n-1)).
    For general p0, the effect-size/sd ratio becomes
      z = rho * (1 - p0) / sqrt(p0 * (1 - p0) / (n-1))
        = rho * sqrt((1 - p0) * (n-1) / p0).
    """
    if p0 is None:
        p0 = 1.0 / S
    if n <= 1 or p0 <= 0 or p0 >= 1:
        return 0.0
    return float(rho * math.sqrt((1 - p0) * (n - 1) / p0))


def observed_z(phi: float, n: int, p0: float) -> float:
    """Standardized z of observed phi against baseline p0 under H_0."""
    if n <= 1 or p0 <= 0 or p0 >= 1:
        return 0.0
    se = math.sqrt(p0 * (1 - p0) / (n - 1))
    return (phi - p0) / se


def p_value_one_sided(phi: float, n: int, p0: float, exact: bool = False) -> float:
    """
    One-sided p-value (upper tail) for observing phi or more, given n-1 pairs.

    exact=True -> exact binomial tail, safer for small n.
    exact=False -> normal approximation (valid for (n-1)*p0 >= 5 etc.)
    """
    k = int(round(phi * (n - 1)))
    if exact:
        return float(stats.binom.sf(k - 1, n - 1, p0))
    z = observed_z(phi, n, p0)
    return float(stats.norm.sf(z))


def s_min_calibration(z_alpha: float, rho: float, n: int) -> int:
    """
    Minimum S for z >= z_alpha at clockwork baseline (p0 = 1/S).
    Derived from z^2 = rho^2 * (S-1)(n-1)  ==>  S = z_alpha^2/(rho^2 (n-1)) + 1.
    """
    if rho <= 0 or n <= 1:
        return 0
    return int(math.ceil((z_alpha ** 2) / (rho ** 2 * (n - 1)) + 1))


@dataclass
class DetectionReport:
    phi: float
    z: float
    p_value: float
    n_pairs: int
    valid: int
    p0: float
    num_states: int
    chain_key: str
    is_watermarked: bool


def detect(
    token_ids: List[int],
    num_states: int,
    chain_key: str,
    secret_key: str,
    overlap_ratio: float = 0.0,
    alpha: float = 0.01,
    exact: bool = False,
) -> DetectionReport:
    """
    Model-free MCL detection with corrected z-score + p-value.

    Uses `get_token_state_soft(...)[0]` (primary state) per token.
    """
    from ltw_watermark.enhanced_mcl import get_token_state_soft

    T = generate_transition_matrix(num_states, chain_key, secret_key)
    p0 = float(np.mean(T > 0))

    if len(token_ids) < 2:
        return DetectionReport(0.0, 0.0, 1.0, 0, 0, p0, num_states, chain_key, False)

    states = [
        get_token_state_soft(int(t), num_states, secret_key, overlap_ratio)[0]
        for t in token_ids
    ]
    n_pairs = len(states) - 1
    valid = sum(1 for i in range(n_pairs) if T[states[i]][states[i + 1]] > 0)
    phi = valid / n_pairs

    z = observed_z(phi, n_pairs + 1, p0)
    p_val = p_value_one_sided(phi, n_pairs + 1, p0, exact=exact)
    return DetectionReport(
        phi=phi,
        z=z,
        p_value=p_val,
        n_pairs=n_pairs,
        valid=valid,
        p0=p0,
        num_states=num_states,
        chain_key=chain_key,
        is_watermarked=p_val < alpha,
    )


def critical_delta_midpoint() -> float:
    """Universal midpoint-threshold critical rate: 1 - 1/sqrt(2) ~ 0.2929."""
    return 1.0 - 1.0 / math.sqrt(2.0)


def critical_delta_alpha(z_alpha: float, rho: float, S: int, n: int, p0: Optional[float] = None) -> float:
    """
    Critical random-sub rate at size-alpha threshold.
    Derived from E[phi|attack] >= p0 + z_alpha * sqrt(p0(1-p0)/(n-1)):
       (1-delta)^2 >= z_alpha / z_expected
    """
    z_exp = expected_z_H1(rho, S, n, p0=p0)
    if z_exp <= 0:
        return 0.0
    ratio = z_alpha / z_exp
    if ratio >= 1.0:
        return 0.0
    return 1.0 - math.sqrt(ratio)


# ---------- HDD-style entropy-weighted detector (Thm 7) ----------
def hdd_lambda(
    valid_flags: List[bool], w_flags: List[bool], S: int
) -> float:
    """
    Neyman-Pearson most-powerful test conditional on known W-profile and ROM null.
    Lambda = log(S) * sum_{i: W_i=1} X_i. (Only watermarked-gate positions count.)
    """
    if S <= 1:
        return 0.0
    log_s = math.log(S)
    return log_s * sum(int(v) for v, w in zip(valid_flags, w_flags) if w)


def hdd_p_value(
    valid_flags: List[bool], w_flags: List[bool], S: int, exact: bool = True
) -> float:
    """
    Under H0 each X_i at a W_i=1 position is Bern(1/S). Count k among K gated positions.
    """
    K = sum(1 for w in w_flags if w)
    if K == 0:
        return 1.0
    k = sum(int(v) for v, w in zip(valid_flags, w_flags) if w)
    if exact:
        return float(stats.binom.sf(k - 1, K, 1.0 / S))
    mean = K * (1.0 / S)
    var = K * (1.0 / S) * (1.0 - 1.0 / S)
    if var <= 0:
        return 1.0
    z = (k - mean) / math.sqrt(var)
    return float(stats.norm.sf(z))
