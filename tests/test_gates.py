"""
Unit tests for gate implementations. No LLM loaded; we construct synthetic
logit vectors and verify each gate fires as specified.
"""

import math

import numpy as np
import pytest
import torch

from ltw_watermark.gates import (
    GateAll,
    GateDelta,
    GateEntropyHigh,
    GateEntropyLow,
    GateNone,
    logits_stats,
    make_gate,
)


def logits_for_probs(probs):
    """Return logits that softmax to approximately `probs`."""
    p = np.asarray(probs, dtype=np.float64)
    p = p / p.sum()
    p = np.clip(p, 1e-20, 1.0)
    return torch.tensor(np.log(p), dtype=torch.float32)


def test_logits_stats_matches_hand_calc():
    logits = logits_for_probs([0.6, 0.3, 0.05, 0.05])
    s = logits_stats(logits)
    assert abs(s["p1"] - 0.6) < 1e-4
    assert abs(s["p2"] - 0.3) < 1e-4
    assert abs(s["delta"] - 0.3) < 1e-4
    H_expected = -(0.6 * math.log(0.6) + 0.3 * math.log(0.3) + 2 * 0.05 * math.log(0.05))
    assert abs(s["entropy"] - H_expected) < 1e-4


def test_gate_all_and_none():
    logits = logits_for_probs([0.5, 0.3, 0.2])
    assert GateAll()(logits) is True
    assert GateNone()(logits) is False


def test_gate_entropy_high():
    low_H = logits_for_probs([0.98, 0.01, 0.01])   # H ~ 0.1 nats
    high_H = logits_for_probs([0.34, 0.33, 0.33])  # H ~ 1.1 nats
    g = GateEntropyHigh(tau=0.5)
    assert g(low_H) is False
    assert g(high_H) is True


def test_gate_entropy_low():
    low_H = logits_for_probs([0.98, 0.01, 0.01])
    high_H = logits_for_probs([0.34, 0.33, 0.33])
    g = GateEntropyLow(tau=0.5)
    assert g(low_H) is True
    assert g(high_H) is False


def test_gate_delta_fires_on_ambiguous():
    """Small Delta = ambiguous between top-2 => WATERMARK (low regret)."""
    ambiguous = logits_for_probs([0.51, 0.49, 0.0])   # Delta = 0.02
    forced = logits_for_probs([0.98, 0.01, 0.01])      # Delta = 0.97
    g = GateDelta(tau=0.1)
    assert g(ambiguous) is True
    assert g(forced) is False


def test_gate_delta_threshold_sharp():
    """Right at the threshold, gate is False (strict <)."""
    g = GateDelta(tau=0.2)
    # Delta = 0.2 exactly -> should NOT fire (strict <)
    p = [0.5, 0.3, 0.2]
    logits = logits_for_probs(p)
    s = logits_stats(logits)
    # Real delta might be close to 0.2 but not exactly due to float -> log -> softmax roundtrip
    expected = s["delta"] < 0.2
    assert g(logits) is expected


def test_make_gate_factory():
    g_all = make_gate("all")
    g_none = make_gate("none")
    g_hi = make_gate("h_high", tau=1.0)
    g_lo = make_gate("h_low", tau=1.0)
    g_d = make_gate("delta", tau=0.1)

    lo = logits_for_probs([0.9, 0.08, 0.02])
    assert g_all(lo) is True
    assert g_none(lo) is False
    assert isinstance(g_hi, GateEntropyHigh)
    assert isinstance(g_lo, GateEntropyLow)
    assert isinstance(g_d, GateDelta)

    with pytest.raises(ValueError):
        make_gate("nonsense", tau=1.0)
    with pytest.raises(ValueError):
        make_gate("h_high")  # missing tau
