"""
Gate signals for watermark position selection.

A `Gate` takes the current-step logits (and optionally a proxy-LM score) and
returns True iff the watermark should be applied at this position.

Implements:
  - GateAll          : always watermark
  - GateEntropyHigh  : watermark iff H(p) > tau                    (SWEET / IE family)
  - GateEntropyLow   : watermark iff H(p) < tau                    (v2 default)
  - GateDelta        : watermark iff (p1 - p2) < tau               (v3 headline)
  - GateSurprisalGap : alias of GateDelta with inverted threshold  (convenience)
  - GatePSurviveOracle : watermark iff pilot P_survive(i) >= tau   (optimal per Thm 9)

All gates return a boolean. Per-position metadata (H, Delta, p1, p2, rank) is
returned alongside via `describe()` so experiment scripts can record it.

Theory reminder (v3 plan Theorems 9 & 10):
  - Thm 9: under local-iid survival, optimal gate thresholds on P_survive.
  - Thm 10: under argmax decoding, gating on Delta < tau caps per-position
            regret (drop in top-1 probability) by tau.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F


def logits_stats(logits: torch.Tensor) -> Dict[str, float]:
    """Compute (p1, p2, delta, entropy, argmax_id) from a (vocab,) logit tensor."""
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = float(-(probs * log_probs).sum().item())
    top2 = torch.topk(probs, k=2)
    p1 = float(top2.values[0].item())
    p2 = float(top2.values[1].item())
    argmax_id = int(top2.indices[0].item())
    return {"p1": p1, "p2": p2, "delta": p1 - p2, "entropy": entropy, "argmax_id": argmax_id}


# -------------------- Gate interface --------------------
class Gate(ABC):
    name: str = "Gate"

    @abstractmethod
    def should_watermark(self, stats: Dict[str, float]) -> bool: ...

    def __call__(self, logits: torch.Tensor) -> bool:
        return self.should_watermark(logits_stats(logits))

    def describe(self, logits: torch.Tensor) -> Dict[str, Any]:
        stats = logits_stats(logits)
        return {"gate": self.name, **stats, "watermark": self.should_watermark(stats)}


@dataclass
class GateAll(Gate):
    name: str = "all"

    def should_watermark(self, stats):
        return True


@dataclass
class GateNone(Gate):
    name: str = "none"

    def should_watermark(self, stats):
        return False


@dataclass
class GateEntropyHigh(Gate):
    """SWEET / Invisible-Entropy family: watermark only at HIGH entropy."""
    tau: float
    name: str = "H_high"

    def should_watermark(self, stats):
        return stats["entropy"] > self.tau


@dataclass
class GateEntropyLow(Gate):
    """v2 default: watermark only at LOW entropy."""
    tau: float
    name: str = "H_low"

    def should_watermark(self, stats):
        return stats["entropy"] < self.tau


@dataclass
class GateDelta(Gate):
    """
    v3 headline: watermark iff Delta = p1 - p2 < tau.
    'Small Delta' means top-1 and top-2 are close -> argmax flip costs little
    regret (per Thm 10). Symmetrically, when Delta is near 1 (forced token)
    we DO NOT gate, because forcing a different token would wreck output.
    """
    tau: float
    name: str = "Delta"

    def should_watermark(self, stats):
        return stats["delta"] < self.tau


@dataclass
class GatePSurviveOracle(Gate):
    """
    Pareto-optimal gate (per Thm 9) with access to a pilot-measured
    P_survive function. `psurv_fn` maps (step_index, logits_stats) -> float.
    """
    tau: float
    psurv_fn: Callable[[int, Dict[str, float]], float]
    name: str = "Psurv"

    def should_watermark(self, stats):
        # step index must be injected separately via __call__ override
        raise NotImplementedError("Use PSurviveOracle via __call__(logits, step=i)")

    def __call__(self, logits: torch.Tensor, step: int = 0) -> bool:
        stats = logits_stats(logits)
        return self.psurv_fn(step, stats) >= self.tau


# -------------------- Factory --------------------
def make_gate(name: str, tau: Optional[float] = None, **kwargs) -> Gate:
    name = name.lower()
    if name in ("all", "none"):
        return GateAll() if name == "all" else GateNone()
    if tau is None:
        raise ValueError(f"gate '{name}' requires tau")
    if name in ("h_high", "entropy_high", "entropyhigh"):
        return GateEntropyHigh(tau=tau)
    if name in ("h_low", "entropy_low", "entropylow"):
        return GateEntropyLow(tau=tau)
    if name in ("delta", "surprisal_gap", "sgg"):
        return GateDelta(tau=tau)
    raise ValueError(f"unknown gate: {name}")


__all__ = [
    "Gate",
    "GateAll",
    "GateNone",
    "GateEntropyHigh",
    "GateEntropyLow",
    "GateDelta",
    "GatePSurviveOracle",
    "logits_stats",
    "make_gate",
]
