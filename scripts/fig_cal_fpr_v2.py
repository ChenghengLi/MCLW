#!/usr/bin/env python3
"""
fig_cal_fpr_v2.py
-----------------
Calibration-anchor empirical FPR plot for the MCLW ICML 2026 paper.

For each state count S in {2, 3, 5, 7, 11}:
  * Load /home/lichen/MCLW/data/v7_min/exp5_<model>_S<S>/records.jsonl
    for the 3 LLMs (llama-3-1-8b-instruct, mistral-7b-instruct-v0-3,
    qwen2-5-7b-instruct).
  * Pool z_nwm (non-watermarked z under MCL) across the 3 LLMs.
  * Empirical FPR = fraction with z_nwm > z_alpha = 2.326.

Overlays:
  * Hoeffding upper bound (per-S, see below)
  * Constant Hoeffding bound exp(-2 z_alpha^2) ~= 0.51%
  * alpha = 1% target line.

Output: /home/lichen/MCLW/icml2026/pictures/fig_cal_fpr.pdf
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- config
DATA_ROOT = Path("/home/lichen/MCLW/data/v7_min")
OUT_PATH = Path("/home/lichen/MCLW/icml2026/pictures/fig_cal_fpr.pdf")

MODELS = [
    "llama-3-1-8b-instruct",
    "mistral-7b-instruct-v0-3",
    "qwen2-5-7b-instruct",
]
S_VALUES = [2, 3, 5, 7, 11]

Z_ALPHA = 2.326    # one-sided alpha = 0.01
ALPHA_PCT = 1.0    # target FPR (%)
RHO = 0.5          # gating fraction used in exp5 (matches summary.json rho)


# ---------------------------------------------------------------- io
def load_records(model: str, S: int) -> list[dict]:
    path = DATA_ROOT / f"exp5_{model}_S{S}" / "records.jsonl"
    with path.open("r") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------- aggregation
def empirical_fpr(S: int) -> tuple[float, int, float]:
    """Pool z_nwm across the 3 LLMs at given S; return (FPR%, n, mean_n_tokens)."""
    zs: list[float] = []
    n_tokens: list[int] = []
    for m in MODELS:
        for r in load_records(m, S):
            z = r.get("z_nwm")
            if z is None or (isinstance(z, float) and math.isnan(z)):
                continue
            zs.append(float(z))
            n_tokens.append(int(r.get("n_tokens_nwm", 0)))
    arr = np.asarray(zs, dtype=float)
    fpr = float((arr > Z_ALPHA).mean()) * 100.0 if len(arr) else float("nan")
    mean_n = float(np.mean(n_tokens)) if n_tokens else float("nan")
    return fpr, len(arr), mean_n


def hoeffding_bound_constant() -> float:
    """exp(-2 z_alpha^2) (in %)."""
    return math.exp(-2.0 * Z_ALPHA * Z_ALPHA) * 100.0


def hoeffding_bound_perS(S: int, n: int, rho: float = RHO) -> float:
    """
    Per-S Hoeffding upper bound (in %):
        2 * exp( -2 * floor((n-1)/2) * (rho/2)^2 * (S-1)^2 / S^2 )
    """
    if S <= 1 or n < 2:
        return float("nan")
    m = (n - 1) // 2
    eps = (rho / 2.0) * (S - 1) / S
    return 2.0 * math.exp(-2.0 * m * eps * eps) * 100.0


# ---------------------------------------------------------------- plot
def render(per_s: dict[int, tuple[float, int, float]]) -> None:
    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(3.0, 2.6))

    Ss = S_VALUES
    fprs = [per_s[S][0] for S in Ss]
    mean_ns = [per_s[S][2] for S in Ss]
    hoeff_perS = [hoeffding_bound_perS(S, int(round(n))) for S, n in zip(Ss, mean_ns)]
    hoeff_const = hoeffding_bound_constant()

    # Empirical FPR (orange, dashed + markers)
    ax.plot(
        Ss, fprs,
        marker="o", markersize=5,
        linestyle="--", linewidth=1.2,
        color="#e07b00",
        label="Empirical FPR",
        zorder=4,
    )

    # Per-S Hoeffding bound (blue, solid)
    ax.plot(
        Ss, hoeff_perS,
        marker="s", markersize=4,
        linestyle="-", linewidth=1.0,
        color="#1f78b4",
        label="Hoeffding bound",
        zorder=3,
    )

    # alpha = 1% target (red dotted horizontal)
    ax.axhline(
        ALPHA_PCT,
        color="#c0392b",
        linestyle=":",
        linewidth=1.0,
        label=r"$\alpha = 1\%$ target",
        zorder=2,
    )

    ax.set_xlabel("State count $S$")
    ax.set_ylabel("Empirical FPR (%)")
    ax.set_title("Empirical FPR vs analytic bound")

    ax.set_xticks(Ss)
    ax.set_xticklabels([str(S) for S in Ss])

    ymax = max(max(fprs) if fprs else 0.0, max(hoeff_perS) if hoeff_perS else 0.0, ALPHA_PCT) * 1.15
    ymax = max(ymax, 8.0)
    ax.set_ylim(0, ymax)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)

    ax.legend(
        loc="upper left",
        frameon=False,
        handlelength=1.5,
        handletextpad=0.4,
        borderaxespad=0.2,
    )

    fig.tight_layout(pad=0.4)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    per_s: dict[int, tuple[float, int, float]] = {}
    print(f"Pooling z_nwm > {Z_ALPHA} across {len(MODELS)} LLMs per S ...")
    print(f"{'S':>3}  {'n':>5}  {'mean_tok':>9}  {'FPR_emp(%)':>11}  "
          f"{'Hoeff_perS(%)':>14}  {'Hoeff_const(%)':>15}")
    h_const = hoeffding_bound_constant()
    for S in S_VALUES:
        fpr, n, mean_n = empirical_fpr(S)
        per_s[S] = (fpr, n, mean_n)
        h_perS = hoeffding_bound_perS(S, int(round(mean_n)))
        print(f"{S:>3}  {n:>5}  {mean_n:>9.1f}  {fpr:>10.2f}   "
              f"{h_perS:>13.4f}   {h_const:>14.4f}")

    render(per_s)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
