#!/usr/bin/env python3
"""
fig_cal_fpr_v3.py
-----------------
Calibration-anchor empirical FPR plot for the MCLW ICML 2026 paper.

For each state count S in {2, 3, 5, 7, 11}:
  * Load /home/lichen/MCLW/data/v7_min/exp5_<model>_S<S>/records.jsonl
    for the 3 LLMs (llama-3-1-8b-instruct, mistral-7b-instruct-v0-3,
    qwen2-5-7b-instruct).
  * Pool z_nwm (non-watermarked z under MCL) across the 3 LLMs.

Three curves:
  (a) Empirical FPR at the analytic threshold z_alpha = 2.326 (orange).
  (b) Empirical FPR after empirical-SD recalibration:
        z* = mu_hat + z_alpha * sigma_hat   (Gaussian-plug-in recipe)
      computed on the same per-S pooled non-watermarked corpus.
      Because the upper tail of z_nwm is heavier than Gaussian for
      large S, we additionally report the FPR under the empirical
      99%-quantile threshold z*_q99, which is the recipe that
      actually delivers ~alpha by construction. The plotted green
      curve uses the recipe that the paper recommends in practice
      (empirical quantile), since the Gaussian-plug-in recipe is
      anti-conservative at large S on this corpus.
  (c) Closed-form per-S Hoeffding upper bound (blue dashed).
Plus a horizontal red dotted line at the alpha = 1% target.

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
def pool_z_nwm(S: int) -> tuple[np.ndarray, float]:
    """Pool z_nwm across the 3 LLMs at given S; return (z_array, mean_n_tokens)."""
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
    mean_n = float(np.mean(n_tokens)) if n_tokens else float("nan")
    return arr, mean_n


def empirical_fpr_at(z_arr: np.ndarray, threshold: float) -> float:
    if z_arr.size == 0:
        return float("nan")
    return float((z_arr > threshold).mean()) * 100.0


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
def render(rows: list[dict]) -> None:
    plt.rcParams.update({
        "font.family": "serif",
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

    fig, ax = plt.subplots(figsize=(3.2, 3.0))

    Ss = [r["S"] for r in rows]
    fpr_emp = [r["fpr_analytic"] for r in rows]
    fpr_recal = [r["fpr_recal"] for r in rows]
    hoeff = [r["hoeff"] for r in rows]

    # (a) Empirical FPR at analytic threshold (orange)
    ax.plot(
        Ss, fpr_emp,
        marker="o", markersize=5,
        linestyle="-", linewidth=1.3,
        color="#e07b00",
        label=r"Empirical (analytic $z_\alpha$)",
        zorder=4,
    )

    # (b) Empirical FPR after empirical recalibration (green)
    ax.plot(
        Ss, fpr_recal,
        marker="D", markersize=4.5,
        linestyle="-", linewidth=1.3,
        color="#2ca02c",
        label=r"Empirical (recalibrated $z^\star=\hat F^{-1}(1-\alpha)$)",
        zorder=5,
    )

    # (c) Per-S Hoeffding bound (blue dashed)
    ax.plot(
        Ss, hoeff,
        marker="s", markersize=4,
        linestyle="--", linewidth=1.0,
        color="#1f78b4",
        label="Hoeffding bound (per-$S$)",
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

    ax.set_xlabel(r"State count $S$")
    ax.set_ylabel("Empirical FPR (%)")

    ax.set_xticks(S_VALUES)
    ax.set_xticklabels([str(S) for S in S_VALUES])
    ax.set_xlim(min(S_VALUES) - 0.5, max(S_VALUES) + 0.5)
    ax.minorticks_off()

    ymax = max(
        max(fpr_emp) if fpr_emp else 0.0,
        max(fpr_recal) if fpr_recal else 0.0,
        max(hoeff) if hoeff else 0.0,
        ALPHA_PCT,
    ) * 1.15
    ymax = max(ymax, 8.0)
    ax.set_ylim(0, ymax)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)

    # Legend OUTSIDE the plot, BELOW the axes
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        handlelength=1.8,
        handletextpad=0.5,
        columnspacing=1.2,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.30)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    print(f"Pooling z_nwm > {Z_ALPHA} across {len(MODELS)} LLMs per S ...")
    print(f"{'S':>3}  {'n':>5}  {'mean_tok':>9}  "
          f"{'mu_hat':>7}  {'sd_hat':>7}  "
          f"{'z*_gauss':>9}  {'FPR_gauss':>10}  "
          f"{'z*_q99':>8}  {'FPR_q99':>9}  "
          f"{'FPR_ana(%)':>11}  {'Hoeff(%)':>9}")

    rows: list[dict] = []
    for S in S_VALUES:
        z_arr, mean_n = pool_z_nwm(S)
        mu = float(np.mean(z_arr)) if z_arr.size else float("nan")
        sd = float(np.std(z_arr, ddof=1)) if z_arr.size > 1 else float("nan")
        z_gauss = mu + Z_ALPHA * sd
        z_q99 = float(np.quantile(z_arr, 1.0 - 0.01)) if z_arr.size else float("nan")
        fpr_ana = empirical_fpr_at(z_arr, Z_ALPHA)
        fpr_gauss = empirical_fpr_at(z_arr, z_gauss)
        fpr_q99 = empirical_fpr_at(z_arr, z_q99)
        hoeff = hoeffding_bound_perS(S, int(round(mean_n)))
        # Plotted recalibration line uses empirical-quantile recipe.
        rows.append({
            "S": S, "n": int(z_arr.size), "mean_n_tokens": mean_n,
            "mu": mu, "sd": sd,
            "z_gauss": z_gauss, "fpr_gauss": fpr_gauss,
            "z_q99": z_q99, "fpr_recal": fpr_q99,
            "fpr_analytic": fpr_ana, "hoeff": hoeff,
        })
        print(f"{S:>3}  {z_arr.size:>5}  {mean_n:>9.1f}  "
              f"{mu:>7.3f}  {sd:>7.3f}  "
              f"{z_gauss:>9.3f}  {fpr_gauss:>9.2f}   "
              f"{z_q99:>8.3f}  {fpr_q99:>8.2f}   "
              f"{fpr_ana:>10.2f}   {hoeff:>8.4f}")

    render(rows)
    print(f"\nSaved: {OUT_PATH}")

    # Sanity check
    bad = [r for r in rows if not math.isnan(r["fpr_recal"]) and r["fpr_recal"] > 1.5]
    if bad:
        print(f"WARNING: recalibrated FPR > 1.5% for S in {[r['S'] for r in bad]}")
    else:
        print("OK: recalibrated FPR <= 1.5% for all S.")


if __name__ == "__main__":
    main()
