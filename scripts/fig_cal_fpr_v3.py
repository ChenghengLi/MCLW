#!/usr/bin/env python3
"""
fig_cal_fpr_v3.py
-----------------
Calibration-anchor empirical FPR plot for the MCLW ICML 2026 paper.

Redesign (v3): show per-model analytic FPR curves as separate lines so
the reader sees model-level variation in calibration drift, while the
recalibrated empirical-quantile curve and the Hoeffding bound stay
pooled across the 3 LLMs (those are corpus-level statistics).

For each state count S in {2, 3, 5, 7, 11}:
  * Load /home/lichen/MCLW/data/v7_min/exp5_<model>_S<S>/records.jsonl
    for the 3 LLMs.
  * Per-model analytic FPR = fraction of z_nwm > z_alpha=2.326.
  * Pooled recalibrated FPR uses the empirical 99%-quantile threshold
    on the pooled non-watermarked z's (recipe the paper recommends).
  * Closed-form per-S Hoeffding upper bound.
  * Horizontal alpha = 1% target.

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
MODEL_LABELS = {
    "llama-3-1-8b-instruct":    "Llama",
    "mistral-7b-instruct-v0-3": "Mistral",
    "qwen2-5-7b-instruct":      "Qwen",
}
# Color = model. Markers = model (consistent w/ headline + decay panels).
MODEL_COLORS = {
    "llama-3-1-8b-instruct":    "#1f4e9d",  # deep blue
    "mistral-7b-instruct-v0-3": "#e07b00",  # orange
    "qwen2-5-7b-instruct":      "#2ca02c",  # green
}
MODEL_MARKERS = {
    "llama-3-1-8b-instruct":    "o",
    "mistral-7b-instruct-v0-3": "s",
    "qwen2-5-7b-instruct":      "^",
}

S_VALUES = [2, 3, 5, 7, 11]

Z_ALPHA = 2.326    # one-sided alpha = 0.01
ALPHA_PCT = 1.0    # target FPR (%)
RHO = 0.5          # gating fraction used in exp5


# ---------------------------------------------------------------- io
def load_records(model: str, S: int) -> list[dict]:
    path = DATA_ROOT / f"exp5_{model}_S{S}" / "records.jsonl"
    with path.open("r") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------- aggregation
def per_model_z_nwm(model: str, S: int) -> tuple[np.ndarray, float]:
    zs: list[float] = []
    n_tokens: list[int] = []
    for r in load_records(model, S):
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
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
    })

    fig, ax = plt.subplots(figsize=(3.2, 3.0))

    Ss = [r["S"] for r in rows]
    fpr_pooled_analytic = [r["fpr_pooled_analytic"] for r in rows]
    fpr_recal = [r["fpr_recal"] for r in rows]
    hoeff = [r["hoeff"] for r in rows]

    # ---- (a) Pooled empirical FPR at analytic threshold ----------------
    ax.plot(
        Ss, fpr_pooled_analytic,
        marker="o", markersize=4.5,
        linestyle="-", linewidth=1.2,
        color="#e07b00",
        label=r"Empirical FPR @ $z_\alpha$",
        zorder=4,
    )

    # ---- (b) Pooled recalibrated empirical-quantile FPR ----------------
    ax.plot(
        Ss, fpr_recal,
        marker="s", markersize=4.5,
        linestyle="-", linewidth=1.4,
        color="#2ca02c",
        label=r"Recalibrated $z^\star=\hat F^{-1}(1-\alpha)$",
        zorder=5,
    )

    # ---- (c) Per-S Hoeffding bound (dashed) ----------------------------
    ax.plot(
        Ss, hoeff,
        linestyle="--", linewidth=1.0,
        color="#1f4e9d",
        marker="",
        label="Hoeffding bound",
        zorder=3,
    )

    # ---- alpha target line ---------------------------------------------
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

    # y range driven by the pooled empirical analytic value.
    all_y = list(fpr_pooled_analytic)
    all_y.extend(fpr_recal)
    all_y.extend(hoeff)
    all_y = [v for v in all_y if not math.isnan(v)]
    ymax = max(max(all_y) * 1.15 if all_y else 8.0, 8.0)
    ax.set_ylim(0, ymax)

    ax.yaxis.grid(True, linestyle=":", linewidth=0.5, color="#bbbbbb", zorder=0)
    ax.set_axisbelow(True)

    # Legend BELOW the axes.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
        fontsize=6.4,
    )

    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.30)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    print(f"Per-model analytic FPR @ z>{Z_ALPHA}; pooled recalibration via "
          f"empirical 99%-quantile.")
    print(f"{'S':>3}  {'L_n':>4} {'M_n':>4} {'Q_n':>4}  "
          f"{'L_FPR':>6} {'M_FPR':>6} {'Q_FPR':>6}  "
          f"{'pool_n':>6}  {'recal_FPR':>9}  {'Hoeff(%)':>9}")

    rows: list[dict] = []
    for S in S_VALUES:
        per_model_arr = {m: per_model_z_nwm(m, S) for m in MODELS}
        pooled = np.concatenate([arr for (arr, _) in per_model_arr.values()])
        per_model_analytic = {
            m: empirical_fpr_at(arr, Z_ALPHA)
            for m, (arr, _) in per_model_arr.items()
        }
        # Recalibration uses pooled empirical 99%-quantile.
        z_q99 = float(np.quantile(pooled, 1.0 - 0.01)) if pooled.size else float("nan")
        fpr_recal = empirical_fpr_at(pooled, z_q99)

        mean_n_pool = float(np.mean([
            mean_n for (_, mean_n) in per_model_arr.values()
            if not math.isnan(mean_n)
        ]))
        hoeff = hoeffding_bound_perS(S, int(round(mean_n_pool)))

        L_arr, _ = per_model_arr["llama-3-1-8b-instruct"]
        M_arr, _ = per_model_arr["mistral-7b-instruct-v0-3"]
        Q_arr, _ = per_model_arr["qwen2-5-7b-instruct"]
        # Pooled-FPR equivalent for the old aggregate row (kept for sanity)
        fpr_pooled_analytic = empirical_fpr_at(pooled, Z_ALPHA)

        rows.append({
            "S": S,
            "per_model_analytic": per_model_analytic,
            "fpr_pooled_analytic": fpr_pooled_analytic,
            "fpr_recal": fpr_recal,
            "hoeff": hoeff,
        })
        print(
            f"{S:>3}  {L_arr.size:>4} {M_arr.size:>4} {Q_arr.size:>4}  "
            f"{per_model_analytic['llama-3-1-8b-instruct']:>5.2f}% "
            f"{per_model_analytic['mistral-7b-instruct-v0-3']:>5.2f}% "
            f"{per_model_analytic['qwen2-5-7b-instruct']:>5.2f}%  "
            f"{pooled.size:>6}  {fpr_recal:>8.2f}%  {hoeff:>8.4f}  "
            f"(pooled analytic={fpr_pooled_analytic:.2f}%)"
        )

    render(rows)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
