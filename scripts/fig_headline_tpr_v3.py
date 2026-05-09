#!/usr/bin/env python3
"""
fig_headline_tpr_v3.py
----------------------
Headline TPR comparison plot for the MCLW ICML 2026 paper.

Improvements over v2:
  * Legend positioned OUTSIDE the plot (below, centered) so bars don't get
    crowded.
  * Tight bbox on save.
  * MCL bars: deeper saturation, bolder edge, hatch-free, sits on top.
  * Numeric value labels above each bar formatted as "XX.X%".
  * Y-axis 0-105 with ticks every 25, very light dotted gridlines.
  * Clean serif typography, no top/right spines, fonttype=42.

Aggregates TPR @ z > 2.326 (one-sided alpha = 0.01) across:
  * 3 LLMs: llama-3-1-8b-instruct, mistral-7b-instruct-v0-3, qwen2-5-7b-instruct
  * domains pooled
  * 3 attack conditions: clean, random-substitution, ZH back-translation

Methods compared: MCL (ours, highlighted), KGW, SWEET.

Output: /home/lichen/MCLW/icml2026/pictures/fig_headline_tpr.pdf
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
OUT_PATH = Path("/home/lichen/MCLW/icml2026/pictures/fig_headline_tpr.pdf")

MODELS = ["llama-3-1-8b-instruct", "mistral-7b-instruct-v0-3", "qwen2-5-7b-instruct"]
METHODS = ["mcl", "kgw", "sweet"]
METHOD_LABELS = {"mcl": "ChainMark (ours)", "kgw": "KGW", "sweet": "SWEET"}
ATTACKS = ["z_clean", "z_random", "z_zh"]
ATTACK_LABELS = ["Clean", "Random-sub", "ZH back-trans"]

Z_THRESHOLD = 2.326  # one-sided alpha = 0.01

# MCL: deep saturated blue (highlight). Baselines: muted but distinguishable.
METHOD_COLORS = {
    "mcl":   "#1b3a8a",   # deep saturated blue
    "kgw":   "#bdbdbd",   # neutral light gray
    "sweet": "#e0b07a",   # warm muted tan
}
METHOD_EDGE = {
    "mcl":   "#0a1d4d",   # very dark blue (bold edge)
    "kgw":   "#7a7a7a",
    "sweet": "#a07a4a",
}
METHOD_LW = {"mcl": 1.4, "kgw": 0.5, "sweet": 0.5}

# ---------------------------------------------------------------- io
def load_records(model: str, method: str) -> list[dict]:
    path = DATA_ROOT / f"exp2_{model}_{method}" / "records.jsonl"
    with path.open("r") as fh:
        return [json.loads(line) for line in fh if line.strip()]

# ---------------------------------------------------------------- aggregation
def tpr_for_method(method: str) -> dict[str, float]:
    pooled: list[dict] = []
    for m in MODELS:
        pooled.extend(load_records(m, method))
    domains = sorted({r.get("domain") for r in pooled})
    out: dict = {}
    for attack in ATTACKS:
        zs = []
        for r in pooled:
            v = r.get(attack)
            if v is None:
                continue
            if isinstance(v, float) and math.isnan(v):
                continue
            zs.append(v)
        zs = np.asarray(zs, dtype=float)
        tpr = float((zs > Z_THRESHOLD).mean()) * 100.0 if len(zs) else float("nan")
        out[attack] = tpr
    out["_n_records"] = len(pooled)
    out["_domains"] = domains
    return out

# ---------------------------------------------------------------- plot
def render(aggregates: dict[str, dict[str, float]]) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
    })

    fig, ax = plt.subplots(figsize=(3.0, 3.0))

    n_attacks = len(ATTACKS)
    n_methods = len(METHODS)
    bar_w = 0.26
    x = np.arange(n_attacks)

    for i, method in enumerate(METHODS):
        vals = [aggregates[method][a] for a in ATTACKS]
        offset = (i - (n_methods - 1) / 2) * bar_w
        is_mcl = method == "mcl"
        bars = ax.bar(
            x + offset,
            vals,
            width=bar_w,
            color=METHOD_COLORS[method],
            edgecolor=METHOD_EDGE[method],
            linewidth=METHOD_LW[method],
            label=METHOD_LABELS[method],
            zorder=4 if is_mcl else 3,
        )
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + 1.8,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=5.6,
                rotation=90,
                fontweight=("bold" if is_mcl else "normal"),
                color=("#0a1d4d" if is_mcl else "#333333"),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ATTACK_LABELS)
    ax.set_ylabel("TPR (%) @ $z>2.326$")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    # Legend OUTSIDE: below the axes, horizontal, centered.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )

    fig.tight_layout(pad=0.3)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

# ---------------------------------------------------------------- main
def main() -> None:
    aggregates: dict[str, dict[str, float]] = {}
    print(f"Aggregating TPR @ z>{Z_THRESHOLD} across {len(MODELS)} LLMs ...")
    for method in METHODS:
        aggregates[method] = tpr_for_method(method)

    print(f"{'method':<6}  {'n':>5}  domains       {'clean':>7}  {'random':>7}  {'zh':>7}")
    for method in METHODS:
        a = aggregates[method]
        print(
            f"{method:<6}  {a['_n_records']:>5}  "
            f"{','.join(a['_domains']):<13}  "
            f"{a['z_clean']:>7.1f}  {a['z_random']:>7.1f}  {a['z_zh']:>7.1f}"
        )

    render(aggregates)
    print(f"\nSaved: {OUT_PATH}")

if __name__ == "__main__":
    main()
