#!/usr/bin/env python3
"""
fig_headline_tpr_v3.py
----------------------
Headline TPR comparison plot for the MCLW ICML 2026 paper.

Design (v3, redesigned):
  * 3-method x 3-attack grouped bars; bar HEIGHT = mean TPR pooled across
    the 3 LLMs (llama-3-1-8b, mistral-7b, qwen2.5-7b).
  * Per-model spread shown as a thin black error bar (min->max of the
    three per-model TPRs) plus three small marker glyphs (one per model)
    overlaid on the bar — so the reader sees not just the aggregate but
    the individual Llama/Mistral/Qwen values for every method-attack cell.
  * Per-model markers: Llama=circle, Mistral=square, Qwen=triangle
    (consistent with the translation-decay panel).
  * MCL bars: deeper saturation, bolder edge, higher zorder so the
    highlighted method sits on top.
  * Legend BELOW the axes (method legend on row 1, model-marker legend
    on row 2). No top/right spines, light dotted y-grid only, fonttype=42.

Aggregates TPR @ z > 2.326 (one-sided alpha = 0.01) over 3 LLMs and 4
domains, separately per (method, attack). The error bar reports the
min/max of the three per-model TPRs for that (method, attack) cell.

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
MODEL_LABELS = {
    "llama-3-1-8b-instruct":    "Llama",
    "mistral-7b-instruct-v0-3": "Mistral",
    "qwen2-5-7b-instruct":      "Qwen",
}
MODEL_MARKERS = {
    "llama-3-1-8b-instruct":    "o",
    "mistral-7b-instruct-v0-3": "s",
    "qwen2-5-7b-instruct":      "^",
}

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
def _tpr(records: list[dict], attack: str) -> float:
    zs = []
    for r in records:
        v = r.get(attack)
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        zs.append(v)
    zs = np.asarray(zs, dtype=float)
    return float((zs > Z_THRESHOLD).mean()) * 100.0 if len(zs) else float("nan")


def aggregates_for_method(method: str) -> dict:
    """Return per-attack pooled mean and per-model TPRs for a method."""
    per_model_recs = {m: load_records(m, method) for m in MODELS}
    pooled_recs = sum(per_model_recs.values(), [])
    out = {"_n_pooled": len(pooled_recs), "_per_model_n": {}}
    for attack in ATTACKS:
        out[attack] = {
            "pooled": _tpr(pooled_recs, attack),
            "per_model": {m: _tpr(per_model_recs[m], attack) for m in MODELS},
        }
    for m in MODELS:
        out["_per_model_n"][m] = len(per_model_recs[m])
    return out


# ---------------------------------------------------------------- plot
def render(aggregates: dict[str, dict]) -> None:
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

    fig, ax = plt.subplots(figsize=(3.2, 3.0))

    n_attacks = len(ATTACKS)
    n_methods = len(METHODS)
    bar_w = 0.26
    x = np.arange(n_attacks)

    for i, method in enumerate(METHODS):
        offset = (i - (n_methods - 1) / 2) * bar_w
        is_mcl = method == "mcl"
        bar_centers = x + offset
        pooled_vals = [aggregates[method][a]["pooled"] for a in ATTACKS]

        # Bar = pooled mean across models
        ax.bar(
            bar_centers,
            pooled_vals,
            width=bar_w,
            color=METHOD_COLORS[method],
            edgecolor=METHOD_EDGE[method],
            linewidth=METHOD_LW[method],
            label=METHOD_LABELS[method],
            zorder=3 if is_mcl else 2,
        )

        # Per-model min/max error bar + per-model markers overlaid on each bar
        for bx, attack in zip(bar_centers, ATTACKS):
            per_model = aggregates[method][attack]["per_model"]
            vals = np.array([per_model[m] for m in MODELS], dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            vmin, vmax = float(vals.min()), float(vals.max())
            if vmax > vmin + 1e-9:
                ax.vlines(
                    bx, vmin, vmax,
                    color="black", linewidth=0.8,
                    zorder=5 if is_mcl else 4,
                )
                # caps
                cap_w = bar_w * 0.18
                ax.hlines([vmin, vmax], bx - cap_w, bx + cap_w,
                          color="black", linewidth=0.8,
                          zorder=5 if is_mcl else 4)
            # Per-model markers (subtle white edge so they show on dark MCL bar)
            for mname in MODELS:
                v = per_model[mname]
                if v is None or math.isnan(v):
                    continue
                ax.plot(
                    bx, v,
                    marker=MODEL_MARKERS[mname],
                    markersize=2.6,
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markeredgewidth=0.45,
                    linestyle="",
                    zorder=6 if is_mcl else 5,
                )

        # Numeric pooled-value label above each bar — placed above the
        # error-bar vmax (or the bar tip if there's no spread), with a
        # 2.5-unit gap so the integer doesn't collide with the cap.
        for bx, attack, v in zip(bar_centers, ATTACKS, pooled_vals):
            per_model = aggregates[method][attack]["per_model"]
            vals = np.array([per_model[m] for m in MODELS], dtype=float)
            vals = vals[~np.isnan(vals)]
            top = float(vals.max()) if vals.size else v
            ax.text(
                bx,
                top + 2.5,
                f"{v:.0f}",
                ha="center",
                va="bottom",
                fontsize=5.6,
                rotation=0,
                fontweight=("bold" if is_mcl else "normal"),
                color=("#0a1d4d" if is_mcl else "#444444"),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(ATTACK_LABELS)
    ax.set_ylabel("TPR (%) @ $z>2.326$")
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    # Two-row legend: methods on row 1, model-marker key on row 2.
    method_handles = [
        plt.Rectangle((0, 0), 1, 1,
                      facecolor=METHOD_COLORS[m],
                      edgecolor=METHOD_EDGE[m],
                      linewidth=METHOD_LW[m],
                      label=METHOD_LABELS[m])
        for m in METHODS
    ]
    model_handles = [
        plt.Line2D([0], [0],
                   marker=MODEL_MARKERS[m],
                   linestyle="",
                   markerfacecolor="white",
                   markeredgecolor="black",
                   markeredgewidth=0.45,
                   markersize=3.2,
                   label=MODEL_LABELS[m])
        for m in MODELS
    ]
    ax.legend(
        handles=method_handles + model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        handlelength=1.1,
        handletextpad=0.4,
        columnspacing=1.0,
        borderaxespad=0.0,
        fontsize=6.6,
    )

    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.30)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    aggregates: dict[str, dict] = {}
    print(f"Aggregating TPR @ z>{Z_THRESHOLD} across {len(MODELS)} LLMs ...")
    for method in METHODS:
        aggregates[method] = aggregates_for_method(method)

    print(f"{'method':<6}  {'n':>5}  "
          f"{'clean':>7}  {'random':>7}  {'zh':>7}  | per-model (L/M/Q)")
    for method in METHODS:
        a = aggregates[method]
        cl = a["z_clean"]; rn = a["z_random"]; zh = a["z_zh"]
        pm_str = " ".join(
            f"{att}=[{cl_pm['per_model']['llama-3-1-8b-instruct']:.1f}/"
            f"{cl_pm['per_model']['mistral-7b-instruct-v0-3']:.1f}/"
            f"{cl_pm['per_model']['qwen2-5-7b-instruct']:.1f}]"
            for att, cl_pm in [("clean", cl), ("rnd", rn), ("zh", zh)]
        )
        print(
            f"{method:<6}  {a['_n_pooled']:>5}  "
            f"{cl['pooled']:>7.1f}  {rn['pooled']:>7.1f}  {zh['pooled']:>7.1f}  | {pm_str}"
        )

    render(aggregates)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
