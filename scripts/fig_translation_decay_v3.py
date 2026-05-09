#!/usr/bin/env python3
"""Translation-robustness decay plot (v3, redesigned) for the MCLW ICML paper.

For every (model, pivot) record in exp3, plot:
    x = delta_eff (effective edit rate under back-translation pivot)
    y = z_post / z_clean    (clipped to 1.2; outliers shown as upward arrows)

Layered visualization (back -> front):
  * Theoretical bound y = (1 - delta_eff)^2 (dashed black)
  * Per-model OLS regression line on the unclipped (delta_eff, ratio)
    pairs: 3 solid colored lines (Llama / Mistral / Qwen) — these are
    the headline new design element so the reader sees model-level slope
    and intercept, not just an undifferentiated cloud.
  * Scatter cloud, faint, color = model (matches regression line),
    marker = pivot (FR=o, DE=s, RU=^, ZH=D). Shape now encodes pivot,
    color encodes model (was the reverse in the previous version) so
    the per-model story dominates.

n=100 records per model x 4 pivots x 3 models = 1200 plotted points.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/home/lichen/MCLW/data/v7_min")
OUT_PATH = Path("/home/lichen/MCLW/icml2026/pictures/fig_translation_decay.pdf")

# Color = MODEL (deep blue / orange / green per design spec).
MODELS = {
    "llama-3-1-8b-instruct":     {"label": "Llama",   "color": "#1f4e9d"},
    "mistral-7b-instruct-v0-3":  {"label": "Mistral", "color": "#e07b00"},
    "qwen2-5-7b-instruct":       {"label": "Qwen",    "color": "#2ca02c"},
}

# Marker = PIVOT.
PIVOTS = {
    "fr": {"label": "FR", "marker": "o"},
    "de": {"label": "DE", "marker": "s"},
    "ru": {"label": "RU", "marker": "^"},
    "zh": {"label": "ZH", "marker": "D"},
}

Y_CLIP = 1.2

# ---------------------------------------------------------------------------
# Matplotlib styling
# ---------------------------------------------------------------------------
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
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
})


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_records(model: str) -> list[dict]:
    path = DATA_ROOT / f"exp3_{model}" / "records.jsonl"
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def collect_points():
    """Return:
        scatter[pivot][model] = (xs_plot, ys_plot, clipped_mask)
        per_model[model] = (xs_unclipped, ys_unclipped) for OLS fit
        raw[pivot] = list of (delta, ratio_unclipped) for stats
    """
    scatter = {p: {m: ([], [], []) for m in MODELS} for p in PIVOTS}
    per_model = {m: ([], []) for m in MODELS}
    raw = {p: [] for p in PIVOTS}

    for model in MODELS:
        recs = load_records(model)
        for rec in recs:
            zc = rec.get("z_clean")
            if zc is None or zc <= 0:
                continue
            for pivot in PIVOTS:
                d = rec.get(f"delta_eff_{pivot}")
                zp = rec.get(f"z_{pivot}")
                if d is None or zp is None:
                    continue
                ratio = zp / zc
                ratio_clip = max(ratio, 0.0)
                clipped = ratio_clip > Y_CLIP
                ratio_clip = min(ratio_clip, Y_CLIP)
                scatter[pivot][model][0].append(d)
                scatter[pivot][model][1].append(ratio_clip)
                scatter[pivot][model][2].append(clipped)
                # For OLS we use the unclipped ratio but cap at 0 from below
                # to avoid pathological negative leverage.
                per_model[model][0].append(d)
                per_model[model][1].append(max(ratio, 0.0))
                raw[pivot].append((d, ratio))
    return scatter, per_model, raw


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(scatter, per_model, out_path: Path):
    fig, ax = plt.subplots(figsize=(3.2, 3.0))

    # --- Theory diagonal y = (1 - x)^2 -----------------------------------
    xs_th = np.linspace(0, 1, 400)
    ax.plot(xs_th, (1 - xs_th) ** 2,
            color="black", linestyle="--",
            linewidth=1.0, zorder=1,
            label=r"Theory $(1-\delta_{\mathrm{eff}})^2$")

    # --- Faint scatter cloud, color=model, marker=pivot ------------------
    for model, mmeta in MODELS.items():
        for pivot, pmeta in PIVOTS.items():
            xs_p, ys_p, clipped = scatter[pivot][model]
            if not xs_p:
                continue
            xs_arr = np.asarray(xs_p)
            ys_arr = np.asarray(ys_p)
            cl_arr = np.asarray(clipped, dtype=bool)

            ax.scatter(
                xs_arr[~cl_arr], ys_arr[~cl_arr],
                s=11,
                marker=pmeta["marker"],
                facecolors=mmeta["color"],
                edgecolors=mmeta["color"],
                linewidths=0.3,
                alpha=0.32,
                zorder=2,
            )
            if cl_arr.any():
                ax.scatter(
                    xs_arr[cl_arr], np.full(cl_arr.sum(), Y_CLIP),
                    s=18,
                    marker=r"$\uparrow$",
                    color=mmeta["color"],
                    alpha=0.55,
                    linewidths=0.3,
                    zorder=2,
                )

    # --- Per-model OLS regression lines ----------------------------------
    # We fit y = a + b*x on unclipped (>=0) ratios over x in [0,1] and
    # draw within the data's observed delta_eff range so the line doesn't
    # extrapolate into empty regions.
    for model, mmeta in MODELS.items():
        xs_m = np.asarray(per_model[model][0], dtype=float)
        ys_m = np.asarray(per_model[model][1], dtype=float)
        if xs_m.size < 2:
            continue
        b, a = np.polyfit(xs_m, ys_m, deg=1)
        x_lo = max(0.0, float(xs_m.min()) - 0.02)
        x_hi = min(1.0, float(xs_m.max()) + 0.02)
        xs_line = np.linspace(x_lo, x_hi, 100)
        ys_line = a + b * xs_line
        # halo behind for visibility on busy cloud
        ax.plot(xs_line, ys_line, color="white", linewidth=2.4, zorder=3.4,
                solid_capstyle="round")
        ax.plot(xs_line, ys_line, color=mmeta["color"], linewidth=1.5,
                zorder=3.6, solid_capstyle="round",
                label=f"{mmeta['label']} fit")

    # --- Axes ------------------------------------------------------------
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, Y_CLIP)
    ax.set_xlabel(r"$\delta_{\mathrm{eff}}$ (effective edit rate)")
    ax.set_ylabel(r"$z_{\mathrm{post}}\,/\,z_{\mathrm{clean}}$")

    ax.axhline(1.0, color="grey", linewidth=0.4, linestyle=":", zorder=0)
    ax.grid(True, linewidth=0.3, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # --- Legend (BELOW axes, two rows) -----------------------------------
    # Row 1: theory + per-model fit lines (the new headline element)
    # Row 2: pivot marker key (so reader can decode marker shapes)
    theory_handle = plt.Line2D([0], [0], color="black", linestyle="--",
                               linewidth=1.0,
                               label=r"$(1-\delta_{\mathrm{eff}})^2$")
    model_handles = [
        plt.Line2D([0], [0], color=meta["color"], linewidth=1.5,
                   label=meta["label"])
        for meta in MODELS.values()
    ]
    pivot_handles = [
        plt.Line2D([0], [0], marker=meta["marker"], linestyle="",
                   markerfacecolor="0.4", markeredgecolor="0.4",
                   markersize=4.0, alpha=0.7,
                   label=meta["label"])
        for meta in PIVOTS.values()
    ]

    ax.legend(
        handles=[theory_handle] + model_handles + pivot_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=4, frameon=False,
        handlelength=1.5,
        handletextpad=0.4, columnspacing=0.9,
        borderaxespad=0.0,
        fontsize=6.6,
    )

    fig.subplots_adjust(left=0.18, right=0.98, top=0.95, bottom=0.30)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def report_stats(raw, per_model):
    print("=== Translation-robustness decay: aggregate stats ===")
    total_above = 0
    total_pts = 0
    for pivot, pts in raw.items():
        if not pts:
            print(f"  {pivot.upper()}: no points")
            continue
        ds = np.array([p[0] for p in pts])
        rs = np.array([p[1] for p in pts])
        bound = (1 - ds) ** 2
        above = int(np.sum(rs >= bound))
        n = len(pts)
        total_above += above
        total_pts += n
        print(f"  {pivot.upper()}: n={n:4d}  mean delta_eff={ds.mean():.3f}  "
              f"mean ratio={rs.mean():.3f}  above-diagonal={above}/{n} "
              f"({above / n * 100:.1f}%)")
    if total_pts:
        print(f"  ALL : n={total_pts:4d}  above-diagonal={total_above}/{total_pts} "
              f"({total_above / total_pts * 100:.1f}%)")
    print("--- per-model OLS fit (y = a + b*x) ---")
    for model, (xs, ys) in per_model.items():
        xs_a = np.asarray(xs); ys_a = np.asarray(ys)
        if xs_a.size < 2:
            continue
        b, a = np.polyfit(xs_a, ys_a, deg=1)
        print(f"  {model:<28s}  n={xs_a.size:4d}  intercept={a:+.3f}  slope={b:+.3f}")


def main():
    scatter, per_model, raw = collect_points()
    make_plot(scatter, per_model, OUT_PATH)
    report_stats(raw, per_model)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
