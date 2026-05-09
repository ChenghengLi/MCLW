#!/usr/bin/env python3
"""Translation-robustness decay plot (v3) for the MCLW ICML paper.

For every (model, pivot) record in exp3, plot:
    x = delta_eff (effective edit rate under back-translation pivot)
    y = z_post / z_clean    (clipped to 1.2; outliers shown as upward arrows)

Overlay theoretical bound y = (1 - delta_eff)^2.
Color by pivot (FR/DE/RU/ZH); marker by model (Llama=o, Mistral=s, Qwen=^).
Combined two-row legend below the figure.
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

MODELS = {
    "llama-3-1-8b-instruct":     {"label": "Llama",   "marker": "o"},
    "mistral-7b-instruct-v0-3":  {"label": "Mistral", "marker": "s"},
    "qwen2-5-7b-instruct":       {"label": "Qwen",    "marker": "^"},
}

# Distinct, paper-friendly pivot colors per spec: FR=blue, DE=green, RU=orange, ZH=red.
PIVOTS = {
    "fr": {"label": "FR", "color": "#1f77b4"},
    "de": {"label": "DE", "color": "#2ca02c"},
    "ru": {"label": "RU", "color": "#ff7f0e"},
    "zh": {"label": "ZH", "color": "#d62728"},
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
    """Return dict pivot -> dict model -> (xs, ys, clipped_mask)."""
    points = {p: {m: ([], [], []) for m in MODELS} for p in PIVOTS}
    raw = {p: [] for p in PIVOTS}  # for stats (uses unclipped ratios)

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
                # clip negative ratios to 0 for plotting
                ratio_plot = max(ratio, 0.0)
                clipped = ratio_plot > Y_CLIP
                ratio_plot = min(ratio_plot, Y_CLIP)
                points[pivot][model][0].append(d)
                points[pivot][model][1].append(ratio_plot)
                points[pivot][model][2].append(clipped)
                raw[pivot].append((d, ratio))
    return points, raw


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(points, out_path: Path):
    fig, ax = plt.subplots(figsize=(3.2, 3.0))

    # Theory diagonal y = (1 - x)^2
    xs = np.linspace(0, 1, 400)
    ax.plot(xs, (1 - xs) ** 2, color="black", linestyle="--",
            linewidth=1.0, zorder=1,
            label=r"Theory: $(1-\delta_{\mathrm{eff}})^2$")

    # Scatter — color by pivot, marker by model
    for pivot, pmeta in PIVOTS.items():
        for model, mmeta in MODELS.items():
            xs_p, ys_p, clipped = points[pivot][model]
            if not xs_p:
                continue
            xs_arr = np.asarray(xs_p)
            ys_arr = np.asarray(ys_p)
            cl_arr = np.asarray(clipped, dtype=bool)

            # Normal points
            ax.scatter(
                xs_arr[~cl_arr], ys_arr[~cl_arr],
                s=14,
                marker=mmeta["marker"],
                facecolors=pmeta["color"],
                edgecolors=pmeta["color"],
                linewidths=0.4,
                alpha=0.5,
                zorder=2,
            )
            # Clipped (outlier) points: arrow up at y=1.2
            if cl_arr.any():
                ax.scatter(
                    xs_arr[cl_arr], np.full(cl_arr.sum(), Y_CLIP),
                    s=22,
                    marker=r"$\uparrow$",
                    color=pmeta["color"],
                    alpha=0.7,
                    linewidths=0.4,
                    zorder=3,
                )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, Y_CLIP)
    ax.set_xlabel(r"$\delta_{\mathrm{eff}}$ (effective edit rate)")
    ax.set_ylabel(r"$z_{\mathrm{post}}\,/\,z_{\mathrm{clean}}$")

    ax.axhline(1.0, color="grey", linewidth=0.4, linestyle=":", zorder=0)
    ax.grid(True, linewidth=0.3, alpha=0.35)

    # Spines: remove top/right
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ----- Single combined legend, outside BELOW the axes ------------------
    pivot_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor=meta["color"],
                   markeredgecolor=meta["color"],
                   markersize=4.5, alpha=0.7,
                   label=f"pivot: {meta['label']}")
        for meta in PIVOTS.values()
    ]
    model_handles = [
        plt.Line2D([0], [0], marker=meta["marker"], linestyle="",
                   markerfacecolor="0.35",
                   markeredgecolor="0.35",
                   markersize=4.5, alpha=0.8,
                   label=f"model: {meta['label']}")
        for meta in MODELS.values()
    ]
    theory_handle = plt.Line2D([0], [0], color="black", linestyle="--",
                               linewidth=1.0,
                               label=r"Theory: $(1-\delta_{\mathrm{eff}})^2$")

    ax.legend(
        handles=pivot_handles + model_handles + [theory_handle],
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=4, frameon=False,
        handlelength=1.1,
        handletextpad=0.4, columnspacing=1.0,
        borderaxespad=0.0,
        fontsize=7,
    )

    fig.tight_layout(pad=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def report_stats(raw):
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


def main():
    points, raw = collect_points()
    make_plot(points, OUT_PATH)
    report_stats(raw)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
