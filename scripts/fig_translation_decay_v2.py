#!/usr/bin/env python3
"""Translation-robustness decay plot for the MCLW ICML paper.

For every (model, pivot) record in exp3, plot:
    x = delta_eff (effective edit rate under back-translation pivot)
    y = z_post / z_clean

Overlay theoretical worst-case bound y = (1 - x)^2.
Color by pivot (FR/DE/RU/ZH); marker by model (Llama=o, Mistral=s, Qwen=^).
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
    "llama-3-1-8b-instruct": {"label": "Llama-3.1-8B", "marker": "o"},
    "mistral-7b-instruct-v0-3": {"label": "Mistral-7B", "marker": "s"},
    "qwen2-5-7b-instruct": {"label": "Qwen2.5-7B", "marker": "^"},
}

# Distinct, paper-friendly pivot colors (colorblind-safe-ish).
PIVOTS = {
    "fr": {"label": "FR", "color": "#1f77b4"},
    "de": {"label": "DE", "color": "#d62728"},
    "ru": {"label": "RU", "color": "#2ca02c"},
    "zh": {"label": "ZH", "color": "#9467bd"},
}


# ---------------------------------------------------------------------------
# Matplotlib styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
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
    """Return dict pivot -> dict model -> (xs, ys)."""
    points = {p: {m: ([], []) for m in MODELS} for p in PIVOTS}
    raw = {p: [] for p in PIVOTS}  # for stats

    for model in MODELS:
        recs = load_records(model)
        for rec in recs:
            zc = rec.get("z_clean")
            if zc is None or zc <= 0:
                # avoid division blow-ups; skip degenerate clean values
                continue
            for pivot in PIVOTS:
                d = rec.get(f"delta_eff_{pivot}")
                zp = rec.get(f"z_{pivot}")
                if d is None or zp is None:
                    continue
                ratio = zp / zc
                points[pivot][model][0].append(d)
                points[pivot][model][1].append(ratio)
                raw[pivot].append((d, ratio))
    return points, raw


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(points, out_path: Path):
    fig, ax = plt.subplots(figsize=(3.0, 2.6))

    # Theoretical diagonal: y = (1 - x)^2 (worst-case bound from M_eff = (1 - delta)^2)
    xs = np.linspace(0, 1, 200)
    ax.plot(xs, (1 - xs) ** 2, color="black", linestyle="--",
            linewidth=1.0, label=r"$(1-\delta)^2$ bound", zorder=1)

    # Scatter
    for pivot, pmeta in PIVOTS.items():
        for model, mmeta in MODELS.items():
            xs_p, ys_p = points[pivot][model]
            if not xs_p:
                continue
            ax.scatter(
                xs_p, ys_p,
                s=10,
                marker=mmeta["marker"],
                facecolors="none",
                edgecolors=pmeta["color"],
                linewidths=0.6,
                alpha=0.75,
                zorder=2,
            )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.2)
    ax.set_xlabel(r"effective edit rate $\delta_{\mathrm{eff}}$")
    ax.set_ylabel(r"$z_{\mathrm{post}}\,/\,z_{\mathrm{clean}}$")

    ax.axhline(1.0, color="grey", linewidth=0.4, linestyle=":", zorder=0)
    ax.grid(True, linewidth=0.3, alpha=0.4)

    # --- Legends -----------------------------------------------------------
    # Pivot color legend (inset, top-right)
    pivot_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor="none",
                   markeredgecolor=meta["color"],
                   markeredgewidth=0.8, markersize=4,
                   label=meta["label"])
        for meta in PIVOTS.values()
    ]
    leg1 = ax.legend(handles=pivot_handles, title="pivot",
                     loc="upper right", frameon=True, framealpha=0.9,
                     handletextpad=0.3, labelspacing=0.25,
                     borderpad=0.3, title_fontsize=7)
    leg1.get_frame().set_linewidth(0.4)
    ax.add_artist(leg1)

    # Model marker legend (inset, lower-left)
    model_handles = [
        plt.Line2D([0], [0], marker=meta["marker"], linestyle="",
                   markerfacecolor="none",
                   markeredgecolor="black",
                   markeredgewidth=0.6, markersize=4,
                   label=meta["label"])
        for meta in MODELS.values()
    ]
    # Bound line entry
    model_handles.append(plt.Line2D([0], [0], color="black",
                                    linestyle="--", linewidth=1.0,
                                    label=r"$(1-\delta)^2$"))
    leg2 = ax.legend(handles=model_handles, loc="lower left",
                     frameon=True, framealpha=0.9,
                     handletextpad=0.3, labelspacing=0.25,
                     borderpad=0.3)
    leg2.get_frame().set_linewidth(0.4)

    fig.tight_layout(pad=0.4)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    points, raw = collect_points()
    make_plot(points, OUT_PATH)
    report_stats(raw)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
