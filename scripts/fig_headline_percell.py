#!/usr/bin/env python3
"""
fig_headline_percell.py
-----------------------
Per-(model, domain) head-to-head TPR view for the MCLW ICML 2026 paper.

Design: side-by-side grouped horizontal bars. 12 (model, domain) rows split
across two side-by-side panels (Clean | ZH back-translation). Each row shows
three thin horizontal bars (ChainMark / KGW / SWEET) whose length encodes
TPR @ z>2.326. ChainMark uses a saturated deep blue; KGW/SWEET use muted
warm greys. The ZH panel is the discriminator: ChainMark bars stretch far
to the right while KGW/SWEET collapse near zero, making MCL's robustness
visually unmistakable in a single glance.

Output: /home/lichen/MCLW/icml2026/pictures/fig_headline_percell.pdf
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# ---------------------------------------------------------------- config
DATA_ROOT = Path("/home/lichen/MCLW/data/v7_min")
OUT_PATH = Path("/home/lichen/MCLW/icml2026/pictures/fig_headline_percell.pdf")

MODELS = [
    ("llama-3-1-8b-instruct", "Llama-3.1-8B"),
    ("qwen2-5-7b-instruct",   "Qwen-2.5-7B"),
    ("mistral-7b-instruct-v0-3", "Mistral-7B"),
]
DOMAINS = ["code", "factual", "wiki", "writing"]
DOMAIN_LABELS = {"code": "code", "factual": "factual",
                 "wiki": "wiki", "writing": "writing"}
METHODS = ["mcl", "kgw", "sweet"]
METHOD_LABELS = {"mcl": "ChainMark", "kgw": "KGW", "sweet": "SWEET"}
METHOD_COLORS = {
    "mcl":   "#1f6ad1",   # saturated deep blue (ChainMark)
    "kgw":   "#9a8a7a",   # muted tan/grey
    "sweet": "#c9b89a",   # lighter muted tan
}
ATTACKS = [("z_clean", "Clean"), ("z_zh", "ZH back-translation")]
Z_THRESHOLD = 2.326

# ---------------------------------------------------------------- io
def load_records(model: str, method: str) -> list[dict]:
    path = DATA_ROOT / f"exp2_{model}_{method}" / "records.jsonl"
    with path.open("r") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def tpr(records: list[dict], domain: str, attack_key: str) -> float:
    zs = []
    for r in records:
        if r.get("domain") != domain:
            continue
        v = r.get(attack_key)
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        zs.append(v)
    if not zs:
        return float("nan")
    return float((np.asarray(zs) > Z_THRESHOLD).mean()) * 100.0


def build_matrix() -> tuple[np.ndarray, list[tuple[str, str]]]:
    """Returns (M, row_meta) where M is (12, 6) and row_meta is the list of
    (model_label, domain) tuples for the 12 rows.
    Column order: clean[mcl,kgw,sweet], zh[mcl,kgw,sweet]."""
    rows: list[list[float]] = []
    row_meta: list[tuple[str, str]] = []
    cache: dict[tuple[str, str], list[dict]] = {}
    for model_id, model_label in MODELS:
        for method in METHODS:
            cache[(model_id, method)] = load_records(model_id, method)
        for domain in DOMAINS:
            row = []
            for attack_key, _ in ATTACKS:
                for method in METHODS:
                    row.append(tpr(cache[(model_id, method)], domain, attack_key))
            rows.append(row)
            row_meta.append((model_label, domain))
    return np.asarray(rows), row_meta


# ---------------------------------------------------------------- plot
def render(M: np.ndarray, row_meta: list[tuple[str, str]]) -> None:
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
        "axes.linewidth": 0.5,
    })

    n_rows = M.shape[0]  # 12

    fig = plt.figure(figsize=(3.25, 4.0))
    # Two side-by-side panels: Clean and ZH back-translation.
    gs = fig.add_gridspec(
        nrows=1, ncols=2,
        wspace=0.10,
        left=0.30, right=0.985,
        top=0.905, bottom=0.10,
    )
    ax_clean = fig.add_subplot(gs[0, 0])
    ax_zh    = fig.add_subplot(gs[0, 1], sharey=ax_clean)

    # Vertical position: each row of "model/domain" occupies one unit; within
    # a row, three sub-bars stacked.
    y_centres = np.arange(n_rows)         # 0..11, top row drawn first via invert
    bar_h = 0.24
    offsets = np.array([+bar_h, 0.0, -bar_h])  # ChainMark on top within group

    for ax, (col0, panel_title) in zip(
        (ax_clean, ax_zh),
        ((0, "Clean"), (3, "ZH back-translation")),
    ):
        for k, method in enumerate(METHODS):
            vals = M[:, col0 + k]
            ax.barh(
                y_centres + offsets[k],
                vals,
                height=bar_h,
                color=METHOD_COLORS[method],
                edgecolor="none",
                linewidth=0,
                zorder=2,
            )
            # Numeric label at end of bar (small, only if bar not tiny).
            for y, v in zip(y_centres + offsets[k], vals):
                if math.isnan(v):
                    continue
                # Place text just to the right of the bar tip; if bar is
                # close to the right edge, place text inside the bar end.
                if v >= 78:
                    tx, ha, color = v - 2, "right", "white"
                else:
                    tx, ha, color = v + 2, "left", "#222222"
                ax.text(
                    tx, y, f"{int(round(v))}",
                    ha=ha, va="center",
                    fontsize=5.6, color=color, zorder=3,
                )

        ax.set_xlim(0, 105)
        ax.set_ylim(-0.6, n_rows - 0.4)
        ax.invert_yaxis()
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=6.5)
        ax.tick_params(axis="x", which="major", length=2, width=0.4, pad=1.5)
        ax.tick_params(axis="y", which="major", length=0)
        # Faint vertical gridlines as reference rails.
        for xv in (25, 50, 75, 100):
            ax.axvline(xv, color="#e8e8e8", linewidth=0.4, zorder=1)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.set_title(panel_title, fontsize=7.8, pad=3)
        # Horizontal separators between models (every 4 rows)
        for k in (3.5, 7.5):
            ax.axhline(y=k, color="#bbbbbb", linewidth=0.4,
                       linestyle=":", zorder=1)

    # Y-axis labels on the LEFT panel only: "Model / domain"
    row_labels = [f"{m} / {d}" for (m, d) in row_meta]
    ax_clean.set_yticks(y_centres)
    ax_clean.set_yticklabels(row_labels, fontsize=6.6)
    # Hide tick labels on the ZH (right) panel y-axis.
    plt.setp(ax_zh.get_yticklabels(), visible=False)
    ax_zh.tick_params(axis="y", which="both", length=0)

    # Single shared x-axis label.
    fig.text(
        0.5 + (0.30 + 0.985)/2 - 0.5,  # mid of plotting area roughly
        0.025, "TPR (%) at $z>2.326$",
        ha="center", va="bottom", fontsize=7.5,
    )

    # Compact legend across the top of the figure.
    legend_handles = [
        Patch(facecolor=METHOD_COLORS["mcl"],   edgecolor="none",
              label="ChainMark"),
        Patch(facecolor=METHOD_COLORS["kgw"],   edgecolor="none",
              label="KGW"),
        Patch(facecolor=METHOD_COLORS["sweet"], edgecolor="none",
              label="SWEET"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5 + (0.30 + 0.985)/2 - 0.5, 0.985),
        ncol=3, frameon=False, fontsize=7,
        handlelength=1.1, handleheight=0.9, handletextpad=0.4,
        columnspacing=1.2,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    M, row_meta = build_matrix()
    print("Matrix (rows = model/domain; cols = clean[mcl,kgw,sweet] | zh[mcl,kgw,sweet]):")
    header = ["clean-mcl", "clean-kgw", "clean-sweet", "zh-mcl", "zh-kgw", "zh-sweet"]
    print(f"{'row':<28}  " + "  ".join(f"{h:>10}" for h in header))
    for (mlabel, dom), row in zip(row_meta, M):
        label = f"{mlabel} / {dom}"
        print(f"{label:<28}  " + "  ".join(f"{v:>10.1f}" for v in row))
    render(M, row_meta)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
