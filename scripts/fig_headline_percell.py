#!/usr/bin/env python3
"""
fig_headline_percell.py
-----------------------
Per-(model, domain) head-to-head TPR heatmap for the MCLW ICML 2026 paper.

Replaces the 12-row table tab:headline-percell. 12 (model, domain) rows x
6 columns (3 methods x 2 attack conditions), with TPR @ z>2.326 in each cell.
Cells are colored on a single shared scale; per-row best is bolded so a
reader can scan for ChainMark dominance at a glance.

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- config
DATA_ROOT = Path("/home/lichen/MCLW/data/v7_min")
OUT_PATH = Path("/home/lichen/MCLW/icml2026/pictures/fig_headline_percell.pdf")

MODELS = [
    ("llama-3-1-8b-instruct", "Llama-3.1-8B"),
    ("qwen2-5-7b-instruct",   "Qwen-2.5-7B"),
    ("mistral-7b-instruct-v0-3", "Mistral-7B"),
]
DOMAINS = ["code", "factual", "wiki", "writing"]
METHODS = ["mcl", "kgw", "sweet"]
METHOD_LABELS = {"mcl": "ChainMark", "kgw": "KGW", "sweet": "SWEET"}
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


def build_matrix() -> tuple[np.ndarray, list[str]]:
    """Returns (M, row_labels) where M is (12, 6).
    Column order: clean[mcl,kgw,sweet], zh[mcl,kgw,sweet]."""
    rows: list[list[float]] = []
    row_labels: list[str] = []
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
            row_labels.append(f"{model_label} / {domain}")
    return np.asarray(rows), row_labels


# ---------------------------------------------------------------- plot
def render(M: np.ndarray, row_labels: list[str]) -> None:
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
    })

    n_rows, n_cols = M.shape  # 12, 6
    fig, ax = plt.subplots(figsize=(3.25, 4.0))

    # Sequential blue cmap; white -> deep blue used for headline figure.
    cmap = LinearSegmentedColormap.from_list(
        "cm_blues", ["#ffffff", "#cfd8ee", "#7a93cc", "#1b3a8a"]
    )

    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Per-row best column index (ChainMark in clean and zh blocks individually).
    clean_best = M[:, 0:3].argmax(axis=1)              # 0..2
    zh_best    = M[:, 3:6].argmax(axis=1) + 3          # 3..5

    # Cell text
    for i in range(n_rows):
        for j in range(n_cols):
            v = M[i, j]
            # Use white text on dark cells, dark text on light cells.
            txt_color = "white" if v >= 60.0 else "#222222"
            is_best = (j == clean_best[i]) or (j == zh_best[i])
            ax.text(
                j, i,
                f"{int(round(v))}",
                ha="center", va="center",
                fontsize=6.6,
                color=txt_color,
                fontweight=("bold" if is_best else "normal"),
            )

    # Column headers: two-row header (Attack block + Method)
    method_labels = [METHOD_LABELS[m] for m in METHODS] * 2
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(method_labels, rotation=40, ha="right", fontsize=6.8)

    # Top-axis labels for the two attack blocks
    ax_top = ax.secondary_xaxis("top")
    ax_top.set_xticks([1, 4])
    ax_top.set_xticklabels(["Clean", "ZH back-trans."], fontsize=7.5)
    ax_top.tick_params(axis="x", which="both", length=0, pad=2)
    for spine in ("top", "right", "left", "bottom"):
        ax_top.spines[spine].set_visible(False)

    # Row labels
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(row_labels, fontsize=6.8)

    # Group separator between Clean and ZH blocks
    ax.axvline(x=2.5, color="#222222", linewidth=0.8)
    # Group separators between models (every 4 rows)
    for k in (3.5, 7.5):
        ax.axhline(y=k, color="#222222", linewidth=0.6)

    # Cell grid (subtle)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.4)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(axis="x", which="major", length=0, pad=2)
    ax.tick_params(axis="y", which="major", length=0, pad=2)

    # Hide the four data-axes spines (already hidden top/right via rcParams,
    # remove left/bottom to keep it clean).
    for spine in ("left", "bottom"):
        ax.spines[spine].set_visible(False)

    # Colorbar: short, on the right
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.55)
    cbar.set_label("TPR (%) @ $z\\!>\\!2.326$", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, length=2, width=0.5)
    cbar.outline.set_linewidth(0.4)

    fig.tight_layout(pad=0.3)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    M, row_labels = build_matrix()
    print("Matrix (rows = model/domain; cols = clean[mcl,kgw,sweet] | zh[mcl,kgw,sweet]):")
    header = ["clean-mcl", "clean-kgw", "clean-sweet", "zh-mcl", "zh-kgw", "zh-sweet"]
    print(f"{'row':<28}  " + "  ".join(f"{h:>10}" for h in header))
    for label, row in zip(row_labels, M):
        print(f"{label:<28}  " + "  ".join(f"{v:>10.1f}" for v in row))
    render(M, row_labels)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
