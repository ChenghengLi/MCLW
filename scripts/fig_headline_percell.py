#!/usr/bin/env python3
"""
fig_headline_percell.py
-----------------------
Per-(model, domain) head-to-head TPR view for the MCLW ICML 2026 paper.

Design: compact dual heatmap. 12 (model, domain) rows × 3 method columns,
shown twice side-by-side (Clean | ZH back-translation), sharing one
sequential blue colormap and one colorbar at the bottom. Sparse text
annotations only where TPR < 50% (the "failure" cells), so the eye is
drawn to where the baselines collapse and ChainMark survives. Replaces
the previous 72-bar grouped barchart, which was visually cluttered.

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
METHOD_LABELS = {"mcl": "MCL", "kgw": "KGW", "sweet": "SWEET"}
ATTACKS = [("z_clean", "Clean"), ("z_zh", "ZH back-translation")]
Z_THRESHOLD = 2.326
ANNOT_THRESHOLD = 50.0  # only annotate cells with TPR strictly below this

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
    (model_label, domain) tuples. Column order:
    clean[mcl,kgw,sweet], zh[mcl,kgw,sweet]."""
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
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.4,
    })

    n_rows = M.shape[0]  # 12
    n_methods = len(METHODS)  # 3

    M_clean = M[:, :3]
    M_zh    = M[:, 3:]

    # Sequential blues; 0..100. White ≈ 0, deep navy ≈ 100.
    cmap = plt.get_cmap("Blues")
    vmin, vmax = 0.0, 100.0

    fig = plt.figure(figsize=(3.2, 4.0))
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        height_ratios=[1.0, 0.035],
        width_ratios=[1.0, 1.0],
        wspace=0.12, hspace=0.20,
        left=0.27, right=0.985,
        top=0.94, bottom=0.085,
    )
    ax_clean = fig.add_subplot(gs[0, 0])
    ax_zh    = fig.add_subplot(gs[0, 1], sharey=ax_clean)
    ax_cbar  = fig.add_subplot(gs[1, :])

    method_xticklabels = [METHOD_LABELS[m] for m in METHODS]

    im = None
    for ax, mat, panel_title in (
        (ax_clean, M_clean, "Clean"),
        (ax_zh,    M_zh,    "ZH back-translation"),
    ):
        im = ax.imshow(
            mat, cmap=cmap, vmin=vmin, vmax=vmax,
            aspect="auto", interpolation="nearest",
        )
        # Methods as a single row of x-tick labels under each panel.
        ax.set_xticks(np.arange(n_methods))
        ax.set_xticklabels(method_xticklabels, fontsize=7.0)
        ax.tick_params(axis="x", which="both", length=0, pad=2)
        ax.tick_params(axis="y", which="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # Title-style panel labels at the top.
        ax.set_title(panel_title, fontsize=8, pad=3)

        # No interior gridlines on the heatmap; only the minor ticks.
        ax.set_xticks(np.arange(-0.5, n_methods, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.tick_params(which="minor", length=0)

        # Annotate every cell with its TPR value. Text colour switches
        # to white on dark cells so contrast stays readable.
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if math.isnan(v):
                    continue
                txt_color = "white" if v >= 60 else "#222222"
                ax.text(
                    j, i, f"{int(round(v))}",
                    ha="center", va="center",
                    fontsize=5.6, color=txt_color, zorder=3,
                )

        # Thin row separators between models (after rows 4 and 8).
        for k in (3.5, 7.5):
            ax.axhline(y=k, color="#cfcfcf", linewidth=0.4, zorder=4)

        # Thin outer border so the panel reads as a contained block.
        ax.add_patch(plt.Rectangle(
            (-0.5, -0.5), n_methods, n_rows,
            fill=False, edgecolor="#9a9a9a", linewidth=0.35, zorder=5,
        ))

    # Compact two-line row labels: "<model>" newline "<domain>" would be too
    # tall; instead use "<model> · <domain>" with a thin midpoint dot. Drop
    # the "Instruct" suffix (already implied) — model names already trimmed.
    row_labels = [f"{m} · {d}" for (m, d) in row_meta]
    ax_clean.set_yticks(np.arange(n_rows))
    ax_clean.set_yticklabels(row_labels, fontsize=6.6)
    plt.setp(ax_zh.get_yticklabels(), visible=False)
    # Invert so first row appears at top (matplotlib default already does
    # this with imshow; keep explicit for clarity if shape changes).

    # Shared horizontal colorbar at the bottom — concise label.
    cbar = fig.colorbar(
        im, cax=ax_cbar, orientation="horizontal",
        ticks=[0, 25, 50, 75, 100],
    )
    cbar.outline.set_linewidth(0.4)
    cbar.ax.tick_params(labelsize=6.5, length=2, width=0.4, pad=1.5)
    cbar.set_label("TPR (%)", fontsize=7.2, labelpad=2)

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
