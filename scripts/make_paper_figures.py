#!/usr/bin/env python3
"""Render the four ICML paper figures.

Outputs four PDFs into icml2026/pictures/:
  - fig_pareto.pdf         (Figure 1, matched-budget Pareto, empirical)
  - fig_theory.pdf         (Figure 2, theory-vs-data scatter, empirical)
  - fig_calibration.pdf    (Figure 3, S*(n,rho,alpha) closed-form curves)
  - fig_robustness.pdf     (Figure 4, Thm 4 robustness curve + empirical pt)

Empirical figures use data/v5_big_N200/summary.json (Gemma 3 270M, S=5,
clockwork, 173 wiki / 20 factual, T=0.7, 18 cells = 3 budgets x 3 gates,
random substitution attack at delta=0.20). Closed-form figures derive
from Theorem 4.1 (calibration) and Theorem 4.4 (robustness).

Usage:
  python3 scripts/make_paper_figures.py
"""

from __future__ import annotations
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "v5_big_N200" / "summary.json"
OUT = REPO / "icml2026" / "pictures"
OUT.mkdir(parents=True, exist_ok=True)

# Model / scheme constants (matches paper text)
S = 5            # states
N_TOK = 100      # max tokens -> n-1 = 99 pairs
SQRT_FACTOR = math.sqrt((S - 1) * (N_TOK - 1))   # approx 19.90
DELTA_ATK = 0.20
SURVIVAL2 = (1.0 - DELTA_ATK) ** 2                # 0.64

# Visual style: matched to a clean ICML look, no seaborn dependency.
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Gate style — consistent across both figures
GATE_STYLE = {
    "H_high": dict(marker="o", color="#1f6ad1", mec="#0e3a85", label=r"$G_{H_{\mathrm{high}}}$"),
    "Delta":  dict(marker="D", color="#ef8a21", mec="#a9540a", label=r"$G_{\Delta}$"),
    "H_low":  dict(marker="s", color="#d44b3e", mec="#7a201a", label=r"$G_{H_{\mathrm{low}}}$"),
}
GATE_ORDER = ["H_high", "Delta", "H_low"]
DOMAIN_ORDER = ["wiki", "factual"]


def load_cells() -> list[dict]:
    with DATA.open() as f:
        return json.load(f)["cells"]


# ---------------------------------------------------------------- Fig 1: Pareto
def fig_pareto(cells: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    titles = {"wiki": r"\textbf{Wiki} ($N{=}173$)", "factual": r"\textbf{Factual} ($N{=}20$)"}
    # pyplot-style titles without LaTeX
    titles = {"wiki": "Wiki (N = 173)", "factual": "Factual (N = 20)"}

    for ax, domain in zip(axes, DOMAIN_ORDER):
        ax.set_title(titles[domain])
        ax.set_xlabel("PPL")
        ax.set_xlim(0, 34)
        ax.set_ylim(0, 20)

        # Trajectories (one dashed line per gate connecting 3 budgets)
        for gate in GATE_ORDER:
            pts = [(c["mean_ppl"], c["mean_z"], c["target_rho"])
                   for c in cells
                   if c["domain"] == domain and c["gate"] == gate]
            pts.sort(key=lambda p: p[2])  # order by rho
            xs, ys, rhos = zip(*pts)
            style = GATE_STYLE[gate]
            ax.plot(xs, ys, linestyle="--", linewidth=1.1,
                    color=style["color"], alpha=0.7, zorder=1)
            ax.plot(xs, ys,
                    linestyle="", marker=style["marker"],
                    markersize=7, markerfacecolor=style["color"],
                    markeredgecolor=style["mec"], markeredgewidth=1.0,
                    label=style["label"] if domain == "wiki" else None,
                    zorder=3)

        # rho labels: only annotate the H_high trajectory (upper curve) and
        # the H_low trajectory (lower curve) to avoid overlaps with Delta.
        for gate in ("H_high", "H_low"):
            pts = [(c["mean_ppl"], c["mean_z"], c["target_rho"])
                   for c in cells
                   if c["domain"] == domain and c["gate"] == gate]
            pts.sort(key=lambda p: p[2])
            for x, y, r in pts:
                if gate == "H_low":
                    ax.annotate(f"ρ={r:g}", (x, y),
                                xytext=(6, -10), textcoords="offset points",
                                color="#555", fontsize=6.5)
                else:
                    ax.annotate(f"ρ={r:g}", (x, y),
                                xytext=(-6, 7), textcoords="offset points",
                                color="#555", fontsize=6.5,
                                ha="right")

        # "better" arrow
        ax.annotate("", xy=(2, 18.5), xytext=(28, 1.5),
                    arrowprops=dict(arrowstyle="->", color="gray",
                                    lw=1.4, alpha=0.6))
        ax.text(3, 18.8, "better", color="gray", fontsize=8,
                style="italic", ha="left")

    axes[0].set_ylabel("detection $z$")

    # Shared legend — bottom center
    handles = [Line2D([0], [0], marker=GATE_STYLE[g]["marker"],
                      linestyle="", markersize=7,
                      markerfacecolor=GATE_STYLE[g]["color"],
                      markeredgecolor=GATE_STYLE[g]["mec"],
                      markeredgewidth=1.0,
                      label=GATE_STYLE[g]["label"])
               for g in GATE_ORDER]
    annotations = [
        r"$G_{H_{\mathrm{high}}}$ (uncertain)",
        r"$G_{\Delta}$ (near-tied)",
        r"$G_{H_{\mathrm{low}}}$ (SWEET/EWD; dominated)",
    ]
    for h, lab in zip(handles, annotations):
        h.set_label(lab)

    fig.legend(handles=handles, ncol=3, frameon=False,
               loc="lower center", bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = OUT / "fig_pareto.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


# ------------------------------------------------------------- Fig 2: Theory
def fig_theory(cells: list[dict]) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    fig.subplots_adjust(wspace=0.30)

    # Panel A — Predicted z = rho_bar * sqrt((S-1)(n-1)) vs observed z
    axA = axes[0]
    axA.set_title(r"(A) detection formula (Thm. 1)", loc="left")
    axA.set_xlabel(r"predicted $z = \bar\rho\sqrt{(S{-}1)(n{-}1)}$")
    axA.set_ylabel("observed $z$")
    axA.set_xlim(0, 20); axA.set_ylim(0, 20)
    axA.plot([0, 20], [0, 20], color="gray", linestyle="--",
             linewidth=1.2, alpha=0.75, label="$y=x$")
    for gate in GATE_ORDER:
        pts = [(c["mean_rho"] * SQRT_FACTOR, c["mean_z"])
               for c in cells if c["gate"] == gate]
        xs, ys = zip(*pts)
        style = GATE_STYLE[gate]
        axA.plot(xs, ys, linestyle="", marker=style["marker"],
                 markersize=6.5, markerfacecolor=style["color"],
                 markeredgecolor=style["mec"], markeredgewidth=1.0,
                 label=style["label"])
    axA.legend(loc="lower right", framealpha=0.95, fontsize=8)

    # Panel B — pre-attack z vs post-attack z at delta=0.20
    axB = axes[1]
    axB.set_title(r"(B) self-healing (Thm. 4, $\delta{=}0.20$)", loc="left")
    axB.set_xlabel("pre-attack $z$")
    axB.set_ylabel("post-attack $z$")
    axB.set_xlim(0, 20); axB.set_ylim(0, 13)
    axB.plot([0, 20], [0, SURVIVAL2 * 20],
             color="gray", linestyle="--", linewidth=1.2, alpha=0.75,
             label=r"$y=(1-\delta)^2 x$")
    for gate in GATE_ORDER:
        pts = [(c["mean_z"], c["random_sub20"]["z_post_mean"])
               for c in cells if c["gate"] == gate]
        xs, ys = zip(*pts)
        style = GATE_STYLE[gate]
        axB.plot(xs, ys, linestyle="", marker=style["marker"],
                 markersize=6.5, markerfacecolor=style["color"],
                 markeredgecolor=style["mec"], markeredgewidth=1.0,
                 label=style["label"])
    axB.legend(loc="upper left", framealpha=0.95, fontsize=8)

    fig.tight_layout()
    out = OUT / "fig_theory.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def fig_calibration() -> Path:
    """Closed-form S*(n, rho, alpha) curves from Theorem 4.1."""
    from math import sqrt
    try:
        from scipy.stats import norm
        z_alpha = lambda a: norm.ppf(1.0 - a)
    except Exception:
        # closed-form alternative for the two alphas we use, accurate to 1e-4
        _Z = {1e-3: 3.0902, 1e-6: 4.7534}
        z_alpha = lambda a: _Z[a]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    n_grid = list(range(50, 2001, 25))
    rho_curves = [(0.25, "#5a5a5a"), (0.50, "#1f6ad1"),
                  (0.75, "#ef8a21"), (1.00, "#d44b3e")]
    panels = [(1e-3, "(A) $\\alpha = 10^{-3}$"),
              (1e-6, "(B) $\\alpha = 10^{-6}$")]

    for ax, (alpha, title) in zip(axes, panels):
        ax.set_title(title, loc="left")
        za2 = z_alpha(alpha) ** 2
        for rho, color in rho_curves:
            S_star = [int(math.ceil(4 * za2 / (rho ** 2 * (n - 1)) + 1))
                      for n in n_grid]
            ax.plot(n_grid, S_star, "-", color=color, linewidth=1.6,
                    label=f"$\\rho = {rho:g}$")
        ax.set_xlabel("text length $n$")
        ax.set_xscale("log")
        ax.set_xticks([50, 100, 200, 500, 1000, 2000])
        ax.set_xticklabels(["50", "100", "200", "500", "1000", "2000"])
        ax.set_ylim(1, 30)
        ax.axhline(2, color="gray", linewidth=0.6, linestyle=":")
        ax.set_yticks([2, 5, 7, 10, 15, 20, 25, 30])
    axes[0].set_ylabel("minimum state count $S^\\star$")
    axes[1].legend(loc="upper right", framealpha=0.95, ncol=2, fontsize=8)
    fig.tight_layout()
    out = OUT / "fig_calibration.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def fig_robustness(cells: list[dict]) -> Path:
    """E[phi | attack](delta) for several rho values + empirical pts at delta=0.20."""
    fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.4))
    p0 = 1.0 / S
    delta_grid = [d / 200.0 for d in range(201)]   # 0..1, step 0.005
    rhos = [(0.25, "#5a5a5a"), (0.50, "#1f6ad1"),
            (0.75, "#ef8a21"), (1.00, "#d44b3e")]

    for rho, color in rhos:
        phi = [p0 + rho * (1.0 - d) ** 2 * (1.0 - p0) for d in delta_grid]
        ax.plot(delta_grid, phi, "-", color=color, linewidth=1.7,
                label=f"$\\rho = {rho:g}$")
        # midpoint threshold line (per-rho)
        tau_mid = p0 + rho * (1.0 - p0) / 2.0
        ax.axhline(tau_mid, color=color, linewidth=0.8, linestyle=":",
                   alpha=0.55)

    # universal critical rate
    delta_star = 1.0 - 1.0 / math.sqrt(2.0)
    ax.axvline(delta_star, color="black", linewidth=1.1, linestyle="--",
               alpha=0.7)
    ax.text(delta_star + 0.005, 0.97,
            r"$\delta^\star = 1{-}1/\sqrt{2}$",
            color="black", fontsize=8, va="top", ha="left", alpha=0.85)

    # empirical (delta = 0.20) points: convert post-attack mean_z back to phi
    # using phi = z * sqrt(p0(1-p0)/(n-1)) + p0
    se_n = math.sqrt(p0 * (1 - p0) / (N_TOK - 1))
    for c in cells:
        if c["target_rho"] == 0.50:   # show one budget for clarity
            phi_post = c["random_sub20"]["z_post_mean"] * se_n + p0
            style = GATE_STYLE[c["gate"]]
            ax.plot(0.20, phi_post, marker=style["marker"], markersize=6,
                    markerfacecolor=style["color"],
                    markeredgecolor=style["mec"], markeredgewidth=0.8,
                    zorder=5)

    # null baseline horizontal
    ax.axhline(p0, color="gray", linewidth=0.8, linestyle=":")
    ax.text(0.97, p0 + 0.015, r"null baseline $p_0 = 1/S$",
            color="gray", fontsize=8, ha="right")

    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"modification fraction $\delta$")
    ax.set_ylabel(r"expected fingerprint $\mathbb{E}[\phi \mid \mathrm{atk}]$")
    ax.set_title(r"Closed-form robustness curve "
                 r"(Thm.~4.4) with empirical points at $\delta{=}0.20$",
                 loc="left", fontsize=9)
    # custom legend including gate markers + rho colors
    line_handles = [Line2D([0], [0], color=c, linewidth=1.7,
                           label=f"$\\rho = {r:g}$") for r, c in rhos]
    pt_handles = [Line2D([0], [0], marker=GATE_STYLE[g]["marker"],
                         linestyle="", markersize=6,
                         markerfacecolor=GATE_STYLE[g]["color"],
                         markeredgecolor=GATE_STYLE[g]["mec"],
                         markeredgewidth=0.8,
                         label=GATE_STYLE[g]["label"])
                  for g in GATE_ORDER]
    leg1 = ax.legend(handles=line_handles, loc="upper right",
                     title="theory", framealpha=0.95, fontsize=8,
                     title_fontsize=8)
    ax.add_artist(leg1)
    ax.legend(handles=pt_handles, loc="center right",
              title=r"empirical ($\delta{=}0.20$, $\rho{=}0.5$)",
              framealpha=0.95, fontsize=8, title_fontsize=8)

    fig.tight_layout()
    out = OUT / "fig_robustness.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


def main() -> None:
    cells = load_cells()
    p1 = fig_pareto(cells)
    p2 = fig_theory(cells)
    p3 = fig_calibration()
    p4 = fig_robustness(cells)
    for p in (p1, p2, p3, p4):
        print(f"wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
