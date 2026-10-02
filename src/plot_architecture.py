"""System architecture diagram for the benchmark + CC-SHAP pipeline.

Static box-and-arrow figure (no experimental data plotted -- this is a
pipeline/system diagram, not a results figure), styled to match the
project's other figures (plot_fef.py / plot_bars.py / plot_radar.py):
same accent color for CC-SHAP (this project's own contribution), same
muted gray for the four baseline explainers and shared infrastructure.

Usage:
    python -m src.plot_architecture --out results/figures/architecture
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ACCENT = "#eb6834"
ACCENT_FILL = "#fde9dd"
MUTED = "#898781"
MUTED_FILL = "#eeedea"
STAGE_FILL = "#f5f5f3"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#0b0b0b"  # blackened per request (was muted gray #52514e)
EDGE = "#c9c7c1"


def box(ax, xy, w, h, text, fill=STAGE_FILL, edge=EDGE, text_color=TEXT_PRIMARY,
        fontsize=10.62, fontweight="normal", zorder=2):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.1, edgecolor=edge, facecolor=fill, zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=text_color, fontweight=fontweight,
             zorder=zorder + 1, linespacing=1.3)
    return patch


def arrow(ax, p0, p1, color=MUTED, lw=1.3, style="-|>", zorder=1):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=12,
                         linewidth=lw, color=color, zorder=zorder,
                         shrinkA=0, shrinkB=0)
    ax.add_patch(a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figures/architecture")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(11, 7.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    # Column 1: data sources
    box(ax, (0.3, 5.7), 2.3, 1.0, "CICIoT2023\n(8-class, 39 feat.,\nbalanced)", fill=MUTED_FILL)
    box(ax, (0.3, 4.2), 2.3, 1.0, "Edge-IIoTset\n(15-class, 42 feat.,\nleakage-corrected)", fill=MUTED_FILL)
    ax.text(1.45, 6.95, "Data (§3.1)", ha="center", fontsize=11.21, fontweight="bold", color=TEXT_SECONDARY)

    # Column 2: preprocessing
    box(ax, (3.1, 4.95), 2.0, 1.0, "Preprocessing\nz-score norm.,\nstratified 75/25 split", fill=STAGE_FILL)
    arrow(ax, (2.6, 6.2), (3.1, 5.6))
    arrow(ax, (2.6, 4.7), (3.1, 5.3))

    # Column 3: models
    box(ax, (5.6, 5.7), 2.0, 1.0, "LightGBM\n(150 trees)", fill=MUTED_FILL)
    box(ax, (5.6, 4.2), 2.0, 1.0, "Compact MLP\n(<5.4k params)", fill=MUTED_FILL)
    ax.text(6.6, 6.95, "Models (§3.2)", ha="center", fontsize=11.21, fontweight="bold", color=TEXT_SECONDARY)
    arrow(ax, (5.1, 5.6), (5.6, 6.1))
    arrow(ax, (5.1, 5.4), (5.6, 4.8))

    # Column 4: explainers
    box(ax, (8.1, 6.35), 2.6, 0.55, "TreeSHAP / KernelSHAP /\nLightweight-KernelSHAP / LIME", fill=MUTED_FILL, fontsize=9.44)
    box(ax, (8.1, 5.55), 2.6, 0.65, "CC-SHAP\n(causal skeleton + per-clique\nattribution consolidation)", fill=ACCENT_FILL, text_color=TEXT_PRIMARY, fontweight="bold", fontsize=9.79)
    ax.text(9.4, 6.95, "Explainers (§3.3)", ha="center", fontsize=11.21, fontweight="bold", color=TEXT_SECONDARY)
    arrow(ax, (7.6, 6.1), (8.1, 6.6))
    arrow(ax, (7.6, 5.0), (8.1, 5.85), color=ACCENT)

    # CC-SHAP offline causal-skeleton sidecar
    box(ax, (8.1, 4.55), 2.6, 0.55, "Offline PC-skeleton\n(learned once; union-find cliques)", fill=ACCENT_FILL, fontsize=9.2, edge=ACCENT)
    arrow(ax, (9.4, 5.1), (9.4, 5.55), color=ACCENT, lw=1.1)

    # Row of harnesses underneath, feeding into metrics
    box(ax, (0.3, 2.55), 3.0, 1.0,
        "Resource-cap harness (§3.4)\nspawn-isolated subprocess,\nCPU-affinity + soft RLIMIT_AS,\nPi4 / Jetson Nano / unconstrained",
        fill=STAGE_FILL, fontsize=8.97)
    box(ax, (3.6, 2.55), 3.0, 1.0,
        "Adversarial harness (§3.5)\nFGSM + multiclass DeepFool (MLP),\nblack-box greedy attack (LightGBM),\nε ∈ {0.05, 0.15, 0.30}",
        fill=STAGE_FILL, fontsize=8.97)

    arrow(ax, (1.45, 4.2), (1.8, 3.55))
    arrow(ax, (6.6, 4.2), (4.5, 3.55))
    arrow(ax, (9.4, 4.55), (5.6, 3.1))

    # Metrics -> composite
    box(ax, (7.1, 2.55), 3.6, 1.0,
        "Metrics (§3.6): fidelity, CCS, MFR, TC,\nsparsity, robustness (top-k overlap)\n→ UMCEF_IoT composite (Eq. 58)",
        fill=STAGE_FILL, fontsize=9.2)
    arrow(ax, (3.3, 3.05), (7.1, 3.05))
    arrow(ax, (6.6, 3.05), (7.1, 3.05))

    # Outputs
    box(ax, (7.9, 0.9), 2.8, 1.0,
        "Section 4 results:\ngrid tables, robustness-by-ε,\nUMCEF_IoT leaderboard, FEF",
        fill=MUTED_FILL, fontsize=9.44)
    arrow(ax, (8.9, 2.55), (8.9, 1.9))

    ax.text(5.5, 0.15,
            "Each box in the resource/adversarial row runs once per (explainer × model × tier) cell.",
            ha="center", fontsize=8.85, color=TEXT_SECONDARY, style="italic")

    fig.tight_layout()
    for ext in ("pdf", "svg"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight")
    print(f"Saved {args.out}.pdf / .svg")


if __name__ == "__main__":
    main()
