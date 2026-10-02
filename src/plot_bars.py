"""Grouped bar chart of the four UMCEF_IoT dimensions per explainer --
FID, EFF, SPAR, ROB (Eq. 58 of Karras et al. 2026).

Why this chart over the radar (src/plot_radar.py): polygon area in a radar
chart is not a linear perceptual channel -- readers over/under-weight a
shape's apparent size relative to the actual numbers, and axis ordering
can bias the impression. A grouped bar chart lets precise values be
compared bar-to-bar directly, which is the more rigorous, reviewer-
expected form for a quantitative results figure. Same underlying data as
plot_radar.py and the UMCEF_IoT leaderboard -- just the more defensible
chart type for a paper.

Each explainer gets its own fixed categorical color (not cycled, not
reassigned by rank -- the dataviz skill's categorical rule), grouped by
metric so the four per-metric clusters are easy to scan left to right.
CC-SHAP keeps a bold black outline so it's still the one the reader's eye
lands on first, without the other four being flattened to uniform gray.

Usage:
    python -m src.plot_bars results/metrics/grid_ciciot2023_20261001_234835.csv \
        results/metrics/robustness_ciciot2023_20261001_234835.csv \
        --title "CICIoT2023 (production scale)" \
        --out results/figures/bars_ciciot2023.pdf
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.plot_radar import LABELS, compute_umcef_components

MUTED_LIGHT = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#0b0b0b"  # blackened per request (was muted gray #52514e)

METRIC_LABELS = ["Fidelity\n(FID)", "Efficiency\n(EFF)", "Sparsity\n(SPAR)", "Robustness\n(ROB)"]
METRIC_COLS = ["FID", "EFF", "SPAR", "ROB"]

# fixed explainer order (not sorted by value) -- color/position follows the
# entity, never its rank, per the dataviz skill's categorical rule
EXPLAINER_ORDER = ["causal_shap", "treeshap", "kernelshap", "lightweight_kernelshap", "lime"]

# fixed categorical palette, first 5 slots of the validated 8-hue order
# (references/palette.md) -- assigned by fixed position, not by value
EXPLAINER_COLOR = {
    "causal_shap": "#2a78d6",              # blue
    "treeshap": "#eb6834",                 # orange
    "kernelshap": "#1baf7a",               # aqua
    "lightweight_kernelshap": "#eda100",   # yellow
    "lime": "#e87ba4",                     # magenta
}


def plot_bars(grid_csv: str, robustness_csv: str, title: str, out_path: str):
    components = compute_umcef_components(grid_csv, robustness_csv)
    explainers = [e for e in EXPLAINER_ORDER if e in components.index]

    n_groups = len(METRIC_COLS)
    n_bars = len(explainers)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("white")

    for i, explainer in enumerate(explainers):
        is_hero = explainer == "causal_shap"
        values = components.loc[explainer, METRIC_COLS].tolist()
        offset = (i - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, values, bar_width * 0.92,
            color=EXPLAINER_COLOR[explainer],
            edgecolor=TEXT_PRIMARY if is_hero else "white",
            linewidth=2.0 if is_hero else 0.8,
            zorder=3,
            label=LABELS.get(explainer, explainer),
        )
        # direct value labels on CC-SHAP's bars only (selective labeling,
        # not a number on every bar) -- it's the series the reader tracks
        if is_hero:
            for rect, val in zip(bars, values):
                ax.annotate(
                    f"{val:.2f}",
                    (rect.get_x() + rect.get_width() / 2, val),
                    textcoords="offset points", xytext=(0, 4),
                    ha="center", fontsize=10.62, fontweight="bold", color=TEXT_PRIMARY,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(METRIC_LABELS, fontsize=12.98, color=TEXT_PRIMARY)
    ax.set_ylabel("Score (higher = better)", fontsize=12.98, color=TEXT_SECONDARY)
    ax.set_ylim(0, 1.08)
    ax.set_title(
        f"Explainer comparison across all four UMCEF_IoT dimensions\n{title}",
        fontsize=15.34, color=TEXT_PRIMARY, pad=14,
    )
    ax.grid(True, axis="y", alpha=0.25, color=MUTED_LIGHT)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # group separators between metric clusters
    for gx in (x[:-1] + 0.5):
        ax.axvline(gx, color=MUTED_LIGHT, lw=0.8, alpha=0.5, zorder=1)

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=11.21, frameon=False,
    )

    note = (
        "CC-SHAP (accent) trails on Fidelity, matches the field on Efficiency,\n"
        "and leads decisively on Sparsity and Robustness."
    )
    fig.text(0.42, -0.02, note, ha="center", fontsize=10.62, color=TEXT_SECONDARY)

    fig.tight_layout()
    base, _ = os.path.splitext(out_path)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    for ext in ("pdf", "svg"):
        path = f"{base}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")
    print(components.round(3).to_string())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("grid_csv")
    parser.add_argument("robustness_csv")
    parser.add_argument("--title", default="")
    parser.add_argument("--out", default="results/figures/bars.png")
    args = parser.parse_args()
    plot_bars(args.grid_csv, args.robustness_csv, args.title, args.out)


if __name__ == "__main__":
    main()
