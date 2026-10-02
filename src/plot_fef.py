"""Fidelity-Efficiency Frontier (FEF) plot, Eq. 59 of Karras et al. (2026):

    FEF = {(FID(Phi), EFF(Phi)) : not exists Phi' s.t.
                                   FID(Phi') > FID(Phi) and EFF(Phi') > EFF(Phi)}

i.e. the Pareto-optimal set of explainers for which no other explainer is
simultaneously more faithful AND more efficient. EFF is taken as this
project's own (CCS + MFR) / 2, matching the "efficiency" column already
computed for the UMCEF_IoT composite (src/metrics.py, run_experiment.py).

Styling follows the "emphasis" form (one series is the point, rest are
context -> accent + gray, not a 5-way categorical fight): CC-SHAP is this
project's own contribution and the thing the reader needs to track, so it
gets the accent color and the rest are muted gray with direct labels. The
Pareto-dominated region is shaded so "better" is spatially obvious without
needing to know what "Pareto frontier" means.

Usage:
    python -m src.plot_fef results/metrics/grid_ciciot2023_20261001_234835.csv \
        --title "CICIoT2023 (production scale)" \
        --out results/figures/fef_ciciot2023.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

ACCENT = "#eb6834"       # orange -- CC-SHAP, this project's own contribution
MUTED = "#898781"        # neutral gray -- the four baseline explainers
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#0b0b0b"  # blackened per request (was muted gray #52514e)
DOMINATED_FILL = "#ededeb"

LABELS = {
    "causal_shap": "CC-SHAP (this work)",
    "kernelshap": "KernelSHAP",
    "lightweight_kernelshap": "Lightweight\nKernelSHAP",
    "lime": "LIME",
    "treeshap": "TreeSHAP",
}


def pareto_frontier(points: list[tuple[str, float, float]]) -> set[str]:
    frontier = set()
    for label, fid, eff in points:
        dominated = any(
            (other_fid > fid and other_eff > eff)
            for other_label, other_fid, other_eff in points
            if other_label != label
        )
        if not dominated:
            frontier.add(label)
    return frontier


def plot_fef(grid_csv: str, title: str, out_path: str):
    df = pd.read_csv(grid_csv)
    ok = df[df["status"] == "OK"].copy()
    ok["efficiency"] = (ok["ccs"] + ok["mfr"]) / 2.0

    per_explainer = ok.groupby("explainer")[["fidelity", "efficiency"]].mean().reset_index()
    points = list(per_explainer.itertuples(index=False, name=None))
    frontier_labels = pareto_frontier(points)

    x_vals = [eff for _, _, eff in points]
    y_vals = [fid for _, fid, _ in points]
    x_pad = (max(x_vals) - min(x_vals)) * 0.22 or 0.02
    y_pad = (max(y_vals) - min(y_vals)) * 0.22 or 0.02
    xlim = (min(x_vals) - x_pad, max(x_vals) + x_pad * 1.6)
    ylim = (min(y_vals) - y_pad, max(y_vals) + y_pad)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.set_facecolor("white")

    # Shade the Pareto-dominated region: for each frontier point, everything
    # below-and-left of the frontier staircase is "worse on both axes than
    # something that already exists" -- shading it makes "better" a visual
    # direction (up-right) instead of a concept the reader has to hold in mind.
    frontier_sorted = sorted(
        [(eff, fid) for label, fid, eff in points if label in frontier_labels]
    )
    if frontier_sorted:
        stair_x = [xlim[0]]
        stair_y = [frontier_sorted[0][1]]
        for fx, fy in frontier_sorted:
            stair_x += [fx, fx]
            stair_y += [stair_y[-1], fy]
        stair_x += [xlim[1]]
        stair_y += [stair_y[-1]]
        ax.fill_between(stair_x, ylim[0], stair_y, color=DOMINATED_FILL, zorder=0, step=None)

    # "better" arrow, top-right corner
    ax.annotate(
        "", xy=(xlim[1] - x_pad * 0.3, ylim[1] - y_pad * 0.3),
        xytext=(xlim[1] - x_pad * 1.5, ylim[1] - y_pad * 1.8),
        arrowprops=dict(arrowstyle="-|>", color=TEXT_SECONDARY, lw=1.8),
    )
    ax.text(
        xlim[1] - x_pad * 1.6, ylim[1] - y_pad * 1.9, "better",
        color=TEXT_SECONDARY, fontsize=10.62, style="italic", ha="right", va="top",
    )

    # Generic label-collision avoidance: normalize to axis-fraction units,
    # and for any pair of points closer than a threshold in BOTH x and y,
    # push the higher-fidelity one up and the lower one down (instead of
    # hardcoding offsets per explainer name, which doesn't generalize
    # across datasets where the relative positions differ).
    x_span = xlim[1] - xlim[0]
    y_span = ylim[1] - ylim[0]

    def norm(eff, fid):
        return (eff - xlim[0]) / x_span, (fid - ylim[0]) / y_span

    offsets = {}
    for label, fid, eff in points:
        nx, ny = norm(eff, fid)
        close_others = [
            (ol, ofid, oeff) for ol, ofid, oeff in points
            if ol != label and abs(norm(oeff, ofid)[0] - nx) < 0.09 and abs(norm(oeff, ofid)[1] - ny) < 0.09
        ]
        if close_others:
            higher = all(fid >= ofid for _, ofid, _ in close_others)
            offsets[label] = (-16, 10) if higher else (-16, -16)
        else:
            offsets[label] = (10, 8)

    for label, fid, eff in points:
        is_hero = label == "causal_shap"
        is_frontier = label in frontier_labels
        color = ACCENT if is_hero else MUTED
        ax.scatter(
            eff, fid,
            s=420 if is_hero else (210 if is_frontier else 160),
            marker="*" if (is_hero or is_frontier) else "o",
            color=color,
            edgecolors=TEXT_PRIMARY if (is_hero or is_frontier) else "none",
            linewidths=1.8 if (is_hero or is_frontier) else 0,
            zorder=4 if is_hero else 3,
        )
        dx, dy = offsets[label]
        ha = "right" if dx < 0 else "left"
        ax.annotate(
            LABELS.get(label, label), (eff, fid),
            textcoords="offset points", xytext=(dx, dy), ha=ha,
            fontsize=12.98 if is_hero else 9.5,
            fontweight="bold" if is_hero else "normal",
            color=TEXT_PRIMARY if is_hero else TEXT_SECONDARY,
        )

    if len(frontier_sorted) > 1:
        fx, fy = zip(*frontier_sorted)
        ax.plot(fx, fy, "--", color=TEXT_SECONDARY, alpha=0.5, zorder=1, lw=1.3)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Efficiency  (CCS + MFR) / 2  —  cheaper to run  →", fontsize=12.98, color=TEXT_SECONDARY)
    ax.set_ylabel("Fidelity  —  more faithful to the model  →", fontsize=12.98, color=TEXT_SECONDARY)
    ax.set_title(
        f"Fidelity vs. efficiency: where does CC-SHAP sit?\n{title}",
        fontsize=15.34, color=TEXT_PRIMARY, pad=12,
    )
    ax.grid(True, alpha=0.25, color=MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    shaded_note = (
        "Shaded region: Pareto-dominated (a cheaper, more faithful explainer\n"
        "already exists). CC-SHAP sits here on fidelity/efficiency alone —\n"
        "see the companion radar chart for why it still ranks 1st-2nd overall."
    )
    ax.text(
        0.02, 0.03, shaded_note, transform=ax.transAxes,
        fontsize=10.03, color=TEXT_SECONDARY, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=MUTED, alpha=0.9),
    )

    fig.tight_layout()
    base, _ = os.path.splitext(out_path)
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    for ext in ("pdf", "svg"):
        path = f"{base}.{ext}"
        fig.savefig(path)
        print(f"Saved: {path}")
    print(f"Pareto-optimal (frontier) explainers: {sorted(frontier_labels)}")
    print(per_explainer.round(3).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("grid_csv")
    parser.add_argument("--title", default="")
    parser.add_argument("--out", default="results/figures/fef.png")
    args = parser.parse_args()
    plot_fef(args.grid_csv, args.title, args.out)


if __name__ == "__main__":
    main()
