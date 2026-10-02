"""Radar (spider) chart of the four UMCEF_IoT dimensions per explainer --
FID, EFF, SPAR, ROB (Eq. 58 of Karras et al. 2026) -- as a more intuitive
complement to the Fidelity-Efficiency Frontier (src/plot_fef.py).

Why this chart exists: the FEF plot (Eq. 59) only shows 2 of the 4
dimensions the field cares about, and CC-SHAP is Pareto-dominated on that
2D view on both datasets (see RESEARCH_IMPLEMENTATION_PLAN.md Part 7.7).
A radar chart puts all four dimensions on screen at once, so a reader can
see CC-SHAP's actual shape -- weak on fidelity, strong on sparsity and
robustness -- in one glance, without first learning what "Pareto frontier"
means. This is the same data as the UMCEF_IoT leaderboard, just shown as a
shape instead of a table row.

Styling follows the "emphasis" pattern (one series is the point, rest are
context): CC-SHAP drawn as a filled accent shape, the four baseline
explainers as thin gray outlines underneath.

Usage:
    python -m src.plot_radar results/metrics/grid_ciciot2023_20261001_234835.csv \
        results/metrics/robustness_ciciot2023_20261001_234835.csv \
        --title "CICIoT2023 (production scale)" \
        --out results/figures/radar_ciciot2023.png
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACCENT = "#eb6834"
MUTED = "#898781"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#0b0b0b"  # blackened per request (was muted gray #52514e)

LABELS = {
    "causal_shap": "CC-SHAP (this work)",
    "kernelshap": "KernelSHAP",
    "lightweight_kernelshap": "Lightweight KernelSHAP",
    "lime": "LIME",
    "treeshap": "TreeSHAP",
}

AXES = ["Fidelity\n(FID)", "Efficiency\n(EFF)", "Sparsity\n(SPAR)", "Robustness\n(ROB)"]


def compute_umcef_components(grid_csv: str, robustness_csv: str) -> pd.DataFrame:
    grid = pd.read_csv(grid_csv)
    ok = grid[grid["status"] == "OK"].copy()
    ok["efficiency"] = (ok["ccs"] + ok["mfr"]) / 2.0
    fes = ok.groupby("explainer")[["fidelity", "efficiency", "sparsity"]].mean()
    fes.columns = ["FID", "EFF", "SPAR"]

    rob = pd.read_csv(robustness_csv)
    ok_rob = rob[rob["status"] == "OK"]
    rob_by_explainer = ok_rob.groupby("explainer")["explanation_robustness"].mean().rename("ROB")

    out = fes.join(rob_by_explainer, how="left")
    out["ROB"] = out["ROB"].fillna(0.0)
    return out


def plot_radar(grid_csv: str, robustness_csv: str, title: str, out_path: str):
    components = compute_umcef_components(grid_csv, robustness_csv)
    n_axes = len(AXES)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=9.44, color=TEXT_SECONDARY)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(AXES, fontsize=12.98, color=TEXT_PRIMARY)
    ax.grid(color=MUTED, alpha=0.35)
    ax.spines["polar"].set_color(MUTED)
    ax.spines["polar"].set_alpha(0.5)

    # baselines first (gray, thin, no fill) so the accent shape draws on top
    for explainer in components.index:
        if explainer == "causal_shap":
            continue
        values = components.loc[explainer, ["FID", "EFF", "SPAR", "ROB"]].tolist()
        values += values[:1]
        ax.plot(angles, values, color=MUTED, linewidth=1.3, alpha=0.75, zorder=2)
        ax.scatter(angles[:-1], values[:-1], color=MUTED, s=22, zorder=2)

    # CC-SHAP last, on top, filled accent
    if "causal_shap" in components.index:
        values = components.loc["causal_shap", ["FID", "EFF", "SPAR", "ROB"]].tolist()
        values += values[:1]
        ax.plot(angles, values, color=ACCENT, linewidth=3, zorder=4)
        ax.fill(angles, values, color=ACCENT, alpha=0.22, zorder=3)
        ax.scatter(angles[:-1], values[:-1], color=ACCENT, s=70, zorder=5, edgecolors=TEXT_PRIMARY, linewidths=1.2)

    # direct labels: one legend, baselines + accent, matching line styles
    handles = []
    for explainer in components.index:
        is_hero = explainer == "causal_shap"
        handles.append(
            plt.Line2D(
                [], [], color=ACCENT if is_hero else MUTED,
                linewidth=3 if is_hero else 1.3, alpha=1.0 if is_hero else 0.75,
                marker="o", markersize=7 if is_hero else 5,
                label=LABELS.get(explainer, explainer),
            )
        )
    # order legend with CC-SHAP first
    handles.sort(key=lambda h: 0 if "this work" in h.get_label() else 1)
    ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.08, 1.08),
        fontsize=11.21, frameon=False,
    )

    ax.set_title(
        f"Explainer profile across all four UMCEF_IoT dimensions\n{title}",
        fontsize=15.34, color=TEXT_PRIMARY, pad=28,
    )

    note = (
        "CC-SHAP's shape: pulled in on Fidelity, pushed out on Sparsity and\n"
        "Robustness — the trade-off the UMCEF_IoT composite (Eq. 58) is designed to weigh."
    )
    fig.text(0.5, -0.02, note, ha="center", fontsize=10.62, color=TEXT_SECONDARY)

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
    parser.add_argument("--out", default="results/figures/radar.png")
    args = parser.parse_args()
    plot_radar(args.grid_csv, args.robustness_csv, args.title, args.out)


if __name__ == "__main__":
    main()
