"""Unified explainer interface: TreeSHAP (exact, tree-only), KernelSHAP
(model-agnostic baseline), LIME, a lightweight/sparsified KernelSHAP
variant (small background + coalition-sample budget) representing the
"lightweight, sparsity-aware explainer" direction called for in Table 23
of Karras et al. (2026), and CC-SHAP (Causal-Consolidated SHAP) -- this
project's own contribution, answering the field's call for "causal,
graph-based... methods" tailored to network telemetry rather than generic
tabular explainers (see src/causal_graph.py for the rationale).

Every explain_* function returns (attributions: np.ndarray[n_samples, n_features],
elapsed_seconds: float) so downstream code can compute CCS/MFR/fidelity
uniformly regardless of which explainer produced the attribution.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer

from src import causal_graph


def explain_treeshap(model, X_background: np.ndarray, X_explain: np.ndarray):
    """Exact TreeSHAP -- only valid for tree ensemble models (LightGBM)."""
    start = time.perf_counter()
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_explain)
    elapsed = time.perf_counter() - start

    attributions = _reduce_shap_output(sv)
    return attributions, elapsed


def explain_kernelshap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_background: np.ndarray,
    X_explain: np.ndarray,
    nsamples: int = 200,
    background_size: int = 50,
):
    """Model-agnostic KernelSHAP baseline -- expensive, O(2^M)-ish in spirit."""
    bg = shap.sample(X_background, background_size)
    start = time.perf_counter()
    explainer = shap.KernelExplainer(predict_fn, bg)
    sv = explainer.shap_values(X_explain, nsamples=nsamples, silent=True)
    elapsed = time.perf_counter() - start

    attributions = _reduce_shap_output(sv)
    return attributions, elapsed


def explain_lightweight_kernelshap(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_background: np.ndarray,
    X_explain: np.ndarray,
):
    """Sparsified/lightweight variant: small background + small coalition
    budget, representing the 'lightweight explainer' research direction
    (Table 23) as an ablation of KernelSHAP rather than a distinct library.
    """
    return explain_kernelshap(
        predict_fn,
        X_background,
        X_explain,
        nsamples=30,
        background_size=10,
    )


def explain_lime(
    predict_proba_fn: Callable[[np.ndarray], np.ndarray],
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    num_features: int | None = None,
    num_samples: int = 5000,
):
    """num_samples is LIME's own internal per-instance perturbation budget
    (its library default is 5000). This is the dominant cost driver for
    LIME in this benchmark: at 5000, explaining N instances costs N*5000
    model predictions, which is what made the CICIoT2023 robustness sweep
    (12 explainer/model/epsilon combos x 30 instances x clean+adversarial)
    take ~25 minutes in practice. Lower it (e.g. 500-1000) for fast dev
    iteration; keep the 5000 default for numbers that go in the writeup,
    since LIME's own stability depends on having enough local samples.
    """
    num_features = num_features or X_background.shape[1]
    explainer = LimeTabularExplainer(
        X_background,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
    )

    attributions = np.zeros((X_explain.shape[0], X_background.shape[1]))
    start = time.perf_counter()
    for i in range(X_explain.shape[0]):
        exp = explainer.explain_instance(
            X_explain[i], predict_proba_fn, num_features=num_features, num_samples=num_samples
        )
        for feat_idx, weight in exp.local_exp[exp.available_labels()[0]]:
            attributions[i, feat_idx] = weight
    elapsed = time.perf_counter() - start

    return attributions, elapsed


def explain_causal_shap(
    base_attributions: np.ndarray,
    causal_adjacency: np.ndarray,
    base_elapsed_s: float,
):
    """CC-SHAP (Causal-Consolidated SHAP): this project's own explainer,
    not a repackaging of an existing library method.

    Takes attributions already produced by a base explainer (TreeSHAP for
    LightGBM, KernelSHAP for the MLP -- CC-SHAP is a *post-processing*
    layer, not a new attribution algorithm from scratch, which is exactly
    why it stays cheap: no extra model queries at explanation time) and a
    causal skeleton (src/causal_graph.build_pc_skeleton, learned ONCE
    offline from the training set). Within each connected component
    ("redundant/entangled feature clique") of the skeleton, keeps only the
    feature with the largest |attribution| per sample and zeroes the rest.

    Rationale: SHAP-family methods are known to split credit ambiguously
    across correlated features (the attribution a redundant duplicate
    receives can flip between near-identical inputs, or between runs with
    different background samples) -- this is exactly the kind of
    instability the survey's Temporal Coherence (TC) metric is designed to
    catch. CC-SHAP's hypothesis: forcing attribution onto the strongest
    signal in each causally-entangled group should raise sparsity and
    stability (TC, adversarial robustness) with a bounded, measurable cost
    to fidelity -- this experiment is designed to test that trade-off
    directly against the paper's other four explainers, not just assert it.

    The wall-clock cost reported here is ONLY the post-processing step
    (union-find + per-sample argmax over cliques), added to the base
    explainer's own elapsed time -- the one-time graph-construction cost
    is accounted separately (see run_experiment.py) exactly like a
    TreeSHAP model's tree-building is not billed against its per-call CCS.
    """
    start = time.perf_counter()
    components = causal_graph.connected_components(causal_adjacency)
    consolidated = np.zeros_like(base_attributions)

    for group in components:
        if len(group) == 1:
            consolidated[:, group[0]] = base_attributions[:, group[0]]
            continue
        group_arr = np.array(group)
        sub = base_attributions[:, group_arr]
        winner_local_idx = np.argmax(np.abs(sub), axis=1)
        for sample_i, local_idx in enumerate(winner_local_idx):
            consolidated[sample_i, group_arr[local_idx]] = sub[sample_i, local_idx]

    post_elapsed = time.perf_counter() - start
    return consolidated, base_elapsed_s + post_elapsed


def explain_dispatch(
    explainer_name: str,
    model_kind: str,
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    feature_names: list[str] | None = None,
    class_names: list[str] | None = None,
    lime_num_samples: int = 5000,
    causal_adjacency: np.ndarray | None = None,
):
    """Top-level, picklable dispatcher used by resource_harness's
    spawn-based subprocess isolation (a lambda/closure isn't picklable and
    also proved unsafe under fork() -- shap's KernelExplainer holds
    background-thread state that doesn't survive a fork cleanly, causing
    the child to hang. Spawn + a plain top-level function avoids both
    problems: the child re-imports everything fresh).

    model_kind: "lightgbm" | "mlp"
    explainer_name == "causal_shap" requires `causal_adjacency` (learned
    once offline via causal_graph.build_pc_skeleton on the training set)
    and uses TreeSHAP as its base for lightgbm, KernelSHAP as its base for
    mlp (the model this project doesn't have exact-tree attribution for).
    """
    if model_kind == "lightgbm":
        proba_fn = model.predict_proba
    elif model_kind == "mlp":
        proba_fn = model.predict_proba
    else:
        raise ValueError(f"unknown model_kind: {model_kind}")

    if explainer_name == "treeshap":
        if model_kind != "lightgbm":
            raise ValueError("treeshap only applies to lightgbm")
        return explain_treeshap(model, X_background, X_explain)
    elif explainer_name == "kernelshap":
        return explain_kernelshap(proba_fn, X_background, X_explain)
    elif explainer_name == "lightweight_kernelshap":
        return explain_lightweight_kernelshap(proba_fn, X_background, X_explain)
    elif explainer_name == "lime":
        return explain_lime(proba_fn, X_background, X_explain, feature_names, class_names, num_samples=lime_num_samples)
    elif explainer_name == "causal_shap":
        if causal_adjacency is None:
            raise ValueError("causal_shap requires causal_adjacency (build once via causal_graph.build_pc_skeleton)")
        if model_kind == "lightgbm":
            base_attr, base_time = explain_treeshap(model, X_background, X_explain)
        else:
            base_attr, base_time = explain_kernelshap(proba_fn, X_background, X_explain)
        return explain_causal_shap(base_attr, causal_adjacency, base_time)
    else:
        raise ValueError(f"unknown explainer_name: {explainer_name}")


def _reduce_shap_output(sv) -> np.ndarray:
    """shap returns a list-of-arrays for multiclass, a single array for
    binary/regression. Reduce to a single [n_samples, n_features] array by
    taking the mean absolute attribution across classes (standard practice
    for a class-agnostic 'which features matter' comparison across methods).
    """
    if isinstance(sv, list):
        stacked = np.stack([np.abs(s) for s in sv], axis=0)  # [n_classes, n, f]
        return stacked.mean(axis=0)
    arr = np.asarray(sv)
    if arr.ndim == 3:  # [n, f, n_classes] (newer shap API for multiclass)
        return np.abs(arr).mean(axis=2)
    return arr
