"""Implements the hardware-centric and fidelity metrics that Karras et al.
(2026, Future Internet 18:83, Section 9) define by equation but do not
compute on real data: CCS, MFR, TC, and the UMCEF composite (Eq. 58).
Also implements deletion/insertion fidelity (standard XAI evaluation,
not paper-specific) as the FID term UMCEF consumes.

All metrics are normalized to [0, 1] where higher = better, per the
survey's convention (Section 9.11).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Fidelity (deletion / insertion AUC)
# ---------------------------------------------------------------------------

def deletion_insertion_fidelity(
    predict_proba_fn,
    X: np.ndarray,
    attributions: np.ndarray,
    y_pred: np.ndarray,
    baseline_value: float = 0.0,
    n_steps: int = 10,
) -> dict:
    """Deletion AUC: how fast predicted-class probability collapses as the
    top-attributed features are masked out (lower area = attributions found
    the truly influential features -> we report 1 - normalized_AUC so
    higher = more faithful, consistent with the paper's "higher is better"
    convention).

    Insertion AUC: inverse test -- start from baseline, add back top
    features, probability should rise quickly if attributions are faithful.
    """
    n_samples, n_features = X.shape
    order = np.argsort(-np.abs(attributions), axis=1)  # most important first
    steps = np.linspace(0, n_features, n_steps + 1).astype(int)

    deletion_curve = np.zeros((n_samples, n_steps + 1))
    insertion_curve = np.zeros((n_samples, n_steps + 1))

    for step_i, k in enumerate(steps):
        X_del = X.copy()
        X_ins = np.full_like(X, baseline_value)
        for s in range(n_samples):
            top_k = order[s, :k]
            X_del[s, top_k] = baseline_value
            X_ins[s, top_k] = X[s, top_k]

        proba_del = predict_proba_fn(X_del)
        proba_ins = predict_proba_fn(X_ins)
        deletion_curve[:, step_i] = proba_del[np.arange(n_samples), y_pred]
        insertion_curve[:, step_i] = proba_ins[np.arange(n_samples), y_pred]

    deletion_auc = deletion_curve.mean(axis=1).mean()
    insertion_auc = insertion_curve.mean(axis=1).mean()

    # faithful explanations: deletion should drop fast (low AUC), insertion
    # should rise fast (high AUC). Combine into one "higher = better" score.
    fidelity = (insertion_auc + (1 - deletion_auc)) / 2.0
    return {
        "fidelity": float(np.clip(fidelity, 0, 1)),
        "deletion_auc": float(deletion_auc),
        "insertion_auc": float(insertion_auc),
    }


# ---------------------------------------------------------------------------
# CCS -- Computational Complexity Score
# ---------------------------------------------------------------------------

def computational_complexity_score(explain_time_s: float, reference_time_s: float) -> float:
    """CCS: normalized inverse of explanation time relative to the fastest
    explainer in the comparison set (reference_time_s). 1.0 = as fast as the
    fastest method; approaches 0 as an explainer gets far slower.
    """
    if explain_time_s <= 0:
        return 1.0
    return float(np.clip(reference_time_s / explain_time_s, 0, 1))


# ---------------------------------------------------------------------------
# MFR -- Memory Footprint Ratio
# ---------------------------------------------------------------------------

def memory_footprint_ratio(peak_memory_mb: float, budget_mb: float) -> float:
    """MFR: how much of the simulated edge-device memory budget the
    explanation process consumed. Returned as 1 - (used / budget), clipped
    to [0, 1], so higher = better (more headroom left).
    """
    if budget_mb <= 0:
        return 0.0
    used_ratio = peak_memory_mb / budget_mb
    return float(np.clip(1 - used_ratio, 0, 1))


# ---------------------------------------------------------------------------
# TC -- Temporal Coherence
# ---------------------------------------------------------------------------

def temporal_coherence(attributions_sequence: np.ndarray) -> float:
    """TC: rank-correlation stability of feature attributions across a
    sliding window of temporally/behaviorally adjacent samples (e.g.
    consecutive flows from the same source). High TC = explanations don't
    flicker between near-identical inputs -- a proxy for explanation
    reliability under the natural drift of live traffic.

    attributions_sequence: [window_size, n_features]
    """
    if attributions_sequence.shape[0] < 2:
        return 1.0
    correlations = []
    for i in range(attributions_sequence.shape[0] - 1):
        rho, _ = spearmanr(attributions_sequence[i], attributions_sequence[i + 1])
        if not np.isnan(rho):
            correlations.append(rho)
    if not correlations:
        return 0.0
    # map correlation [-1, 1] -> [0, 1]
    mean_rho = float(np.mean(correlations))
    return float(np.clip((mean_rho + 1) / 2, 0, 1))


# ---------------------------------------------------------------------------
# Robustness (explanation stability under adversarial perturbation)
# ---------------------------------------------------------------------------

def explanation_robustness(attr_clean: np.ndarray, attr_perturbed: np.ndarray, top_k: int = 5) -> float:
    """Top-k feature-set Jaccard overlap between clean and perturbed
    explanations, averaged over samples. Higher = more robust.
    """
    n_samples = attr_clean.shape[0]
    scores = []
    for i in range(n_samples):
        top_clean = set(np.argsort(-np.abs(attr_clean[i]))[:top_k])
        top_pert = set(np.argsort(-np.abs(attr_perturbed[i]))[:top_k])
        union = top_clean | top_pert
        inter = top_clean & top_pert
        scores.append(len(inter) / len(union) if union else 1.0)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# UMCEF_IoT composite (Eq. 58): 0.4*FID + 0.3*EFF + 0.2*ROB + 0.1*SPAR
# ---------------------------------------------------------------------------

@dataclass
class UMCEFComponents:
    fidelity: float          # FID
    efficiency: float        # EFF -- combine CCS and MFR
    robustness: float        # ROB
    sparsity: float          # SPAR -- fraction of near-zero attributions


def sparsity_score(attributions: np.ndarray, threshold: float = 1e-3) -> float:
    near_zero = np.abs(attributions) < threshold
    return float(near_zero.mean())


def umcef_iot(components: UMCEFComponents) -> float:
    fid = np.clip(components.fidelity, 0, 1)
    eff = np.clip(components.efficiency, 0, 1)
    rob = np.clip(components.robustness, 0, 1)
    spar = np.clip(components.sparsity, 0, 1)
    return float(0.4 * fid + 0.3 * eff + 0.2 * rob + 0.1 * spar)
