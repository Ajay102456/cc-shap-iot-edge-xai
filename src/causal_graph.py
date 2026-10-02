"""Lightweight causal-skeleton discovery over tabular IoT flow features.

Motivation (the actual research gap this addresses): both source documents
converge on the same complaint -- SHAP/LIME are imported from generic
tabular ML and are "not tailored to network telemetry" (Nascita et al.,
2025), and the field calls for "causal, graph-based, or protocol-aware
methods" to replace or augment them (Nivash et al., 2026; Gad, 2025).
Nothing in the compared-explainer literature actually does this cheaply
enough for edge deployment -- full PC-algorithm / FCI causal discovery is
itself expensive (that's why it's absent from resource-constrained IoT-XAI
work). This module implements a deliberately BOUNDED PC-skeleton (max
conditioning-set size = 1, not the unbounded PC algorithm) so causal
structure discovery itself stays within a "lightweight, edge-aware"
budget, consistent with the survey's own resource-aware framing -- the
boundedness is not a compromise hidden from the reader, it's the design.

The skeleton is learned ONCE, offline, from the training/background set
(exactly like how TreeSHAP's cost is dominated by tree-building, which
also happens offline) -- so its cost does not count against per-sample
explanation latency in the same way KernelSHAP's iterative sampling does.
This is intentional and is the whole point: causal structure amortizes,
generic model-agnostic sampling does not.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def _fisher_z_test(r: float, n: int, cond_set_size: int, alpha: float) -> bool:
    """Returns True if i and j are judged conditionally independent
    (i.e., the edge should be removed) given a partial correlation r
    estimated with `cond_set_size` conditioning variables and n samples.
    """
    df = n - cond_set_size - 3
    if df <= 0:
        return False  # not enough samples to test -- keep the edge (conservative)
    r = np.clip(r, -0.999999, 0.999999)
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(df)
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return p_value > alpha


def _partial_corr_order1(corr: np.ndarray, i: int, j: int, k: int) -> float:
    r_ij, r_ik, r_jk = corr[i, j], corr[i, k], corr[j, k]
    denom = np.sqrt((1 - r_ik**2) * (1 - r_jk**2))
    if denom < 1e-10:
        return r_ij
    return (r_ij - r_ik * r_jk) / denom


def build_pc_skeleton(X: np.ndarray, alpha: float = 0.05, max_cond_set_size: int = 1) -> np.ndarray:
    """Bounded PC-skeleton discovery. Returns a symmetric boolean adjacency
    matrix [n_features, n_features] (diagonal False): True where no tested
    conditioning set rendered the pair independent, i.e. a plausible direct
    (non-mediated, up to conditioning-set-size) dependency remains.

    max_cond_set_size=1 means: test marginal independence (order 0), then
    for surviving edges, test independence conditioned on each single other
    variable (order 1). This is PC's first two stages, deliberately not
    extended further -- see module docstring for why that's the design.
    """
    n, p = X.shape
    corr = np.corrcoef(X, rowvar=False)
    adjacency = ~np.eye(p, dtype=bool)  # start fully connected (minus diagonal)

    # Order 0: marginal independence
    for i in range(p):
        for j in range(i + 1, p):
            if _fisher_z_test(corr[i, j], n, cond_set_size=0, alpha=alpha):
                adjacency[i, j] = adjacency[j, i] = False

    # Order 1: conditioned on each other single variable
    if max_cond_set_size >= 1:
        for i in range(p):
            for j in range(i + 1, p):
                if not adjacency[i, j]:
                    continue
                for k in range(p):
                    if k == i or k == j:
                        continue
                    r_ij_k = _partial_corr_order1(corr, i, j, k)
                    if _fisher_z_test(r_ij_k, n, cond_set_size=1, alpha=alpha):
                        adjacency[i, j] = adjacency[j, i] = False
                        break

    return adjacency


def build_effect_size_skeleton(X: np.ndarray, min_abs_r: float = 0.3) -> np.ndarray:
    """Ablation control: a 'causal-looking' skeleton built from a practical
    effect-size threshold on marginal |correlation| alone, with no
    significance test and no order-1 conditional-independence refinement.

    Added after peer review flagged that, at the production sample sizes
    used in this paper (n >= 10,000), the Fisher-z significance test in
    build_pc_skeleton() conflates statistical significance with practical
    relevance: at that sample size, even very weak correlations clear
    alpha=0.05, so the "causal" and order-0 "correlated-only" skeletons
    both collapse into one all-encompassing clique spanning every feature
    (confirmed empirically, see ablation_causal.py output). That makes any
    comparison against a size-matched random grouping vacuous -- there is
    only one clique to permute within. This function produces a genuinely
    sparser, more selective grouping by filtering on effect size rather
    than (or in addition to) significance, which is the only way to get a
    non-trivial random-clique ablation at this sample size.
    """
    p = X.shape[1]
    corr = np.corrcoef(X, rowvar=False)
    adjacency = np.abs(corr) >= min_abs_r
    np.fill_diagonal(adjacency, False)
    return adjacency


def random_cliques_matching(adjacency: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Ablation control: returns a new adjacency matrix whose connected
    components have the EXACT same size distribution as `adjacency`'s real
    causal-skeleton cliques, but with feature membership assigned by random
    shuffle rather than by any conditional-independence test. Used to ask
    whether CC-SHAP's consolidation mechanism needs causal structure at all,
    or would produce similar sparsity/robustness gains from forcing the
    same degree of top-1-per-group consolidation on arbitrarily grouped
    features (see manuscript Section 4.4 ablation).
    """
    components = connected_components(adjacency)
    sizes = [len(c) for c in components]
    p = adjacency.shape[0]
    perm = rng.permutation(p)

    new_adjacency = np.zeros((p, p), dtype=bool)
    cursor = 0
    for size in sizes:
        group = perm[cursor:cursor + size]
        cursor += size
        for a in group:
            for b in group:
                if a != b:
                    new_adjacency[a, b] = True
    return new_adjacency


def connected_components(adjacency: np.ndarray) -> list[list[int]]:
    """Union-find over the skeleton -> redundant/entangled feature cliques.
    Singletons (no surviving edges) are their own component of size 1.
    """
    p = adjacency.shape[0]
    parent = list(range(p))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(p):
        for j in range(i + 1, p):
            if adjacency[i, j]:
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(p):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())
