"""Ablation + bootstrap-CI companion experiment, added in response to peer
review: does CC-SHAP's sparsity/robustness advantage come from the CAUSAL
skeleton specifically, or merely from forcing top-1-per-group consolidation
on any grouping of comparable granularity?

Two ablation controls, both applied via the same consolidation mechanism
(explainers.explain_causal_shap) so only the *grouping* varies:

  - corr_only:    build_pc_skeleton(max_cond_set_size=0) -- marginal
                   correlation clustering only, skipping the order-1
                   conditional-independence refinement that makes the real
                   skeleton "causal" rather than merely "correlated".
  - random_clique: causal_graph.random_cliques_matching -- same clique-size
                   distribution as the real skeleton, membership shuffled.

Also computes a bootstrap 95% CI (1000 resamples over the explained-sample
batch) for fidelity, sparsity, and robustness-at-eps=0.15, for: real
CC-SHAP, both ablations, and the KernelSHAP baseline (the "near-tied"
UMCEF_IoT comparator on CICIoT2023).

Usage:
    python -m src.ablation_causal --data ciciot2023 --n-per-class 10000 --n-explain 100
    python -m src.ablation_causal --data edge-iiotset --n-per-class 1000 --n-explain 100
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from src import adversarial, causal_graph, data, explainers, metrics, models

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")


def per_sample_fidelity(proba_fn, X, attributions, y_pred, baseline_value=0.0, n_steps=10):
    """Same curve construction as metrics.deletion_insertion_fidelity, but
    returns one fidelity value per sample instead of collapsing to a single
    mean, so a bootstrap CI can resample over samples."""
    n_samples, n_features = X.shape
    order = np.argsort(-np.abs(attributions), axis=1)
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
        proba_del = proba_fn(X_del)
        proba_ins = proba_fn(X_ins)
        deletion_curve[:, step_i] = proba_del[np.arange(n_samples), y_pred]
        insertion_curve[:, step_i] = proba_ins[np.arange(n_samples), y_pred]

    deletion_auc = deletion_curve.mean(axis=1)
    insertion_auc = insertion_curve.mean(axis=1)
    fid = np.clip((insertion_auc + (1 - deletion_auc)) / 2.0, 0, 1)
    return fid


def per_sample_sparsity(attributions, threshold=1e-3):
    return (np.abs(attributions) < threshold).mean(axis=1)


def per_sample_robustness(attr_clean, attr_perturbed, top_k=5):
    n = attr_clean.shape[0]
    scores = np.zeros(n)
    for i in range(n):
        top_clean = set(np.argsort(-np.abs(attr_clean[i]))[:top_k])
        top_pert = set(np.argsort(-np.abs(attr_perturbed[i]))[:top_k])
        union = top_clean | top_pert
        inter = top_clean & top_pert
        scores[i] = len(inter) / len(union) if union else 1.0
    return scores


def bootstrap_ci(values: np.ndarray, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = values[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(values.mean()), float(lo), float(hi)


def consolidate(base_attr, adjacency):
    consolidated, _ = explainers.explain_causal_shap(base_attr, adjacency, base_elapsed_s=0.0)
    return consolidated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["ciciot2023", "edge-iiotset"], required=True)
    ap.add_argument("--n-per-class", type=int, default=10000)
    ap.add_argument("--n-explain", type=int, default=100)
    ap.add_argument("--seed", type=int, default=43)  # distinct from production seed=42: independent ablation run
    ap.add_argument("--eps", type=float, default=0.15)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[1/5] Loading {args.data} and training models (seed={args.seed})...")
    if args.data == "ciciot2023":
        ds = data.load_ciciot2023(n_per_class=args.n_per_class, random_state=args.seed)
    else:
        ds = data.load_edge_iiotset(n_per_class=args.n_per_class, random_state=args.seed)

    n_classes = len(ds.class_names)
    lgb_model = models.train_lightgbm(ds.X_train, ds.y_train, n_classes, random_state=args.seed)
    mlp = models.train_mlp(ds.X_train, ds.y_train, ds.X_train.shape[1], n_classes, random_state=args.seed)
    lgb_acc = accuracy_score(ds.y_test, lgb_model.predict(ds.X_test))
    mlp_acc = accuracy_score(ds.y_test, mlp.predict(ds.X_test))
    print(f"      lgb_acc={lgb_acc:.4f} mlp_acc={mlp_acc:.4f}")

    idx = rng.choice(ds.X_test.shape[0], size=min(args.n_explain, ds.X_test.shape[0]), replace=False)
    X_explain = ds.X_test[idx]
    y_lgb = lgb_model.predict(X_explain)
    y_mlp = mlp.predict(X_explain)

    print("[2/5] Building causal skeleton + ablation adjacencies...")
    causal_adj = causal_graph.build_pc_skeleton(ds.X_train, max_cond_set_size=1)
    corr_only_adj = causal_graph.build_pc_skeleton(ds.X_train, max_cond_set_size=0)
    effect_adj = causal_graph.build_effect_size_skeleton(ds.X_train, min_abs_r=0.3)
    random_adj = causal_graph.random_cliques_matching(effect_adj, rng)  # matched to the SPARSER effect-size partition

    def clique_sizes(adj):
        return sorted((len(c) for c in causal_graph.connected_components(adj)), reverse=True)

    causal_sizes = clique_sizes(causal_adj)
    effect_sizes = clique_sizes(effect_adj)
    print(f"      causal skeleton (alpha=0.05, order<=1): {int(causal_adj.sum() / 2)} edges, "
          f"{len(causal_sizes)} cliques, sizes={causal_sizes}")
    print(f"      corr-only (order-0): {int(corr_only_adj.sum() / 2)} edges, "
          f"{len(clique_sizes(corr_only_adj))} cliques, sizes={clique_sizes(corr_only_adj)}")
    print(f"      effect-size (|r|>=0.3, no significance test): {int(effect_adj.sum() / 2)} edges, "
          f"{len(effect_sizes)} cliques, sizes={effect_sizes}")
    print(f"      random (matched to effect-size partition): "
          f"{len(clique_sizes(random_adj))} cliques, sizes={clique_sizes(random_adj)}")

    print("[3/5] Base attributions (TreeSHAP/lgb, KernelSHAP/mlp)...")
    base_attr_lgb, _ = explainers.explain_treeshap(lgb_model, ds.X_train, X_explain)
    base_attr_mlp, _ = explainers.explain_kernelshap(mlp.predict_proba, ds.X_train, X_explain)

    variants = {
        "causal_shap": (causal_adj, causal_adj),
        "corr_only_ablation": (corr_only_adj, corr_only_adj),
        "effect_size_shap": (effect_adj, effect_adj),
        "random_clique_ablation": (random_adj, random_adj),  # matched to effect_adj's (sparser) partition
    }

    print("[4/5] Fidelity + sparsity + CI (clean inputs)...")
    rows = []
    sample_arrays = {}  # for later robustness-perturbation step
    for name, (adj_lgb, adj_mlp) in variants.items():
        for model_name, base_attr, X, y_pred, proba_fn in [
            ("lightgbm", base_attr_lgb, X_explain, y_lgb, lgb_model.predict_proba),
            ("mlp", base_attr_mlp, X_explain, y_mlp, mlp.predict_proba),
        ]:
            adj = adj_lgb if model_name == "lightgbm" else adj_mlp
            consolidated = consolidate(base_attr, adj)
            fid_arr = per_sample_fidelity(proba_fn, X, consolidated, y_pred)
            spar_arr = per_sample_sparsity(consolidated)
            fid_mean, fid_lo, fid_hi = bootstrap_ci(fid_arr)
            spar_mean, spar_lo, spar_hi = bootstrap_ci(spar_arr)
            rows.append({
                "variant": name, "model": model_name,
                "fidelity_mean": fid_mean, "fidelity_ci_lo": fid_lo, "fidelity_ci_hi": fid_hi,
                "sparsity_mean": spar_mean, "sparsity_ci_lo": spar_lo, "sparsity_ci_hi": spar_hi,
            })
            sample_arrays[(name, model_name)] = consolidated

    # also CI the KernelSHAP baseline itself (the "near-tied" comparator) for both models
    kernelshap_lgb, _ = explainers.explain_kernelshap(lgb_model.predict_proba, ds.X_train, X_explain)
    kernelshap_mlp = base_attr_mlp  # KernelSHAP IS the MLP base explainer already
    for model_name, attr, X, y_pred, proba_fn in [
        ("lightgbm", kernelshap_lgb, X_explain, y_lgb, lgb_model.predict_proba),
        ("mlp", kernelshap_mlp, X_explain, y_mlp, mlp.predict_proba),
    ]:
        fid_arr = per_sample_fidelity(proba_fn, X, attr, y_pred)
        spar_arr = per_sample_sparsity(attr)
        fid_mean, fid_lo, fid_hi = bootstrap_ci(fid_arr)
        spar_mean, spar_lo, spar_hi = bootstrap_ci(spar_arr)
        rows.append({
            "variant": "kernelshap_baseline", "model": model_name,
            "fidelity_mean": fid_mean, "fidelity_ci_lo": fid_lo, "fidelity_ci_hi": fid_hi,
            "sparsity_mean": spar_mean, "sparsity_ci_lo": spar_lo, "sparsity_ci_hi": spar_hi,
        })
        sample_arrays[("kernelshap_baseline", model_name)] = attr

    print(f"[5/5] Robustness under adversarial perturbation (eps={args.eps})...")
    X_adv_mlp = adversarial.deepfool_perturb(mlp.model, X_explain, y_mlp, epsilon=args.eps, n_classes=n_classes)
    X_adv_lgb = adversarial.boundary_perturb(lgb_model.predict_proba, X_explain, y_lgb, epsilon=args.eps, rng=rng)

    base_attr_lgb_adv, _ = explainers.explain_treeshap(lgb_model, ds.X_train, X_adv_lgb)
    base_attr_mlp_adv, _ = explainers.explain_kernelshap(mlp.predict_proba, ds.X_train, X_adv_mlp)
    kernelshap_lgb_adv, _ = explainers.explain_kernelshap(lgb_model.predict_proba, ds.X_train, X_adv_lgb)

    rob_rows = []
    for name, (adj_lgb, adj_mlp) in variants.items():
        cons_lgb_adv = consolidate(base_attr_lgb_adv, adj_lgb)
        cons_mlp_adv = consolidate(base_attr_mlp_adv, adj_mlp)
        rob_lgb = per_sample_robustness(sample_arrays[(name, "lightgbm")], cons_lgb_adv)
        rob_mlp = per_sample_robustness(sample_arrays[(name, "mlp")], cons_mlp_adv)
        for model_name, rob_arr in [("lightgbm", rob_lgb), ("mlp", rob_mlp)]:
            mean, lo, hi = bootstrap_ci(rob_arr)
            rob_rows.append({"variant": name, "model": model_name, "epsilon": args.eps,
                              "robustness_mean": mean, "robustness_ci_lo": lo, "robustness_ci_hi": hi})

    rob_kernelshap_lgb = per_sample_robustness(kernelshap_lgb, kernelshap_lgb_adv)
    rob_kernelshap_mlp = per_sample_robustness(kernelshap_mlp, base_attr_mlp_adv)
    for model_name, rob_arr in [("lightgbm", rob_kernelshap_lgb), ("mlp", rob_kernelshap_mlp)]:
        mean, lo, hi = bootstrap_ci(rob_arr)
        rob_rows.append({"variant": "kernelshap_baseline", "model": model_name, "epsilon": args.eps,
                          "robustness_mean": mean, "robustness_ci_lo": lo, "robustness_ci_hi": hi})

    df = pd.DataFrame(rows)
    rob_df = pd.DataFrame(rob_rows)
    merged = df.merge(rob_df, on=["variant", "model"], how="left")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"ablation_{args.data}_{stamp}.csv")
    merged.to_csv(out_path, index=False)

    clique_path = os.path.join(RESULTS_DIR, f"cliques_{args.data}_{stamp}.txt")
    with open(clique_path, "w") as f:
        f.write(f"causal skeleton (alpha=0.05, order<=1) clique sizes (descending): {causal_sizes}\n")
        f.write(f"n_cliques={len(causal_sizes)}, n_singletons={sum(1 for s in causal_sizes if s == 1)}, "
                f"largest={max(causal_sizes)}, n_features={causal_adj.shape[0]}\n")
        f.write(f"effect-size (|r|>=0.3) clique sizes (descending): {effect_sizes}\n")
        f.write(f"n_cliques={len(effect_sizes)}, n_singletons={sum(1 for s in effect_sizes if s == 1)}, "
                f"largest={max(effect_sizes)}, n_features={effect_adj.shape[0]}\n")

    print(f"\nSaved: {out_path}")
    print(f"Saved: {clique_path}")
    print("\n=== Summary ===")
    print(merged.round(3).to_string())


if __name__ == "__main__":
    main()
