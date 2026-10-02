"""Main experiment runner: model x explainer x resource-tier grid, with
fidelity, CCS, MFR, TC, adversarial robustness, and the UMCEF_IoT composite.

Usage:
    python -m src.run_experiment --data synthetic --n-explain 40
    python -m src.run_experiment --data ciciot2023 --n-per-class 5000

Writes a results CSV to results/metrics/ and prints a summary table.
"""
from __future__ import annotations

import os

# Must be set before numpy/torch/lightgbm are imported anywhere in this
# process: pins BLAS/OMP thread pools to 1 so the resource-harness child
# processes (which use os.fork()) never inherit a multi-threaded BLAS state
# that can deadlock post-fork (only the forking thread survives fork(); a
# lock held by another thread at fork time stays locked forever in the
# child). Single-threaded BLAS is also the realistic condition on a
# single/few-core edge device, so this is not just a workaround.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score

from src import adversarial, causal_graph, data, explainers, metrics, models
from src.resource_harness import RESOURCE_TIERS, run_under_tier

warnings.filterwarnings("ignore", category=ConvergenceWarning)  # LIME's internal ridge fit on tiny samples

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "metrics")


def build_explainer_jobs(ds: data.Dataset, lgb_model, mlp: models.TrainedMLP):
    """Returns a dict: explainer_name -> {model_name -> model_object}.

    Jobs are dispatched through explainers.explain_dispatch (a top-level,
    picklable function) rather than lambdas/closures, because the
    resource-harness subprocess uses `spawn` -- lambdas aren't picklable,
    and closures over live shap/model state proved unsafe even under the
    previously-tried `fork` (see resource_harness.py docstring).
    """
    jobs = {
        "treeshap": {"lightgbm": lgb_model},
        "kernelshap": {"lightgbm": lgb_model, "mlp": mlp},
        "lightweight_kernelshap": {"lightgbm": lgb_model, "mlp": mlp},
        "lime": {"lightgbm": lgb_model, "mlp": mlp},
        "causal_shap": {"lightgbm": lgb_model, "mlp": mlp},
    }
    return jobs


def run_grid(ds: data.Dataset, n_explain: int, tiers: list[str], seed: int = 42, lime_num_samples: int = 5000,
             mlp_attack: str = "deepfool") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_classes = len(ds.class_names)

    print(f"[1/4] Training models on {ds.source} "
          f"({ds.X_train.shape[0]} train / {ds.X_test.shape[0]} test, {ds.X_train.shape[1]} features)")
    lgb_model = models.train_lightgbm(ds.X_train, ds.y_train, n_classes, random_state=seed)
    mlp = models.train_mlp(ds.X_train, ds.y_train, ds.X_train.shape[1], n_classes, random_state=seed)

    lgb_acc = accuracy_score(ds.y_test, lgb_model.predict(ds.X_test))
    mlp_acc = accuracy_score(ds.y_test, mlp.predict(ds.X_test))
    print(f"      LightGBM test accuracy: {lgb_acc:.4f}")
    print(f"      CompactMLP test accuracy: {mlp_acc:.4f} ({mlp.model.param_count()} params)")

    idx = rng.choice(ds.X_test.shape[0], size=min(n_explain, ds.X_test.shape[0]), replace=False)
    X_explain = ds.X_test[idx]
    y_lgb = lgb_model.predict(X_explain)
    y_mlp = mlp.predict(X_explain)

    print("      Building CC-SHAP causal skeleton (bounded PC, once, offline)...")
    t0 = time.perf_counter()
    causal_adjacency = causal_graph.build_pc_skeleton(ds.X_train)
    n_edges = int(causal_adjacency.sum() / 2)
    n_features = causal_adjacency.shape[0]
    print(f"      skeleton: {n_edges} edges over {n_features} features "
          f"({time.perf_counter() - t0:.2f}s, one-time offline cost)")

    jobs = build_explainer_jobs(ds, lgb_model, mlp)
    reference_times = {}  # explainer_name -> fastest observed time, for CCS

    rows = []
    print("[2/4] Running explainer x model x resource-tier grid...")
    for explainer_name, model_jobs in jobs.items():
        for model_name, model_obj in model_jobs.items():
            X_ex = X_explain
            y_pred = y_lgb if model_name == "lightgbm" else y_mlp
            proba_fn = lgb_model.predict_proba if model_name == "lightgbm" else mlp.predict_proba

            for tier_name in tiers:
                tier = RESOURCE_TIERS[tier_name]
                print(f"      {explainer_name:24s} | {model_name:10s} | {tier_name}")

                (attr_and_time, run_result) = run_under_tier(
                    explainers.explain_dispatch, tier,
                    explainer_name, model_name, model_obj, ds.X_train, X_ex,
                    feature_names=ds.feature_names, class_names=ds.class_names,
                    lime_num_samples=lime_num_samples, causal_adjacency=causal_adjacency,
                )
                if run_result.error or attr_and_time is None:
                    rows.append({
                        "explainer": explainer_name, "model": model_name, "tier": tier_name,
                        "status": "FAILED", "error": run_result.error,
                        "elapsed_s": run_result.elapsed_s, "peak_memory_mb": run_result.peak_memory_mb,
                    })
                    continue

                attributions, explain_time_s = attr_and_time
                key = (explainer_name,)
                reference_times[key] = min(reference_times.get(key, explain_time_s), explain_time_s)

                fid = metrics.deletion_insertion_fidelity(proba_fn, X_ex, attributions, y_pred)
                ccs = metrics.computational_complexity_score(explain_time_s, reference_times[key])
                mfr = metrics.memory_footprint_ratio(run_result.peak_memory_mb, tier.memory_mb)
                spar = metrics.sparsity_score(attributions)

                # temporal coherence over the explained batch as a proxy sliding window
                tc = metrics.temporal_coherence(attributions)

                rows.append({
                    "explainer": explainer_name, "model": model_name, "tier": tier_name,
                    "status": "OK",
                    "elapsed_s": explain_time_s,
                    "peak_memory_mb": run_result.peak_memory_mb,
                    "fidelity": fid["fidelity"],
                    "deletion_auc": fid["deletion_auc"],
                    "insertion_auc": fid["insertion_auc"],
                    "ccs": ccs,
                    "mfr": mfr,
                    "sparsity": spar,
                    "temporal_coherence": tc,
                    "attributions_npy": attributions,  # kept in-memory only, stripped before CSV export
                })

    df = pd.DataFrame(rows)

    print(f"[3/4] Robustness under adversarial perturbation (epsilon sweep, mlp_attack={mlp_attack})...")
    robustness_rows = run_robustness_sweep(ds, lgb_model, mlp, jobs, X_explain, y_lgb, y_mlp, rng,
                                            lime_num_samples=lime_num_samples, causal_adjacency=causal_adjacency,
                                            mlp_attack=mlp_attack, n_classes=n_classes)
    robustness_df = pd.DataFrame(robustness_rows)

    print("[4/4] Computing UMCEF_IoT composite...")
    umcef_df = compute_umcef_leaderboard(df, robustness_df)

    return df, robustness_df, umcef_df


def run_robustness_sweep(ds, lgb_model, mlp, jobs, X_explain, y_lgb, y_mlp, rng, epsilons=(0.05, 0.15, 0.3),
                          lime_num_samples=5000, causal_adjacency=None, mlp_attack="deepfool", n_classes=None):
    rows = []
    for eps in epsilons:
        if mlp_attack == "deepfool":
            X_adv_mlp = adversarial.deepfool_perturb(mlp.model, X_explain, y_mlp, epsilon=eps, n_classes=n_classes)
        else:
            X_adv_mlp = adversarial.fgsm_perturb(mlp.model, X_explain, y_mlp, epsilon=eps)
        X_adv_lgb = adversarial.boundary_perturb(lgb_model.predict_proba, X_explain, y_lgb, epsilon=eps, rng=rng)

        acc_mlp_clean = accuracy_score(y_mlp, mlp.predict(X_explain))
        acc_mlp_adv = accuracy_score(y_mlp, mlp.predict(X_adv_mlp))
        acc_lgb_clean = accuracy_score(y_lgb, lgb_model.predict(X_explain))
        acc_lgb_adv = accuracy_score(y_lgb, lgb_model.predict(X_adv_lgb))

        for explainer_name in ["treeshap", "kernelshap", "lightweight_kernelshap", "lime", "causal_shap"]:
            for model_name, X_clean, X_adv, y_pred, acc_drop in [
                ("mlp", X_explain, X_adv_mlp, y_mlp, acc_mlp_clean - acc_mlp_adv),
                ("lightgbm", X_explain, X_adv_lgb, y_lgb, acc_lgb_clean - acc_lgb_adv),
            ]:
                if model_name not in jobs.get(explainer_name, {}):
                    continue
                model_obj = jobs[explainer_name][model_name]
                try:
                    attr_clean, _ = explainers.explain_dispatch(
                        explainer_name, model_name, model_obj, ds.X_train, X_clean,
                        feature_names=ds.feature_names, class_names=ds.class_names,
                        lime_num_samples=lime_num_samples, causal_adjacency=causal_adjacency,
                    )
                    attr_adv, _ = explainers.explain_dispatch(
                        explainer_name, model_name, model_obj, ds.X_train, X_adv,
                        feature_names=ds.feature_names, class_names=ds.class_names,
                        lime_num_samples=lime_num_samples, causal_adjacency=causal_adjacency,
                    )
                except Exception as e:  # noqa: BLE001
                    rows.append({"explainer": explainer_name, "model": model_name, "epsilon": eps,
                                 "status": "FAILED", "error": str(e)})
                    continue
                rob = metrics.explanation_robustness(attr_clean, attr_adv)
                rows.append({
                    "explainer": explainer_name, "model": model_name, "epsilon": eps,
                    "status": "OK",
                    "explanation_robustness": rob,
                    "model_accuracy_drop": acc_drop,
                    "attack": mlp_attack if model_name == "mlp" else "boundary_blackbox",
                })
    return rows


def compute_umcef_leaderboard(grid_df: pd.DataFrame, robustness_df: pd.DataFrame) -> pd.DataFrame:
    """UMCEF_IoT (Eq. 58, Karras et al. 2026): 0.4*FID + 0.3*EFF + 0.2*ROB + 0.1*SPAR,
    aggregated per explainer (mean across models/tiers for FID/EFF/SPAR, mean
    across models/epsilons for ROB) so the fidelity-vs-robustness-vs-sparsity
    trade-off (e.g. CC-SHAP's real result: high sparsity+robustness, lower
    fidelity) is visible as ONE transparent composite number per explainer,
    not cherry-picked from whichever raw metric looks best.
    """
    if grid_df.empty:
        return pd.DataFrame()
    ok = grid_df[grid_df["status"] == "OK"].copy()
    if ok.empty:
        return pd.DataFrame()
    ok["efficiency"] = (ok["ccs"] + ok["mfr"]) / 2.0

    fid_eff_spar = ok.groupby("explainer")[["fidelity", "efficiency", "sparsity"]].mean()
    fid_eff_spar.columns = ["FID", "EFF", "SPAR"]

    if not robustness_df.empty and "status" in robustness_df.columns:
        ok_rob = robustness_df[robustness_df["status"] == "OK"]
        rob_by_explainer = ok_rob.groupby("explainer")["explanation_robustness"].mean().rename("ROB")
    else:
        rob_by_explainer = pd.Series(dtype=float, name="ROB")

    leaderboard = fid_eff_spar.join(rob_by_explainer, how="left")
    leaderboard["ROB"] = leaderboard["ROB"].fillna(0.0)  # no robustness data -> conservative (not "trustworthy")

    leaderboard["UMCEF_IoT"] = (
        0.4 * leaderboard["FID"] + 0.3 * leaderboard["EFF"] + 0.2 * leaderboard["ROB"] + 0.1 * leaderboard["SPAR"]
    )
    return leaderboard.sort_values("UMCEF_IoT", ascending=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic", "ciciot2023", "edge-iiotset"], default="synthetic")
    parser.add_argument("--n-per-class", type=int, default=5000)
    parser.add_argument("--label-grouping", choices=["8class", "raw"], default="8class",
                         help="CICIoT2023 only: 8class (default, matches published benchmarks) or raw (all 34 categories)")
    parser.add_argument("--n-explain", type=int, default=30, help="# test samples to explain (keep small: KernelSHAP is O(2^M)-ish)")
    parser.add_argument("--tiers", nargs="+", default=["tier_pi4", "tier_jetson_nano", "tier_unconstrained"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lime-num-samples", type=int, default=5000,
                         help="LIME's internal per-instance perturbation budget (library default 5000). "
                              "This is the dominant cost driver for LIME -- lower it (e.g. 500) for fast "
                              "dev iteration; keep 5000 for numbers that go in the writeup.")
    parser.add_argument("--adversarial-attack", choices=["fgsm", "deepfool"], default="deepfool",
                         help="MLP robustness-sweep attack. 'deepfool' (default) is the true iterative "
                              "multiclass DeepFool used by Munilla & Khammas (2026), for direct comparability "
                              "to that paper. 'fgsm' is faster (single gradient step) for quick dev iteration. "
                              "LightGBM always uses the black-box boundary attack (DeepFool needs gradients).")
    args = parser.parse_args()

    if args.data == "synthetic":
        ds = data.make_synthetic_iot_traffic(random_state=args.seed)
        warnings.warn("Using SYNTHETIC data -- pipeline validation only, not a real experiment result.")
    elif args.data == "ciciot2023":
        ds = data.load_ciciot2023(n_per_class=args.n_per_class, label_grouping=args.label_grouping, random_state=args.seed)
    else:
        ds = data.load_edge_iiotset(n_per_class=args.n_per_class, random_state=args.seed)

    df, robustness_df, umcef_df = run_grid(ds, n_explain=args.n_explain, tiers=args.tiers, seed=args.seed,
                                            lime_num_samples=args.lime_num_samples, mlp_attack=args.adversarial_attack)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = os.path.join(RESULTS_DIR, f"grid_{ds.source}_{stamp}.csv")
    robustness_path = os.path.join(RESULTS_DIR, f"robustness_{ds.source}_{stamp}.csv")
    umcef_path = os.path.join(RESULTS_DIR, f"umcef_{ds.source}_{stamp}.csv")

    df.drop(columns=["attributions_npy"], errors="ignore").to_csv(metrics_path, index=False)
    robustness_df.to_csv(robustness_path, index=False)
    umcef_df.to_csv(umcef_path)

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {robustness_path}")
    print(f"Saved: {umcef_path}")
    print("\n=== Summary (mean fidelity / ccs / mfr / tc by explainer, OK rows only) ===")
    ok_df = df[df["status"] == "OK"]
    if not ok_df.empty:
        summary = ok_df.groupby("explainer")[["fidelity", "ccs", "mfr", "sparsity", "temporal_coherence"]].mean()
        print(summary.round(3).to_string())

    print("\n=== Robustness summary (mean explanation_robustness by explainer/epsilon) ===")
    ok_rob = robustness_df[robustness_df.get("status") == "OK"] if not robustness_df.empty else robustness_df
    if ok_rob is not None and not ok_rob.empty:
        rob_summary = ok_rob.groupby(["explainer", "epsilon"])["explanation_robustness"].mean()
        print(rob_summary.round(3).to_string())

    print("\n=== UMCEF_IoT leaderboard (0.4*FID + 0.3*EFF + 0.2*ROB + 0.1*SPAR, Eq. 58) ===")
    if not umcef_df.empty:
        print(umcef_df.round(3).to_string())


if __name__ == "__main__":
    main()
