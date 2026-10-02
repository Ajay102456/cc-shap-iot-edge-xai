# System Architecture & Experiment Process

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│  DATA LAYER (src/data.py)                                                │
│                                                                            │
│   CICIoT2023 (real)     Edge-IIoTset          Synthetic generator        │
│   8-class grouped,      (leakage-fixed         (pipeline dev/test        │
│   39 flow features      split required)        only, not for results)    │
│         └───────────────────┬───────────────────────┘                   │
│                              ▼                                           │
│              Dataset(X_train, X_test, y_train, y_test,                  │
│                       feature_names, class_names)                       │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  MODEL LAYER (src/models.py)                                             │
│                                                                            │
│   ┌────────────────────────┐        ┌───────────────────────────────┐   │
│   │  LightGBM                │        │  CompactMLP (torch, CPU)       │   │
│   │  150 trees, edge-scale   │        │  ~4-5k params -- edge-deploy   │   │
│   └────────────┬─────────────┘        └───────────────┬─────────────────┘   │
└────────────────┼──────────────────────────────────────┼──────────────────┘
                  ▼                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  EXPLAINABILITY LAYER (src/explainers.py)                                │
│                                                                            │
│  TreeSHAP  KernelSHAP  Lightweight-  LIME   CC-SHAP  ← novel contribution │
│  (exact,   (model-     KernelSHAP           (causal-consolidated,        │
│  LGBM      agnostic    (small bg+           post-processes a base        │
│  only)     baseline)   coalition cap)       explainer's attributions)    │
│                                                    │                      │
│                                                    ▼                      │
│                          src/causal_graph.py                            │
│                     bounded PC-skeleton (cond.≤1)                        │
│                learned ONCE offline from X_train → feature cliques       │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  RESOURCE-CAP SIMULATION (src/resource_harness.py)                       │
│                                                                            │
│  Each (explainer, model, tier) cell runs in an ISOLATED SPAWNED PROCESS: │
│                                                                            │
│    tier_pi4          tier_jetson_nano        tier_unconstrained          │
│    4 cores / 1.5GB    4 cores / 2.5GB         all cores / 8GB (baseline) │
│         └──────────────────┴──────────────────────┘                      │
│                             ▼                                            │
│      CPU-affinity cap + soft RLIMIT_AS memory cap + bounded timeout      │
│      (OOM / allocator retry-storm → recorded as a result, never hangs)  │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  ADVERSARIAL LAYER (src/adversarial.py)                                  │
│                                                                            │
│   MLP (differentiable):            LightGBM (non-differentiable):        │
│    - FGSM (fast, 1-step)            - boundary_perturb (black-box,       │
│    - DeepFool (true multiclass,       greedy, finite-difference          │
│      iterative, ε-clipped)            feature sensitivity)               │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  METRICS LAYER (src/metrics.py)                                          │
│                                                                            │
│   Fidelity  CCS(speed)  MFR(memory)  Temporal-Coherence  Sparsity  ROB   │
│                                    │                                      │
│                                    ▼                                      │
│      UMCEF_IoT = 0.4·FID + 0.3·EFF + 0.2·ROB + 0.1·SPAR   (Eq. 58)       │
└───────────────────────────────┬──────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  OUTPUT: results/metrics/{grid,robustness,umcef}_<source>_<ts>.csv       │
│          + console summary tables + UMCEF leaderboard                    │
└──────────────────────────────────────────────────────────────────────────┘
```

## Experiment Process (per `run_experiment.py` invocation)

```
 START: python -m src.run_experiment --data ciciot2023 [flags]
   │
   ▼
 ┌─ Load dataset ───────────────────────────────────────────────┐
 │  data.load_ciciot2023() → stratified subsample, 8-class      │
 │  group, z-score scale, train/test split                      │
 └───────────────────────────────┬───────────────────────────────┘
                                  ▼
 ┌─[1/4] Train models ───────────────────────────────────────────┐
 │  LightGBM.fit(X_train,y_train)   CompactMLP.fit(...)          │
 │  → print test accuracy for both                               │
 └───────────────────────────────┬───────────────────────────────┘
                                  ▼
 ┌─ Build CC-SHAP causal skeleton (once, offline) ────────────────┐
 │  causal_graph.build_pc_skeleton(X_train) → adjacency matrix    │
 └───────────────────────────────┬───────────────────────────────┘
                                  ▼
 ┌─[2/4] Explainer × Model × Tier grid ───────────────────────────┐
 │                                                                  │
 │  for explainer in {TreeSHAP, KernelSHAP, LW-KernelSHAP,        │
 │                     LIME, CC-SHAP}:                             │
 │    for model in {LightGBM, MLP}:        (skip incompatible)    │
 │      for tier in {pi4, jetson_nano, unconstrained}:             │
 │        ┌────────────────────────────────────────────┐          │
 │        │ spawn subprocess under resource cap          │          │
 │        │  → explain_dispatch(...) → attributions       │          │
 │        │  → measure: latency, peak memory, OOM/timeout │          │
 │        └────────────────────────────────────────────┘          │
 │        compute: fidelity, CCS, MFR, sparsity, temporal-coh.    │
 │        append row to grid_df                                    │
 └───────────────────────────────┬───────────────────────────────┘
                                  ▼
 ┌─[3/4] Adversarial robustness sweep ────────────────────────────┐
 │  for epsilon in {0.05, 0.15, 0.30}:                             │
 │    X_adv_mlp = DeepFool(MLP, X, y, epsilon)   [or FGSM]        │
 │    X_adv_lgb = boundary_perturb(LightGBM, X, y, epsilon)       │
 │    for explainer in {all 5}:                                    │
 │      for model in {compatible}:                                 │
 │        attr_clean = explain(X)      attr_adv = explain(X_adv)  │
 │        robustness = top-k feature-overlap(attr_clean, attr_adv)│
 │        append row to robustness_df                              │
 └───────────────────────────────┬───────────────────────────────┘
                                  ▼
 ┌─[4/4] UMCEF_IoT composite ─────────────────────────────────────┐
 │  join grid_df (FID, EFF, SPAR) + robustness_df (ROB) per        │
 │  explainer → UMCEF_IoT = 0.4FID+0.3EFF+0.2ROB+0.1SPAR           │
 │  → sort into leaderboard                                        │
 └───────────────────────────────┬───────────────────────────────┘
                                  ▼
 ┌─ Save + report ─────────────────────────────────────────────────┐
 │  results/metrics/grid_*.csv                                     │
 │  results/metrics/robustness_*.csv                               │
 │  results/metrics/umcef_*.csv                                    │
 │  console: summary table + robustness table + leaderboard         │
 └─────────────────────────────────────────────────────────────────┘
   END
```
