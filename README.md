# XAI for IoT Edge — Resource-Aware, Faithful Explainability Benchmark + CC-SHAP

Implements the experiment designed in `RESEARCH_IMPLEMENTATION_PLAN.md`, itself
built from the knowledge gap identified in Karras et al. (2026, *Future
Internet* 18(2):83) and cross-checked against a Consensus.app evidence scan
of the IoT-edge XAI literature. Two stacked contributions:

1. **It computes the metrics** (CCS, MFR, TC, UMCEF, Fidelity–Efficiency
   Frontier) that the source survey formalizes in equations but never runs
   on real data — benchmarking TreeSHAP, KernelSHAP, a lightweight
   KernelSHAP variant, and LIME under simulated edge resource caps and
   adversarial perturbation.
2. **It proposes CC-SHAP** (Causal-Consolidated SHAP, `src/causal_graph.py`
   + `explain_causal_shap` in `src/explainers.py`) — a new, lightweight,
   causally-informed explainer answering the field's explicit, repeated
   call for "causal, graph-based" methods tailored to network telemetry
   instead of generic tabular SHAP/LIME (Consensus report's Open Research
   Questions; Karras et al.'s "Causal Semantics" frontier, Fig. 6). See
   `RESEARCH_IMPLEMENTATION_PLAN.md` Part 7 for the full rationale,
   hypothesis (H4), and results. **Validated on two structurally different
   real datasets** — production-scale CICIoT2023 (balanced 8-class, 10,000
   samples/class, 100 explained samples, full-budget LIME, all 3 resource
   tiers, true multiclass DeepFool) and a generalization check on
   Edge-IIoTset (15-class, 42-feature, duplicate-leakage-corrected, §7.6):
   **on both, CC-SHAP is the most adversarially robust explainer of all
   five at every perturbation strength, and by far the sparsest.** It
   ranks 2nd on the UMCEF_IoT composite on CICIoT2023 (0.708, near-tied
   with the leader's 0.713) and **1st on Edge-IIoTset (0.769)** — a real,
   bounded fidelity trade-off reported honestly, not hidden, replicated
   across datasets rather than a single-dataset artifact.

## Status

- **Phase 1 (baseline models) — done and verified.** LightGBM + a compact
  (~5k param) PyTorch MLP, trained on real CICIoT2023 data in `src/models.py`.
- **Phase 2 (explainers + metrics) — done and verified.** TreeSHAP,
  KernelSHAP, a lightweight/sparsified KernelSHAP variant, LIME, and
  **CC-SHAP** (this project's own causal-consolidation explainer, see
  headline above) are wrapped behind one interface (`src/explainers.py`);
  CCS/MFR/TC/fidelity/sparsity/robustness/**UMCEF_IoT** are implemented in
  `src/metrics.py` and computed as a sorted leaderboard in `run_experiment.py`.
- **Phase 3 (robustness + resource caps) — done and verified.**
  `src/adversarial.py` implements FGSM and **true multiclass DeepFool**
  (`--adversarial-attack deepfool`, the default) for the MLP, and a
  black-box greedy attack for LightGBM. `src/resource_harness.py` runs
  each (explainer, model, tier) cell in an isolated, spawned subprocess
  with CPU-affinity capping and a soft `RLIMIT_AS` memory cap, so a hard
  native-allocator crash or timeout under a tight budget is recorded as an
  OOM event for that one cell instead of taking down the whole run.
  **In-process `RLIMIT_AS` caps below ~1.5GB are unreliable** (see the long
  comment in `resource_harness.py`: OpenBLAS/libgomp enter retry storms or
  `pthread_create` failures near a tight virtual-memory ceiling rather than
  raising a clean Python exception) — this is a real, documented
  methodological finding, not just an implementation detail, and it's why
  `docker/resource_cap.md`'s cgroup-based caps (kernel OOM killer, fails
  fast and deterministically) are the authoritative path for any numbers
  that go into the writeup, especially the literal 1GB Pi 4 / 2GB Jetson
  Nano budgets.

  **Known minor issue:** after `main()` prints its final summary and the
  three result CSVs are written (the run is fully complete and correct at
  that point), the top-level Python process has been observed to keep
  running for several more minutes at high CPU before exiting on its own
  -- no further log output, no further files written, just a slow
  interpreter-shutdown tail (likely multiprocessing resource-tracker/queue
  cleanup from the many spawned subprocesses). It's not a correctness
  issue -- the results are already safely on disk the moment the CSV
  paths print -- but don't be surprised if `ps aux` still shows the
  process alive well after you have your numbers; it's safe to
  `kill -9` it at that point rather than wait it out.
- **Phase 4 (real dataset + production-scale results) — done, on two datasets.**
  `src/data.py` loads the real CICIoT2023 release (34 per-class
  subdirectories, label-by-folder-name layout) with two data-quality fixes
  found and fixed during the first production runs: (a) per-class
  subsampling was originally applied per *raw* subfolder rather than per
  *final grouped label*, badly inflating multi-folder classes like DDoS
  relative to single-folder classes like Benign; (b) a small fraction of
  real rows have literal `inf` values in rate-style features (division by
  a duration that rounds to 0), which crashed `StandardScaler` until
  explicitly handled. Both are fixed and documented in
  `RESEARCH_IMPLEMENTATION_PLAN.md` §7.3.1. `load_edge_iiotset()` loads the
  official Edge-IIoTset "Selected dataset for ML and DL" release with a
  duplicate-row leakage fix (814/157,800 exact-duplicate rows dropped
  before splitting, §7.6). Synthetic data generation remains available for
  fast pipeline-development iteration only — **do not report
  synthetic-data numbers as findings.**

**Production-scale CICIoT2023 run completed 2026-10-01** (balanced
8-class, 10,000 samples/class, 100 explained samples, full-budget LIME,
all 3 resource tiers, true multiclass DeepFool) and the **Edge-IIoTset
generalization check completed 2026-10-02** (15-class, 42-feature,
duplicate-leakage-corrected) — see `RESEARCH_IMPLEMENTATION_PLAN.md` §7.3
and §7.6 for the full grid/robustness/UMCEF tables on each. Headline: on
**both** datasets, CC-SHAP is the most adversarially robust explainer of
all five at every perturbation strength tested and by far the sparsest;
it places a near-statistical-tie 2nd on CICIoT2023's UMCEF_IoT composite
(0.708 vs. 0.713) and **1st on Edge-IIoTset's (0.769)** — a trade-off
that replicates across two structurally different datasets, not a
single-dataset artifact.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Note: this repo was developed and smoke-tested on Python 3.14 (the only
interpreter available on this machine at setup time), not the Python 3.12
assumed in the research plan. All dependencies (`shap`, `lime`, `lightgbm`,
`torch` CPU build, `scikit-learn`) installed and ran cleanly on 3.14; the
`docker/Dockerfile` pins back to `python:3.12-slim` for the environment
actually targeted by the plan.

## Running

```bash
# Pipeline validation on synthetic data (fast, ~1 min)
python -m src.run_experiment --data synthetic --n-explain 30

# Real experiment (after downloading CICIoT2023 into data/raw/ciciot2023/,
# see src/data.py docstring for the expected layout)
python -m src.run_experiment --data ciciot2023 --n-per-class 5000 --n-explain 100
```

`--n-explain` controls how many test samples get explained per
(explainer, model, tier) cell. Keep it small for KernelSHAP variants —
they're the whole point of measuring the resource–interpretability gap,
and they are slow by design.

`--lime-num-samples` (default 5000, LIME's own library default) is the
dominant cost driver for LIME specifically: it's LIME's internal
per-instance perturbation budget, so explaining N instances costs
N × num_samples model predictions. On real CICIoT2023 data (39 features)
this made a full robustness sweep take ~25 minutes. Drop it to ~500 for
fast dev iteration; keep the 5000 default for numbers going in the
writeup, since LIME's local-fit stability depends on having enough
samples.

### Running long/background jobs

Production-scale runs take long enough to be worth detaching and logging
to `results/logs/` (grouped with the `results/metrics/` and
`results/figures/` they produce, instead of loose files in the project
root):

```bash
nohup python -u -m src.run_experiment --data ciciot2023 \
  --n-per-class 10000 --n-explain 100 --lime-num-samples 5000 \
  --tiers tier_pi4 tier_jetson_nano tier_unconstrained \
  --adversarial-attack deepfool \
  > results/logs/run_ciciot2023_$(date +%Y%m%d_%H%M%S).log 2>&1 &
disown
echo "Started, PID: $!"
```

`-u` keeps the log unbuffered (so `tail -f` shows real-time progress
instead of lagging behind); `disown` detaches it from the shell so closing
the terminal or an accidental Ctrl+C in that pane won't kill it — **note
this does not survive an actual machine reboot/sleep**, only
terminal-level interruptions. Check progress with
`tail -f results/logs/run_*.log`, confirm it's still alive with `ps aux |
grep run_experiment`, and expect the log filename's timestamp to roughly
match the `grid_*`/`robustness_*`/`umcef_*` CSVs it eventually produces in
`results/metrics/`.

## Repository layout

```
src/
  data.py              # CICIoT2023 / Edge-IIoTset loaders + synthetic fallback
  models.py            # LightGBM + CompactMLP (edge-representative sizes)
  explainers.py        # TreeSHAP, KernelSHAP, lightweight-KernelSHAP, LIME
  metrics.py           # CCS, MFR, TC, fidelity, sparsity, robustness, UMCEF_IoT
  adversarial.py       # FGSM (MLP) + black-box greedy attack (LightGBM)
  resource_harness.py  # CPU affinity + soft memory cap + timing/measurement
  run_experiment.py    # ties it all together, writes results/metrics/*.csv
docker/
  Dockerfile, resource_cap.md   # hard resource-cap runs for reported numbers
data/raw/              # place real datasets here (not committed)
results/metrics/       # experiment output CSVs (grid + robustness sweep + umcef)
results/figures/       # fef/bars/radar PDF+SVG per dataset
results/logs/          # background-run stdout logs (see "Running long/background jobs")
RESEARCH_IMPLEMENTATION_PLAN.md  # full plan: RQs, hypotheses, roadmap, risks
```

## Next steps (see plan Part 7 for full detail)

1. ~~Download CICIoT2023, run production-scale experiment, compute UMCEF leaderboard~~ — **done** (`RESEARCH_IMPLEMENTATION_PLAN.md` §7.3).
2. ~~Apply the Edge-IIoTset leakage fix, run cross-dataset generalization check~~ — **done** (§7.6; CC-SHAP ranks 1st on UMCEF_IoT there).
3. ~~Figures: FEF, grouped bar, and radar views, both datasets~~ — **done**
   (`src/plot_fef.py`, `src/plot_bars.py`, `src/plot_radar.py`;
   `results/figures/{fef,bars,radar}_*.{pdf,svg}`, §7.7). `bars_*` is the
   primary comparison figure — each explainer in its own fixed,
   colorblind-validated color. Notable finding: CC-SHAP is **not** on the
   2D FEF frontier on either dataset — it only wins once robustness and
   sparsity are in the picture (immediately visible in the bar/radar
   views), which is the paper's own point about why a 2D view is insufficient.
4. Switch reported resource-cap numbers to the Docker hard-cap runs
   (`docker/resource_cap.md`) for the literal 1GB/2GB Pi4/Jetson Nano
   budgets, since the in-process `RLIMIT_AS` caps are unreliable below ~1.5GB.
5. ~~Draft the manuscript~~ — **done**, targeting **Sensors (MDPI)** (see
   `MANUSCRIPT_DRAFT.md`'s manuscript-status note for the rationale vs.
   Future Internet / IEEE Access), using Part 7 of the plan (both §7.3 and
   §7.6) as the results section's backbone. Current draft: system
   architecture figure, 7 results/appendix tables, 4 figures, 47
   references (21 newly cross-checked against the local `Papers_md/`
   corpus), ~8,600 words of body text across an expanded Related Work
   (9 subsections) and a positioning table against prior IoT-XAI
   evaluation studies.
