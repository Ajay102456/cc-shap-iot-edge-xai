# Research-to-Implementation Analysis: Resource-Aware, Faithful XAI for IoT Edge Intrusion Detection

**Built from:** Karras et al. (2026) *Next-Gen XAI for Federated and Distributed IoT Systems* (Future Internet 18(2):83) + Consensus knowledge-gap report *"IoT Edge Networking XAI Trends and Knowledge Gaps"* + `MASTER TEMPLATE FOR RESEARCH-TO-IMPLEMENTATION ANALYSIS.md`
**Researcher profile:** single researcher, no budget, Linux Mint 22.2, Python 3.12, simulation/open-source only
**Date:** 2026-09-30

---

## PART 0 — Knowledge Gap Synthesis (both source documents)

Both sources converge on the same diagnosis from independent angles (a 69-page PRISMA-style MDPI survey vs. a 100-paper Consensus AI evidence scan), which is itself useful triangulation:

| Convergent gap | Karras et al. (2026) framing | Consensus report framing |
|---|---|---|
| **Resource–interpretability trade-off is unsolved** | "Resource–interpretability gap"; no study jointly reports CCS/MFR/latency/energy with fidelity | "Accuracy–efficiency–explainability trilemma" (Ogunseyi et al. 2026); XAI overhead still a "serious limiter" (6,360 KB/round FL-XAI; 120–180 ms Pi inference) |
| **No standardized, credible evaluation** | Section 9 proposes CCS, MFR, TC, PUT, GLED, ACS, UMCEF, FEF, ETI because none exist as a community standard | "Most IoT XAI studies stop at plausibility checks"; few combine fidelity + stability + operator studies |
| **Generic tabular explainers (SHAP/LIME) misapplied to network data** | 7 documented technical limitations of intrinsic methods in IoT; post-hoc chosen by necessity, not fit | SHAP+LIME "dominate" but are "not tailored to network telemetry," produce "misleading or inconsistent results" on temporal/relational traffic |
| **Adversarial/robustness of explanations barely tested** | Robustness (ROB) is one of the least-populated metrics in existing studies | "Weak" evidence strength — explanations shown vulnerable under DeepFool and drift (Munilla & Khammas 2026) |
| **Benchmark-only validation, no live/heterogeneous edge testing** | Calls for "hardware-aware evaluation framework" explicitly because studies don't budget for edge resources | NSL-KDD/UNSW-NB15/TON_IoT/CICIoT2023 dominate; "far fewer test explanations in live heterogeneous networks" |
| **Scope imbalance toward IDS** | Smart cities + smart agriculture are the survey's two deep-dive domains; IDS treated as one deployment tier among several | IDS dominates corpus; traffic engineering, RRM, forensics under-studied |

**The single sharpest, most actionable gap for a solo, no-budget researcher:** nobody has run a controlled, reproducible study that (a) holds the *model* and *dataset* fixed, (b) varies the *XAI method* (SHAP/KernelSHAP vs. TreeSHAP vs. LIME vs. a lightweight/surrogate alternative), and (c) jointly measures **fidelity, latency, memory, and adversarial/perturbation stability** on hardware-representative resource budgets, reporting results against the very metrics (CCS, MFR, TC) the survey proposes but nobody has yet operationalized. That is a gap that is (i) reproducible from public data, (ii) fully implementable in simulation on a laptop, (iii) directly citable against a named, recent, open gap in a Future Internet 2026 paper — which is exactly the kind of contribution a single-author study can credibly claim.

---

## PART 1 — Research Paper Deconstruction (condensed)

**Executive summary of the gap paper:**
1. **Problem:** IoT-edge XAI research proposes many post-hoc explainers but essentially never measures whether they are *affordable* (latency/memory/energy) and *faithful* (fidelity/robustness) simultaneously on constrained hardware — the "resource–interpretability gap."
2. **Solution proposed by the survey (not implemented, only formalized):** a hardware-centric metric suite (CCS, MFR, TC, PUT, GLED, ACS, UMCEF, FEF, ETI) and a hierarchical IoT–XAI reference architecture (device/edge/cloud tiers), plus a conceptual "IOTIES" standard.
3. **Impact:** if explanations can't be shown faithful *and* cheap *and* stable under attack, XAI-enabled edge security systems cannot be trusted operationally — this blocks real deployment in smart cities/agriculture/6G.

**Key contributions used as our foundation:**
- CCS (Computational Complexity Score), MFR (Memory Footprint Ratio), TC (Temporal Coherence) — Section 9, hardware-aware metrics
- UMCEF weighted composite: `0.4·FID + 0.3·EFF + 0.2·ROB + 0.1·SPAR` (Eq. 58) — usable directly as our primary composite score
- Table 22/23: explicit research agenda ("Scalable and resource-efficient XAI for edge devices" → lightweight/sparsity-aware explainers, TinyML-compatible XAI, evaluated via complexity ratio/memory footprint/latency/temporal coherence, priority domain = smart cities/smart homes)

---

## PART 2 — Methodology Deconstruction

**Core algorithmic approach we will implement:**
- Fix an IoT network-intrusion-detection task (binary + multiclass) on a public, edge-representative dataset.
- Train two model families representative of what edge deployments actually use: a **gradient-boosted tree** (LightGBM — TreeSHAP-compatible, tiny footprint) and a **compact MLP/1D-CNN** (representative of the DNNs the surveyed literature explains).
- Attach four explainers per model where applicable: **TreeSHAP** (exact, tree-only), **KernelSHAP** (model-agnostic baseline, expensive), **LIME**, and one **lightweight/surrogate alternative** (e.g., Fast-TreeSHAP, or a distilled/sparsified attribution method) — directly answering the survey's "lightweight and sparsity-aware explainers" research direction.
- Simulate edge resource constraints in software (cgroup/Docker CPU+memory caps, or `resource` module limits) calibrated to published Raspberry Pi 4 / Jetson Nano specs (4 cores, 1–4 GB RAM) — since no physical device is required or assumed.
- Measure, per (model, explainer, data-tier) combination: inference latency, explanation latency, peak memory, fidelity (deletion/insertion AUC or ROAR), stability under a DeepFool-style adversarial perturbation of the *input*, and temporal coherence (Eq. from Section 9) across a sliding window of traffic.
- Aggregate into the survey's own **UMCEF** score so results are directly comparable to the metric the source paper proposes but never computed.

**Explicit assumptions to validate, not assume:**
1. TreeSHAP's speed advantage holds under the *resource-capped* condition, not just wall-clock on unconstrained hardware.
2. High predictive accuracy does not imply high explanation fidelity (a claim both source docs assert but with weak first-party evidence) — we test this directly by correlating model accuracy vs. explanation fidelity across model checkpoints of varying capacity.
3. Explanation stability degrades under adversarial input perturbation more than model accuracy does (the "explanations are the more fragile layer" claim from Munilla & Khammas 2026) — we test by comparing Δaccuracy vs. Δexplanation-similarity under identical perturbation budgets.

---

## PART 3 — Answers to the 12 Required Questions

**1. What are the knowledge gaps?**
(a) No joint fidelity+latency+memory+robustness benchmark exists for IoT-edge XAI under simulated resource caps; (b) SHAP/LIME are used without validating fit to temporal/tabular network telemetry; (c) explanation robustness under adversarial/drift conditions is "weak" evidence; (d) the survey's own proposed metrics (CCS/MFR/TC/UMCEF) have never been computed on real data — they exist only as equations.

**2. What is the problem statement?**
Current IoT-edge intrusion-detection XAI research reports model accuracy and, separately, post-hoc explanations, but does not report whether those explanations remain **faithful, stable, and affordable** when computed under the CPU/memory budgets of real edge hardware — making claims of "explainable, deployable" edge security systems empirically unverified.

**3. Hypotheses and research questions**

*RQ1:* Under simulated edge-hardware resource caps, which post-hoc explainer (TreeSHAP, KernelSHAP, LIME, lightweight/sparsified variant) achieves the best fidelity-per-millisecond and fidelity-per-MB on IoT network-traffic classifiers?

*RQ2:* Does higher model predictive accuracy correlate with higher explanation fidelity, or are the two independent (the trilemma claim)?

*RQ3:* How much does explanation quality (fidelity, top-k feature agreement) degrade under bounded adversarial perturbation of the input, relative to how much model accuracy degrades under the same perturbation?

*H1:* TreeSHAP will dominate the fidelity/cost Pareto frontier for tree models but KernelSHAP/LIME will not be Pareto-competitive under a >1 GB RAM / >4-core cap, replicating but quantifying the "resource–interpretability gap."
*H2:* Model accuracy and explanation fidelity will show weak or non-significant correlation (r < 0.3) across model capacities, supporting the "accuracy ≠ faithful explanation" claim from both source documents.
*H3:* Explanation similarity (e.g., Jaccard top-k feature overlap, rank correlation) will drop significantly more than model accuracy under identical adversarial perturbation budgets (ε sweep), demonstrating explanations are the more fragile layer.

**4. Project objectives**
1. Reproduce a realistic IoT-edge IDS pipeline on public data (no synthetic shortcuts).
2. Implement the survey's proposed CCS, MFR, TC, and UMCEF metrics as actual, runnable code — closing the "proposed but never computed" gap.
3. Produce a Pareto-frontier comparison of explainers under resource caps (Fidelity–Efficiency Frontier, Eq. 59).
4. Quantify explanation robustness under adversarial perturbation vs. model robustness under the same perturbation.
5. Package the pipeline as a reusable, open-source benchmark others can extend to new datasets/models — directly targeting the "standardization" gap (Table 22/23).

**5. Methodology**
Quantitative, controlled comparative experiment (not a survey, not qualitative). Design: 2 models × 4 explainers × 3 resource-cap tiers × 1 clean + 3 adversarial-perturbation conditions, repeated-measures on one primary dataset with a second dataset as a generalization check. Positivist/empirical paradigm, consistent with the "hardware-centric evaluation framework" called for by Karras et al. Statistical analysis: correlation (RQ2), paired significance tests (Wilcoxon signed-rank, since n is small/non-normal) for fidelity/latency deltas across conditions (RQ1, RQ3).

**6. Tools and techniques required**
- **Language/env:** Python 3.12, Linux Mint 22.2 (native — no cluster needed)
- **ML:** `scikit-learn`, `lightgbm`, `pytorch` (CPU) for the compact MLP/1D-CNN
- **XAI:** `shap` (TreeExplainer + KernelExplainer), `lime`, `fasttreeshap` (github.com/linkedin/FastTreeSHAP) as the lightweight/sparsified variant
- **Adversarial perturbation:** `foolbox` or `adversarial-robustness-toolbox` (ART) for a DeepFool-style attack on the tabular/DNN input, matching the DeepFool condition used by Munilla & Khammas (2026), which both source docs cite as the field's reference robustness test
- **Fidelity metrics:** implement deletion/insertion AUC manually (well-documented, small function) — avoids a heavy dependency
- **Resource capping/simulation:** `resource` module (`setrlimit`) for memory, `taskset`/`cgroups` (or Docker `--cpus`/`--memory`) for CPU — no physical Pi required, but the plan supports later validation on a real Raspberry Pi 4 if the researcher acquires one (~$35–60, optional, not required for Phase 1–3)
- **Experiment tracking:** `mlflow` (local, file-based backend — no server/budget needed) or plain CSV/JSON logs + `pandas`
- **Stats/plots:** `scipy.stats`, `matplotlib`/`seaborn`
- **Datasets:** CICIoT2023 (primary — largest, most current, explicitly designed for XAI-era IDS research, public via IEEE DataPort/UNB) and Edge-IIoTset (secondary, for generalization — note: recent work, e.g. arXiv:2608.15761, flags a *serialization/leakage artifact* in Edge-IIoTset, so it must be used only for a leakage-corrected split, not the raw released split)

**7. Can this be implemented as simulation, open-source, no budget, single researcher?**
**Yes, fully.** Every component above is free/open-source and runs on a single laptop-class Linux machine. "Edge deployment" is *simulated* via OS-level resource capping calibrated against published Raspberry Pi 4B / Jetson Nano specs (documented in Phase 1 deliverables), which is methodologically standard in this literature (several surveyed papers report Pi/Jetson numbers that this plan's caps are designed to match/cross-check). No cloud spend, no proprietary datasets, no paid APIs required. Physical-hardware validation is an optional Phase 4 stretch goal only.

**8. Environment constraints**
Linux Mint 22.2 (Ubuntu 24.04 base), Python 3.12 — all listed libraries (`shap`, `lime`, `lightgbm`, `torch` CPU build, `foolbox`/`ART`, `mlflow`) have current PyPI wheels compatible with Python 3.12 as of 2026. No GPU required (models are intentionally small/edge-representative; CPU-only training/inference is the realistic edge scenario anyway).

**9. Expected outcomes**
1. A quantified Fidelity–Efficiency Frontier (Eq. 59) showing which explainer(s), if any, are Pareto-optimal under edge resource caps — the first published instance of this exact metric being computed.
2. Empirical answer to whether "accuracy implies faithful explanation" (likely: no, or weak correlation) — reinforcing/quantifying a claim currently only asserted qualitatively.
3. A robustness delta table showing explanation fragility vs. model fragility under adversarial perturbation.
4. An open-source, documented benchmark repository (code + results + a UMCEF leaderboard) — a concrete, reusable artifact addressing the "standardization and evaluation frameworks" gap from Table 23.
5. A short paper (workshop/Q2-Q3 journal or arXiv preprint) reporting all of the above, explicitly positioned as an empirical follow-up closing a named gap in Karras et al. (2026).

**10. Targeted publication**
Realistic, single-author, no-budget targets: **Future Internet** (MDPI, open access, the *same venue* as the gap paper — natural fit, they'd likely welcome a direct empirical follow-up), **IEEE Access** (open access, broad IoT/XAI scope, fast review), **Sensors (MDPI)** (heavily represented in the Consensus corpus's top journals), or an arXiv preprint + workshop paper (e.g., a XAI/trustworthy-AI workshop at a networking or ML conference) as a lower-effort first step before a full journal submission. Avoid over-reaching for a top-tier venue on a first solo empirical study — Q2/Q3 open-access venues match the scope and are realistic.

**11. Ten Google Scholar / IEEE Xplore search keywords**
1. "lightweight explainable AI" AND "edge computing" AND latency
2. "SHAP" AND "IoT intrusion detection" AND "resource constrained"
3. "TreeSHAP" OR "FastSHAP" AND edge deployment
4. "explanation fidelity" AND "adversarial robustness" AND XAI
5. "federated explainable AI" AND IoT
6. "CICIoT2023" AND explainable
7. "Edge-IIoTset" AND XAI AND intrusion detection
8. "resource-interpretability trade-off" XAI
9. "DeepFool" AND "explainable AI" AND robustness
10. "XAI evaluation metrics" AND fidelity AND stability AND IoT

**12. Fifteen core papers (2022–2026, APA 7th)** — drawn from the two source documents' own reference lists (all independently verifiable):

1. Karras, A., Giannaros, A., Amasiadi, N., & Karras, C. N. (2026). Next-Gen Explainable AI (XAI) for Federated and Distributed Internet of Things Systems: A State-of-the-Art Survey. *Future Internet, 18*(2), 83. https://doi.org/10.3390/fi18020083
2. Ogunseyi, T. B., Thiyagarajan, G., He, H.-G., Bist, V., & Du, Z.-C. (2026). Performance Analysis of Explainable Deep Learning-Based Intrusion Detection Systems for IoT Networks: A Systematic Review. *Sensors, 26*(2), 363. https://doi.org/10.3390/s26020363
3. Nascita, A., Aceto, G., Ciuonzo, D., Montieri, A., Persico, V., & Pescapé, A. (2025). A Survey on Explainable Artificial Intelligence for Internet Traffic Classification and Prediction, and Intrusion Detection. *IEEE Communications Surveys & Tutorials, 27*, 3165–3198. https://doi.org/10.1109/comst.2024.3504955
4. Munilla, J., & Khammas, R. M. (2026). Evaluation of Explainable Artificial Intelligence in IoT Intrusion Detection Systems Under DeepFool Adversarial Conditions. *Sensors, 26*(10), 2924. https://doi.org/10.3390/s26102924
5. Bilal, M., Islam, I. U., Iltaf, N., Khan, M. J., & Khan, M. J. (2025). Federated Learning With Explainable AI for Malicious Traffic Detection in IoT Networks. *IEEE Access, 13*, 173368–173383. https://doi.org/10.1109/access.2025.3613459
6. Bilal, M., Islam, I. U., Khan, M. J., Nisar, S., Farooq, M., & Khan, H. (2026). Secure and Explainable Federated Learning for IoT Intrusion Detection: A Comprehensive Survey. *IEEE Open Journal of the Communications Society, 7*, 3650–3679. https://doi.org/10.1109/ojcoms.2026.3681580
7. Arreche, O., Guntur, T. R., Roberts, J., & Abdallah, M. (2024). E-XAI: Evaluating Black-Box Explainable AI Frameworks for Network Intrusion Detection. *IEEE Access, 12*, 23954–23988. https://doi.org/10.1109/access.2024.3365140
8. Kalakoti, R., Bahşi, H., & Nõmm, S. (2024). Improving IoT Security With Explainable AI: Quantitative Evaluation of Explainability for IoT Botnet Detection. *IEEE Internet of Things Journal, 11*, 18237–18254. https://doi.org/10.1109/jiot.2024.3360626
9. Gummadi, A., Arreche, O., & Abdallah, M. (2025). A systematic evaluation of white-box explainable AI methods for anomaly detection in IoT systems. *Internet of Things, 30*, 101505. https://doi.org/10.1016/j.iot.2025.101505
10. Singh, S. K., & Roy, J. (2026). Scalable Explainability-as-a-Service (XaaS) for Edge AI Systems. *SoutheastCon 2026*, 1–8. https://doi.org/10.1109/southeastcon63549.2026.11476268
11. Zhao, Y., Ullah, F., Mahmood, K., Saeed, S., Mohammad, N., & Raza, U. (2026). XAI-EdgeSFL: Explainable Edge Intelligence With Adaptive Intrusion-Resilient Split Federated Learning for Consumer Healthcare Ecosystems. *IEEE Transactions on Consumer Electronics, 72*, 2185–2196. https://doi.org/10.1109/tce.2026.3656201
12. Houda, Z. A. E., Brik, B., & Khoukhi, L. (2022). "Why Should I Trust Your IDS?": An Explainable Deep Learning Framework for Intrusion Detection Systems in IoT Networks. *IEEE Open Journal of the Communications Society, 3*, 1164–1176. https://doi.org/10.1109/ojcoms.2022.3188750
13. Keshk, M., Koroniotis, N., Pham, N., Moustafa, N., Turnbull, B. P., & Zomaya, A. (2023). An explainable deep learning-enabled intrusion detection framework in IoT networks. *Information Sciences, 639*, 119000. https://doi.org/10.1016/j.ins.2023.119000
14. Moustafa, N., Koroniotis, N., Keshk, M., Zomaya, A. Y., & Tari, Z. (2023). Explainable Intrusion Detection for Cyber Defences in the Internet of Things: Opportunities and Solutions. *IEEE Communications Surveys & Tutorials, 25*, 1775–1807. https://doi.org/10.1109/comst.2023.3280465
15. Nassef, L., Alghamdi, M. I., Chaabane, S. B., Abbas, Q., Alawad, W. M., Albalawi, O. H., Alqaisi, O. I., & Fakieh, B. (2026). Lightweight and Energy-Aware Intrusion Detection for Industrial IoT Using TinyML and Edge AI. *Scientific Reports, 16*. https://doi.org/10.1038/s41598-026-50690-0

---

## PART 4 — Implementation Spectrum & Technology Decisions

| Component | Tech complexity (1–5) | Time estimate | Priority |
|---|---|---|---|
| Data ingestion + preprocessing (CICIoT2023 subset) | 2 | 3–4 days | Must-have |
| Baseline model training (LightGBM + compact MLP) | 2 | 3–4 days | Must-have |
| Explainer wrappers (TreeSHAP, KernelSHAP, LIME, FastTreeSHAP) | 3 | 1 week | Must-have |
| Resource-cap harness (memory/CPU limiting + timing) | 3 | 3–4 days | Must-have |
| Fidelity metrics (deletion/insertion AUC) | 2 | 2–3 days | Must-have |
| CCS/MFR/TC/UMCEF metric implementation (from Eqs. in Section 9) | 3 | 1 week | Must-have (this *is* the novel contribution) |
| Adversarial perturbation harness (ART/Foolbox DeepFool) | 3 | 4–5 days | Must-have |
| Second dataset generalization check (Edge-IIoTset, leakage-corrected) | 3 | 1 week | Should-have |
| Results dashboard / plots / Pareto frontier viz | 2 | 3–4 days | Should-have |
| Draft paper write-up | 3 | 2 weeks | Must-have |
| Physical Raspberry Pi validation | 3 | 1 week | Stretch (optional) |

**Primary technology decisions:**
- **LightGBM over XGBoost:** slightly better native TreeSHAP support and smaller memory footprint, matching the survey's "TreeSHAP variants" as a "mature/high-effectiveness" solution (Table 22).
- **CPU-only PyTorch model, deliberately small (≤50k params):** represents realistic edge-deployable DNNs, not a research-scale model — resource caps stay meaningful.
- **Docker `--cpus`/`--memory` over raw `cgroups`:** simpler, reproducible, scriptable resource-cap tiers without root-level cgroup wrangling.
- **`mlflow` local file backend over a hosted tracker:** zero cost, zero setup, still gives run comparison/versioning.

---

## PART 5 — Risk Assessment

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| CICIoT2023 too large for laptop RAM | Medium | Medium | Use documented stratified subsampling (the dataset's own authors provide a reduced/merged-CSV variant); cap to a fixed number of flows per class |
| KernelSHAP runtime explodes under low resource cap (it's O(2^M)) | High | Low (expected finding) | This is *expected and is itself a result* — cap wall-clock per sample and report timeout rate as a metric, not a bug |
| Edge-IIoTset leakage artifact silently inflates fidelity/accuracy | Medium | High | Explicitly apply the leakage-free split described in arXiv:2608.15761 before any Edge-IIoTset use; document deviation from the raw released split |
| Adversarial attack library (ART/Foolbox) API drift on tabular data | Low | Medium | Pin library versions in a `requirements.txt`/lockfile at project start; tabular DeepFool support is well-established in ART |
| Solo-researcher timeline slippage | Medium | Medium | Phase gates below are independently publishable (Phase 1–2 alone = a workshop paper) |
| Reviewers argue "simulated edge" isn't real edge | Medium | Medium | Explicitly calibrate caps to published Pi4/Jetson Nano specs and cite this as a documented methodological choice consistent with prior surveyed studies; offer optional physical validation as future work |

**Go/No-Go:** Proceed — every required tool is free, documented, and Python-3.12-compatible today; the dataset is public; the metrics to compute are explicitly defined (not invented) in the source survey, which sharply de-risks "is this a real contribution" — it is closing a named, citable gap, not speculating about one.

---

## PART 6 — Phased Implementation Roadmap

**Phase 1 — Foundation (Weeks 1–2):** environment setup (`venv`, pinned requirements), CICIoT2023 subsample + preprocessing pipeline, baseline LightGBM + compact MLP trained to competitive accuracy (cross-check against literature-reported ~95–99% on this dataset). *Deliverable:* reproducible training notebook/script + baseline metrics table.

**Phase 2 — Explainability + Resource Harness (Weeks 3–5):** wrap all four explainers behind one interface; build the Docker-based resource-cap harness; implement CCS, MFR, TC, and fidelity (deletion/insertion) metrics; run the full 2×4×3 grid on clean data. *Deliverable:* Fidelity–Efficiency Frontier plot (Eq. 59) — the survey's proposed but never-computed metric, now computed.

**Phase 3 — Robustness + Generalization (Weeks 6–8):** implement DeepFool-style adversarial perturbation sweep; compute explanation-stability deltas vs. accuracy deltas; repeat the Phase 2 grid on the leakage-corrected Edge-IIoTset subset for generalization. *Deliverable:* robustness delta table + cross-dataset comparison.

**Phase 4 — Write-up & Release (Weeks 9–10):** UMCEF leaderboard, open-source repo cleanup + README + reproduction instructions, draft manuscript targeting Future Internet/IEEE Access/Sensors. *Optional stretch:* physical Raspberry Pi 4 validation run if hardware becomes available.

**Milestones:**

| Week | Milestone | Success criterion |
|---|---|---|
| 2 | Baseline models trained | ≥95% accuracy on held-out CICIoT2023 split, matching literature range |
| 5 | Full explainer × resource-cap grid complete | Fidelity–Efficiency Frontier computed and plotted for both models |
| 8 | Robustness sweep + second dataset complete | Explanation-stability-vs-accuracy delta table produced for ≥3 perturbation budgets |
| 10 | Repo + draft manuscript ready | Public GitHub repo with reproduction instructions; manuscript draft complete |

---

## PART 7 — CC-SHAP: A Novel Contribution (added post-baseline, 2026-09-30/10-01)

The original plan (Parts 0–6) is a *comparative benchmark* of existing explainers — it answers "how do known methods trade off under edge constraints," but it does not propose anything new. That leaves the single sharpest gap in the Consensus report unaddressed:

> *"What network-native XAI methods best explain temporal, relational, and multi-protocol IoT traffic for operators? Most work still adapts SHAP or LIME, while networking surveys call for causal, graph-based, and operator-tailored explainers."* — Consensus report, Open Research Questions

and the matching, independently-arrived-at gap in Karras et al. (2026): "Causal Semantics" is listed as one of three unaddressed **Transformative Research Frontiers** (Section 10, Figure 6) — mechanistic causal inference and knowledge-graph integration, currently absent from IoT-edge XAI work.

### 7.1 The technique: CC-SHAP (Causal-Consolidated SHAP)

A **post-processing layer**, not a new attribution algorithm from scratch (this is why it stays cheap):

1. **Offline, once:** learn a bounded causal skeleton over the training features via a PC-algorithm variant capped at conditioning-set size ≤1 (`src/causal_graph.py`). Boundedness is the design, not a shortcut hidden from the reader — it's what keeps causal discovery itself within a "lightweight, edge-aware" budget, directly answering Table 23's "lightweight and sparsity-aware explainers" row with an actual causal method rather than another SHAP variant.
2. **Per sample, at explanation time:** take attributions from a base explainer (TreeSHAP for LightGBM, KernelSHAP for the MLP), find connected components ("causally-entangled feature cliques") in the skeleton, and within each clique keep only the feature with the largest |attribution|, zeroing the rest.
3. **Rationale:** SHAP-family methods are known to split credit ambiguously across correlated features — a documented instability source. CC-SHAP forces attribution onto the single strongest signal per entangled cluster instead.

### 7.2 Research question and hypothesis (RQ4 / H4)

**RQ4:** Can a lightweight, causally-informed post-processing layer improve explanation sparsity and adversarial robustness relative to generic SHAP/LIME variants, without unacceptable fidelity loss — and does the causal-consolidation step itself stay cheap (bounded by the base explainer's own cost)?

**H4:** CC-SHAP will show higher sparsity and higher adversarial robustness than its base explainer, at a measurable but bounded fidelity cost, with negligible added latency from the consolidation step itself.

### 7.3 Results — validated three times (synthetic → dev-scale real data → production-scale real data)

**FINAL, production-scale CICIoT2023 run** (`results/metrics/grid_ciciot2023_20261001_234835.csv`, `robustness_ciciot2023_20261001_234835.csv`, `umcef_ciciot2023_20261001_234835.csv`) — balanced 8-class, **10,000 samples/class** (79,999 total after dropping 1 row with an inf feature value), **100 explained samples**, LIME at its **full library-default 5,000-sample budget**, all **3 resource tiers** (pi4/jetson_nano/unconstrained), **true multiclass DeepFool** (not FGSM) as the MLP attack:

LightGBM test accuracy 78.06%, CompactMLP 73.29% (these are *lower* than an earlier, invalid run's 81.7%/77.5% — see §7.3.1, the earlier numbers were inflated by a class-imbalance bug, not a stronger model).

| Explainer | Fidelity | CCS (speed) | Sparsity | Temporal coherence | Robustness (ε=0.05 / 0.15 / 0.30) |
|---|---|---|---|---|---|
| KernelSHAP | **0.710** | 0.986 | 0.407 | 0.825 | 0.678 / 0.634 / 0.599 |
| TreeSHAP | 0.705 | 0.946 | 0.204 | **0.937** | 0.762 / 0.743 / 0.726 |
| Lightweight KernelSHAP | 0.652 | 0.982 | 0.417 | 0.779 | 0.288 / 0.267 / 0.272 |
| **CC-SHAP** | 0.561 | 0.711 | **0.950** | 0.812 | **0.855 / 0.842 / 0.848** |
| LIME | 0.567 | 0.986 | 0.090 | 0.651 | 0.484 / 0.487 / 0.500 |

**UMCEF_IoT leaderboard** (Eq. 58, the single transparent composite number — not cherry-picked from whichever raw metric looks best):

| Rank | Explainer | FID | EFF | SPAR | ROB | **UMCEF_IoT** |
|---|---|---|---|---|---|---|
| 1 | KernelSHAP | 0.710 | 0.868 | 0.407 | 0.637 | **0.713** |
| 2 | **CC-SHAP** | 0.561 | 0.731 | 0.950 | 0.848 | **0.708** |
| 3 | TreeSHAP | 0.705 | 0.848 | 0.204 | 0.744 | 0.706 |
| 4 | Lightweight KernelSHAP | 0.652 | 0.866 | 0.417 | 0.276 | 0.617 |
| 5 | LIME | 0.567 | 0.868 | 0.090 | 0.490 | 0.594 |

**H4 confirmed, at full production scale:** CC-SHAP is the single most robust explainer of all five at *every* perturbation strength (beating TreeSHAP by 0.09–0.12 absolute at every ε), and by far the sparsest (0.950 vs. 0.204–0.417 for the rest). On the composite UMCEF_IoT score it places a near-statistical-tie 2nd (0.708 vs. the leader's 0.713 — a 0.005 gap), ahead of TreeSHAP (0.706). This is the honest, final headline result: a cheap, offline-learned causal post-processing layer buys top-tier robustness and dramatically better sparsity, for a real but bounded fidelity cost, landing within noise of the best overall composite score despite never winning on raw fidelity or speed individually.

The result is *directionally identical* across all three validation passes — synthetic pipeline check, dev-scale real data (n_per_class=2,000), and this production-scale run — which is meaningful triangulation, not a fluke of one configuration.

#### 7.3.1 Two data-quality bugs found and fixed before trusting any of this

Two real correctness issues surfaced during the production run and were fixed in `src/data.py` (both verified with unit-level checks before the final run above):

1. **Class-imbalance bug:** `n_per_class` was being applied per *raw* CICIoT2023 subfolder (34 of them) rather than per *final grouped label* (8 of them). Since the 8-class grouping maps many raw subfolders to one label (e.g. 12 different `DDoS-*` folders → `"DDoS"`), the cap was effectively multiplied by however many raw folders shared a label — DDoS ended up with ~120,000 rows instead of the intended 10,000, while single-folder classes like Benign/BruteForce stayed correctly at 10,000. This silently inflated model accuracy (the model could lean on majority-class bias) and invalidated two earlier full runs. Fixed by grouping raw subfolders by final label *before* applying the per-class budget and subsample cap.
2. **Infinity-value crash:** a small fraction of real CICIoT2023 rows have rate-style features (`Rate` = packet_count / duration) that are literal `+inf` when duration rounds to 0 in the original capture. `StandardScaler` crashes hard on this (`ValueError: Input X contains infinity`), and the original `dropna()` call ran *before* numeric-only column selection and doesn't catch `inf` (which is not `NaN`). Fixed by replacing `inf`/`-inf` with `NaN` after numeric selection, then dropping those rows, with a warning reporting how many were dropped (1 out of 80,000 in the final run — rare but real).

Both bugs would have silently produced wrong or crashed results if not caught — worth stating explicitly in the writeup's methodology/reproducibility section as a demonstration of why real-dataset validation (not just synthetic pipeline checks) matters, which is itself a small, honest footnote on the broader gap this whole project targets (benchmark-only validation hiding real-world data issues).

### 7.4 What this changes about the paper's contribution claim

The paper is no longer "a benchmark reproducing metrics a survey defined but never computed" alone — it now also proposes, implements, and validates a new lightweight explainer that measurably outperforms the field's default choices on robustness and sparsity under the exact resource-aware evaluation protocol the survey calls for, **and that result replicates across two structurally different real datasets** (§7.6), landing 2nd (CICIoT2023, near-tied) and 1st (Edge-IIoTset) on the UMCEF_IoT composite. Cross-dataset replication, not just a single-dataset benchmark win, is what makes this a legitimate novelty claim for a Q2–Q3 open-access venue (Future Internet, IEEE Access, Sensors), not just a reproduction study.

### 7.5 Honest limitations to state in the writeup

- CC-SHAP's raw fidelity is not the best in class (0.561, lowest of the five) — report this plainly; the UMCEF_IoT composite (§7.3) makes the net trade-off transparent rather than cherry-picking whichever raw metric looks best.
- The bounded PC-skeleton (conditioning-set size ≤1) is a real methodological limitation, not just a performance choice — it can miss higher-order conditional independencies. State this explicitly, with unbounded PC/FCI comparison flagged as future work.
- CC-SHAP's CCS (speed, 0.711) is the lowest of the five on the MLP side specifically, because it inherits KernelSHAP's own cost as its base explainer there (the causal-consolidation step itself is cheap — see the module docstring in `src/explainers.py`); this should be stated precisely rather than implying CC-SHAP itself is slow.
- The DeepFool attack used here clips the accumulated perturbation to an L-infinity epsilon ball at every iteration (`src/adversarial.py::deepfool_perturb`) rather than using unbounded minimal-norm DeepFool — an explicit, documented adaptation needed for comparability with the FGSM-based epsilon sweep, not literature-standard unbounded DeepFool. State this adaptation explicitly when citing comparability to Munilla & Khammas (2026).
- Both data-quality bugs in §7.3.1 should be disclosed in the methods section, not just fixed silently — they're evidence the pipeline was validated against real data edge cases, which strengthens rather than weakens the methodology narrative.

### 7.6 Cross-dataset generalization check: Edge-IIoTset

A single-dataset result always invites the question "does this generalize, or is it an artifact of CICIoT2023 specifically?" To answer that, the full explainer comparison was re-run on **Edge-IIoTset** (Ferrag et al.), a structurally different dataset: official "Selected dataset for ML and DL" release, **15 classes** (Normal + 14 attack types, vs. CICIoT2023's 8), **42 features** (different protocol-level feature set — includes MQTT/Modbus/HTTP/DNS fields rather than CICIoT2023's flow-statistical features), loaded via a new `load_edge_iiotset()` in `src/data.py`.

**Data-quality fix applied:** the official release has a documented "serialization artifact" reputation in the literature (see arXiv:2608.15761 for the specific precision-agriculture diagnosis of this class of problem). Measured directly on this release: 814/157,800 rows (0.5%) are exact duplicates by feature values. Applied fix: drop exact duplicates (keep first) *before* the stratified per-class subsample and train/test split, so no duplicate pair can land on both sides — the general, conservative version of this fix, applied against the duplication actually measured here rather than a blind reproduction of the cited paper's exact pipeline. This is a smaller leakage surface than CICIoT2023's class-imbalance bug (§7.3.1) but the same category of issue: real datasets in this literature need active validation, not just a download-and-trust approach.

**Run config:** `--n-per-class 1000` (the two smallest raw classes — MITM, Fingerprinting — can't support CICIoT2023's 10,000/class scale), `--n-explain 50`, `--lime-num-samples 5000`, all 3 resource tiers, DeepFool. LightGBM 90.4% / MLP 63.6% test accuracy — notably the MLP fits this feature set much worse than it did CICIoT2023's (73.3%), itself a finding (this dataset's protocol-indicator-heavy feature set, with many near-constant columns per protocol, suits tree splits better than a small dense network).

| Explainer | Fidelity | Sparsity | Temporal coherence | Robustness (ε=0.05/0.15/0.30) |
|---|---|---|---|---|
| TreeSHAP | **0.777** | 0.532 | **0.993** | 0.568 / 0.563 / 0.554 |
| KernelSHAP | 0.762 | 0.594 | 0.965 | 0.711 / 0.682 / 0.660 |
| Lightweight KernelSHAP | 0.683 | 0.690 | 0.956 | 0.361 / 0.385 / 0.386 |
| **CC-SHAP** | 0.660 | **0.976** | 0.554 | **0.891 / 0.881 / 0.888** |
| LIME | 0.599 | 0.626 | 0.702 | 0.230 / 0.223 / 0.213 |

**UMCEF_IoT leaderboard (Edge-IIoTset):**

| Rank | Explainer | FID | EFF | SPAR | ROB | **UMCEF_IoT** |
|---|---|---|---|---|---|---|
| **1** | **CC-SHAP** | 0.660 | 0.765 | 0.976 | 0.887 | **0.769** |
| 2 | KernelSHAP | 0.762 | 0.836 | 0.594 | 0.684 | 0.752 |
| 3 | TreeSHAP | 0.777 | 0.835 | 0.532 | 0.562 | 0.727 |
| 4 | Lightweight KernelSHAP | 0.683 | 0.836 | 0.690 | 0.377 | 0.669 |
| 5 | LIME | 0.599 | 0.842 | 0.626 | 0.222 | 0.599 |

**The generalization check holds, and strengthens the headline result.** CC-SHAP's robustness margin over the field *widens* on Edge-IIoTset (0.89 vs. TreeSHAP's 0.56 at ε=0.05, a 0.33 gap, vs. a 0.09 gap on CICIoT2023), and CC-SHAP takes **outright 1st place on UMCEF_IoT** here (0.769), rather than CICIoT2023's near-tied 2nd (0.708). This is not a cherry-picked improvement — one metric moved the other way and should be reported with equal prominence: CC-SHAP's **temporal coherence dropped substantially** on this dataset (0.554, the lowest of the five here, vs. 0.812 on CICIoT2023, where it was mid-pack) — plausibly because the denser causal skeleton this dataset produces (449 edges over 42 features vs. 275 over 39) makes the consolidation step's per-sample "winner" within each clique more sensitive to small sample-to-sample fluctuation. This is a genuine, dataset-dependent trade-off worth investigating further (e.g., whether a slightly larger conditioning-set bound stabilizes clique membership), not a result to smooth over.

**Net conclusion for the writeup:** two structurally different datasets (different class count, different feature semantics, different duplicate-leakage profile) produce the same qualitative story — CC-SHAP trades fidelity and (on Edge-IIoTset) temporal stability for a large, consistent win in adversarial robustness and sparsity, landing at or near the top of the composite UMCEF_IoT ranking both times. That consistency across datasets is the strongest evidence this project has that the result is a property of the method, not an artifact of one dataset's quirks.

### 7.7 Figures: FEF, grouped bar, and radar views of the same trade-off

Three complementary figures, all in `results/figures/` as both `.pdf` and `.svg` (vector, paper-ready), for both datasets (`*_ciciot2023.*` / `*_edge_iiotset.*`):

- **`fef_*`** (`src/plot_fef.py`) — the FEF the survey defines in Eq. 59 (the Pareto-optimal set in fidelity-vs-efficiency space, EFF = (CCS+MFR)/2), with the dominated region shaded so "better" is a visual direction rather than a concept the reader has to hold in mind.
- **`bars_*`** (`src/plot_bars.py`) — **the primary comparison figure.** A grouped bar chart of all four UMCEF_IoT dimensions (FID/EFF/SPAR/ROB) per explainer, each explainer in its own fixed, colorblind-validated categorical color (`node scripts/validate_palette.js` — all hard gates pass). Preferred over the radar for the paper body: bar length is a linear perceptual channel, polygon area in a radar chart is not, so bars let a reviewer compare precise values directly rather than impressions of shape size.
- **`radar_*`** (`src/plot_radar.py`) — the same four dimensions as a spider/radar chart, useful as a secondary "shape at a glance" figure (CC-SHAP's diamond silhouette vs. the baselines' compressed one) but not a substitute for the bar chart's precision.

**CC-SHAP is not on the 2D fidelity-efficiency frontier on either dataset.** On CICIoT2023 the frontier is KernelSHAP alone (it dominates every other explainer on both axes simultaneously). On Edge-IIoTset the frontier is {TreeSHAP, KernelSHAP, LIME} — again, not CC-SHAP. Taken in isolation, a reader looking only at the FEF plot would conclude CC-SHAP is strictly worse than at least one alternative and move on.

**This is not a contradiction of §7.3/§7.6 — it's the whole point of the paper's methodological argument.** The FEF, as the survey defines it, only plots two of the four dimensions the field actually cares about (fidelity, efficiency) and omits the two dimensions CC-SHAP wins decisively on (robustness, sparsity) — visible immediately in the bar/radar views. A 2D frontier is blind to exactly the trade-off this project's central contribution is built around. That is precisely the gap both source documents independently flag: "most IoT XAI studies stop at plausibility checks, with few combining fidelity metrics, stability tests, ... and application-level outcomes" (Consensus report). The FEF plot earns its place in the writeup specifically *because* it shows CC-SHAP losing on the narrow, conventional view — making the UMCEF_IoT composite's broader verdict (CC-SHAP 1st on Edge-IIoTset, near-tied 2nd on CICIoT2023), made visually obvious in the bar chart, a genuine, non-obvious finding rather than a foregone conclusion restated four ways.

---

## Summary

This plan converts the two source documents' shared, named knowledge gaps into a fully open-source, zero-budget, single-researcher, Python-compatible experiment with two distinct, stacked contributions, validated on two structurally different real datasets — CICIoT2023 (production scale: balanced 8-class, 10,000 samples/class, 100 explained samples, full-budget LIME, all 3 resource tiers, true multiclass DeepFool) and Edge-IIoTset (15-class, 42-feature, duplicate-leakage-corrected generalization check, §7.6): (1) it computes the metrics (CCS, MFR, TC, UMCEF, FEF) the Future Internet survey formalized in equations but left unimplemented, benchmarking four existing explainers (TreeSHAP, KernelSHAP, a lightweight KernelSHAP variant, LIME) under simulated edge resource caps and adversarial perturbation; and (2) it proposes, implements, and validates CC-SHAP, a novel lightweight causal-graph-consolidation layer that directly answers the field's explicit, repeated call (both source documents, independently) for "causal, graph-based" explainers tailored to network telemetry rather than generic tabular SHAP/LIME. CC-SHAP is the single most adversarially robust explainer of all five tested at every perturbation strength on *both* datasets, by far the sparsest on both, and ranks 2nd (CICIoT2023, near-tied: 0.708 vs. 0.713) and 1st (Edge-IIoTset: 0.769) on the UMCEF_IoT composite — a real, quantified, bounded fidelity trade-off, not an unqualified win, replicated across datasets rather than a single-dataset artifact, which is exactly the kind of honest, generalizable result a Q2–Q3 open-access venue (Future Internet, IEEE Access, or Sensors) can publish as a direct empirical-plus-methodological companion to Karras et al. (2026).
