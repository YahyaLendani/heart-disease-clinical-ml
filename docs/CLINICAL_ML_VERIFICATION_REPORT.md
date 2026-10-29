# Clinical ML System Verification & Methodological Audit Report

**Status:** Revised after the evaluation-protocol review (see Section 6)  
**Test Suite Execution:** 81 / 81 passed (~45 s on 4 CPU cores) | CI: full Ruff rule set, Python 3.10–3.13 tests, pipeline smoke run  
**Scope:** Data Hygiene, Feature Engineering, Deep & Tabular Architectures, Uncertainty Quantification, Causal Inference, Counterfactual Recourse, Algorithmic Fairness, Nelder-Mead Simplex Optimization, Vectorized GNN/DCA, Multi-Core Parallelism, and Pipeline Governance.

---

## 1. Executive Summary

An end-to-end architectural and mathematical verification audit was conducted across the repository’s test suites (`test_data_leakage.py`, `test_advancements.py`, `test_methodological_upgrades.py`, `test_reading_list_implementations.py`) and execution orchestrator (`run_pipeline.py`). The system satisfies strict clinical machine learning safety invariants: **zero train-test data leakage**, **patient-level and center-level partition isolation**, **distribution-free conformal coverage guarantees**, **deconfounded doubly robust causal estimation**, **strictly immutable demographic constraints during counterfactual recourse**, and **deterministic pipeline reproducibility**.

---

## 2. Mathematical Correctness & Clinical Safety Invariants

### 2.1 Zero Data Leakage & Multi-Level Partition Isolation
Clinical ML models frequently suffer from optimistic bias due to patient-level overlap or global preprocessing prior to splitting. The test suite enforces four distinct isolation invariants:
1. **Patient & Admission Isolation (`TestMIMICNoPatientOverlap`):** Uses `GroupShuffleSplit` keyed on `subject_id` (`random_state=42`, `test_size=0.2`). Set intersection assertions verify:
   $$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset \quad \text{and} \quad \mathcal{H}_{\text{train}} \cap \mathcal{H}_{\text{test}} = \emptyset$$
   where $\mathcal{S}$ and $\mathcal{H}$ denote patient (`subject_id`) and admission (`hadm_id`) sets, preventing longitudinal leakage across multiple ICU stays.
2. **Leave-One-Center-Out (LOCO) Domain Isolation (`TestCrossCenterPartitioning`):** Across the four clinical sites (`cleveland`, `hungarian`, `switzerland`, `va_long_beach`), partitioning guarantees $\mathcal{P}_{\text{train}} \cap \mathcal{P}_{\text{test\_site}} = \emptyset$, evaluating true out-of-distribution (OOD) cross-hospital transferability.
3. **Fit-on-Train Preprocessing (`TestScalerFitOnlyOnTrain`, `TestImputerFitOnlyOnTrain`):**
   - `StandardScaler` parameters $(\hat{\mu}_{\text{train}}, \hat{\sigma}_{\text{train}})$ are estimated exclusively via `fit_transform(X_train)`, centering training features ($|\bar{X}_{\text{train}}| < 0.1$) while applying `transform(X_test)` without refitting.
   - `SimpleImputer` (median strategy) is verified by injecting synthetic `np.nan` into `X_test` and confirming exact recovery of the training median ($|\hat{x}_{\text{test, imputed}} - \text{median}(X_{\text{train}})| < 10^{-6}$).
4. **Leakage-Free Feature Engineering (`test_nan_propagation_without_leakage`):** `add_cardiovascular_features` propagates `NaN` values naturally through arithmetic operations (`pd.isna(out.loc[0, "rpp"])`) rather than imputing globally before cross-validation splits.

---

### 2.2 Physiological Feature Engineering & Target Polarity Sanity
1. **Cardiovascular Domain Indices (`TestCardiovascularFeatures`):**
   - **Rate Pressure Product (Myocardial Oxygen Demand):**
     $$\text{RPP} = \frac{\text{trestbps} \times \text{thalach}}{100}$$
   - **Chronotropic Index / Ratio:**
     $$\text{Chronotropic Ratio} = \frac{\text{thalach}}{220 - \text{age}}$$
   - **Ischemia Burden & Vessel-Weighted Ischemia:**
     $$\text{Ischemia Burden} = \text{oldpeak} \times \text{exang}, \qquad \text{Vessel Ischemia} = \text{ca} \times \text{oldpeak}$$
2. **Epidemiological Polarity Invariants (`TestTargetPolaritySanity`):**
   Automated guards prevent silent label inversion ($0 \leftrightarrow 1$ across heterogeneous UCI/multi-center encodings) by asserting clinically established Pearson correlations with the disease label $Y$:
   - Exercise-induced angina: $\rho(\text{exang}, Y) > 0.20$
   - ST depression: $\rho(\text{oldpeak}, Y) > 0.20$
   - Peak heart rate achieved: $\rho(\text{thalach}, Y) < -0.20$

---

### 2.3 Causal Inference & Unmeasured Confounding Sensitivity
Observational clinical data is subject to confounding by indication (sicker patients receive more aggressive interventions).
1. **Doubly Robust AIPW (`test_doubly_robust_aipw_recovers_true_causal_effect_under_confounding`):**
   Evaluated on a synthetic structural causal model ($N=1200$) where confounder $X_0 \sim \mathcal{N}(0,1)$ drives both treatment assignment $T$ and outcome $Y$, with a true Average Treatment Effect (ATE) of $\tau = +0.20$:
   $$\hat{\tau}_{\text{AIPW}} = \frac{1}{N} \sum_{i=1}^{N} \left[ \hat{\mu}_1(X_i) - \hat{\mu}_0(X_i) + \frac{T_i (Y_i - \hat{\mu}_1(X_i))}{\hat{e}(X_i)} - \frac{(1 - T_i)(Y_i - \hat{\mu}_0(X_i))}{1 - \hat{e}(X_i)} \right]$$
   - Naive observational risk difference exhibits severe upward confounding bias ($> 0.26$).
   - Cross-fitted AIPW successfully deconfounds the estimate ($|\hat{\tau}_{\text{AIPW}} - 0.20| < 0.06$) with valid 95% confidence intervals ($\text{CI}_{\text{lower}} < \hat{\tau}_{\text{AIPW}} < \text{CI}_{\text{upper}}$).
2. **VanderWeele & Ding (2017) E-Value (`test_vanderweele_e_value_exact_formula`):**
   Quantifies the minimum strength of association an unmeasured confounder must have with both treatment and outcome to explain away the observed Risk Ratio ($\text{RR}$):
   $$\text{E-value} = \text{RR}^* + \sqrt{\text{RR}^* (\text{RR}^* - 1)}, \qquad \text{where } \text{RR}^* = \begin{cases} \text{RR} & \text{if } \text{RR} \ge 1 \\ 1/\text{RR} & \text{if } \text{RR} < 1 \end{cases}$$
   - Verified to within $10^{-6}$ tolerance for risk-increasing ($\text{RR}=2.0 \implies 2 + \sqrt{2} \approx 3.414214$) and protective ($\text{RR}=0.5 \implies \text{RR}^*=2.0$) regimes.
   - Null-crossing confidence intervals ($\text{CI}_{\text{bound}} = 0.95 \le 1.0$ for $\text{RR} > 1$) deterministically return $\text{E-value}_{\text{CI}} = 1.0$.

---

### 2.4 Distribution-Free Conformal Prediction & Calibration
1. **Split & Mondrian Conformal Prediction (`test_split_and_mondrian_conformal_finite_sample_coverage`):**
   `SplitConformalClassifier(alpha=0.10, mondrian=True)` computes class-conditional nonconformity quantiles $\hat{q}_{1-\alpha}^{(y)}$ separately for $y \in \{0, 1\}$, guaranteeing finite-sample class-conditional coverage regardless of prevalence shift:
   $$\mathbb{P}\big(Y_{\text{test}} \in \hat{C}(X_{\text{test}}) \mid Y_{\text{test}} = y\big) \ge 1 - \alpha$$
   - Verified marginal coverage $\ge 0.86$ and class-conditional positive/negative coverages $\ge 0.85$ at $\alpha = 0.10$.
   - Partition probabilities strictly satisfy the conservation law:
     $$p_{\text{low\_risk}} + p_{\text{high\_risk}} + p_{\text{ambiguous}} + p_{\text{empty}} = 1.0$$
2. **Post-Hoc Temperature Scaling & Deep Ensembles (`TestCalibrationTemperatureScaling`, `TestDeepEnsembles`):**
   - `ModelWithTemperature` optimizes scalar temperature $T > 0$ on validation logits $z / T$ to reduce Expected Calibration Error ($\text{ECE} \ge 0$).
   - `DeepEnsemble` ($M=3$) decomposes predictive uncertainty into mean probability $\bar{p}$, non-negative epistemic variance $\sigma^2_{\text{epistemic}} \ge 0$, predictive Shannon entropy $\mathcal{H}(\bar{p})$, and binary abstention flags (`is_uncertain`).

---

### 2.5 Actionable Counterfactual Recourse & Demographic Safety
`test_clinical_counterfactual_immutability_and_validity` verifies `ClinicalCounterfactualGenerator` via projected gradient optimization over a differentiable clinical network:
1. **Recourse Validity:** Reduces a high-risk patient’s predicted risk from $P(Y=1 \mid x) > 0.85$ to $P(Y=1 \mid x_{\text{cf}}) \le 0.25$ (`validity_achieved == True`).
2. **Strict Demographic Immutability:** Unmodifiable attributes (`immutable_features=("age", "sex")`) have gradients/updates zeroed out at every step, yielding exact zero perturbation:
   $$\max_{j \in \mathcal{I}_{\text{immutable}}} |x_{\text{cf}, j} - x_j| = 0.0$$
3. **Physiological Feasibility:** Counterfactual states are clamped within empirical training box constraints:
   $$l_j - 10^{-5} \le x_{\text{cf}, j} \le u_j + 10^{-5} \quad \forall j$$

---

### 2.6 Algorithmic Fairness & Equal Opportunity Mitigation
1. **Subgroup Auditing (`TestFairnessMetrics`):** `compute_subgroup_metrics` accurately computes Sensitivity ($\text{TPR} = \frac{\text{TP}}{\text{TP}+\text{FN}}$), False Negative Rate ($\text{FNR} = 1 - \text{TPR}$), Specificity ($\text{TNR} = \frac{\text{TN}}{\text{TN}+\text{FP}}$), and False Positive Rate ($\text{FPR} = 1 - \text{TNR}$). In cardiac screening, bounding $\Delta \text{FNR}$ across sex/age groups prevents systematic under-diagnosis in underrepresented cohorts.
2. **Post-Processing Threshold Optimization (`TestFairnessMitigation`):** `find_equal_opportunity_thresholds` calibrates group-specific decision thresholds $(\tau_0, \tau_1)$ to minimize Equal Opportunity Difference $|\text{TPR}_{A=0}(\tau_0) - \text{TPR}_{A=1}(\tau_1)| \le \epsilon_{\text{tol}}$ while maximizing joint $F_1$ utility.

---

### 2.7 Advanced Neural Architectures & Explainability
`test_reading_list_implementations.py` validates tensor invariants, symmetry, and differentiability across four architectural families:
- **TabTransformer:** Contextualizes categorical embeddings via multi-head self-attention, returning attention maps of shape $(B, H, N_{\text{cat}}, N_{\text{cat}})$ and verified non-null gradients (`param.grad is not None`) across all trainable parameters.
- **Patient Similarity GNNs (GCN & GAT):** `build_knn_patient_graph` constructs a symmetric cosine k-NN adjacency matrix ($A = A^\top$) with explicit self-loops ($\text{diag}(A) = \mathbf{1}$). `normalize_adjacency_gcn` computes $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ for `GCNClassifier` and `GATClassifier` node classification.
- **Multimodal Fusion (`early`, `late`, `cross_attention`):** Integrates 1D physiological waveforms (`ResNet1DBlock` and `ViTPatchEncoder` with patch size 16) with tabular EHR features.
- **Grad-CAM Attribution:** Extracts layer activations and backward gradients, producing min-max normalized saliency maps bounded in $[0, 1]$ with proper hook cleanup (`gradcam.close()`).

---

## 3. Numerical Stability & Edge-Case Analysis

| Component / Module | Potential Numerical Hazard | Implemented Safeguard & Verification Status |
| :--- | :--- | :--- |
| **Chronotropic Ratio** (`models`) | Division by zero or sign inversion if `age >= 220` | Standard adult cohorts satisfy $\text{age} \in [18, 100]$ ($220 - \text{age} \ge 120$). Guarding $220 - \text{age}$ with $\max(220 - \text{age}, \epsilon)$ is maintained as a best-practice invariant. |
| **Propensity Scores & IPTW / AIPW** (`evaluation`) | Exploding inverse weights $\frac{1}{\hat{e}(X)}$ or $\frac{1}{1-\hat{e}(X)}$ when $\hat{e}(X) \to 0 \text{ or } 1$ | Verified in `TestCausalInference`: `compute_propensity_scores` strictly bounds $\hat{e}(X) \in (0, 1)$ via overlap clipping, preventing division-by-zero in Horvitz-Thompson and AIPW estimators. |
| **GCN Spectral Normalization** (`models`) | Isolated nodes causing zero degree $d_i = 0$ in $D^{-1/2}$ | `build_knn_patient_graph` enforces self-loops ($\text{diag}(A) = 1$), guaranteeing $\tilde{d}_i \ge 1.0 \ \forall i$ prior to computing $\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$. |
| **Deep Ensemble Entropy** (`training`) | $\log(0) \to -\infty$ when $\bar{p} \in \{0, 1\}$ | Probability clamping $\bar{p} \in [\epsilon, 1-\epsilon]$ prevents `NaN` generation in $-\bar{p}\log\bar{p} - (1-\bar{p})\log(1-\bar{p})$. |
| **Temperature Scaling** (`models`) | Temperature $T \le 0$ inverting logit polarity during optimization | Verified in `test_temperature_scaling_optimization`: `optimal_temperature > 0.0` is strictly enforced via log-parameterization or positivity constraints. |
| **Grad-CAM Normalization** (`evaluation`) | Zero-gradient dead ReLU layers causing $\frac{0}{0}$ in min-max scaling | Verified in `test_gradcam_attribution`: output CAM is strictly finite and bounded in $[0.0, 1.0]$. |
| **Deep Network Training** (`training/train_model.py`) | Exploding gradients in MLP/Transformer training | `run_pipeline.py` and training scripts apply gradient clipping, weight decay ($10^{-4}$), dropout ($0.25$), and early stopping (`patience=12`). |

---

## 4. Pipeline Governance, Reproducibility & Automation Audit

Inspection of `run_pipeline.py` confirms a fail-safe, reproducible execution architecture:
1. **Test Gate:** CI runs the full pytest suite on every push and pull request. Locally, `heart-pipeline --run-tests` runs it before the pipeline and aborts on any failure (it is no longer run by default on every pipeline invocation).
2. **Deterministic Seeding:** A unified `--seed` argument (default `42`) is propagated across every training and experiment script, including the reading-list suite (which previously hard-coded 42). CV folds have their own seeds and train single-threaded, so results are identical whether folds run serially or in parallel (`tests/test_optimizations.py`).
3. **Automated Thesis Synchronization (`TestThesisBlockBuilders`):** Results from ensembles, LOCO cross-center evaluation, and fairness analyses are programmatically formatted into delimited Markdown blocks (`ENSEMBLE_AUTO_START/END`, `LOCO_AUTO_START/END`, `FAIRNESS_AUTO_START/END`), eliminating manual copy-paste transcription errors between experimental artifacts and thesis tables.

---

## 5. Summary Statement

All 81 unit, integration and end-to-end tests pass, and the full pipeline regenerates every artifact from the committed data. The corrections in Section 6 were needed before the reported numbers could be relied on; the remaining limitation is sample size (184 holdout patients for the primary heart cohort, 3 test deaths for the MIMIC demonstration), which the reported confidence intervals and paired DeLong tests make explicit.

---

## 6. Corrections from the Evaluation-Protocol Review

The earlier version of this report missed the following. Each is fixed and, where testable, covered by `tests/test_optimizations.py`.

| Issue | Effect | Fix |
| :--- | :--- | :--- |
| Best HeartMLP optimizer chosen by **test accuracy** | Optimistic reported test score | Chosen by validation loss |
| Hyperparameter grid (`tune`) ranked by **test accuracy** | Same | Ranked by repeated-CV ROC-AUC, then validation loss |
| F1/Youden "optimal thresholds" tuned **and** scored on the test set | Optimistic threshold metrics | Tuned on validation (out-of-fold for ensembles), applied to test |
| HeartMLP and MIMIC MLP CV early-stopped on the **fold being scored** | Optimistic CV scores | Separate early-stopping split inside each fold |
| Conformal calibrated on raw probabilities, evaluated on temperature-scaled ones | Coverage guarantee did not apply | Same probability scale for calibration and test |
| MIMIC leakage assertion compared positions from two different index spaces | `mimic_train_baselines.py` and `mimic_train_mlp.py` failed on every run | Compare subject IDs |
| `os.cpu_count() or 1 if n_jobs == -1 else n` operator precedence | `--n-jobs` ignored in 3 scripts | `evaluation.cv_utils.resolve_n_jobs` |
| `--device cuda` / `--xgb-device cuda` defaults in `run_pipeline.py` | Pipeline crashed on machines without a GPU | `auto` default |
| Pipeline passed the Cleveland file to the fairness stage | Silently undid the move to the 918-patient cohort | LOCO and fairness always use the multi-center cohort |
| Single 61-patient holdout used to rank models | Rankings within noise | Bootstrap 95% CIs on every test metric; repeated stratified CV for every model family |
| Equal Opportunity "100% disparity reduction" reported alone | Hid that both thresholds were 0.05 (flag almost everyone) | False-positive rate and accuracy before/after are reported, with a warning |
| Equal Opportunity search maximised F1 | Near-zero thresholds on small or high-prevalence validation sets | Youden's J objective (flagging everyone scores 0); warning when any group threshold is near 0 |
| Zero-coded `chol` (172 patients) and `trestbps` (1) in the multi-center cohort | `chol = 0` identified Zurich (93.5% prevalence): a site shortcut | Recoded as missing in `prepare_heart_combined.py` |
| Kaggle placeholders `ca = 4` (4) and `thal = 0` (2) in the Cleveland file | Invalid category values used as measurements | Recoded as missing in `clean_data.py` |
| "All covariates balanced below \|SMD\| = 0.10" after weighting | Not supported: 7 of 12 remain above 0.10 (largest 0.264) | Actual balance computed and reported by the suite |
| Only 2 Random Forest / XGBoost configurations compared | Under-tuned baselines | 8-configuration CV grid each; full search tables saved |
| Models ranked by point estimates only | Rankings not tested | Paired DeLong tests with Holm correction (0 of 91 pairs significant on the 918-patient cohort) |
| Hard-coded findings in thesis templates ("ensembles reduce variance", "linear models degrade sharply") | Text contradicted the data | Findings generated from the result tables |
| 61-patient Cleveland split as primary benchmark | Confidence intervals about ±0.08 | 918-patient cohort is primary (±0.04); Cleveland kept as sensitivity analysis |

