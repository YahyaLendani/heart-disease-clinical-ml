# Reading List Implementation Benchmark Summary

This report summarizes the empirical performance of all architectures and methods implemented from `AI_Thesis_Reading-List.csv`.

## 1. Architectural Benchmark Table

Holdout: one stratified 60/20/20 split (61 test patients), with stratified bootstrap 95% CIs. CV: repeated stratified 5-fold x 3 (mean ± std over folds). Model weights are chosen by validation loss in both settings.

| Model | Holdout Accuracy | Holdout ROC-AUC [95% CI] | Holdout Brier | CV Accuracy | CV ROC-AUC | CV Brier |
| --- | --- | --- | --- | --- | --- | --- |
| TabTransformer | 0.8197 | 0.9080 [0.8300, 0.9719] | 0.1247 | 0.8047 ± 0.0530 | 0.8944 ± 0.0270 | 0.1348 ± 0.0217 |
| GCN (Kipf & Welling) | 0.8033 | 0.9102 [0.8322, 0.9719] | 0.1305 | 0.8300 ± 0.0366 | 0.9151 ± 0.0248 | 0.1189 ± 0.0175 |
| GAT (Velickovic et al.) | 0.8197 | 0.8972 [0.8192, 0.9632] | 0.1353 | 0.8244 ± 0.0408 | 0.9100 ± 0.0296 | 0.1238 ± 0.0241 |
| Multimodal Cross-Attention (synthetic-signal demo) † | 0.6721 | 0.6905 [0.5541, 0.8214] | 0.2227 | 0.6457 ± 0.0793 | 0.6817 ± 0.0690 | 0.2276 ± 0.0258 |

† Demonstration only: the signal channel is synthesised from the tabular features, so this row is not a like-for-like comparison.

## 2. Methodology & Rigor Summary
- **Temperature Scaling (Guo et al., 2017):** Validation-tuned optimal T `1.511` on trained `HeartMLP` reduced holdout test ECE from `0.1256` to `0.1150` (Validation ECE: `0.1183` -> `0.1211`).
- **Deep Ensembles (Lakshminarayanan et al., 2017):** Average epistemic predictive variance is `0.0112` across 61 holdout patients.
- **Split & Mondrian Conformal Prediction (Angelopoulos & Bates, 2021):** Calibrated and evaluated on the same (uncalibrated) MLP probabilities; the 90% finite-sample coverage target achieved `0.9016` empirical coverage (Case: `1.0000`, Control: `0.8182`) with average set size `1.344` and `0.344` clinical abstention rate.
- **Conformal Efficiency-Coverage Trade-Off:** Operating at $\alpha=0.15$ (85% coverage guarantee) cuts the ambiguous deferral rate from `34.4%` down to `23.0%` with `88.5%` empirical test coverage.
- **Causal IPTW Analysis (Pearl & Mackenzie, 2018):** Observational risk difference is `0.4623`, whereas confounder-isolated Inverse Probability Weighting (IPTW) reveals a causal ATE of `0.0638`, demonstrating a confounding bias of `0.3985`.
- **Covariate Balance (IPTW, treatment `exang`):** maximum |SMD| fell from `0.934` to `0.321`, but `5` of `12` covariates remain at or above the 0.10 balance threshold (age, ca, cp, thalach, trestbps), so residual measured confounding cannot be ruled out.
- **Doubly Robust AIPW & E-Value (Chernozhukov et al., 2018; VanderWeele & Ding, 2017):**
  - *Symptomatic Proxy (exang):* Cross-fitted AIPW ATE = `-0.0112` (95% CI: [-0.1923, 0.1700]), E-value = `1.186`.
  - *Modifiable Metabolic Treatment (Hypertension SBP >= 140):* Cross-fitted AIPW ATE = `0.0669` (95% CI: [-0.0473, 0.1812]), E-value = `1.569` (CI E-value = `1.000`).
- **Algorithmic Fairness (Barocas et al., 2020):** Validation-tuned Equal Opportunity thresholds (t0=0.586, t1=0.050) reduced holdout test TPR disparity from `0.2381` to `0.0000` (100.0% reduction), while the false-positive rate moved from `0.212` to `0.394` and accuracy from `0.803` to `0.787` — at least one group threshold is near 0, so this parity comes from flagging almost every patient in that group and is not a useful operating point.
- **Clinical Counterfactual Recourse (Wachter et al., 2017; Ustun et al., 2019):** Achieved `100.0%` recourse validity across high-risk test patients with `8.32` average features modified, reducing mean predicted risk by `0.590` while strictly maintaining `0.000000` immutable demographic violation.
- **Closed-Loop Conformal Recourse Transition:** `100.0%` of Conformal High-Risk patients ($\hat{C}(x) = \{1\}$) successfully transitioned into Low-Risk $\{0\}$ or Deferral $\{0, 1\}$ following sparse actionable recourse.

## 3. LOCO Domain-Shift Conformal Generalization Table

| Evaluation Center | N Samples | Disease Prevalence | Fixed Threshold Acc | 90% Conformal Coverage | Avg Set Size | Abstain Rate |
| --- | --- | --- | --- | --- | --- | --- |
| cleveland | 303 | 45.9% | 65.3% | 80.9% | 1.22 | 22.4% |
| hungarian | 293 | 36.2% | 77.5% | 91.1% | 1.32 | 32.1% |
| switzerland | 123 | 93.5% | 64.2% | 87.8% | 1.38 | 38.2% |
| va_long_beach | 199 | 74.4% | 73.9% | 88.9% | 1.32 | 31.7% |