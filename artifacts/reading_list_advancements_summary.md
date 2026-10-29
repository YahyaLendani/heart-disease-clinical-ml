# Reading List Implementation Benchmark Summary

This report summarizes the empirical performance of all architectures and methods implemented from `AI_Thesis_Reading-List.csv`.

## 1. Architectural Benchmark Table

Holdout: one stratified 60/20/20 split (184 test patients), with stratified bootstrap 95% CIs. CV: repeated stratified 5-fold x 3 (mean ± std over folds). Model weights are chosen by validation loss in both settings.

| Model | Holdout Accuracy | Holdout ROC-AUC [95% CI] | Holdout Brier | CV Accuracy | CV ROC-AUC | CV Brier |
| --- | --- | --- | --- | --- | --- | --- |
| TabTransformer | 0.7989 | 0.8855 [0.8311, 0.9305] | 0.1377 | 0.7916 ± 0.0260 | 0.8659 ± 0.0285 | 0.1471 ± 0.0163 |
| GCN (Kipf & Welling) | 0.7989 | 0.8871 [0.8370, 0.9335] | 0.1350 | 0.8112 ± 0.0255 | 0.8781 ± 0.0259 | 0.1381 ± 0.0158 |
| GAT (Velickovic et al.) | 0.8098 | 0.8815 [0.8311, 0.9291] | 0.1376 | 0.8021 ± 0.0276 | 0.8737 ± 0.0327 | 0.1421 ± 0.0205 |
| Multimodal Cross-Attention (synthetic-signal demo) † | 0.6630 | 0.6985 [0.6201, 0.7775] | 0.2179 | 0.6837 ± 0.0297 | 0.7037 ± 0.0426 | 0.2159 ± 0.0117 |

† Demonstration only: the signal channel is synthesised from the tabular features, so this row is not a like-for-like comparison.

## 2. Methodology & Rigor Summary
- **Temperature Scaling (Guo et al., 2017):** Validation-tuned optimal T `1.455` on trained `HeartMLP` reduced holdout test ECE from `0.0727` to `0.0674` (Validation ECE: `0.0731` -> `0.0641`).
- **Deep Ensembles (Lakshminarayanan et al., 2017):** Average epistemic predictive variance is `0.0049` across 184 holdout patients.
- **Split & Mondrian Conformal Prediction (Angelopoulos & Bates, 2021):** Calibrated and evaluated on the same (uncalibrated) MLP probabilities; the 90% finite-sample coverage target achieved `0.8859` empirical coverage (Case: `0.8725`, Control: `0.9024`) with average set size `1.255` and `0.255` clinical abstention rate.
- **Conformal Efficiency-Coverage Trade-Off:** Operating at $\alpha=0.15$ (85% coverage guarantee) cuts the ambiguous deferral rate from `25.5%` down to `15.2%` with `86.4%` empirical test coverage.
- **Causal IPTW Analysis (Pearl & Mackenzie, 2018):** Observational risk difference is `0.4463`, whereas confounder-isolated Inverse Probability Weighting (IPTW) reveals a causal ATE of `0.0750`, demonstrating a confounding bias of `0.3712`.
- **Covariate Balance (IPTW, treatment `exang`):** maximum |SMD| fell from `0.961` to `0.264`, but `7` of `12` covariates remain at or above the 0.10 balance threshold (ca, cp, oldpeak, sex, slope, thal, trestbps), so residual measured confounding cannot be ruled out.
- **Doubly Robust AIPW & E-Value (Chernozhukov et al., 2018; VanderWeele & Ding, 2017):**
  - *Symptomatic Proxy (exang):* Cross-fitted AIPW ATE = `0.0596` (95% CI: [-0.0473, 0.1666]), E-value = `1.457`.
  - *Modifiable Metabolic Treatment (Hypertension SBP >= 140):* Cross-fitted AIPW ATE = `-0.0133` (95% CI: [-0.0691, 0.0424]), E-value = `1.184` (CI E-value = `1.000`).
- **Algorithmic Fairness (Barocas et al., 2020):** Validation-tuned Equal Opportunity thresholds (t0=0.295, t1=0.505) reduced holdout test TPR disparity from `0.2261` to `0.2154` (4.7% reduction), while the false-positive rate moved from `0.268` to `0.268` and accuracy from `0.788` to `0.783`.
- **Clinical Counterfactual Recourse (Wachter et al., 2017; Ustun et al., 2019):** Achieved `80.0%` recourse validity across high-risk test patients with `7.76` average features modified, reducing mean predicted risk by `0.519` while strictly maintaining `0.000000` immutable demographic violation.
- **Closed-Loop Conformal Recourse Transition:** `100.0%` of Conformal High-Risk patients ($\hat{C}(x) = \{1\}$) successfully transitioned into Low-Risk $\{0\}$ or Deferral $\{0, 1\}$ following sparse actionable recourse.

## 3. LOCO Domain-Shift Conformal Generalization Table

| Evaluation Center | N Samples | Disease Prevalence | Fixed Threshold Acc | 90% Conformal Coverage | Avg Set Size | Abstain Rate |
| --- | --- | --- | --- | --- | --- | --- |
| cleveland | 303 | 45.9% | 65.3% | 80.9% | 1.22 | 22.4% |
| hungarian | 293 | 36.2% | 77.5% | 91.1% | 1.32 | 32.1% |
| switzerland | 123 | 93.5% | 64.2% | 87.8% | 1.38 | 38.2% |
| va_long_beach | 199 | 74.4% | 73.9% | 88.9% | 1.32 | 31.7% |