# Reading List Implementation Benchmark Summary

This report summarizes the empirical performance of all architectures and methods implemented from `AI_Thesis_Reading-List.csv`.

## 1. Architectural Benchmark Table

Holdout: one stratified 60/20/20 split (61 test patients), with stratified bootstrap 95% CIs. CV: repeated stratified 5-fold x 3 (mean ± std over folds). Model weights are chosen by validation loss in both settings.

| Model | Holdout Accuracy | Holdout ROC-AUC [95% CI] | Holdout Brier | CV Accuracy | CV ROC-AUC | CV Brier |
| --- | --- | --- | --- | --- | --- | --- |
| TabTransformer | 0.8197 | 0.9091 [0.8323, 0.9697] | 0.1211 | 0.8047 ± 0.0420 | 0.8897 ± 0.0296 | 0.1396 ± 0.0233 |
| GCN (Kipf & Welling) | 0.7705 | 0.9037 [0.8236, 0.9654] | 0.1394 | 0.8190 ± 0.0286 | 0.9096 ± 0.0277 | 0.1232 ± 0.0170 |
| GAT (Velickovic et al.) | 0.8197 | 0.9015 [0.8247, 0.9643] | 0.1364 | 0.8068 ± 0.0392 | 0.9012 ± 0.0290 | 0.1290 ± 0.0220 |
| Multimodal Cross-Attention (synthetic-signal demo) † | 0.6721 | 0.6905 [0.5541, 0.8214] | 0.2227 | 0.6457 ± 0.0793 | 0.6817 ± 0.0690 | 0.2276 ± 0.0258 |

† Demonstration only: the signal channel is synthesised from the tabular features, so this row is not a like-for-like comparison.

## 2. Methodology & Rigor Summary
- **Temperature Scaling (Guo et al., 2017):** Validation-tuned optimal T `1.522` on trained `HeartMLP` reduced holdout test ECE from `0.1060` to `0.0787` (Validation ECE: `0.1060` -> `0.1245`).
- **Deep Ensembles (Lakshminarayanan et al., 2017):** Average epistemic predictive variance is `0.0106` across 61 holdout patients.
- **Split & Mondrian Conformal Prediction (Angelopoulos & Bates, 2021):** Calibrated and evaluated on the same (uncalibrated) MLP probabilities; the 90% finite-sample coverage target achieved `0.9508` empirical coverage (Case: `1.0000`, Control: `0.9091`) with average set size `1.443` and `0.443` clinical abstention rate.
- **Conformal Efficiency-Coverage Trade-Off:** Operating at $\alpha=0.15$ (85% coverage guarantee) cuts the ambiguous deferral rate from `44.3%` down to `23.0%` with `88.5%` empirical test coverage.
- **Causal IPTW Analysis (Pearl & Mackenzie, 2018):** Observational risk difference is `0.4623`, whereas confounder-isolated Inverse Probability Weighting (IPTW) reveals a causal ATE of `0.0657`, demonstrating a confounding bias of `0.3966`.
- **Doubly Robust AIPW & E-Value (Chernozhukov et al., 2018; VanderWeele & Ding, 2017):**
  - *Symptomatic Proxy (exang):* Cross-fitted AIPW ATE = `-0.0060` (95% CI: [-0.1844, 0.1724]), E-value = `1.130`.
  - *Modifiable Metabolic Treatment (Hypertension SBP >= 140):* Cross-fitted AIPW ATE = `0.0712` (95% CI: [-0.0420, 0.1845]), E-value = `1.594` (CI E-value = `1.000`).
- **Algorithmic Fairness (Barocas et al., 2020):** Validation-tuned Equal Opportunity thresholds (t0=0.050, t1=0.050) reduced holdout test TPR disparity from `0.2381` to `0.0000` (100.0% reduction), while the false-positive rate moved from `0.212` to `0.485` and accuracy from `0.803` to `0.738` — both thresholds are near 0, so this parity comes from flagging almost every patient and is not a useful operating point.
- **Clinical Counterfactual Recourse (Wachter et al., 2017; Ustun et al., 2019):** Achieved `100.0%` recourse validity across high-risk test patients with `8.56` average features modified, reducing mean predicted mortality risk by `-0.592` while strictly maintaining `0.000000` immutable demographic violation.
- **Closed-Loop Conformal Recourse Transition:** `100.0%` of Conformal High-Risk patients ($\hat{C}(x) = \{1\}$) successfully transitioned into Low-Risk $\{0\}$ or Deferral $\{0, 1\}$ following sparse actionable recourse.

## 3. LOCO Domain-Shift Conformal Generalization Table

| Evaluation Center | N Samples | Disease Prevalence | Fixed Threshold Acc | 90% Conformal Coverage | Avg Set Size | Abstain Rate |
| --- | --- | --- | --- | --- | --- | --- |
| cleveland | 303 | 45.9% | 71.3% | 87.1% | 1.28 | 27.7% |
| hungarian | 293 | 36.2% | 80.5% | 93.2% | 1.35 | 34.8% |
| switzerland | 123 | 93.5% | 54.5% | 82.1% | 1.43 | 43.1% |
| va_long_beach | 199 | 74.4% | 69.8% | 84.4% | 1.25 | 24.6% |