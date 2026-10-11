# Reading List Implementation Benchmark Summary

This report summarizes the empirical performance of all architectures and methods implemented from `AI_Thesis_Reading-List.csv`.

## 1. Architectural Benchmark Table

| Model | Test Accuracy | Test F1 | Test ROC-AUC | Test PR-AUC | Test Brier Score |
| --- | --- | --- | --- | --- | --- |
| TabTransformer | 0.7869 | 0.8060 | 0.8528 | 0.8775 | 0.1556 |
| GCN (Kipf & Welling) | 0.7705 | 0.7879 | 0.8680 | 0.8872 | 0.1516 |
| GAT (Velickovic et al.) | 0.8033 | 0.8235 | 0.9015 | 0.9316 | 0.1309 |
| Multimodal Cross-Attention (ResNet/ViT surrogate) | 0.5902 | 0.5098 | 0.5963 | 0.6942 | 0.2666 |

## 2. Methodology & Rigor Summary
- **Temperature Scaling (Guo et al., 2017):** Validation-tuned optimal T `1.824` reduced holdout test ECE from `0.1653` to `0.1001` (Validation ECE: `0.3656` -> `0.2665`).
- **Deep Ensembles (Lakshminarayanan et al., 2017):** Average epistemic predictive variance is `0.0194` across 61 holdout patients.
- **Causal Analysis (Pearl & Mackenzie, 2018):** Observational risk difference is `-0.4623`, whereas Inverse Probability Weighting (IPTW) reveals a causal ATE of `-0.4585`, demonstrating a confounding bias of `-0.0037`.
- **Algorithmic Fairness (Barocas et al., 2020):** Validation-tuned Equal Opportunity thresholds (t0=0.050, t1=0.050) reduced holdout test TPR disparity from `0.1115` to `0.0000` (100.0% reduction).