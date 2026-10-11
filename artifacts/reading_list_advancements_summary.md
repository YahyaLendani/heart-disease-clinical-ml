# Reading List Implementation Benchmark Summary

This report summarizes the empirical performance of all architectures and methods implemented from `AI_Thesis_Reading-List.csv`.

## 1. Architectural Benchmark Table

| Model | Test Accuracy | Test F1 | Test ROC-AUC | Test PR-AUC | Test Brier Score |
| --- | --- | --- | --- | --- | --- |
| TabTransformer | 0.7869 | 0.8060 | 0.8517 | 0.8756 | 0.1540 |
| GCN (Kipf & Welling) | 0.7869 | 0.8060 | 0.8842 | 0.9006 | 0.1399 |
| GAT (Velickovic et al.) | 0.8525 | 0.8657 | 0.8820 | 0.9002 | 0.1338 |
| Multimodal Cross-Attention (ResNet/ViT) | 0.4098 | 0.4375 | 0.4623 | 0.4699 | 0.3761 |

## 2. Methodology & Rigor Summary
- **Temperature Scaling (Guo et al., 2017):** Recalibrated ECE from `0.3395` to `0.2414` (Optimal T: `1.800`).
- **Deep Ensembles (Lakshminarayanan et al., 2017):** Average epistemic predictive variance is `0.0201` across 61 holdout patients.
- **Causal Analysis (Pearl & Mackenzie, 2018):** Observational risk difference is `-0.4623`, whereas Inverse Probability Weighting (IPTW) reveals a causal ATE of `-0.4585`, demonstrating a confounding bias of `-0.0037`.
- **Algorithmic Fairness (Barocas et al., 2020):** Equal Opportunity threshold tuning reduced demographic sensitivity disparity by `100.0%` (from `0.0238` to `0.0000`).