# Graph-Augmented Clinical Transformer — Multi-Cohort & Large-Scale Benchmark

## 1. UCI Multi-Center Heart Disease (N=918, 5-Fold Stratified CV)

| Model | ROC-AUC | PR-AUC | Accuracy | F1 | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TabTransformer (No Graph) | 0.8786 ± 0.0198 | 0.8854 ± 0.0293 | 0.8115 ± 0.0181 | 0.8289 ± 0.0154 | 0.1407 ± 0.0119 |
| Inductive GAT (No Transformer) | 0.8785 ± 0.0271 | 0.8828 ± 0.0244 | 0.8170 ± 0.0249 | 0.8317 ± 0.0254 | 0.1379 ± 0.0173 |
| Graph-Augmented Transformer (Ours) | 0.8887 ± 0.0150 | 0.8874 ± 0.0269 | 0.8181 ± 0.0244 | 0.8329 ± 0.0250 | 0.1350 ± 0.0094 |

## 2. MIMIC-IV ICU Mortality (N=275, 5-Fold Patient-Grouped CV)

| Model | ROC-AUC | PR-AUC | Accuracy | F1 | Brier |
| :--- | :---: | :---: | :---: | :---: | :---: |
| TabTransformer (No Graph) | 0.7318 ± 0.1547 | 0.2712 ± 0.1582 | 0.9021 ± 0.0419 | 0.1371 ± 0.1718 | 0.0850 ± 0.0109 |
| Inductive GAT (No Transformer) | 0.7191 ± 0.1443 | 0.2710 ± 0.2112 | 0.9091 ± 0.0380 | 0.1238 ± 0.1524 | 0.0865 ± 0.0175 |
| Graph-Augmented Transformer (Ours) | 0.6951 ± 0.1269 | 0.1940 ± 0.1349 | 0.8474 ± 0.0728 | 0.0444 ± 0.0889 | 0.1198 ± 0.0173 |

## 3. Large-Scale Cardiovascular Cohort (OpenML 45547, N=68,589, Mini-Batch Inductive)

| Model | ROC-AUC | PR-AUC | Accuracy | F1 | Brier | ECE | Train Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TabTransformer (No Graph) | 0.7951 | 0.7738 | 0.7279 | 0.7132 | 0.1840 | 0.0135 | 19.4s |
| Graph-Augmented Transformer (Ours) | 0.8009 | 0.7826 | 0.7363 | 0.7230 | 0.1817 | 0.0244 | 32.7s |
