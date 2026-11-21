# Clinical Guideline Benchmark: CardioGraph CDS v3 vs. Framingham Risk Score (2008)

Evaluated on **10,288** untouched holdout screening patients (OpenML 45547).
Paired DeLong test: **+0.0991 ROC-AUC** (`z = 22.913`, `p = 3.48e-116`).

| Model | Holdout ROC-AUC [95% CI] | PR-AUC | Brier ↓ | ECE ↓ | Net Benefit @ 50% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CardioGraph CDS v3 (PLR+UniMP Ensemble)** | **0.7996** [0.7910, 0.8074] | **0.7808** | **0.1824** | **0.0180** | **0.2226** |
| **Framingham 2008 (Recalibrated)** | 0.7005 [0.6904, 0.7100] | 0.6594 | 0.2213 | 0.0521 | 0.1325 |
| **Framingham 2008 (Raw 10-yr)** | 0.7005 [0.6904, 0.7100] | 0.6594 | 0.3988 | 0.4045 | — |
