# Clinical Guideline Showdown: CardioGraph CDS v3 vs. ESC SCORE2 (2021) & Framingham (2008)

Evaluated on **10,288** untouched holdout screening patients (OpenML 45547).
- **vs. European ESC SCORE2 (2021, High-Risk / Poland):** **+0.1232 ROC-AUC** (paired DeLong `z = 25.809`, `p = 6.96e-147`)
- **vs. US Framingham (2008):** **+0.0991 ROC-AUC** (paired DeLong `z = 22.913`, `p = 3.48e-116`)

| Model / Clinical Guideline | Holdout ROC-AUC [95% CI] | PR-AUC | Brier ↓ | ECE ↓ | Net Benefit @ 50% |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CardioGraph CDS v3 (PLR+UniMP Ensemble)** | **0.7996** [0.7910, 0.8074] | **0.7808** | **0.1824** | **0.0180** | **0.2226** |
| **ESC SCORE2 2021 High-Risk/Poland (Recalibrated)** | 0.6764 [0.6667, 0.6862] | 0.6422 | 0.2259 | 0.0301 | 0.1227 |
| **ESC SCORE2 2021 High-Risk/Poland (Raw 10-yr)** | 0.6764 [0.6667, 0.6862] | 0.6422 | 0.4706 | 0.4744 | — |
| **Framingham 2008 (Recalibrated)** | 0.7005 [0.6904, 0.7100] | 0.6594 | 0.2213 | 0.0521 | 0.1325 |
| **Framingham 2008 (Raw 10-yr)** | 0.7005 [0.6904, 0.7100] | 0.6594 | 0.3988 | 0.4045 | — |
