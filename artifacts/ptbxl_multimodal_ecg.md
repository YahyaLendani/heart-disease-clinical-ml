# Real 12-Lead ECG Multimodal Cross-Attention Benchmark (PhysioNet PTB-XL v1.0.3)

Evaluated on **200** untouched official **Fold-10** patients (`787` Train folds 1–8 / `200` Val fold 9, **strict zero `patient_id` overlap**; Task: Myocardial Infarction / ST-T Ischemia `MI+STTC` vs. Normal Sinus `NORM`).
- **Multimodal vs. Tabular Demographics Only:** paired DeLong `z = 5.784`, `p = 7.29e-09`
- **Mean Learned ECG Gate Reliance:** `g_ecg = 0.407`

| Architecture | Holdout ROC-AUC [95% CI] | PR-AUC | F1 | Sensitivity | Specificity | Brier ↓ | ECE ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tabular Demographics Only** | 0.7621 [0.6947, 0.8217] | 0.7506 | 0.6761 | 0.7200 | 0.5900 | 0.1977 | 0.0770 |
| **12-Lead ECG 1D-ResNet** | 0.8935 [0.8484, 0.9300] | 0.9051 | 0.7337 | 0.6200 | 0.9300 | 0.1755 | 0.1633 |
| **Multimodal Cross-Attention (ECG + Tabular)** | **0.9462** [0.9175, 0.9711] | **0.9529** | **0.8411** | **0.9000** | **0.7600** | **0.1115** | **0.0805** |
