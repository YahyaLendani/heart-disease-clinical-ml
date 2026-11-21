# External Site Transfer: Zero-Shot vs. Zero-Label Weighted Conformal vs. Labeled `calibrate-site`

Base bundle: `3.1.0-plr-unimp-zeropickle-cds` (OpenML 45547, N=68,589).
Compares three deployment regimes across shifted target clinics (50/50 local split):
1. **Zero-Shot:** Unadapted source bundle.
2. **Zero-Label Weighted Conformal (Tibshirani et al., NeurIPS 2019):** Density-ratio importance weighting using *unlabeled* target features only.
3. **2-Second Labeled `calibrate-site`:** Post-blend Platt recalibration + local Mondrian quantiles.

| Target Clinic / Cohort | Eval N | Prev. | Zero-Shot ECE | Recal. ECE | Zero-Shot 90% Cov | Zero-Label Weighted 90% Cov | Labeled `calibrate-site` 90% Cov | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Framingham Heart Study (Real Prospective 10-Yr CHD, N=4,172)** | 2086 | 15.0% | 0.4233 | **0.0061** | 66.8% | 78.8% | **90.9%** | 0.6759 |
| **UCI 4-Center Hospital Consortium (External Dataset)** | 459 | 55.3% | 0.1841 | **0.0370** | 78.0% | 78.6% | **88.9%** | 0.5507 |
| **Geriatric Cardiology Clinic (Age >= 58, High Prevalence)** | 1450 | 63.7% | 0.0254 | **0.0236** | 92.6% | 86.7% | **92.3%** | 0.7226 |
| **Preventive Wellness Clinic (Age < 50, Low Prevalence)** | 1584 | 35.0% | 0.0223 | **0.0266** | 87.1% | 92.9% | **90.3%** | 0.8309 |
