# External Site Transfer & Local Recalibration Benchmark (`calibrate-site`)

Base bundle: `3.0.0-plr-unimp-3paradigm-cds` (OpenML 45547, N=68,589).
Each target clinic is split 50/50 into a local calibration slice and an untouched local holdout slice.

| Target Clinic / Cohort | Eval N | Prevalence | Zero-Shot ECE | Recalibrated ECE | Zero-Shot 90% Cov | Recalibrated 90% Cov | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UCI 4-Center Hospital Consortium (External Dataset)** | 459 | 55.3% | 0.1875 | **0.0425** | 78.2% | **88.7%** | 0.5437 |
| **Geriatric Cardiology Clinic (Age >= 58, High Prevalence)** | 1450 | 63.7% | 0.0246 | **0.0266** | 92.7% | **92.5%** | 0.7252 |
| **Preventive Wellness Clinic (Age < 50, Low Prevalence)** | 1584 | 35.0% | 0.0254 | **0.0248** | 87.5% | **90.2%** | 0.8341 |
