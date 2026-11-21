# External Site Transfer & Local Recalibration Benchmark (`calibrate-site`)

Base bundle: `3.1.0-plr-unimp-zeropickle-cds` (OpenML 45547, N=68,589).
Each target clinic is split 50/50 into a local calibration slice and an untouched local holdout slice.

| Target Clinic / Cohort | Eval N | Prevalence | Zero-Shot ECE | Recalibrated ECE | Zero-Shot 90% Cov | Recalibrated 90% Cov | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Framingham Heart Study (Real Prospective 10-Yr CHD, N=4,172)** | 2086 | 15.0% | 0.4233 | **0.0061** | 66.8% | **90.9%** | 0.6759 |
| **UCI 4-Center Hospital Consortium (External Dataset)** | 459 | 55.3% | 0.1841 | **0.0370** | 78.0% | **88.9%** | 0.5507 |
| **Geriatric Cardiology Clinic (Age >= 58, High Prevalence)** | 1450 | 63.7% | 0.0254 | **0.0236** | 92.6% | **92.3%** | 0.7226 |
| **Preventive Wellness Clinic (Age < 50, Low Prevalence)** | 1584 | 35.0% | 0.0223 | **0.0266** | 87.1% | **90.3%** | 0.8309 |
