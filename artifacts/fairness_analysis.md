# Demographic Fairness and Clinical Disparity Analysis

Evaluates whether the clinical risk model demonstrates parity across biological sex and age brackets.

### Subgroup Performance Table

| subgroup | n_samples | prevalence_pct | accuracy | f1 | roc_auc | sensitivity_tpr | specificity_tnr | false_negative_rate | false_positive_rate | selection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All Patients | 184 | 55.4348 | 0.8478 | 0.8692 | 0.9075 | 0.9118 | 0.7683 | 0.0882 | 0.2317 | 0.6087 |
| Female Patients (sex=0) | 35 | 22.8571 | 0.8857 | 0.7500 | 0.9213 | 0.7500 | 0.9259 | 0.2500 | 0.0741 | 0.2286 |
| Male Patients (sex=1) | 149 | 63.0872 | 0.8389 | 0.8788 | 0.8870 | 0.9255 | 0.6909 | 0.0745 | 0.3091 | 0.6980 |
| Younger (<50 yrs) | 59 | 42.3729 | 0.8983 | 0.8750 | 0.9482 | 0.8400 | 0.9412 | 0.1600 | 0.0588 | 0.3898 |
| Middle-aged (50-64 yrs) | 104 | 57.6923 | 0.8269 | 0.8636 | 0.8807 | 0.9500 | 0.6591 | 0.0500 | 0.3409 | 0.6923 |
| Senior (>=65 yrs) | 21 | 80.9524 | 0.8095 | 0.8824 | 0.9118 | 0.8824 | 0.5000 | 0.1176 | 0.5000 | 0.8095 |

### Key Demographic Fairness Findings

- **Equal Opportunity Difference (Sex):** `0.1755`
- **Disparate Impact Ratio (Female vs. Male):** `0.3275`
- **Missed Disease Rate (FNR):** Female = `25.0%`, Male = `7.4%`
