# Demographic Fairness and Clinical Disparity Analysis

Evaluates whether the clinical risk model demonstrates parity across biological sex and age brackets.

### Subgroup Performance Table

| subgroup | n_samples | prevalence_pct | accuracy | f1 | roc_auc | sensitivity_tpr | specificity_tnr | false_negative_rate | false_positive_rate | selection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All Patients | 184 | 55.4348 | 0.8533 | 0.8696 | 0.9280 | 0.8824 | 0.8171 | 0.1176 | 0.1829 | 0.5707 |
| Female Patients (sex=0) | 35 | 22.8571 | 0.9143 | 0.8000 | 0.8935 | 0.7500 | 0.9630 | 0.2500 | 0.0370 | 0.2000 |
| Male Patients (sex=1) | 149 | 63.0872 | 0.8389 | 0.8750 | 0.9195 | 0.8936 | 0.7455 | 0.1064 | 0.2545 | 0.6577 |
| Younger (<50 yrs) | 59 | 42.3729 | 0.8644 | 0.8182 | 0.9694 | 0.7200 | 0.9706 | 0.2800 | 0.0294 | 0.3220 |
| Middle-aged (50-64 yrs) | 104 | 57.6923 | 0.8462 | 0.8769 | 0.9170 | 0.9500 | 0.7045 | 0.0500 | 0.2955 | 0.6731 |
| Senior (>=65 yrs) | 21 | 80.9524 | 0.8571 | 0.9091 | 0.9265 | 0.8824 | 0.7500 | 0.1176 | 0.2500 | 0.7619 |

### Key Demographic Fairness Findings

- **Equal Opportunity Difference (Sex):** `0.1436`
- **Disparate Impact Ratio (Female vs. Male):** `0.3041`
- **Missed Disease Rate (FNR):** Female = `25.0%`, Male = `10.6%`
