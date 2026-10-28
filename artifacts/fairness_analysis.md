# Demographic Fairness and Clinical Disparity Analysis

Evaluates whether the clinical risk model demonstrates parity across biological sex and age brackets.

### Subgroup Performance Table

| subgroup | n_samples | prevalence_pct | accuracy | f1 | roc_auc | sensitivity_tpr | specificity_tnr | false_negative_rate | false_positive_rate | selection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All Patients | 184 | 55.4348 | 0.8533 | 0.8708 | 0.9303 | 0.8922 | 0.8049 | 0.1078 | 0.1951 | 0.5815 |
| Female Patients (sex=0) | 35 | 22.8571 | 0.9143 | 0.8000 | 0.9074 | 0.7500 | 0.9630 | 0.2500 | 0.0370 | 0.2000 |
| Male Patients (sex=1) | 149 | 63.0872 | 0.8389 | 0.8763 | 0.9217 | 0.9043 | 0.7273 | 0.0957 | 0.2727 | 0.6711 |
| Younger (<50 yrs) | 59 | 42.3729 | 0.8644 | 0.8182 | 0.9741 | 0.7200 | 0.9706 | 0.2800 | 0.0294 | 0.3220 |
| Middle-aged (50-64 yrs) | 104 | 57.6923 | 0.8558 | 0.8855 | 0.9193 | 0.9667 | 0.7045 | 0.0333 | 0.2955 | 0.6827 |
| Senior (>=65 yrs) | 21 | 80.9524 | 0.8095 | 0.8824 | 0.9412 | 0.8824 | 0.5000 | 0.1176 | 0.5000 | 0.8095 |

### Key Demographic Fairness Findings

- **Equal Opportunity Difference (Sex):** `0.1543`
- **Disparate Impact Ratio (Female vs. Male):** `0.2980`
- **Missed Disease Rate (FNR):** Female = `25.0%`, Male = `9.6%`
