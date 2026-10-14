# Demographic Fairness and Clinical Disparity Analysis

Evaluates whether the clinical risk model demonstrates parity across biological sex and age brackets.

### Subgroup Performance Table

| subgroup | n_samples | prevalence_pct | accuracy | f1 | roc_auc | sensitivity_tpr | specificity_tnr | false_negative_rate | false_positive_rate | selection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All Patients | 61 | 45.9016 | 0.7869 | 0.7797 | 0.8810 | 0.8214 | 0.7576 | 0.1786 | 0.2424 | 0.5082 |
| Female Patients (sex=0) | 23 | 30.4348 | 0.7826 | 0.6667 | 0.8839 | 0.7143 | 0.8125 | 0.2857 | 0.1875 | 0.3478 |
| Male Patients (sex=1) | 38 | 55.2632 | 0.7895 | 0.8182 | 0.8655 | 0.8571 | 0.7059 | 0.1429 | 0.2941 | 0.6053 |
| Younger (<50 yrs) | 18 | 27.7778 | 0.9444 | 0.9091 | 1.0000 | 1.0000 | 0.9231 | 0.0000 | 0.0769 | 0.3333 |
| Middle-aged (50-64 yrs) | 34 | 52.9412 | 0.7059 | 0.7368 | 0.8438 | 0.7778 | 0.6250 | 0.2222 | 0.3750 | 0.5882 |
| Senior (>=65 yrs) | 9 | 55.5556 | 0.7778 | 0.8000 | 0.8000 | 0.8000 | 0.7500 | 0.2000 | 0.2500 | 0.5556 |

### Key Demographic Fairness Findings

- **Equal Opportunity Difference (Sex):** `0.1429`
- **Disparate Impact Ratio (Female vs. Male):** `0.5747`
- **Missed Disease Rate (FNR):** Female = `28.6%`, Male = `14.3%`
