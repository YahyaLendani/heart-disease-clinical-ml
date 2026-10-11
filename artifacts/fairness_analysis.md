# Demographic Fairness and Clinical Disparity Analysis

Evaluates whether the clinical risk model demonstrates parity across biological sex and age brackets.

### Subgroup Performance Table

| subgroup | n_samples | prevalence_pct | accuracy | f1 | roc_auc | sensitivity_tpr | specificity_tnr | false_negative_rate | false_positive_rate | selection_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All Patients | 61 | 54.0984 | 0.7377 | 0.7576 | 0.8517 | 0.7576 | 0.7143 | 0.2424 | 0.2857 | 0.5410 |
| Female Patients (sex=0) | 18 | 72.2222 | 0.7222 | 0.8000 | 0.8308 | 0.7692 | 0.6000 | 0.2308 | 0.4000 | 0.6667 |
| Male Patients (sex=1) | 43 | 46.5116 | 0.7442 | 0.7317 | 0.8522 | 0.7500 | 0.7391 | 0.2500 | 0.2609 | 0.4884 |
| Younger (<50 yrs) | 14 | 85.7143 | 0.7857 | 0.8800 | 0.7917 | 0.9167 | 0.0000 | 0.0833 | 1.0000 | 0.9286 |
| Middle-aged (50-64 yrs) | 40 | 50.0000 | 0.7250 | 0.7027 | 0.8700 | 0.6500 | 0.8000 | 0.3500 | 0.2000 | 0.4250 |
| Senior (>=65 yrs) | 7 | 14.2857 | 0.7143 | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.0000 | 0.3333 | 0.4286 |

### Key Demographic Fairness Findings

- **Equal Opportunity Difference (Sex):** `0.0192`
- **Disparate Impact Ratio (Female vs. Male):** `1.3651`
- **Missed Disease Rate (FNR):** Female = `23.1%`, Male = `25.0%`
