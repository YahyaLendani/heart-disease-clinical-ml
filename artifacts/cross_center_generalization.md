# Leave-One-Center-Out (LOCO) Cross-Hospital Generalization

This experiment tests hospital generalizability by training on 3 clinical centers and testing on an unseen 4th center.

### Transfer ROC-AUC by Unseen Hospital Center

| test_center | Ensemble (Transfer) | Logistic Regression | Random Forest | XGBoost | Mean ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| cleveland | 0.8432 | 0.5833 | 0.8626 | 0.8441 | 0.7833 |
| hungarian | 0.8761 | 0.8588 | 0.8696 | 0.8675 | 0.8680 |
| switzerland | 0.7043 | 0.6359 | 0.7239 | 0.7163 | 0.6951 |
| va_long_beach | 0.6918 | 0.7085 | 0.6848 | 0.6689 | 0.6885 |

### Complete Metric Breakdown

| test_center | model | train_size | test_size | transfer_accuracy | transfer_f1 | transfer_roc_auc | transfer_pr_auc | transfer_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cleveland | Logistic Regression | 615 | 303 | 0.6205 | 0.4601 | 0.5833 | 0.5823 | 0.3019 |
| cleveland | Random Forest | 615 | 303 | 0.7822 | 0.7591 | 0.8626 | 0.8577 | 0.1600 |
| cleveland | XGBoost | 615 | 303 | 0.7525 | 0.7331 | 0.8441 | 0.8298 | 0.1630 |
| cleveland | Ensemble (Transfer) | 615 | 303 | 0.7855 | 0.7490 | 0.8432 | 0.7912 | 0.1672 |
| hungarian | Logistic Regression | 625 | 293 | 0.7918 | 0.7336 | 0.8588 | 0.8226 | 0.1526 |
| hungarian | Random Forest | 625 | 293 | 0.7099 | 0.6931 | 0.8696 | 0.8044 | 0.1930 |
| hungarian | XGBoost | 625 | 293 | 0.7679 | 0.7258 | 0.8675 | 0.7761 | 0.1581 |
| hungarian | Ensemble (Transfer) | 625 | 293 | 0.7713 | 0.7287 | 0.8761 | 0.8232 | 0.1605 |
| switzerland | Logistic Regression | 795 | 123 | 0.5691 | 0.7072 | 0.6359 | 0.9502 | 0.2671 |
| switzerland | Random Forest | 795 | 123 | 0.6911 | 0.8061 | 0.7239 | 0.9686 | 0.1867 |
| switzerland | XGBoost | 795 | 123 | 0.6098 | 0.7419 | 0.7163 | 0.9585 | 0.2212 |
| switzerland | Ensemble (Transfer) | 795 | 123 | 0.6423 | 0.7684 | 0.7043 | 0.9626 | 0.2197 |
| va_long_beach | Logistic Regression | 719 | 199 | 0.7136 | 0.8055 | 0.7085 | 0.8678 | 0.1868 |
| va_long_beach | Random Forest | 719 | 199 | 0.6683 | 0.7643 | 0.6848 | 0.8443 | 0.2031 |
| va_long_beach | XGBoost | 719 | 199 | 0.6533 | 0.7454 | 0.6689 | 0.8239 | 0.2200 |
| va_long_beach | Ensemble (Transfer) | 719 | 199 | 0.6884 | 0.7801 | 0.6918 | 0.8473 | 0.1972 |
