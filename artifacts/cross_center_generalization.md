# Leave-One-Center-Out (LOCO) Cross-Hospital Generalization

This experiment tests hospital generalizability by training on 3 clinical centers and testing on an unseen 4th center.

### Transfer ROC-AUC by Unseen Hospital Center

| test_center | Ensemble (Transfer) | Logistic Regression | Random Forest | XGBoost | Mean ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| cleveland | 0.8171 | 0.6014 | 0.8622 | 0.8147 | 0.7738 |
| hungarian | 0.8732 | 0.8819 | 0.8608 | 0.8510 | 0.8667 |
| switzerland | 0.7065 | 0.6315 | 0.7511 | 0.7163 | 0.7014 |
| va_long_beach | 0.7201 | 0.7373 | 0.7141 | 0.6919 | 0.7158 |

### Complete Metric Breakdown

| test_center | model | train_size | test_size | transfer_accuracy | transfer_f1 | transfer_roc_auc | transfer_pr_auc | transfer_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cleveland | Logistic Regression | 615 | 303 | 0.6205 | 0.5228 | 0.6014 | 0.5911 | 0.2865 |
| cleveland | Random Forest | 615 | 303 | 0.7327 | 0.7429 | 0.8622 | 0.8532 | 0.1674 |
| cleveland | XGBoost | 615 | 303 | 0.6799 | 0.7034 | 0.8147 | 0.7991 | 0.1925 |
| cleveland | Ensemble (Transfer) | 615 | 303 | 0.7360 | 0.7351 | 0.8171 | 0.7596 | 0.1790 |
| hungarian | Logistic Regression | 625 | 293 | 0.7474 | 0.7218 | 0.8819 | 0.8463 | 0.1796 |
| hungarian | Random Forest | 625 | 293 | 0.7235 | 0.7076 | 0.8608 | 0.7793 | 0.2038 |
| hungarian | XGBoost | 625 | 293 | 0.7406 | 0.7099 | 0.8510 | 0.7551 | 0.1827 |
| hungarian | Ensemble (Transfer) | 625 | 293 | 0.7509 | 0.7245 | 0.8732 | 0.8073 | 0.1827 |
| switzerland | Logistic Regression | 795 | 123 | 0.6504 | 0.7772 | 0.6315 | 0.9493 | 0.2317 |
| switzerland | Random Forest | 795 | 123 | 0.6504 | 0.7749 | 0.7511 | 0.9712 | 0.1993 |
| switzerland | XGBoost | 795 | 123 | 0.6423 | 0.7708 | 0.7163 | 0.9635 | 0.2160 |
| switzerland | Ensemble (Transfer) | 795 | 123 | 0.6748 | 0.7959 | 0.7065 | 0.9641 | 0.2113 |
| va_long_beach | Logistic Regression | 719 | 199 | 0.7337 | 0.8239 | 0.7373 | 0.8742 | 0.1692 |
| va_long_beach | Random Forest | 719 | 199 | 0.7387 | 0.8231 | 0.7141 | 0.8674 | 0.1882 |
| va_long_beach | XGBoost | 719 | 199 | 0.7186 | 0.8028 | 0.6919 | 0.8422 | 0.2054 |
| va_long_beach | Ensemble (Transfer) | 719 | 199 | 0.7387 | 0.8231 | 0.7201 | 0.8642 | 0.1813 |
