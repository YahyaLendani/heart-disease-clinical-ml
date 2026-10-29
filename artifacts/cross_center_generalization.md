# Leave-One-Center-Out (LOCO) Cross-Hospital Generalization

This experiment tests hospital generalizability by training on 3 clinical centers and testing on an unseen 4th center.

### Transfer ROC-AUC by Unseen Hospital Center

| test_center | Ensemble (Transfer) | Logistic Regression | Random Forest | XGBoost | Mean ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| cleveland | 0.8411 | 0.5833 | 0.8626 | 0.8406 | 0.7819 |
| hungarian | 0.8761 | 0.8588 | 0.8696 | 0.8676 | 0.8680 |
| switzerland | 0.7065 | 0.6359 | 0.7239 | 0.7147 | 0.6952 |
| va_long_beach | 0.6930 | 0.7085 | 0.6848 | 0.6716 | 0.6895 |

### Complete Metric Breakdown

| test_center | model | train_size | test_size | transfer_accuracy | transfer_f1 | transfer_roc_auc | transfer_pr_auc | transfer_brier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cleveland | Logistic Regression | 615 | 303 | 0.6205 | 0.4601 | 0.5833 | 0.5823 | 0.3019 |
| cleveland | Random Forest | 615 | 303 | 0.7822 | 0.7591 | 0.8626 | 0.8577 | 0.1600 |
| cleveland | XGBoost | 615 | 303 | 0.7492 | 0.7324 | 0.8406 | 0.8259 | 0.1659 |
| cleveland | Ensemble (Transfer) | 615 | 303 | 0.7921 | 0.7586 | 0.8411 | 0.7868 | 0.1680 |
| hungarian | Logistic Regression | 625 | 293 | 0.7918 | 0.7336 | 0.8588 | 0.8226 | 0.1526 |
| hungarian | Random Forest | 625 | 293 | 0.7099 | 0.6931 | 0.8696 | 0.8044 | 0.1930 |
| hungarian | XGBoost | 625 | 293 | 0.7713 | 0.7287 | 0.8676 | 0.7749 | 0.1572 |
| hungarian | Ensemble (Transfer) | 625 | 293 | 0.7747 | 0.7317 | 0.8761 | 0.8228 | 0.1601 |
| switzerland | Logistic Regression | 795 | 123 | 0.5691 | 0.7072 | 0.6359 | 0.9502 | 0.2671 |
| switzerland | Random Forest | 795 | 123 | 0.6911 | 0.8061 | 0.7239 | 0.9686 | 0.1867 |
| switzerland | XGBoost | 795 | 123 | 0.6504 | 0.7749 | 0.7147 | 0.9581 | 0.2187 |
| switzerland | Ensemble (Transfer) | 795 | 123 | 0.6423 | 0.7684 | 0.7065 | 0.9632 | 0.2189 |
| va_long_beach | Logistic Regression | 719 | 199 | 0.7136 | 0.8055 | 0.7085 | 0.8678 | 0.1868 |
| va_long_beach | Random Forest | 719 | 199 | 0.6683 | 0.7643 | 0.6848 | 0.8443 | 0.2031 |
| va_long_beach | XGBoost | 719 | 199 | 0.6432 | 0.7341 | 0.6716 | 0.8305 | 0.2201 |
| va_long_beach | Ensemble (Transfer) | 719 | 199 | 0.6884 | 0.7801 | 0.6930 | 0.8481 | 0.1972 |
