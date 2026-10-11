| model | test_accuracy | test_f1 | test_roc_auc | test_pr_auc | test_brier_score | optimal_threshold_f1 | f1_at_optimal_threshold | optimal_threshold_youden | youden_j |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.8033 | 0.8182 | 0.8734 | 0.8924 | 0.1482 | 0.3263 | 0.8451 | 0.3770 | 0.6288 |
| Random Forest | 0.7705 | 0.7879 | 0.8690 | 0.8847 | 0.1504 | 0.3965 | 0.8286 | 0.4859 | 0.5985 |
| Hybrid Stacking Ensemble (Ours) | 0.7869 | 0.8000 | 0.8690 | 0.8862 | 0.1616 | 0.1877 | 0.8333 | 0.3590 | 0.5985 |
| Hybrid Soft-Voting Ensemble | 0.7705 | 0.7879 | 0.8658 | 0.8826 | 0.1566 | 0.4236 | 0.8235 | 0.4236 | 0.5985 |
| XGBoost | 0.7377 | 0.7576 | 0.8517 | 0.8632 | 0.1793 | 0.1004 | 0.8533 | 0.1004 | 0.6126 |
| PyTorch MLP | 0.7541 | 0.7692 | 0.8398 | 0.8569 | 0.1893 | 0.1149 | 0.8333 | 0.1149 | 0.5877 |