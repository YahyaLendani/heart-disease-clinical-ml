| model | test_accuracy | test_f1 | test_roc_auc | test_pr_auc | test_brier_score | optimal_threshold_f1 | f1_at_optimal_threshold | optimal_threshold_youden | youden_j |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PyTorch MLP | 0.8033 | 0.7857 | 0.9177 | 0.9044 | 0.1271 | 0.3680 | 0.8667 | 0.3680 | 0.7468 |
| Hybrid Soft-Voting Ensemble | 0.8197 | 0.8000 | 0.9091 | 0.8923 | 0.1247 | 0.2665 | 0.8485 | 0.2665 | 0.6970 |
| Hybrid Stacking Ensemble (Ours) | 0.8033 | 0.7857 | 0.9026 | 0.8866 | 0.1336 | 0.2427 | 0.8308 | 0.2427 | 0.6613 |
| Logistic Regression | 0.7869 | 0.7636 | 0.9026 | 0.8964 | 0.1345 | 0.3078 | 0.8525 | 0.3078 | 0.7165 |
| Random Forest | 0.8197 | 0.8000 | 0.8907 | 0.8831 | 0.1372 | 0.5398 | 0.8148 | 0.5398 | 0.6645 |
| XGBoost | 0.7869 | 0.7797 | 0.8810 | 0.8660 | 0.1552 | 0.4779 | 0.8197 | 0.4779 | 0.6504 |