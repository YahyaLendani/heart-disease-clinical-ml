| model | test_accuracy | test_f1 | test_roc_auc | test_pr_auc | test_brier_score | optimal_threshold_f1 | f1_at_optimal_threshold | optimal_threshold_youden | youden_j |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| XGBoost | 0.8533 | 0.8708 | 0.9303 | 0.9423 | 0.1044 | 0.5867 | 0.8867 | 0.6008 | 0.7506 |
| Hybrid Soft-Voting Ensemble | 0.8261 | 0.8462 | 0.9259 | 0.9415 | 0.1120 | 0.3138 | 0.8684 | 0.6481 | 0.7064 |
| Hybrid Stacking Ensemble (Ours) | 0.8261 | 0.8476 | 0.9244 | 0.9402 | 0.1158 | 0.2715 | 0.8634 | 0.6494 | 0.6992 |
| Random Forest | 0.8207 | 0.8421 | 0.9164 | 0.9299 | 0.1205 | 0.4001 | 0.8649 | 0.6528 | 0.6891 |
| PyTorch MLP | 0.8152 | 0.8333 | 0.9042 | 0.9202 | 0.1259 | 0.2938 | 0.8624 | 0.6457 | 0.6648 |
| Logistic Regression | 0.8043 | 0.8252 | 0.9034 | 0.9271 | 0.1255 | 0.3926 | 0.8465 | 0.6734 | 0.6891 |