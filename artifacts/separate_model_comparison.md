| source | model | test_accuracy | test_f1 | test_roc_auc | best_val_accuracy | best_epoch | test_pr_auc | test_brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_baseline | XGBoost | 0.8033 | 0.8000 | 0.8788 | 0.8204 |  |  |  |
| pytorch_mlp | MLP_ADAM | 0.8033 | 0.7931 | 0.8918 | 0.7551 | 41.0000 | 0.8420 | 0.1408 |
| heart_baseline | Random Forest | 0.8033 | 0.7931 | 0.8761 | 0.8352 |  |  |  |
| pytorch_mlp | MLP_SGD | 0.7869 | 0.7719 | 0.8755 | 0.7755 | 76.0000 | 0.8465 | 0.1680 |
| heart_baseline | Logistic Regression | 0.7869 | 0.7636 | 0.8853 | 0.8111 |  |  |  |