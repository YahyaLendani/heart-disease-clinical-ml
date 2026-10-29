| source | model | test_accuracy | test_f1 | test_roc_auc | best_val_accuracy | best_epoch | test_pr_auc | test_brier_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| heart_baseline | XGBoost | 0.8315 | 0.8517 | 0.9094 | 0.8109 |  |  |  |
| heart_baseline | Random Forest | 0.8152 | 0.8396 | 0.9002 | 0.8164 |  |  |  |
| pytorch_mlp | MLP_ADAM | 0.8098 | 0.8325 | 0.8852 | 0.8095 | 36.0000 | 0.8976 | 0.1393 |
| heart_baseline | Logistic Regression | 0.8043 | 0.8269 | 0.8841 | 0.8012 |  |  |  |
| pytorch_mlp | MLP_SGD | 0.7935 | 0.8137 | 0.8790 | 0.7959 | 75.0000 | 0.8993 | 0.1480 |