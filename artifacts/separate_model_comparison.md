| source | model | test_accuracy | test_f1 | test_roc_auc | best_val_accuracy | best_epoch |
| --- | --- | --- | --- | --- | --- | --- |
| heart_baseline | Logistic Regression | 0.8033 | 0.8235 | 0.8712 | 0.8091 |  |
| heart_baseline | XGBoost | 0.7705 | 0.7941 | 0.8745 | 0.8009 |  |
| heart_baseline | Random Forest | 0.7705 | 0.7879 | 0.8658 | 0.8257 |  |
| pytorch_mlp | MLP_ADAM | 0.5574 | 0.4706 | 0.7251 | 0.6327 | 1.0000 |
| pytorch_mlp | MLP_SGD | 0.5410 | 0.3913 | 0.7219 | 0.5714 | 1.0000 |