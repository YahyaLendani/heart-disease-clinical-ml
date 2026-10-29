# Paired Model Comparison (DeLong test, Holm-adjusted)

61 shared test patients (28 with disease) from `cleaned_heart_disease.csv`. `auc_diff` = AUC(model_a) - AUC(model_b); 0 of 91 pairs differ at Holm-adjusted p < 0.05.

| model_a | model_b | auc_a | auc_b | auc_diff | se | diff_ci_low | diff_ci_high | z | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ensemble base: PyTorch MLP | MLP_SGD | 0.9167 | 0.8690 | 0.0476 | 0.0222 | 0.0041 | 0.0912 | 2.1423 | 0.0322 | 1.0000 | False |
| MLP_SGD | MLP_ADAM | 0.8690 | 0.8977 | -0.0287 | 0.0141 | -0.0562 | -0.0011 | -2.0410 | 0.0412 | 1.0000 | False |
| Hybrid Soft-Voting Ensemble | MLP_SGD | 0.9145 | 0.8690 | 0.0455 | 0.0227 | 0.0009 | 0.0900 | 1.9981 | 0.0457 | 1.0000 | False |
| MLP_SGD | GCN (Kipf & Welling) | 0.8690 | 0.9102 | -0.0411 | 0.0224 | -0.0851 | 0.0028 | -1.8339 | 0.0667 | 1.0000 | False |
| MLP_SGD | TabTransformer | 0.8690 | 0.9080 | -0.0390 | 0.0219 | -0.0818 | 0.0039 | -1.7830 | 0.0746 | 1.0000 | False |
| Ensemble base: Random Forest | Hybrid Soft-Voting Ensemble | 0.8853 | 0.9145 | -0.0292 | 0.0167 | -0.0619 | 0.0035 | -1.7501 | 0.0801 | 1.0000 | False |
| Random Forest | Hybrid Soft-Voting Ensemble | 0.8772 | 0.9145 | -0.0373 | 0.0219 | -0.0802 | 0.0056 | -1.7060 | 0.0880 | 1.0000 | False |
| Random Forest | Ensemble base: PyTorch MLP | 0.8772 | 0.9167 | -0.0395 | 0.0234 | -0.0854 | 0.0064 | -1.6861 | 0.0918 | 1.0000 | False |
| Hybrid Stacking Ensemble (Ours) | MLP_SGD | 0.9069 | 0.8690 | 0.0379 | 0.0236 | -0.0084 | 0.0842 | 1.6032 | 0.1089 | 1.0000 | False |
| Ensemble base: XGBoost | Hybrid Soft-Voting Ensemble | 0.8842 | 0.9145 | -0.0303 | 0.0201 | -0.0697 | 0.0091 | -1.5058 | 0.1321 | 1.0000 | False |
| Ensemble base: Random Forest | Hybrid Stacking Ensemble (Ours) | 0.8853 | 0.9069 | -0.0216 | 0.0148 | -0.0507 | 0.0074 | -1.4587 | 0.1447 | 1.0000 | False |
| XGBoost | MLP_SGD | 0.9004 | 0.8690 | 0.0314 | 0.0221 | -0.0120 | 0.0747 | 1.4189 | 0.1559 | 1.0000 | False |
| Logistic Regression | MLP_SGD | 0.9004 | 0.8690 | 0.0314 | 0.0225 | -0.0127 | 0.0755 | 1.3945 | 0.1632 | 1.0000 | False |
| Ensemble base: Random Forest | Ensemble base: PyTorch MLP | 0.8853 | 0.9167 | -0.0314 | 0.0230 | -0.0765 | 0.0137 | -1.3646 | 0.1724 | 1.0000 | False |
| Ensemble base: XGBoost | Hybrid Stacking Ensemble (Ours) | 0.8842 | 0.9069 | -0.0227 | 0.0170 | -0.0561 | 0.0106 | -1.3360 | 0.1815 | 1.0000 | False |
| Random Forest | Hybrid Stacking Ensemble (Ours) | 0.8772 | 0.9069 | -0.0298 | 0.0227 | -0.0743 | 0.0148 | -1.3100 | 0.1902 | 1.0000 | False |
| Random Forest | XGBoost | 0.8772 | 0.9004 | -0.0233 | 0.0180 | -0.0585 | 0.0119 | -1.2960 | 0.1950 | 1.0000 | False |
| Ensemble base: XGBoost | Ensemble base: PyTorch MLP | 0.8842 | 0.9167 | -0.0325 | 0.0271 | -0.0856 | 0.0207 | -1.1977 | 0.2310 | 1.0000 | False |
| Ensemble base: Logistic Regression | MLP_SGD | 0.8994 | 0.8690 | 0.0303 | 0.0255 | -0.0197 | 0.0803 | 1.1870 | 0.2352 | 1.0000 | False |
| Random Forest | TabTransformer | 0.8772 | 0.9080 | -0.0308 | 0.0270 | -0.0838 | 0.0222 | -1.1405 | 0.2541 | 1.0000 | False |
| XGBoost | Hybrid Soft-Voting Ensemble | 0.9004 | 0.9145 | -0.0141 | 0.0125 | -0.0386 | 0.0104 | -1.1259 | 0.2602 | 1.0000 | False |
| Random Forest | GCN (Kipf & Welling) | 0.8772 | 0.9102 | -0.0330 | 0.0307 | -0.0931 | 0.0271 | -1.0761 | 0.2819 | 1.0000 | False |
| XGBoost | Ensemble base: Random Forest | 0.9004 | 0.8853 | 0.0152 | 0.0141 | -0.0126 | 0.0429 | 1.0716 | 0.2839 | 1.0000 | False |
| MLP_SGD | GAT (Velickovic et al.) | 0.8690 | 0.8972 | -0.0281 | 0.0273 | -0.0816 | 0.0253 | -1.0321 | 0.3020 | 1.0000 | False |
| Hybrid Soft-Voting Ensemble | MLP_ADAM | 0.9145 | 0.8977 | 0.0168 | 0.0166 | -0.0157 | 0.0492 | 1.0128 | 0.3112 | 1.0000 | False |
| Ensemble base: PyTorch MLP | MLP_ADAM | 0.9167 | 0.8977 | 0.0189 | 0.0189 | -0.0182 | 0.0561 | 1.0000 | 0.3173 | 1.0000 | False |
| Random Forest | MLP_ADAM | 0.8772 | 0.8977 | -0.0206 | 0.0227 | -0.0651 | 0.0240 | -0.9042 | 0.3659 | 1.0000 | False |
| Logistic Regression | Random Forest | 0.9004 | 0.8772 | 0.0233 | 0.0264 | -0.0284 | 0.0749 | 0.8828 | 0.3774 | 1.0000 | False |
| XGBoost | Ensemble base: PyTorch MLP | 0.9004 | 0.9167 | -0.0162 | 0.0192 | -0.0539 | 0.0214 | -0.8444 | 0.3984 | 1.0000 | False |
| Ensemble base: Random Forest | GCN (Kipf & Welling) | 0.8853 | 0.9102 | -0.0249 | 0.0303 | -0.0843 | 0.0346 | -0.8207 | 0.4118 | 1.0000 | False |
| Ensemble base: Random Forest | TabTransformer | 0.8853 | 0.9080 | -0.0227 | 0.0281 | -0.0777 | 0.0323 | -0.8096 | 0.4182 | 1.0000 | False |
| Hybrid Soft-Voting Ensemble | Hybrid Stacking Ensemble (Ours) | 0.9145 | 0.9069 | 0.0076 | 0.0094 | -0.0108 | 0.0260 | 0.8070 | 0.4197 | 1.0000 | False |
| Ensemble base: XGBoost | GCN (Kipf & Welling) | 0.8842 | 0.9102 | -0.0260 | 0.0340 | -0.0925 | 0.0406 | -0.7649 | 0.4444 | 1.0000 | False |
| Ensemble base: PyTorch MLP | GAT (Velickovic et al.) | 0.9167 | 0.8972 | 0.0195 | 0.0255 | -0.0305 | 0.0694 | 0.7642 | 0.4447 | 1.0000 | False |
| Random Forest | Ensemble base: Logistic Regression | 0.8772 | 0.8994 | -0.0222 | 0.0298 | -0.0806 | 0.0362 | -0.7443 | 0.4567 | 1.0000 | False |
| Logistic Regression | Ensemble base: PyTorch MLP | 0.9004 | 0.9167 | -0.0162 | 0.0220 | -0.0594 | 0.0270 | -0.7363 | 0.4616 | 1.0000 | False |
| GCN (Kipf & Welling) | GAT (Velickovic et al.) | 0.9102 | 0.8972 | 0.0130 | 0.0177 | -0.0217 | 0.0476 | 0.7344 | 0.4627 | 1.0000 | False |
| Ensemble base: XGBoost | TabTransformer | 0.8842 | 0.9080 | -0.0238 | 0.0326 | -0.0877 | 0.0401 | -0.7299 | 0.4655 | 1.0000 | False |
| Logistic Regression | Hybrid Soft-Voting Ensemble | 0.9004 | 0.9145 | -0.0141 | 0.0194 | -0.0521 | 0.0240 | -0.7249 | 0.4685 | 1.0000 | False |
| Ensemble base: Logistic Regression | Hybrid Soft-Voting Ensemble | 0.8994 | 0.9145 | -0.0152 | 0.0214 | -0.0572 | 0.0269 | -0.7069 | 0.4796 | 1.0000 | False |
| XGBoost | Ensemble base: XGBoost | 0.9004 | 0.8842 | 0.0162 | 0.0230 | -0.0288 | 0.0612 | 0.7068 | 0.4797 | 1.0000 | False |
| Ensemble base: Random Forest | MLP_SGD | 0.8853 | 0.8690 | 0.0162 | 0.0234 | -0.0296 | 0.0620 | 0.6946 | 0.4873 | 1.0000 | False |
| MLP_ADAM | GCN (Kipf & Welling) | 0.8977 | 0.9102 | -0.0124 | 0.0182 | -0.0481 | 0.0232 | -0.6848 | 0.4935 | 1.0000 | False |
| Ensemble base: Logistic Regression | Ensemble base: PyTorch MLP | 0.8994 | 0.9167 | -0.0173 | 0.0255 | -0.0672 | 0.0326 | -0.6800 | 0.4965 | 1.0000 | False |
| Logistic Regression | Ensemble base: Random Forest | 0.9004 | 0.8853 | 0.0152 | 0.0232 | -0.0304 | 0.0607 | 0.6523 | 0.5142 | 1.0000 | False |
| Hybrid Soft-Voting Ensemble | GAT (Velickovic et al.) | 0.9145 | 0.8972 | 0.0173 | 0.0267 | -0.0349 | 0.0696 | 0.6495 | 0.5160 | 1.0000 | False |
| Random Forest | GAT (Velickovic et al.) | 0.8772 | 0.8972 | -0.0200 | 0.0332 | -0.0852 | 0.0451 | -0.6025 | 0.5468 | 1.0000 | False |
| Ensemble base: Logistic Regression | Ensemble base: Random Forest | 0.8994 | 0.8853 | 0.0141 | 0.0246 | -0.0341 | 0.0623 | 0.5720 | 0.5673 | 1.0000 | False |
| Ensemble base: Random Forest | MLP_ADAM | 0.8853 | 0.8977 | -0.0124 | 0.0218 | -0.0551 | 0.0303 | -0.5713 | 0.5678 | 1.0000 | False |
| TabTransformer | GAT (Velickovic et al.) | 0.9080 | 0.8972 | 0.0108 | 0.0190 | -0.0264 | 0.0480 | 0.5700 | 0.5687 | 1.0000 | False |
| Logistic Regression | Ensemble base: XGBoost | 0.9004 | 0.8842 | 0.0162 | 0.0305 | -0.0436 | 0.0760 | 0.5322 | 0.5946 | 1.0000 | False |
| Hybrid Stacking Ensemble (Ours) | MLP_ADAM | 0.9069 | 0.8977 | 0.0092 | 0.0176 | -0.0252 | 0.0436 | 0.5239 | 0.6004 | 1.0000 | False |
| Random Forest | Ensemble base: Random Forest | 0.8772 | 0.8853 | -0.0081 | 0.0157 | -0.0389 | 0.0227 | -0.5164 | 0.6055 | 1.0000 | False |
| Ensemble base: XGBoost | MLP_ADAM | 0.8842 | 0.8977 | -0.0135 | 0.0268 | -0.0661 | 0.0391 | -0.5041 | 0.6142 | 1.0000 | False |
| Ensemble base: Logistic Regression | Ensemble base: XGBoost | 0.8994 | 0.8842 | 0.0152 | 0.0308 | -0.0452 | 0.0755 | 0.4922 | 0.6226 | 1.0000 | False |
| XGBoost | Hybrid Stacking Ensemble (Ours) | 0.9004 | 0.9069 | -0.0065 | 0.0133 | -0.0325 | 0.0195 | -0.4888 | 0.6250 | 1.0000 | False |
| Ensemble base: XGBoost | MLP_SGD | 0.8842 | 0.8690 | 0.0152 | 0.0313 | -0.0462 | 0.0765 | 0.4839 | 0.6285 | 1.0000 | False |
| MLP_ADAM | TabTransformer | 0.8977 | 0.9080 | -0.0103 | 0.0218 | -0.0531 | 0.0325 | -0.4707 | 0.6379 | 1.0000 | False |
| Ensemble base: PyTorch MLP | Hybrid Stacking Ensemble (Ours) | 0.9167 | 0.9069 | 0.0097 | 0.0207 | -0.0309 | 0.0504 | 0.4701 | 0.6383 | 1.0000 | False |
| Ensemble base: PyTorch MLP | TabTransformer | 0.9167 | 0.9080 | 0.0087 | 0.0226 | -0.0356 | 0.0530 | 0.3831 | 0.7017 | 1.0000 | False |
| Logistic Regression | GCN (Kipf & Welling) | 0.9004 | 0.9102 | -0.0097 | 0.0257 | -0.0601 | 0.0406 | -0.3793 | 0.7045 | 1.0000 | False |
| XGBoost | GCN (Kipf & Welling) | 0.9004 | 0.9102 | -0.0097 | 0.0257 | -0.0602 | 0.0407 | -0.3786 | 0.7050 | 1.0000 | False |
| Ensemble base: Logistic Regression | Hybrid Stacking Ensemble (Ours) | 0.8994 | 0.9069 | -0.0076 | 0.0200 | -0.0468 | 0.0317 | -0.3782 | 0.7052 | 1.0000 | False |
| Logistic Regression | TabTransformer | 0.9004 | 0.9080 | -0.0076 | 0.0202 | -0.0472 | 0.0320 | -0.3752 | 0.7075 | 1.0000 | False |
| XGBoost | TabTransformer | 0.9004 | 0.9080 | -0.0076 | 0.0207 | -0.0481 | 0.0329 | -0.3667 | 0.7138 | 1.0000 | False |
| Ensemble base: Logistic Regression | GCN (Kipf & Welling) | 0.8994 | 0.9102 | -0.0108 | 0.0296 | -0.0689 | 0.0473 | -0.3652 | 0.7150 | 1.0000 | False |
| Ensemble base: Random Forest | GAT (Velickovic et al.) | 0.8853 | 0.8972 | -0.0119 | 0.0328 | -0.0762 | 0.0524 | -0.3629 | 0.7167 | 1.0000 | False |
| Ensemble base: XGBoost | GAT (Velickovic et al.) | 0.8842 | 0.8972 | -0.0130 | 0.0367 | -0.0850 | 0.0590 | -0.3534 | 0.7238 | 1.0000 | False |
| Hybrid Stacking Ensemble (Ours) | GAT (Velickovic et al.) | 0.9069 | 0.8972 | 0.0097 | 0.0276 | -0.0443 | 0.0638 | 0.3530 | 0.7241 | 1.0000 | False |
| Random Forest | MLP_SGD | 0.8772 | 0.8690 | 0.0081 | 0.0230 | -0.0370 | 0.0532 | 0.3525 | 0.7245 | 1.0000 | False |
| Ensemble base: Logistic Regression | TabTransformer | 0.8994 | 0.9080 | -0.0087 | 0.0250 | -0.0577 | 0.0404 | -0.3457 | 0.7296 | 1.0000 | False |
| Logistic Regression | Hybrid Stacking Ensemble (Ours) | 0.9004 | 0.9069 | -0.0065 | 0.0192 | -0.0441 | 0.0311 | -0.3384 | 0.7350 | 1.0000 | False |
| Hybrid Soft-Voting Ensemble | TabTransformer | 0.9145 | 0.9080 | 0.0065 | 0.0232 | -0.0389 | 0.0519 | 0.2802 | 0.7793 | 1.0000 | False |
| Ensemble base: PyTorch MLP | GCN (Kipf & Welling) | 0.9167 | 0.9102 | 0.0065 | 0.0234 | -0.0393 | 0.0523 | 0.2777 | 0.7813 | 1.0000 | False |
| Random Forest | Ensemble base: XGBoost | 0.8772 | 0.8842 | -0.0070 | 0.0278 | -0.0615 | 0.0474 | -0.2531 | 0.8002 | 1.0000 | False |
| Hybrid Soft-Voting Ensemble | GCN (Kipf & Welling) | 0.9145 | 0.9102 | 0.0043 | 0.0236 | -0.0419 | 0.0505 | 0.1837 | 0.8543 | 1.0000 | False |
| Ensemble base: PyTorch MLP | Hybrid Soft-Voting Ensemble | 0.9167 | 0.9145 | 0.0022 | 0.0139 | -0.0251 | 0.0294 | 0.1557 | 0.8763 | 1.0000 | False |
| XGBoost | MLP_ADAM | 0.9004 | 0.8977 | 0.0027 | 0.0178 | -0.0322 | 0.0376 | 0.1519 | 0.8792 | 1.0000 | False |
| Logistic Regression | MLP_ADAM | 0.9004 | 0.8977 | 0.0027 | 0.0195 | -0.0355 | 0.0409 | 0.1388 | 0.8896 | 1.0000 | False |
| Logistic Regression | GAT (Velickovic et al.) | 0.9004 | 0.8972 | 0.0032 | 0.0237 | -0.0433 | 0.0498 | 0.1368 | 0.8912 | 1.0000 | False |
| Hybrid Stacking Ensemble (Ours) | GCN (Kipf & Welling) | 0.9069 | 0.9102 | -0.0032 | 0.0248 | -0.0518 | 0.0453 | -0.1310 | 0.8958 | 1.0000 | False |
| Logistic Regression | Ensemble base: Logistic Regression | 0.9004 | 0.8994 | 0.0011 | 0.0084 | -0.0153 | 0.0175 | 0.1291 | 0.8972 | 1.0000 | False |
| XGBoost | GAT (Velickovic et al.) | 0.9004 | 0.8972 | 0.0032 | 0.0265 | -0.0488 | 0.0553 | 0.1224 | 0.9026 | 1.0000 | False |
| TabTransformer | GCN (Kipf & Welling) | 0.9080 | 0.9102 | -0.0022 | 0.0233 | -0.0478 | 0.0435 | -0.0930 | 0.9259 | 1.0000 | False |
| Ensemble base: Logistic Regression | GAT (Velickovic et al.) | 0.8994 | 0.8972 | 0.0022 | 0.0281 | -0.0528 | 0.0571 | 0.0772 | 0.9385 | 1.0000 | False |
| Ensemble base: Logistic Regression | MLP_ADAM | 0.8994 | 0.8977 | 0.0016 | 0.0222 | -0.0418 | 0.0450 | 0.0733 | 0.9416 | 1.0000 | False |
| XGBoost | Ensemble base: Logistic Regression | 0.9004 | 0.8994 | 0.0011 | 0.0207 | -0.0395 | 0.0417 | 0.0523 | 0.9583 | 1.0000 | False |
| Ensemble base: Random Forest | Ensemble base: XGBoost | 0.8853 | 0.8842 | 0.0011 | 0.0230 | -0.0440 | 0.0461 | 0.0471 | 0.9624 | 1.0000 | False |
| Hybrid Stacking Ensemble (Ours) | TabTransformer | 0.9069 | 0.9080 | -0.0011 | 0.0249 | -0.0499 | 0.0477 | -0.0435 | 0.9653 | 1.0000 | False |
| MLP_ADAM | GAT (Velickovic et al.) | 0.8977 | 0.8972 | 0.0005 | 0.0257 | -0.0499 | 0.0510 | 0.0210 | 0.9832 | 1.0000 | False |
| Logistic Regression | XGBoost | 0.9004 | 0.9004 | 0.0000 | 0.0174 | -0.0341 | 0.0341 | 0.0000 | 1.0000 | 1.0000 | False |