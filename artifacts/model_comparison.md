# Paired Model Comparison (DeLong test, Holm-adjusted)

184 shared test patients (102 with disease) from `cleaned_heart_disease_combined.csv`. `auc_diff` = AUC(model_a) - AUC(model_b); 0 of 91 pairs differ at Holm-adjusted p < 0.05.

| model_a | model_b | auc_a | auc_b | auc_diff | se | diff_ci_low | diff_ci_high | z | p_value | p_holm | significant_holm_0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ensemble base: PyTorch MLP | Hybrid Stacking Ensemble (Ours) | 0.8767 | 0.9004 | -0.0237 | 0.0104 | -0.0442 | -0.0032 | -2.2654 | 0.0235 | 1.0000 | False |
| Logistic Regression | XGBoost | 0.8841 | 0.9094 | -0.0252 | 0.0112 | -0.0472 | -0.0033 | -2.2542 | 0.0242 | 1.0000 | False |
| Ensemble base: PyTorch MLP | Hybrid Soft-Voting Ensemble | 0.8767 | 0.9003 | -0.0236 | 0.0106 | -0.0443 | -0.0028 | -2.2254 | 0.0261 | 1.0000 | False |
| XGBoost | GAT (Velickovic et al.) | 0.9094 | 0.8815 | 0.0279 | 0.0127 | 0.0029 | 0.0528 | 2.1910 | 0.0284 | 1.0000 | False |
| Ensemble base: XGBoost | Ensemble base: PyTorch MLP | 0.9075 | 0.8767 | 0.0307 | 0.0140 | 0.0032 | 0.0583 | 2.1880 | 0.0287 | 1.0000 | False |
| XGBoost | TabTransformer | 0.9094 | 0.8855 | 0.0239 | 0.0116 | 0.0012 | 0.0466 | 2.0662 | 0.0388 | 1.0000 | False |
| XGBoost | MLP_SGD | 0.9094 | 0.8790 | 0.0304 | 0.0152 | 0.0005 | 0.0602 | 1.9960 | 0.0459 | 1.0000 | False |
| XGBoost | Ensemble base: Logistic Regression | 0.9094 | 0.8853 | 0.0240 | 0.0122 | 0.0002 | 0.0479 | 1.9737 | 0.0484 | 1.0000 | False |
| XGBoost | Ensemble base: PyTorch MLP | 0.9094 | 0.8767 | 0.0326 | 0.0179 | -0.0025 | 0.0677 | 1.8222 | 0.0684 | 1.0000 | False |
| Ensemble base: Logistic Regression | Hybrid Soft-Voting Ensemble | 0.8853 | 0.9003 | -0.0149 | 0.0086 | -0.0318 | 0.0019 | -1.7350 | 0.0827 | 1.0000 | False |
| Ensemble base: Logistic Regression | Hybrid Stacking Ensemble (Ours) | 0.8853 | 0.9004 | -0.0151 | 0.0087 | -0.0322 | 0.0021 | -1.7225 | 0.0850 | 1.0000 | False |
| Hybrid Stacking Ensemble (Ours) | MLP_SGD | 0.9004 | 0.8790 | 0.0214 | 0.0126 | -0.0032 | 0.0460 | 1.7034 | 0.0885 | 1.0000 | False |
| Hybrid Soft-Voting Ensemble | MLP_SGD | 0.9003 | 0.8790 | 0.0213 | 0.0125 | -0.0033 | 0.0458 | 1.6986 | 0.0894 | 1.0000 | False |
| Logistic Regression | Hybrid Stacking Ensemble (Ours) | 0.8841 | 0.9004 | -0.0163 | 0.0097 | -0.0353 | 0.0028 | -1.6753 | 0.0939 | 1.0000 | False |
| Logistic Regression | Hybrid Soft-Voting Ensemble | 0.8841 | 0.9003 | -0.0161 | 0.0096 | -0.0350 | 0.0028 | -1.6744 | 0.0941 | 1.0000 | False |
| Ensemble base: XGBoost | MLP_SGD | 0.9075 | 0.8790 | 0.0285 | 0.0170 | -0.0049 | 0.0618 | 1.6741 | 0.0941 | 1.0000 | False |
| XGBoost | GCN (Kipf & Welling) | 0.9094 | 0.8871 | 0.0222 | 0.0137 | -0.0047 | 0.0492 | 1.6176 | 0.1057 | 1.0000 | False |
| Random Forest | XGBoost | 0.9002 | 0.9094 | -0.0092 | 0.0058 | -0.0207 | 0.0022 | -1.5764 | 0.1149 | 1.0000 | False |
| Hybrid Soft-Voting Ensemble | GAT (Velickovic et al.) | 0.9003 | 0.8815 | 0.0188 | 0.0123 | -0.0054 | 0.0430 | 1.5214 | 0.1282 | 1.0000 | False |
| Hybrid Stacking Ensemble (Ours) | GAT (Velickovic et al.) | 0.9004 | 0.8815 | 0.0189 | 0.0124 | -0.0055 | 0.0433 | 1.5199 | 0.1285 | 1.0000 | False |
| XGBoost | MLP_ADAM | 0.9094 | 0.8852 | 0.0242 | 0.0160 | -0.0072 | 0.0555 | 1.5108 | 0.1308 | 1.0000 | False |
| Logistic Regression | Ensemble base: XGBoost | 0.8841 | 0.9075 | -0.0233 | 0.0155 | -0.0536 | 0.0070 | -1.5090 | 0.1313 | 1.0000 | False |
| Ensemble base: Logistic Regression | Ensemble base: XGBoost | 0.8853 | 0.9075 | -0.0221 | 0.0148 | -0.0510 | 0.0068 | -1.4987 | 0.1339 | 1.0000 | False |
| Ensemble base: XGBoost | GAT (Velickovic et al.) | 0.9075 | 0.8815 | 0.0259 | 0.0178 | -0.0089 | 0.0608 | 1.4608 | 0.1441 | 1.0000 | False |
| Random Forest | GAT (Velickovic et al.) | 0.9002 | 0.8815 | 0.0187 | 0.0128 | -0.0064 | 0.0437 | 1.4585 | 0.1447 | 1.0000 | False |
| Ensemble base: XGBoost | MLP_ADAM | 0.9075 | 0.8852 | 0.0222 | 0.0161 | -0.0094 | 0.0539 | 1.3770 | 0.1685 | 1.0000 | False |
| Logistic Regression | Random Forest | 0.8841 | 0.9002 | -0.0160 | 0.0117 | -0.0389 | 0.0069 | -1.3721 | 0.1700 | 1.0000 | False |
| Random Forest | MLP_SGD | 0.9002 | 0.8790 | 0.0212 | 0.0156 | -0.0093 | 0.0517 | 1.3595 | 0.1740 | 1.0000 | False |
| Random Forest | Ensemble base: PyTorch MLP | 0.9002 | 0.8767 | 0.0234 | 0.0178 | -0.0115 | 0.0584 | 1.3131 | 0.1892 | 1.0000 | False |
| Ensemble base: XGBoost | GCN (Kipf & Welling) | 0.9075 | 0.8871 | 0.0203 | 0.0162 | -0.0114 | 0.0520 | 1.2567 | 0.2089 | 1.0000 | False |
| Hybrid Stacking Ensemble (Ours) | GCN (Kipf & Welling) | 0.9004 | 0.8871 | 0.0133 | 0.0107 | -0.0077 | 0.0343 | 1.2379 | 0.2158 | 1.0000 | False |
| Hybrid Soft-Voting Ensemble | GCN (Kipf & Welling) | 0.9003 | 0.8871 | 0.0132 | 0.0106 | -0.0077 | 0.0340 | 1.2373 | 0.2160 | 1.0000 | False |
| Hybrid Stacking Ensemble (Ours) | MLP_ADAM | 0.9004 | 0.8852 | 0.0152 | 0.0124 | -0.0092 | 0.0395 | 1.2228 | 0.2214 | 1.0000 | False |
| Hybrid Soft-Voting Ensemble | MLP_ADAM | 0.9003 | 0.8852 | 0.0151 | 0.0124 | -0.0093 | 0.0394 | 1.2135 | 0.2249 | 1.0000 | False |
| Ensemble base: XGBoost | TabTransformer | 0.9075 | 0.8855 | 0.0220 | 0.0182 | -0.0136 | 0.0576 | 1.2096 | 0.2264 | 1.0000 | False |
| Ensemble base: Random Forest | Ensemble base: PyTorch MLP | 0.8973 | 0.8767 | 0.0206 | 0.0172 | -0.0131 | 0.0542 | 1.1977 | 0.2310 | 1.0000 | False |
| XGBoost | Ensemble base: Random Forest | 0.9094 | 0.8973 | 0.0121 | 0.0101 | -0.0078 | 0.0320 | 1.1908 | 0.2337 | 1.0000 | False |
| Random Forest | Ensemble base: Logistic Regression | 0.9002 | 0.8853 | 0.0148 | 0.0125 | -0.0097 | 0.0393 | 1.1851 | 0.2360 | 1.0000 | False |
| Random Forest | TabTransformer | 0.9002 | 0.8855 | 0.0147 | 0.0126 | -0.0100 | 0.0394 | 1.1682 | 0.2427 | 1.0000 | False |
| Ensemble base: Random Forest | MLP_SGD | 0.8973 | 0.8790 | 0.0183 | 0.0163 | -0.0136 | 0.0502 | 1.1229 | 0.2615 | 1.0000 | False |
| Ensemble base: Random Forest | GAT (Velickovic et al.) | 0.8973 | 0.8815 | 0.0158 | 0.0142 | -0.0121 | 0.0436 | 1.1110 | 0.2666 | 1.0000 | False |
| Logistic Regression | Ensemble base: Random Forest | 0.8841 | 0.8973 | -0.0132 | 0.0129 | -0.0385 | 0.0122 | -1.0174 | 0.3090 | 1.0000 | False |
| Hybrid Stacking Ensemble (Ours) | TabTransformer | 0.9004 | 0.8855 | 0.0149 | 0.0151 | -0.0146 | 0.0444 | 0.9930 | 0.3207 | 1.0000 | False |
| Hybrid Soft-Voting Ensemble | TabTransformer | 0.9003 | 0.8855 | 0.0148 | 0.0150 | -0.0145 | 0.0441 | 0.9910 | 0.3217 | 1.0000 | False |
| Random Forest | GCN (Kipf & Welling) | 0.9002 | 0.8871 | 0.0130 | 0.0134 | -0.0133 | 0.0394 | 0.9705 | 0.3318 | 1.0000 | False |
| Ensemble base: Random Forest | Ensemble base: XGBoost | 0.8973 | 0.9075 | -0.0102 | 0.0107 | -0.0311 | 0.0107 | -0.9533 | 0.3404 | 1.0000 | False |
| MLP_SGD | MLP_ADAM | 0.8790 | 0.8852 | -0.0062 | 0.0066 | -0.0191 | 0.0067 | -0.9453 | 0.3445 | 1.0000 | False |
| Random Forest | MLP_ADAM | 0.9002 | 0.8852 | 0.0149 | 0.0163 | -0.0170 | 0.0468 | 0.9184 | 0.3584 | 1.0000 | False |
| Ensemble base: Logistic Regression | Ensemble base: Random Forest | 0.8853 | 0.8973 | -0.0120 | 0.0135 | -0.0384 | 0.0145 | -0.8874 | 0.3749 | 1.0000 | False |
| XGBoost | Hybrid Soft-Voting Ensemble | 0.9094 | 0.9003 | 0.0091 | 0.0103 | -0.0111 | 0.0293 | 0.8815 | 0.3780 | 1.0000 | False |
| XGBoost | Hybrid Stacking Ensemble (Ours) | 0.9094 | 0.9004 | 0.0090 | 0.0105 | -0.0115 | 0.0295 | 0.8576 | 0.3911 | 1.0000 | False |
| Ensemble base: XGBoost | Hybrid Soft-Voting Ensemble | 0.9075 | 0.9003 | 0.0072 | 0.0084 | -0.0093 | 0.0237 | 0.8522 | 0.3941 | 1.0000 | False |
| Ensemble base: XGBoost | Hybrid Stacking Ensemble (Ours) | 0.9075 | 0.9004 | 0.0071 | 0.0086 | -0.0099 | 0.0240 | 0.8173 | 0.4137 | 1.0000 | False |
| Ensemble base: Random Forest | TabTransformer | 0.8973 | 0.8855 | 0.0118 | 0.0147 | -0.0171 | 0.0407 | 0.8027 | 0.4221 | 1.0000 | False |
| GCN (Kipf & Welling) | GAT (Velickovic et al.) | 0.8871 | 0.8815 | 0.0056 | 0.0071 | -0.0083 | 0.0196 | 0.7900 | 0.4295 | 1.0000 | False |
| Ensemble base: Random Forest | MLP_ADAM | 0.8973 | 0.8852 | 0.0121 | 0.0167 | -0.0206 | 0.0447 | 0.7246 | 0.4687 | 1.0000 | False |
| Ensemble base: Random Forest | GCN (Kipf & Welling) | 0.8973 | 0.8871 | 0.0102 | 0.0143 | -0.0178 | 0.0381 | 0.7130 | 0.4758 | 1.0000 | False |
| Ensemble base: PyTorch MLP | GCN (Kipf & Welling) | 0.8767 | 0.8871 | -0.0104 | 0.0157 | -0.0412 | 0.0204 | -0.6626 | 0.5076 | 1.0000 | False |
| MLP_SGD | GCN (Kipf & Welling) | 0.8790 | 0.8871 | -0.0081 | 0.0124 | -0.0325 | 0.0162 | -0.6541 | 0.5130 | 1.0000 | False |
| Ensemble base: Logistic Regression | MLP_SGD | 0.8853 | 0.8790 | 0.0063 | 0.0099 | -0.0131 | 0.0257 | 0.6399 | 0.5222 | 1.0000 | False |
| Ensemble base: Logistic Regression | Ensemble base: PyTorch MLP | 0.8853 | 0.8767 | 0.0086 | 0.0137 | -0.0183 | 0.0355 | 0.6267 | 0.5308 | 1.0000 | False |
| Random Forest | Ensemble base: XGBoost | 0.9002 | 0.9075 | -0.0073 | 0.0122 | -0.0313 | 0.0167 | -0.5962 | 0.5510 | 1.0000 | False |
| Ensemble base: PyTorch MLP | MLP_ADAM | 0.8767 | 0.8852 | -0.0085 | 0.0146 | -0.0371 | 0.0201 | -0.5819 | 0.5606 | 1.0000 | False |
| Logistic Regression | MLP_SGD | 0.8841 | 0.8790 | 0.0051 | 0.0099 | -0.0142 | 0.0245 | 0.5200 | 0.6031 | 1.0000 | False |
| Logistic Regression | Ensemble base: PyTorch MLP | 0.8841 | 0.8767 | 0.0074 | 0.0157 | -0.0234 | 0.0382 | 0.4716 | 0.6372 | 1.0000 | False |
| Ensemble base: PyTorch MLP | TabTransformer | 0.8767 | 0.8855 | -0.0087 | 0.0209 | -0.0497 | 0.0322 | -0.4180 | 0.6759 | 1.0000 | False |
| Random Forest | Ensemble base: Random Forest | 0.9002 | 0.8973 | 0.0029 | 0.0070 | -0.0109 | 0.0166 | 0.4097 | 0.6821 | 1.0000 | False |
| Ensemble base: Logistic Regression | GAT (Velickovic et al.) | 0.8853 | 0.8815 | 0.0038 | 0.0100 | -0.0157 | 0.0234 | 0.3833 | 0.7015 | 1.0000 | False |
| Ensemble base: Random Forest | Hybrid Stacking Ensemble (Ours) | 0.8973 | 0.9004 | -0.0031 | 0.0084 | -0.0196 | 0.0133 | -0.3704 | 0.7111 | 1.0000 | False |
| Ensemble base: Random Forest | Hybrid Soft-Voting Ensemble | 0.8973 | 0.9003 | -0.0030 | 0.0085 | -0.0196 | 0.0136 | -0.3521 | 0.7247 | 1.0000 | False |
| MLP_SGD | TabTransformer | 0.8790 | 0.8855 | -0.0065 | 0.0184 | -0.0425 | 0.0295 | -0.3515 | 0.7252 | 1.0000 | False |
| Logistic Regression | GCN (Kipf & Welling) | 0.8841 | 0.8871 | -0.0030 | 0.0091 | -0.0209 | 0.0149 | -0.3272 | 0.7435 | 1.0000 | False |
| Logistic Regression | GAT (Velickovic et al.) | 0.8841 | 0.8815 | 0.0026 | 0.0095 | -0.0161 | 0.0213 | 0.2758 | 0.7827 | 1.0000 | False |
| Ensemble base: PyTorch MLP | GAT (Velickovic et al.) | 0.8767 | 0.8815 | -0.0048 | 0.0180 | -0.0400 | 0.0304 | -0.2663 | 0.7900 | 1.0000 | False |
| TabTransformer | GAT (Velickovic et al.) | 0.8855 | 0.8815 | 0.0039 | 0.0149 | -0.0253 | 0.0332 | 0.2645 | 0.7914 | 1.0000 | False |
| Logistic Regression | Ensemble base: Logistic Regression | 0.8841 | 0.8853 | -0.0012 | 0.0048 | -0.0106 | 0.0082 | -0.2489 | 0.8034 | 1.0000 | False |
| MLP_ADAM | GAT (Velickovic et al.) | 0.8852 | 0.8815 | 0.0037 | 0.0150 | -0.0256 | 0.0330 | 0.2476 | 0.8044 | 1.0000 | False |
| Ensemble base: Logistic Regression | GCN (Kipf & Welling) | 0.8853 | 0.8871 | -0.0018 | 0.0084 | -0.0183 | 0.0147 | -0.2131 | 0.8312 | 1.0000 | False |
| MLP_SGD | GAT (Velickovic et al.) | 0.8790 | 0.8815 | -0.0025 | 0.0134 | -0.0287 | 0.0237 | -0.1880 | 0.8509 | 1.0000 | False |
| XGBoost | Ensemble base: XGBoost | 0.9094 | 0.9075 | 0.0019 | 0.0130 | -0.0235 | 0.0273 | 0.1477 | 0.8826 | 1.0000 | False |
| Hybrid Soft-Voting Ensemble | Hybrid Stacking Ensemble (Ours) | 0.9003 | 0.9004 | -0.0001 | 0.0008 | -0.0018 | 0.0015 | -0.1436 | 0.8859 | 1.0000 | False |
| Ensemble base: PyTorch MLP | MLP_SGD | 0.8767 | 0.8790 | -0.0023 | 0.0160 | -0.0336 | 0.0291 | -0.1420 | 0.8871 | 1.0000 | False |
| MLP_ADAM | GCN (Kipf & Welling) | 0.8852 | 0.8871 | -0.0019 | 0.0136 | -0.0286 | 0.0248 | -0.1406 | 0.8882 | 1.0000 | False |
| TabTransformer | GCN (Kipf & Welling) | 0.8855 | 0.8871 | -0.0017 | 0.0164 | -0.0338 | 0.0305 | -0.1020 | 0.9187 | 1.0000 | False |
| Logistic Regression | TabTransformer | 0.8841 | 0.8855 | -0.0013 | 0.0138 | -0.0283 | 0.0257 | -0.0954 | 0.9240 | 1.0000 | False |
| Logistic Regression | MLP_ADAM | 0.8841 | 0.8852 | -0.0011 | 0.0122 | -0.0251 | 0.0229 | -0.0879 | 0.9300 | 1.0000 | False |
| Random Forest | Hybrid Stacking Ensemble (Ours) | 0.9002 | 0.9004 | -0.0002 | 0.0094 | -0.0186 | 0.0182 | -0.0255 | 0.9797 | 1.0000 | False |
| Random Forest | Hybrid Soft-Voting Ensemble | 0.9002 | 0.9003 | -0.0001 | 0.0093 | -0.0184 | 0.0182 | -0.0128 | 0.9898 | 1.0000 | False |
| MLP_ADAM | TabTransformer | 0.8852 | 0.8855 | -0.0002 | 0.0199 | -0.0392 | 0.0387 | -0.0120 | 0.9904 | 1.0000 | False |
| Ensemble base: Logistic Regression | MLP_ADAM | 0.8853 | 0.8852 | 0.0001 | 0.0114 | -0.0222 | 0.0224 | 0.0105 | 0.9916 | 1.0000 | False |
| Ensemble base: Logistic Regression | TabTransformer | 0.8853 | 0.8855 | -0.0001 | 0.0156 | -0.0307 | 0.0305 | -0.0077 | 0.9939 | 1.0000 | False |