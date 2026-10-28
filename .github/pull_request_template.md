## Summary of Changes

<!-- Provide a concise scientific and technical overview of what this PR introduces or fixes. -->

## Related Issue(s)

- Closes #<!-- issue number -->

## Type of Change

- [ ] **Model Architecture** (`models/` — e.g., MLP, TabTransformer, GCN, GAT, Multimodal)
- [ ] **Training / Optimization** (`training/` — e.g., loss functions, early stopping, ensembles, MIMIC-IV)
- [ ] **Clinical Evaluation & Trustworthiness** (`evaluation/` or `experiments/` — e.g., Conformal Prediction, Calibration, Causal AIPW/IPTW, Fairness, SHAP, Counterfactual Recourse)
- [ ] **Data Pipeline / EDA** (`clean_data.py`, `eda.py`, `mimic_prepare_dataset.py`)
- [ ] **Bug Fix / Refactoring / CI**
- [ ] **Documentation / Thesis Results Update**

## Scientific & Reproducibility Checklist

- [ ] **Zero Data Leakage Verified:** All preprocessing transforms (`StandardScaler`, `OneHotEncoder`), imputation values, calibration temperatures, and conformal quantiles are fit strictly on training/validation partitions.
- [ ] **Deterministic Reproducibility:** Random seeds (`--seed 42`) are properly propagated across NumPy, PyTorch, and scikit-learn splits.
- [ ] **Immutable Clinical Constraints Preserved:** If modifying counterfactual recourse or perturbation logic, non-modifiable attributes (`age`, `sex`) remain strictly locked (`0.000000` violation).
- [ ] **Unit & Integration Tests Passing:** `pytest tests/ -v --tb=short` passes locally across all test modules.
- [ ] **Code Formatting & Linting:** `ruff check .` reports no syntax or style regressions.
- [ ] **Artifact & Documentation Sync:** Updated `README.md` or ran `python update_thesis_results.py --task all` if empirical metrics changed.

## Empirical Benchmark Impact (If Applicable)

<!-- Paste before/after metrics from artifacts/baseline_comparison.md or artifacts/reading_list_advancements_summary.md -->

| Metric | Previous | Updated | Delta |
| :--- | :---: | :---: | :---: |
| Test ROC-AUC | | | |
| Test Brier Score | | | |
| 90% Conformal Coverage | | | |

