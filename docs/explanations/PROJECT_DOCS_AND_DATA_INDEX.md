# Project Docs and Data Index

This file keeps the same guide-style format but focuses on project documentation/config files and dataset artifacts rather than Python code walkthrough.

## 1) What This File Covers

1. Main documentation files and what each is used for.
2. Configuration/dependency reference.
3. Core dataset files and generated artifact files.
4. Suggested reading order for thesis/report preparation.

---

## 2) Documentation Files

### `README.md`
- Primary operational runbook.
- Contains command examples for:
  - heart pipeline
  - baseline models
  - MIMIC dataset/training
  - thesis updater tasks
  - GPU flags

### `thesis_draft.md`
- Main thesis manuscript draft.
- Contains manually written narrative + auto-generated blocks.
- Auto-updated sections are inserted by `update_thesis_results.py` using markers.

### `AI_Thesis_Reading-List.csv`
- Primary literature and theoretical reading list for the thesis.
- Contains 19 curated seminal papers spanning deep learning foundations, Adam optimization, tabular transformers, attention mechanisms, graph neural networks, multimodal clinical AI, vision transformers, SHAP/Grad-CAM interpretability, calibration, deep ensembles, algorithmic fairness, causal inference, and healthcare AI ethics.

### `docs/explanations/README.md`
- Local index for explanation files.
- Recommended reading sequence.

---

## 3) Config and Dependency Files

### `requirements.txt`
- Python package requirements for this project:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `torch`
  - `xgboost`

### `.venv/` (environment folder)
- Local Python virtual environment.
- Used to run scripts with pinned dependencies.

---

## 4) Core Dataset Files

### `heart.csv`
- Main heart dataset source (preferred when present).

### `cleve.mod`
- Fallback heart dataset source.

### `cleaned_heart_disease.csv`
- Canonical cleaned heart dataset produced by `clean_data.py`.
- Input to EDA, PyTorch training, and heart baselines.

### `mimic_hospital_mortality.csv`
- Engineered admissions-level MIMIC mortality dataset.
- Produced by `mimic_prepare_dataset.py`.
- Input to `mimic_train_baselines.py`.

---

## 5) Main Artifact Groups (`artifacts/`)

### Training artifacts (PyTorch)
- `heart_mlp_sgd.pt`
- `heart_mlp_adam.pt`
- `training_history_sgd.csv`
- `training_history_adam.csv`
- `confusion_matrix_sgd.csv`
- `confusion_matrix_adam.csv`

### Heart baseline artifacts
- `baseline_comparison.csv`
- `baseline_comparison.md`

### MIMIC baseline artifacts
- `mimic_baseline_comparison.csv`
- `mimic_baseline_comparison.md`

### EDA artifacts
- `eda_missingness.csv`
- `eda_outliers_iqr.csv`
- `eda_target_group_stats.csv`
- `eda_target_correlations.csv`
- `eda_target_separation.csv`

### Tuning artifacts
- `tuning_results.csv`
- `best_hyperparameters.json`

---

## 6) Thesis Auto-Update Markers (Reference)

### In `thesis_draft.md`
- MIMIC results block markers:
  - `AUTO_RESULTS_START`
  - `AUTO_RESULTS_END`

- Heart baseline block markers:
  - `AUTO_HEART_RESULTS_START`
  - `AUTO_HEART_RESULTS_END`

- Heart EDA block markers:
  - `AUTO_HEART_EDA_START`
  - `AUTO_HEART_EDA_END`

These markers are read/written by `update_thesis_results.py`.

---

## 7) Suggested Reading Order (Docs/Data Focus)

1. `README.md`
2. `requirements.txt`
3. `cleaned_heart_disease.csv` (inspect columns)
4. `mimic_hospital_mortality.csv` (inspect columns)
5. `artifacts/baseline_comparison.csv`
6. `artifacts/mimic_baseline_comparison.csv`
7. `thesis_draft.md`

---

## 8) Companion Explanation Files

- For code walkthrough: `PROJECT_LINE_BY_LINE_GUIDE.md`
- For code navigation by script section: `PROJECT_EVERY_LINE_INDEX.md`
