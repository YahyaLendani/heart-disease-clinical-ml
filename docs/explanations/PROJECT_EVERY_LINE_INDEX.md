# Project Every-Line Index (Python Files)

This file is a navigation index for detailed code reading. It keeps the same guide-style format but focuses on where to inspect line-level logic in each script.

## 1) How to Use This Index

1. Start with the script you want to understand.
2. Open the script side-by-side with `PROJECT_LINE_BY_LINE_GUIDE.md`.
3. Use the "Key Line Regions" below to jump to the most important sections.
4. Read top-to-bottom for control flow, then revisit helper functions.

---

## 2) File Reading Order (Recommended)

1. `clean_data.py`
2. `eda.py`
3. `train_model.py`
4. `baseline_models.py`
5. `run_pipeline.py`
6. `mimic_prepare_dataset.py`
7. `mimic_train_baselines.py`
8. `tune_train_model.py`
9. `update_thesis_results.py`
10. `validate_scratch_mlp.py`

---

## 3) `clean_data.py` Key Line Regions

### Input and schema
- Imports and `COLUMNS` definition.
- `build_arg_parser()`.
- `load_raw_data(...)` source selection logic.

### Cleaning logic
- `clean_dataset(...)` conversion/imputation/target normalization.

### Execution entry
- `main()` save + summary printing.

---

## 4) `eda.py` Key Line Regions

### EDA helper functions
- `_compute_missingness(...)`
- `_compute_outlier_counts(...)`
- `_compute_target_group_stats(...)`
- `_compute_target_separation(...)`
- `_compute_target_correlations(...)`

### Main EDA flow
- `run_eda(...)` console output + artifact exports.
- `main()` dataset loading and target normalization.

---

## 5) `train_model.py` Key Line Regions

### Core model definitions
- `ScratchLinear`
- `HeartMLP`

### Training/evaluation helpers
- `set_seed(...)`
- `smooth_binary_targets(...)`
- `make_loss(...)`
- `evaluate_loss(...)`
- `evaluate_accuracy(...)`
- `predict_binary(...)`

### Training engine
- `train_one_optimizer(...)` epoch loop, early stopping, scheduler, artifact saving.

### Runtime setup
- `build_arg_parser(...)`
- `resolve_torch_device(...)`
- `main()` preprocessing, tensor creation, DataLoader setup, optimizer comparison output.

---

## 6) `baseline_models.py` Key Line Regions

### Configuration and model params
- RF/XGB baseline/candidate dictionaries.

### Data/preprocessing
- `load_data(...)`
- `build_preprocessors(...)`

### Evaluation and model selection
- `evaluate_model(...)`
- `cross_validate_model(...)`
- `select_best_params(...)`

### Hardware + orchestration
- `resolve_xgb_device(...)`
- `main()` CV, holdout eval, early stopping for XGB, result exports, thesis update trigger.

---

## 7) `mimic_prepare_dataset.py` Key Line Regions

### Data loading and guards
- `build_arg_parser(...)`
- `_safe_read_csv(...)`

### Feature engineering pipeline
- `build_dataset(...)`:
  - timestamp parsing
  - LOS feature construction
  - ICU aggregation
  - diagnosis indicator construction
  - merges and imputations

### Save/output
- `main()` output CSV + cohort summary.

---

## 8) `mimic_train_baselines.py` Key Line Regions

### Input and device handling
- `build_arg_parser(...)`
- `resolve_xgb_device(...)`
- `load_data(...)`

### Metrics and CV flow
- `evaluate_holdout(...)`
- `evaluate_cv(...)`

### End-to-end training
- `main()` class imbalance handling, model list setup, CV + holdout scoring, exports, thesis update.

---

## 9) `run_pipeline.py` Key Line Regions

### Script runner utility
- `run_script(...)` subprocess invocation and error propagation.

### CLI + orchestration
- `build_arg_parser(...)`
- `main()` full sequence:
  - clean
  - EDA
  - PyTorch train
  - optional baselines
  - optional thesis update

---

## 10) `tune_train_model.py` Key Line Regions

### Tuning config and parsing
- `build_arg_parser(...)`
- `parse_metrics(...)`

### Trial execution
- `run_trial(...)` command composition + stdout parsing.

### Ranking/saving
- `main()` grid generation, trial loop, CSV + JSON export.

---

## 11) `update_thesis_results.py` Key Line Regions

### Block marker controls
- auto-block start/end markers for MIMIC, heart baseline, heart EDA.

### Block builders
- `build_mimic_results_block(...)`
- `build_heart_results_block(...)`
- `build_heart_eda_block(...)`

### Insert/replace strategy
- `replace_or_insert_block(...)`.

### Task entry points
- `main()` branches for `mimic`, `heart`, `all`.

---

## 12) `validate_scratch_mlp.py` Key Line Regions

- synthetic batch generation
- forward shape assertion
- backward gradient assertions
- optimizer-step parameter-change assertion

---

## 13) Companion Files

- For conceptual walkthrough: `PROJECT_LINE_BY_LINE_GUIDE.md`
- For docs/data-focused reference: `PROJECT_DOCS_AND_DATA_INDEX.md`
- For quick navigation: `README.md` in the same folder
