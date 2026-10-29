# Project Every-Line Index (Python Files)

This file is a navigation index for detailed code reading. It keeps the same guide-style format but focuses on where to inspect line-level logic in each script.

## 1) How to Use This Index

1. Start with the script you want to understand.
2. Open the script side-by-side with `PROJECT_LINE_BY_LINE_GUIDE.md`.
3. Use the "Key Line Regions" below to jump to the most important sections.
4. Read top-to-bottom for control flow, then revisit helper functions.

---

## 2) File Reading Order (Recommended)

1. `project_paths.py`
2. `clean_data.py`
3. `eda.py`
4. `training/train_model.py` (with `training/torch_utils.py`)
5. `training/baseline_models.py` (with `evaluation/cv_utils.py`, `training/device_utils.py`)
6. `run_pipeline.py`
7. `mimic_prepare_dataset.py`
8. `training/mimic_train_baselines.py`
9. `experiments/tune_train_model.py`
10. `evaluation/update_thesis_results.py`
11. `validate_scratch_mlp.py`

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

## 5) `training/train_model.py` Key Line Regions

### Core model definitions
- `ScratchLinear`, `HeartMLP` are in `models/heart_mlp.py`.

### Training/evaluation helpers
- `smooth_binary_targets(...)`, `make_loss(...)`, `make_optimizer(...)`
- `predict_proba(...)` (full-batch inference)
- `training/torch_utils.py`: `set_seed`, `iterate_minibatches`, `predict_logits`, `clone_state`

### Training engine
- `fit_with_early_stopping(...)`: epoch loop, single validation pass, scheduler, early stopping.
- `train_one_optimizer(...)`: holdout training, validation-tuned thresholds, bootstrap CIs, artifact saving.

### Cross-validation
- `_run_cv_fold(...)`: one fold with its own early-stopping split and seed (runs in a worker process).
- `run_mlp_cross_validation(...)`: repeated stratified k-fold, folds in parallel.

### Runtime setup
- `build_arg_parser(...)`, `load_heart_dataset(...)`
- `main(argv)`: splits, preprocessing, CV + holdout per optimizer, selection by validation loss, outputs.

---

## 6) `training/baseline_models.py` Key Line Regions

### Configuration and model params
- `RF_BASE_PARAMS`/`RF_PARAM_GRID`, `XGB_BASE_PARAMS`/`XGB_PARAM_GRID` and `expand_grid(...)` (8 configurations each; `n_jobs=1`, folds run in parallel instead).

### Data/preprocessing
- `load_data(...)`
- `build_preprocessors(...)`

### Evaluation and model selection
- `holdout_result(...)`, `evaluate_model(...)`: validation-tuned thresholds and bootstrap CIs.
- `cross_validate_model(...)`: repeated CV with the single-pass scorer.
- `select_best_params(...)`: cross-validates every grid configuration; returns the winner, its CV scores and the full search table.

### Hardware + orchestration
- `training/device_utils.py`: `resolve_xgb_device(...)`, `align_xgb_inference_device(...)`, `cv_n_jobs_for(...)`
- `main(argv)`: CV, holdout eval, early stopping for XGB, result exports, thesis update.

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

## 8) `training/mimic_train_baselines.py` Key Line Regions

### Input and device handling
- `build_arg_parser(...)`, `load_data(...)`; devices via `training/device_utils.py`.

### Splitting, metrics and CV flow
- `split_train_val_test(...)`: patient-grouped splits and subject-ID leakage assertions.
- `evaluate_holdout(...)`: validation-tuned thresholds, bootstrap CIs, `test_events`.
- `evaluate_cv(...)`: repeated patient-grouped CV with the single-pass scorer.

### End-to-end training
- `main(argv)`: small-sample warning, class-imbalance handling, model setup, CV + holdout scoring, exports, thesis update.

---

## 9) `run_pipeline.py` Key Line Regions

### Step registry
- `STEPS` and `run_step(...)`: in-process subcommands with timing.

### CLI + orchestration
- `build_arg_parser(...)`
- `run_full_pipeline(...)`: data preparation (combined cohort by default), EDA, PyTorch training, optional baselines/ensemble/LOCO/fairness/reading list, `compare-models`, optional thesis update.
- `main(argv)`: dispatches `heart-pipeline <step>` or the full pipeline.

---

## 10) `experiments/tune_train_model.py` Key Line Regions

### Tuning config
- `build_arg_parser(...)`, `save_learning_rate_plot(...)`

### Trial execution
- `run_trial(...)`: calls `training.train_model.main(argv)` in-process.

### Ranking/saving
- `main(argv)`: grid generation, trial loop, ranking by repeated-CV ROC-AUC, CSV + JSON export.

---

## 11) `evaluation/update_thesis_results.py` Key Line Regions

### Block marker controls
- Auto-block start/end markers for MIMIC, heart baseline, heart EDA, SHAP, ensemble, LOCO, fairness.

### Block builders
- `build_mimic_results_block(...)` (with `small_sample_note(...)`)
- `build_heart_results_block(...)`
- `build_heart_eda_block(...)`

### Insert/replace strategy
- `replace_or_insert_block(...)`.

### Task entry points
- `main(argv)` branches for `mimic`, `heart`, `all`.

---

## 11b) `evaluation/model_comparison.py` Key Line Regions

- `save_test_predictions(...)` / `load_test_predictions(...)`: per-patient test probabilities shared across scripts.
- `_midrank(...)`, `delong_test(...)`: fast DeLong test for correlated AUCs.
- `holm_adjust(...)`, `compare_all_models(...)`, `main(argv)`: pairwise table with Holm-adjusted p-values.

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
