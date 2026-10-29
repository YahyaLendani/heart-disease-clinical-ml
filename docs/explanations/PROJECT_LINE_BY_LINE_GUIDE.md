# Project Line-by-Line Guide (Step-by-Step)

This file explains the project from start to finish and then explains each source file in execution order.

## 1) Big-Picture Step-by-Step Flow

1. Install the package and dependencies: `pip install -e ".[dev]"`.
2. `heart-pipeline clean` builds `data/processed/cleaned_heart_disease.csv` from `data/raw/heart.csv` (or fallback `data/raw/cleve.mod`).
3. `heart-pipeline eda` prints EDA and exports EDA CSV files in `artifacts/`.
4. `heart-pipeline train-mlp` trains the scratch PyTorch MLP (SGD/Adam) with repeated CV and saves training artifacts.
5. Optionally `heart-pipeline baselines` trains LR/RF/XGBoost heart baselines.
6. Optionally `heart-pipeline prepare-mimic` builds the MIMIC mortality dataset.
7. Optionally `heart-pipeline mimic-baselines` / `mimic-mlp` for MIMIC modeling.
8. Optionally `heart-pipeline update-thesis` refreshes thesis result sections.
9. Or run `heart-pipeline` with flags to orchestrate steps 2–5 (and the optional stages) in one process.

All file locations come from `project_paths.py`; dataset arguments accept bare names that are looked up in `data/processed` and `data/raw`.

---

## 2) `requirements.txt` (dependency lines)

- `pandas`: DataFrame operations, cleaning, CSV I/O.
- `numpy`: numeric operations and effect size/math in EDA/MIMIC.
- `scikit-learn`: preprocessing, splitting, metrics, CV, classical models.
- `torch`: scratch MLP, autograd training loop.
- `xgboost`: gradient-boosted tree baseline model.
- `scipy`, `joblib`: rank statistics for fast bootstrap AUCs; process-parallel CV folds.
- `shap`, `matplotlib`: explanations and figures.

---

## 3) `clean_data.py` line-by-line (top to bottom)

### Imports and constants
- `import argparse`: enables command-line flags.
- `from pathlib import Path`: robust file paths.
- `import pandas as pd`: tabular data operations.
- `COLUMNS = [...]`: explicit schema for non-CSV Cleveland-style input.

### `build_arg_parser()`
- Creates parser with description.
- Adds optional `--input`.
- If omitted, script auto-detects source dataset.

### `load_raw_data(base_dir, input_arg)`
- Resolves source path from CLI or fallback order: `heart.csv` then `cleve.mod`.
- Validates source exists.
- If `.csv`:
  - loads with `pd.read_csv`.
  - normalizes column names to lowercase/trimmed.
- Else:
  - loads whitespace-delimited format with predefined `COLUMNS`.
- Returns loaded DataFrame and path used.

### `clean_dataset(df)`
- Copies input to avoid mutating caller data.
- Trims whitespace for object/string columns.
- Replaces `?` with proper missing marker `pd.NA`.
- Attempts numeric conversion for known numeric candidate fields.
- Drops duplicate rows.
- Splits features into numeric/categorical by dtype.
- Numeric missing values: median imputation.
- Categorical missing values: mode imputation.
- If no `target` but has `diagnosis`, creates binary `target` (`sick` -> 1 else 0).
- Forces `target` numeric int (coerce, fill 0, cast int).
- Returns cleaned DataFrame.

### `main()`
- Parses args.
- Defines output path `cleaned_heart_disease.csv`.
- Loads raw dataset.
- Cleans dataset.
- Saves cleaned CSV.
- Prints source used and row/missingness summary.

---

## 4) `eda.py` line-by-line (top to bottom)

### Imports
- `Path`: output path building.
- `pandas`, `numpy`: EDA calculations.

### `_compute_missingness(df)`
- Calculates missing counts per column.
- Converts to missing percentages.
- Builds standardized missingness table.
- Sorts by highest missing count.

### `_compute_outlier_counts(df, numeric_cols)`
- Iterates numeric columns.
- Converts each feature to numeric safely.
- Computes Q1/Q3 and IQR.
- Uses IQR rule bounds (`Q1 - 1.5*IQR`, `Q3 + 1.5*IQR`).
- Counts outliers and percent per feature.
- Returns sorted outlier table.

### `_compute_target_group_stats(df, numeric_cols)`
- Groups by `target`.
- Aggregates mean/median/std/min/max per numeric feature.
- Flattens hierarchical columns to single names.

### `_compute_target_separation(df, numeric_cols)`
- Splits data into target 0 vs target 1.
- For each numeric feature:
  - computes class means.
  - computes difference (`target1 - target0`).
  - computes pooled standard deviation.
  - computes Cohen’s d effect size.
- Returns sorted separation table by absolute effect size.

### `_compute_target_correlations(df, numeric_cols)`
- For each numeric feature except `target`:
  - computes Pearson and Spearman correlation with target.
- Sorts by absolute Pearson strength.

### `run_eda(df, output_dir)`
- Prints dataset shape and duplicate count.
- Prints missing-value section.
- Prints dtype section.
- Prints target distribution (count + percent).
- Prints numeric `describe()` summary.
- Prints top outlier summary.
- Prints top levels for categorical columns.
- Prints means by target.
- Prints full target-stratified stats.
- Prints top correlations with target.
- Prints top class-separation features.
- Creates output directory if needed.
- Writes EDA artifact CSVs:
  - `eda_missingness.csv`
  - `eda_outliers_iqr.csv`
  - `eda_target_group_stats.csv`
  - `eda_target_correlations.csv`
  - `eda_target_separation.csv`

### `main()`
- Ensures `cleaned_heart_disease.csv` exists.
- Loads cleaned data.
- Builds/normalizes `target`.
- Calls `run_eda(...)` with `artifacts/` path.

---

## 5) `training/train_model.py` step by step

`HeartMLP` and `ScratchLinear` live in `models/heart_mlp.py`; seeding and tensor batching live in `training/torch_utils.py`.

### `make_loss`, `make_optimizer`, `smooth_binary_targets`
- `bce` or `bce_weighted` (`pos_weight = negatives / positives`).
- SGD (momentum 0.9) or Adam, both with weight decay.
- Optional label smoothing toward 0.5.

### `fit_with_early_stopping(...)`
- Data stays in tensors on the target device; `iterate_minibatches` shuffles an index with a seeded generator (no `DataLoader`).
- Each epoch: mini-batch updates with optional gradient clipping, then **one** full-batch validation pass that yields both validation loss and accuracy.
- `ReduceLROnPlateau` on validation loss; early stopping with patience; the best weights are restored.

### `train_one_optimizer(...)`
- Trains on the holdout train split with early stopping on the validation split.
- Predicts test and validation probabilities once each.
- Decision thresholds (F1, Youden) are tuned on validation and applied to test.
- Adds stratified bootstrap 95% CIs to every test metric; saves checkpoint, confusion matrix, history and curves when asked.

### `_run_cv_fold(...)` and `run_mlp_cross_validation(...)`
- Repeated stratified k-fold (`--cv-folds`, `--cv-repeats`) on train+val.
- Inside each fold, a separate split of the training portion is used for early stopping, so the scored fold never picks the stopping epoch.
- Each fold has its own seed (`seed + 1000 * fold`), so results do not depend on run order; on CPU the folds run in parallel processes with one thread each.

### `main(argv=None)`
- Loads the dataset via `project_paths.resolve_data_path`, makes an 80/20 train+val / test split and an inner train/val split.
- Runs CV and holdout training per optimizer; writes `mlp_cv_fold_metrics_*.csv`, `mlp_cv_summary.csv`, `mlp_holdout_summary.csv`.
- The "best optimizer" is the one with the lowest **validation** loss; the test split is never consulted for selection.
- Optional SHAP analysis on the selected model. Returns the result rows (used by `tune` and `overfit`).

---

## 6) `training/baseline_models.py` step by step

### Imports and parameter dictionaries
- RF/XGB baseline and candidate parameter sets, all with `n_jobs=1`: on a few hundred rows thread start-up costs more than it saves, so parallelism happens across CV folds instead.

### `load_data()` / `build_preprocessors(X)`
- Reads the cleaned dataset and builds scaled (linear models) and unscaled (trees) preprocessors; imputation is fit inside each fold.

### `cross_validate_model(...)`
- Repeated stratified k-fold CV using `evaluation.cv_utils.multi_metric_scorer`, which calls `predict_proba` once per fold and returns all five metrics.
- Folds run in parallel (`training.device_utils.cv_n_jobs_for`), except for GPU XGBoost.

### `select_best_params(...)`
- Cross-validates the baseline and candidate configurations and keeps the better CV ROC-AUC.
- Returns the chosen parameters **and** their CV scores, so the chosen configuration is not cross-validated a second time.

### `holdout_result(...)` / `evaluate_model(...)`
- Test metrics at the 0.5 threshold, validation-tuned F1/Youden thresholds applied to test, and bootstrap 95% CIs.

### `main(argv=None)`
- Splits 60/20/20, selects RF/XGB parameters, fits LR and RF, fits XGBoost with early stopping on the validation split.
- Sorts the summary by CV ROC-AUC, writes `baseline_comparison.{csv,md}` and figures; `--update-thesis` calls `evaluation.update_thesis_results.main` in-process.

Device selection (`--xgb-device auto|cpu|cuda`) lives in `training/device_utils.py`.

---

## 7) `mimic_prepare_dataset.py` line-by-line (top to bottom)

### Imports and defaults
- Loads required libs.
- Defines default local MIMIC root path.

### `build_arg_parser()`
- Allows custom `--mimic-root` and output filename.

### `_safe_read_csv(path)`
- Validates path exists.
- Reads CSV or raises clear error.

### `build_dataset(mimic_root)`
- Reads admissions/patients/icustays/diagnoses tables.
- Parses datetime columns for admissions.
- Engineers LOS features (`hospital_los_days`, `ed_los_hours`).
- Normalizes mortality target (`hospital_expire_flag`).
- Keeps patient demographics.
- Aggregates ICU stay counts/LOS and first unit.
- Cleans diagnosis codes and versions.
- Creates diagnosis indicator features:
  - circulatory
  - diabetes
  - CKD
- Aggregates diagnosis burden features per admission.
- Merges all feature blocks into one admissions-level dataset.
- Fills ICU/diagnosis missing values.
- Selects final modeling feature list.
- Performs type-aware missing-value imputation.
- Deduplicates by `hadm_id`.
- Returns final DataFrame.

### `main()`
- Parses args.
- Builds dataset.
- Saves output CSV.
- Prints size + mortality rate + output location.

---

## 8) `training/mimic_train_baselines.py` step by step

### Imports, constants and parser
- Shares scoring (`evaluation.cv_utils`) and device logic (`training.device_utils`) with the heart scripts.
- `MIN_TEST_EVENTS_FOR_RANKING = 10`: below this many test deaths the script warns that holdout rankings are unstable.
- Parser: input file, folds, `--cv-repeats`, `--seed`, output dir, thesis update, XGB device, SHAP flag.

### `load_data(path)` / `make_preprocessor(X)`
- Reads the admissions dataset, separates `hospital_expire_flag`, keeps `subject_id` for patient-grouped splitting.

### `split_train_val_test(...)`
- `StratifiedGroupKFold` on `subject_id` for test and validation, so no patient appears in two partitions.
- The leakage assertions compare actual subject IDs (an earlier version mixed index spaces and failed every run).

### `evaluate_holdout(...)`
- Expired-class F1/recall/precision, ROC-AUC, PR-AUC, Brier; thresholds tuned on validation; bootstrap 95% CIs; `test_events` column.

### `evaluate_cv(...)`
- Repeated patient-grouped stratified CV with the single-pass scorer.

### `main(argv=None)`
- Splits, reports deaths per split (and the small-sample warning), fits LR/RF/XGB, sorts by CV PR-AUC, writes tables and figures, optional SHAP and thesis update.

The companion `training/mimic_train_mlp.py` uses the same split; its CV early-stops on a separate patient-grouped split inside each fold.

> The bundled MIMIC-IV demo has 275 admissions and 15 deaths. Treat MIMIC outputs as a demonstration that the pipeline runs on ICU data, not as model rankings.

---

## 9) `run_pipeline.py` (the `heart-pipeline` command)

### `STEPS`
- Maps each subcommand (`clean`, `prepare-combined`, `eda`, `train-mlp`, `baselines`, `ensemble`, `loco`, `fairness`, `reading-list`, `compare-models`, `mimic-*`, `tune`, `overfit`, `update-thesis`, `test`, ...) to a module with `main(argv)`.

### `run_step(step, argv)`
- Imports the module and calls `main(argv)` in the same process, printing the elapsed time.

### `run_full_pipeline(args)`
1. Optional `--run-tests`.
2. Data preparation: `prepare-combined` for the default 918-patient cohort (`clean` when `--dataset cleaned_heart_disease.csv`), then `eda`.
3. `train-mlp`; optional `baselines`, `ensemble`, `loco`, `fairness` (LOCO and fairness always use the multi-center cohort), `reading-list`.
4. `compare-models`: paired DeLong tests across every model scored on the shared test split.
5. Optional thesis updates.

---

## 10) `experiments/tune_train_model.py` step by step

- Builds the Cartesian product of optimizer, learning rate, dropout, gradient clipping and label smoothing.
- `run_trial(...)` calls `training.train_model.main(argv)` in-process (no subprocess or stdout parsing) with a per-trial output folder.
- Ranks trials by repeated-CV ROC-AUC, then validation loss. Test accuracy is recorded for reference only.
- Saves `tuning_results.csv`, `best_hyperparameters.json` and a learning-rate vs CV ROC-AUC plot.

---

## 11) `evaluation/update_thesis_results.py` step by step

### Markers and parser
- HTML-comment markers delimit auto-managed blocks (MIMIC, heart, heart EDA, SHAP, ensemble, LOCO, fairness).
- `--task mimic|heart|all`; `--dataset`/`--comparison` override the task-specific `--mimic-*` / `--heart-*` defaults.

### Block builders
- `build_mimic_results_block` adds `small_sample_note(...)` when the test split has fewer than 10 deaths.
- `build_heart_results_block`, `build_heart_eda_block`, `build_shap_block` and the advanced blocks format artifact CSVs as Markdown.

### `main(argv=None)`
- Resolves paths (bare dataset names are looked up in `data/processed`), rebuilds the requested blocks and writes `thesis_draft.md`. Text outside the markers is never touched, so the hand-written discussion must be updated separately.

---

## 12) `evaluation/model_comparison.py` step by step

- `save_test_predictions(...)`: each training script writes per-patient test probabilities to `artifacts/test_predictions/<source>.csv`.
- `load_test_predictions(...)`: merges files that share the dataset and test rows (all scripts use the same seed-42 first split).
- `delong_test(...)`: fast DeLong test (Sun & Xu, 2014) for two correlated AUCs; returns the difference, its 95% CI and p-value.
- `holm_adjust(...)`: Holm step-down correction across all pairs.
- `main(argv)`: writes `model_comparison.{csv,md}`.

---

## 13) `validate_scratch_mlp.py` and key files

`validate_scratch_mlp.py` seeds the RNGs, runs one forward/backward/step on a synthetic batch and asserts that the output shape is right, gradients are non-zero and parameters change.

Key data and artifact files:
- `data/raw/heart.csv`, `data/raw/cleve.mod`, `data/raw/processed.*.data`: source files.
- `data/processed/cleaned_heart_disease_combined.csv`: primary 918-patient cohort (zero-coded cholesterol and blood pressure recoded as missing).
- `data/processed/cleaned_heart_disease.csv`: Cleveland-only cohort (Kaggle placeholder codes `ca = 4`, `thal = 0` recoded as missing).
- `data/processed/mimic_hospital_mortality.csv`: MIMIC-IV demo admissions dataset.
- `artifacts/baseline_comparison.csv`, `artifacts/mlp_holdout_summary.csv`, `artifacts/reading_list_advancements_summary.csv`: results with 95% CIs and repeated-CV columns.
- `artifacts/model_comparison.md`: paired DeLong tests.
- `artifacts/hyperparameter_search_*.csv`: every configuration tried for Random Forest and XGBoost.
- `artifacts/cleveland/`: the same benchmarks on the Cleveland-only cohort.

---

## 14) If you want truly literal line-by-line for each exact line number

This guide explains each statement in execution order. If you want the **literal** version (every line number including blank lines and separators), I can generate a second file that enumerates every single line as:

`<file>:<line_number> -> explanation`

for all `.py` files in the repo.