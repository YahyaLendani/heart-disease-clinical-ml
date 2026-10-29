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

## 8) `mimic_train_baselines.py` line-by-line (top to bottom)

### Imports and parser
- Similar ML stack to heart baselines.
- Includes metrics tailored for class imbalance and probability ranking.
- Parser accepts input file, folds, output dir, thesis update, and XGB device.

### `resolve_xgb_device(...)`
- Same GPU probe/fallback logic as heart baseline script.

### `load_data(path)`
- Reads MIMIC admissions dataset.
- Splits target `hospital_expire_flag` from features.
- Drops ID columns from feature matrix.

### `make_preprocessor(X)`
- StandardScaler for numeric + OHE for categorical.

### `evaluate_holdout(...)`
- Computes holdout metrics:
  - accuracy
  - expired-class F1/recall/precision
  - ROC-AUC
  - PR-AUC
- Prints classification report and returns dict.

### `evaluate_cv(...)`
- Stratified CV with same mortality-aware metric set.
- Returns mean/std for all tracked metrics.

### `main()`
- Parses args.
- Loads data and resolves XGB device.
- Creates stratified train/test split.
- Computes `scale_pos_weight` from class imbalance.
- Builds LR/RF/XGB pipelines.
- Runs CV + holdout evaluation per model.
- Sorts summary prioritizing PR-AUC and minority metrics.
- Saves CSV and markdown tables.
- If `--update-thesis`, triggers thesis updater for MIMIC block.

---

## 9) `run_pipeline.py` line-by-line (top to bottom)

### Imports
- Parser + subprocess + interpreter path handling.

### `run_script(script_name, extra_args=None)`
- Resolves script path.
- Verifies file exists.
- Executes using current Python interpreter.
- Raises if return code is non-zero.

### `build_arg_parser()`
- Exposes heart training options, GPU options, baseline options, and thesis update options.

### `main()`
- Parses args.
- Executes sequential pipeline:
  1) `clean_data.py`
  2) `eda.py`
  3) `train_model.py` with forwarded training args
  4) optional `baseline_models.py` with forwarded baseline args
  5) optional thesis update (`--update-thesis`)
  6) optional full thesis update (`--update-thesis-all`)
- Prints completion banner.

---

## 10) `tune_train_model.py` line-by-line (top to bottom)

### Imports
- Uses itertools for grid search combinations.
- Uses regex to parse training stdout.
- Uses pandas/json for saving tuning outputs.

### `build_arg_parser()`
- Defines tuning grid and shared training args.

### `parse_metrics(stdout, optimizer)`
- Regex-extracts validation and test accuracy from trainer summary line.

### `run_trial(...)`
- Builds subprocess command for `train_model.py` with one parameter combination.
- Runs process and captures stdout/stderr.
- Parses metrics if successful output format exists.
- Returns trial record dictionary.

### `main()`
- Locates `train_model.py`.
- Creates output directory.
- Builds Cartesian product of all candidate hyperparameters.
- Runs all trials sequentially.
- Drops failed/empty metric rows.
- Ranks by test then validation accuracy.
- Saves ranked CSV and best-config JSON.
- Prints best configuration summary.

---

## 11) `update_thesis_results.py` line-by-line (top to bottom)

### Imports and markers
- Uses argparse/path/pandas.
- Defines HTML-style start/end markers for auto-managed blocks:
  - MIMIC results
  - Heart results
  - Heart EDA findings

### `build_arg_parser()`
- Adds flexible paths for thesis, datasets, comparison tables, EDA artifacts.
- Adds `--task` with `mimic`, `heart`, or `all`.

### Formatting and summary helpers
- `to_markdown_table`: converts DataFrame to markdown table with formatted numbers.
- `cohort_summary_mimic`: computes MIMIC cohort summary stats.
- `choose_primary_metric`: selects ranking metric from available columns.
- `build_mimic_results_block`: creates markdown block for MIMIC results.
- `infer_heart_target` + `cohort_summary_heart`: normalize target and compute heart cohort summary.
- `build_heart_results_block`: builds heart baseline results markdown block.
- `_top_records`: helper for top-N extraction.
- `build_heart_eda_block`: builds heart EDA findings block from EDA artifact CSVs.

### `replace_or_insert_block(...)`
- Replaces existing marker block if present.
- Otherwise inserts block before references or inside template results section.

### `main()`
- Parses args and reads thesis file.
- For `mimic` task:
  - validates files/columns
  - builds MIMIC block
  - writes thesis
- For `heart` task:
  - validates files
  - builds heart baseline block
  - optionally adds heart EDA block if EDA artifact files exist
  - writes thesis
- For `all` task:
  - validates both MIMIC + heart inputs
  - updates MIMIC block, heart block, and optional heart EDA block
  - writes thesis once at end

---

## 12) `validate_scratch_mlp.py` line-by-line (top to bottom)

- Imports `HeartMLP` and `set_seed` from trainer module.
- Seeds RNG.
- Creates synthetic input batch and binary targets.
- Runs forward pass and asserts output shape.
- Saves pre-step parameters.
- Backpropagates BCEWithLogits loss.
- Asserts gradients exist and are non-zero.
- Runs one optimizer step.
- Compares parameters pre/post step and asserts they changed.
- Prints success, loss, and total parameter delta.

---

## 13) What each important data/artifact file is for

- `heart.csv`: main heart dataset used by cleaner.
- `cleve.mod`: fallback heart dataset.
- `cleaned_heart_disease.csv`: canonical cleaned input for EDA/train/baselines.
- `mimic_hospital_mortality.csv`: engineered MIMIC admissions-level mortality dataset.
- `artifacts/training_history_*.csv`: epoch-level training curves.
- `artifacts/confusion_matrix_*.csv`: confusion matrices for PyTorch runs.
- `artifacts/heart_mlp_*.pt`: saved model checkpoints.
- `artifacts/baseline_comparison.csv`: heart baseline leaderboard.
- `artifacts/mimic_baseline_comparison.csv`: MIMIC baseline leaderboard.
- `artifacts/eda_*.csv`: exported EDA tables for thesis/reporting.

---

## 14) If you want truly literal line-by-line for each exact line number

This guide explains each statement in execution order. If you want the **literal** version (every line number including blank lines and separators), I can generate a second file that enumerates every single line as:

`<file>:<line_number> -> explanation`

for all `.py` files in the repo.