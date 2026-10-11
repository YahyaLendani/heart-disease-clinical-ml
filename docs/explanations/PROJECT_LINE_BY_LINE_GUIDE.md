# Project Line-by-Line Guide (Step-by-Step)

This file explains the project from start to finish and then explains each source file in execution order.

## 1) Big-Picture Step-by-Step Flow

1. Install dependencies from `requirements.txt`.
2. Run `clean_data.py` to build `cleaned_heart_disease.csv` from `heart.csv` (or fallback `cleve.mod`).
3. Run `eda.py` to print EDA and export EDA artifact CSV files in `artifacts/`.
4. Run `train_model.py` to train the scratch PyTorch MLP (SGD/Adam) and save training artifacts.
5. Optionally run `baseline_models.py` to train LR/RF/XGBoost heart baselines.
6. Optionally run `mimic_prepare_dataset.py` to build MIMIC mortality dataset.
7. Optionally run `mimic_train_baselines.py` for MIMIC baseline modeling.
8. Optionally run `update_thesis_results.py` to auto-refresh thesis result sections.
9. Or run `run_pipeline.py` to orchestrate steps 2–5 (and thesis update flags).

---

## 2) `requirements.txt` (dependency lines)

- `pandas`: DataFrame operations, cleaning, CSV I/O.
- `numpy`: numeric operations and effect size/math in EDA/MIMIC.
- `scikit-learn`: preprocessing, splitting, metrics, CV, classical models.
- `torch`: scratch MLP, autograd training loop.
- `xgboost`: gradient-boosted tree baseline model.

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

## 5) `train_model.py` line-by-line (top to bottom)

### Imports
- Standard libs: paths, arg parsing, deep copy, RNG.
- Data/scikit tools for preprocessing and metrics.
- PyTorch core, nn, functional, and data utilities.

### `set_seed(seed)`
- Seeds Python, NumPy, and Torch RNG.
- Seeds all CUDA devices if available.
- Sets deterministic cuDNN mode for reproducibility.

### `ScratchLinear`
- Defines a custom linear layer using explicit trainable tensors (`nn.Parameter`).
- `reset_parameters`: Kaiming init for weights, zero init for bias.
- `forward`: manual matrix multiply + bias.

### `HeartMLP`
- Defines 3-layer scratch MLP:
  - `fc1`: input -> 64
  - `fc2`: 64 -> 32
  - `fc3`: 32 -> 1
- Forward pass:
  - linear -> ReLU -> dropout
  - linear -> ReLU -> dropout
  - final linear -> squeeze to shape `(batch,)`.

### `smooth_binary_targets(targets, smoothing)`
- Applies label smoothing for binary targets.
- If smoothing is 0, returns original targets unchanged.

### `make_loss(loss_name, y_train_tensor)`
- Returns `BCEWithLogitsLoss` for `bce`.
- For `bce_weighted`, computes `pos_weight = negatives/positives` and returns weighted BCE loss.

### `evaluate_loss(...)`
- Sets eval mode.
- Runs no-grad pass over loader.
- Moves batch tensors to selected device.
- Computes average validation/test loss.

### `evaluate_accuracy(...)`
- Eval mode + no-grad.
- Computes sigmoid probabilities.
- Thresholds at 0.5.
- Returns sklearn accuracy.

### `predict_binary(...)`
- Same as evaluation path, but returns binary predictions list only.

### `train_one_optimizer(...)`
- Creates optimizer (`SGD` or `Adam`) from CLI choice.
- Creates `ReduceLROnPlateau` scheduler.
- Initializes best-checkpoint trackers.
- Epoch loop:
  - train mode.
  - batch loop: move to device, optional label smoothing, forward, loss, backward, optional grad clipping, optimizer step.
  - computes train and validation metrics.
  - scheduler step on val loss.
  - appends history row.
  - early stopping by patience.
- Loads best weights.
- Evaluates on test split.
- Prints report and confusion matrix.
- Saves optional checkpoint and confusion matrix; always saves history if path passed.
- Returns metrics dictionary.

### `build_arg_parser()`
- Defines CLI options for optimizer/loss/epochs/training controls.
- Includes regularization controls (`dropout`, `weight_decay`, `grad_clip`, `label_smoothing`).
- Includes GPU control flag `--device` (`auto`, `cpu`, `cuda`).

### `resolve_torch_device(requested_device)`
- Enforces CPU if requested.
- Enforces CUDA and raises if unavailable when explicitly requested.
- Uses auto-detect for `auto`.

### `main()`
- Parses args and seeds all RNGs.
- Loads cleaned dataset and normalizes target.
- Splits into train/val/test with stratification.
- Builds `ColumnTransformer`:
  - StandardScaler on numeric.
  - OneHotEncoder on categorical.
- Transforms train/val/test.
- Converts arrays to tensors.
- Resolves device and sets DataLoader `pin_memory` for GPU.
- Builds criterion and moves it to selected device.
- Runs selected optimizer(s).
- Saves artifacts into output directory.
- Prints optimizer comparison and best optimizer.

---

## 6) `baseline_models.py` line-by-line (top to bottom)

### Imports and parameter dictionaries
- Imports ML stack and XGBoost + GPU detection dependencies.
- Defines RF/XGB baseline and candidate parameter sets.

### `load_data()`
- Reads cleaned heart dataset.
- Ensures binary `target` exists.
- Splits into features `X` and labels `y`.

### `build_preprocessors(X)`
- Returns two preprocessors:
  - scaled numeric + OHE categorical for linear models.
  - passthrough numeric + OHE categorical for tree models.

### `evaluate_model(...)`
- Predicts class and probabilities.
- Computes accuracy, F1, ROC-AUC.
- Prints classification report.
- Returns summary dict.

### `cross_validate_model(...)`
- Stratified K-fold CV.
- Computes mean/std for accuracy, F1, ROC-AUC.

### `select_best_params(...)`
- Builds baseline and candidate pipelines.
- CV-evaluates both.
- Chooses param set with better CV ROC-AUC.

### `build_arg_parser()`
- Adds baseline CLI options: early stopping rounds, eval split, folds, output dir, thesis update flag, XGBoost device flag.

### `resolve_xgb_device(requested_device)`
- CPU immediate if requested.
- Auto: CPU if no torch CUDA.
- Else attempts tiny XGBoost CUDA fit probe.
- Falls back to CPU for `auto`; raises for strict `cuda` if unavailable.

### `main()`
- Parses args and loads data.
- Resolves XGBoost device and prints it.
- Builds train/test split and XGB train/val split for early stopping.
- Selects best RF and XGB params by CV.
- Builds and CV-evaluates models.
- Fits LR and RF pipelines and evaluates holdout.
- Fits XGBoost separately with early stopping.
- Builds summary table and sorts by holdout metrics.
- Saves CSV + markdown comparison to artifacts.
- If `--update-thesis`, runs `update_thesis_results.py` for heart block refresh.

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