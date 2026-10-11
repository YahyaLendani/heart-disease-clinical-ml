# Heart Disease Pipeline

This project is split into three clear stages:

1. **Clean data** → `clean_data.py`
2. **Run EDA** → `eda.py`
3. **Train model (PyTorch)** → `train_model.py`
4. **Train baseline models** → `baseline_models.py`

## Project Structure (Refactored)

- `models/`
	- shared architecture modules (e.g., `models/heart_mlp.py`)
- `training/`
	- `training/train_model.py`
	- `training/baseline_models.py`
	- `training/mimic_train_baselines.py`
- `evaluation/`
	- `evaluation/compare_separate_runs.py`
	- `evaluation/update_thesis_results.py`
- `experiments/`
	- `experiments/tune_train_model.py`
	- `experiments/analyze_overfitting_network_size.py`
- `configs/`
	- `configs/requirements.txt`
- Existing root commands are preserved via compatibility wrappers, so you can continue using:
	- `python train_model.py`
	- `python baseline_models.py`
	- `python mimic_train_baselines.py`

## Quick Start (One Command)

Run the full pipeline:

```bash
python run_pipeline.py
```

Run with GPU (PyTorch + XGBoost where available):

```bash
python run_pipeline.py --device cuda --xgb-device cuda
```

Run full pipeline + baseline models:

```bash
python run_pipeline.py --run-baselines
```

Run full pipeline + baseline models with XGBoost early stopping settings:

```bash
python run_pipeline.py --run-baselines --xgb-early-stopping-rounds 25 --xgb-eval-size 0.2
```

Run full pipeline + baselines with cross-validation folds:

```bash
python run_pipeline.py --run-baselines --cv-folds 5
```

Run full pipeline with a custom validation split for MLP holdout training:

```bash
python run_pipeline.py --val-size 0.2
```

Run full pipeline and auto-update thesis results block at the end:

```bash
python run_pipeline.py --update-thesis
```

Run full pipeline and auto-update both heart + MIMIC thesis result blocks:

```bash
python run_pipeline.py --update-thesis-all
```

Run complete AI Thesis reading list suite (TabTransformer, GNNs, Multimodal, Calibration, Deep Ensembles, Causal, Fairness):

```bash
python run_reading_list_suite.py
# or via main pipeline:
python run_pipeline.py --run-reading-list
```

## Step-by-Step

### 1) Clean dataset

```bash
python clean_data.py
```

`clean_data.py` automatically uses `heart.csv` if present in the project folder; otherwise it falls back to `cleve.mod`.

Output:
- `cleaned_heart_disease.csv`

### 2) Perform EDA

```bash
python eda.py
```

Output includes:
- dataset shape and duplicates
- missing values
- data types
- target distribution
- numeric statistics
- top category levels
- means by target class

### 3) Train and evaluate model

```bash
python train_model.py
```

Run PyTorch MLP with 5-fold cross-validation (default) and holdout test evaluation:

```bash
python train_model.py --optimizer both --cv-folds 5
```

Change validation split used for holdout training (default: `0.2`):

```bash
python train_model.py --val-size 0.2
```

Force GPU for PyTorch training:

```bash
python train_model.py --device cuda
```

Current model:
- PyTorch MLP classifier with preprocessing (`StandardScaler` + `OneHotEncoder`)

Metrics printed:
- accuracy
- precision/recall/F1 classification report
- confusion matrix

Save artifacts (best model + confusion matrix):

```bash
python train_model.py --optimizer both --save-checkpoint --save-confusion
```

Train longer for better accuracy over time:

```bash
python train_model.py --optimizer both --loss bce_weighted --epochs 60 --patience 12 --save-checkpoint --save-confusion
```

Train with smaller/larger MLP hidden layers:

```bash
python train_model.py --optimizer adam --hidden-dims 32 16
python train_model.py --optimizer adam --hidden-dims 128 64
```

Use stricter early stopping (minimum validation-loss improvement):

```bash
python train_model.py --optimizer both --min-delta 0.001
```

Automatic hyperparameter tuning (dropout, grad clip, label smoothing):

```bash
python tune_train_model.py --epochs 30 --output-dir artifacts
```

Tune learning rate explicitly (example sweep):

```bash
python tune_train_model.py --epochs 30 --learning-rates 0.0005 0.001 0.005 --output-dir artifacts
```

Learning-rate tuning also saves:
- `artifacts/learning_rate_vs_accuracy.png`

Saved in `artifacts/`:
- `heart_mlp_sgd.pt`, `heart_mlp_adam.pt`
- `confusion_matrix_sgd.csv`, `confusion_matrix_adam.csv`
- `training_history_sgd.csv`, `training_history_adam.csv`
- `training_curves_sgd.png`, `training_curves_adam.png`
- `mlp_cv_fold_metrics_sgd.csv`, `mlp_cv_fold_metrics_adam.csv`
- `mlp_cv_summary.csv`
- `tuning_results.csv`, `best_hyperparameters.json`

### 4) Train baseline models

```bash
python baseline_models.py
```

With XGBoost early stopping settings:

```bash
python baseline_models.py --xgb-early-stopping-rounds 25 --xgb-eval-size 0.2
```

XGBoost early-stopping curve image is saved to:
- `artifacts/xgb_learning_curve.png`

With cross-validation folds:

```bash
python baseline_models.py --cv-folds 5
```

Use GPU for XGBoost in baseline training:

```bash
python baseline_models.py --cv-folds 5 --xgb-device cuda
```

Train heart baselines and auto-update thesis results block:

```bash
python baseline_models.py --cv-folds 5 --update-thesis
```

Baselines trained:
- Logistic Regression
- Random Forest
- XGBoost

Metrics printed per model:
- accuracy
- F1-score
- ROC-AUC
- classification report

At the end, a comparison table ranks all baseline models.
The table includes holdout test metrics and cross-validation metrics (mean/std).

### 5) Train every model separately, then compare

This workflow logs each separate run to `artifacts/separate_model_runs.csv`.

Train PyTorch MLP with SGD:

```bash
python train_model.py --optimizer sgd --save-run-metrics
```

Train PyTorch MLP with Adam:

```bash
python train_model.py --optimizer adam --save-run-metrics
```

Train heart Logistic Regression baseline only:

```bash
python baseline_models.py --model logreg --save-run-metrics
```

Train heart Random Forest baseline only:

```bash
python baseline_models.py --model rf --save-run-metrics
```

Train heart XGBoost baseline only:

```bash
python baseline_models.py --model xgb --save-run-metrics
```

Compare all separately trained models:

```bash
python compare_separate_runs.py
```

Outputs:
- `artifacts/separate_model_runs.csv`
- `artifacts/separate_model_comparison.csv`
- `artifacts/separate_model_comparison.md`

### 6) Analyze overfitting across smaller/larger networks

Run architecture-size overfitting analysis (small/base/large by default):

```bash
python analyze_overfitting_network_size.py --optimizer adam --epochs 40 --cv-folds 3
```

Custom architectures:

```bash
python analyze_overfitting_network_size.py --architectures small:16,8 base:64,32 large:128,64
```

Outputs:
- `artifacts/overfitting_by_network_size.csv`
- `artifacts/overfitting_by_network_size.md`
- per-architecture training artifacts under `artifacts/overfit_<label>/`

## Concepts Implemented (Requested)

### 1) PyTorch autograd
- Training uses `loss.backward()` to compute gradients automatically.
- Optimizer updates are done with `optimizer.step()` and reset via `optimizer.zero_grad()`.

### 2) Training loops
- Explicit epoch/batch loop is implemented in `train_model.py`.
- Per-epoch **train loss** and **validation loss** are printed.

### 3) Loss functions
- Supported options:
	- `bce` → `BCEWithLogitsLoss`
	- `bce_weighted` → `BCEWithLogitsLoss(pos_weight=...)` for class imbalance

Example:

```bash
python train_model.py --loss bce_weighted
```

### 4) Optimization: SGD vs Adam
- Supports `SGD` and `Adam`, or both in one run.
- Adam settings follow the original paper defaults (`betas=(0.9, 0.999)`, `eps=1e-8`).
- Comparison table is printed after training.

Examples:

```bash
python train_model.py --optimizer sgd
python train_model.py --optimizer adam
python train_model.py --optimizer both
```

### 5) Overfitting controls
- Validation split for monitoring generalization.
- Dropout in the MLP.
- L2 regularization via `weight_decay`.
- Early stopping using validation loss + patience.

Example:

```bash
python train_model.py --epochs 80 --patience 12 --dropout 0.25 --weight-decay 1e-4
```

## AI Thesis Reading List & References

The project codebase and manuscript (`thesis_draft.md`) are guided by the 19 seminal papers and texts documented in [`AI_Thesis_Reading-List.csv`](file:///c:/Users/yahya/Documents/Final%20Project/AI_Thesis_Reading-List.csv):

1. **Foundations (Deep Learning & Optimization):**
   - Goodfellow, Bengio, & Courville (2016) — *Deep Learning* (Ch. 6 Feedforward, Ch. 7 Regularization, Ch. 8 Optimization).
   - Kingma & Ba (2015) — *Adam: A Method for Stochastic Optimization* (`1412.6980v9.pdf`).
2. **Tabular Deep Learning & Attention:**
   - Huang et al. (2020) — *TabTransformer: Tabular Data Modeling Using Contextual Embeddings*.
   - Vaswani et al. (2017) — *Attention Is All You Need* (multi-head scaled dot-product self-attention).
3. **Graph Neural Networks (GNN):**
   - Kipf & Welling (2017) — *Semi-Supervised Classification with Graph Convolutional Networks (GCN)*.
   - Veličković et al. (2018) — *Graph Attention Networks (GAT)*.
   - Zhang et al. (2020) — *Graph-BERT: Node Representation Learning with Graph Transformer*.
4. **Multimodal Machine Learning & Vision:**
   - Baltrušaitis, Ahuja, & Morency (2018) — *Multimodal Machine Learning: A Survey* (5 core challenges).
   - Dosovitskiy et al. (2020) — *An Image Is Worth 16x16 Words: Vision Transformer (ViT)*.
   - He et al. (2016) — *Deep Residual Learning for Image Recognition (ResNet)*.
5. **Model Explainability (XAI):**
   - Lundberg & Lee (2017) — *SHAP: A Unified Approach to Interpreting Model Predictions* (game-theoretic Shapley values).
   - Selvaraju et al. (2017) — *Grad-CAM: Visual Explanations from Deep Networks*.
6. **Calibration & Uncertainty Quantification:**
   - Guo et al. (2017) — *On Calibration of Modern Neural Networks* (Brier Score, ECE, temperature scaling).
   - Lakshminarayanan, Pritzel, & Blundell (2017) — *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*.
7. **Algorithmic Fairness, Causality & Clinical Ethics:**
   - Barocas, Hardt, & Narayanan (2020) — *Fairness and Machine Learning: Limitations & Opportunities*.
   - Pearl & Mackenzie (2018) — *The Book of Why: The New Science of Cause and Effect* (Ladder of Causation).
   - Bryson (2019) — *Ethics of Artificial Intelligence and Machine Learning*.
8. **Healthcare AI & Scientific Writing:**
   - Marques et al. (2021) — *Artificial Intelligence in Healthcare: Past, Present and Future*.
   - Patel & Peyton Jones (2013) — *How to Write a Great Research Paper*.

## Python Interpreter (Your Environment)

This project now uses a single GPU-enabled virtual environment: `.venv`.

Activate it in PowerShell:

```bash
.\.venv\Scripts\Activate.ps1
```

Then run scripts normally with `python`, for example:

```bash
python run_pipeline.py --device cuda --xgb-device cuda
```

## MIMIC-IV Demo: Hospital Mortality

Build a flat admissions-level dataset from MIMIC demo tables:

```bash
python mimic_prepare_dataset.py
```

Train baseline models on hospital mortality:

```bash
python mimic_train_baselines.py --cv-folds 5 --output-dir artifacts
```

Use GPU for XGBoost in MIMIC baseline training:

```bash
python mimic_train_baselines.py --cv-folds 5 --output-dir artifacts --xgb-device cuda
```

Train MIMIC baselines and auto-update thesis results block:

```bash
python mimic_train_baselines.py --cv-folds 5 --output-dir artifacts --update-thesis
```

Outputs:
- `mimic_hospital_mortality.csv`
- `artifacts/mimic_baseline_comparison.csv`
- `artifacts/mimic_baseline_comparison.md`

Auto-insert latest MIMIC cohort stats + baseline table into thesis draft:

```bash
python update_thesis_results.py
```

Auto-insert latest heart cohort stats + baseline table into thesis draft:

```bash
python update_thesis_results.py --task heart --dataset cleaned_heart_disease.csv --comparison artifacts/baseline_comparison.csv
```

If EDA artifacts exist (`artifacts/eda_*.csv`), the same command also auto-inserts a Heart EDA Findings block into the thesis.

Auto-insert both MIMIC and heart result blocks in one command:

```bash
python update_thesis_results.py --task all
```
