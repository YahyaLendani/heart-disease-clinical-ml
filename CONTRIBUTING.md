# Contributing to the Cardiovascular Clinical ML Pipeline

Thank you for your interest in contributing to this research repository! This project implements a reproducible, publication-grade clinical machine learning framework spanning tabular deep learning, graph neural networks, uncertainty quantification, conformal prediction, causal inference, and algorithmic fairness for cardiovascular risk and ICU mortality prediction.

To preserve scientific rigor, clinical safety invariants, and deterministic reproducibility, please follow the guidelines below.

---

## 1. Development Environment Setup

### Prerequisites
- **Python:** 3.10, 3.11, 3.12, or 3.13
- **Git:** 2.35+
- **Optional:** NVIDIA CUDA-capable GPU for accelerated PyTorch/XGBoost training (all scripts gracefully support `--device cpu` and `--xgb-device cpu`).

### Installation

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YahyaLendani/heart-disease-clinical-ml.git
   cd heart-disease-clinical-ml
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows PowerShell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies and editable package with developer extras:
   ```bash
   pip install --upgrade pip
   pip install -e ".[dev]"
   ```

---

## 2. Architectural & Methodological Standards

Contributions are organized into four modular packages:

- `models/` — Neural architectures (`HeartMLP`, `TabTransformer`, `GCN`, `GAT`, Multimodal Cross-Attention surrogates).
- `training/` — Training loops, classical baselines (`LogisticRegression`, `RandomForest`, `XGBoost`), MIMIC-IV mortality pipelines, and hybrid stacking/voting ensembles.
- `evaluation/` — Calibration (Temperature Scaling, ECE, Brier Score), SHAP explainability, demographic fairness (`Equal Opportunity`, disparate impact), and thesis table synchronization.
- `experiments/` — Leave-One-Center-Out (LOCO) cross-hospital generalization, Split/Mondrian Conformal Prediction, Doubly Robust Causal Inference (IPTW/AIPW, E-values), and Counterfactual Clinical Recourse.

### Key Scientific Guardrails
1. **Zero Data Leakage:** All feature scalers (`StandardScaler`), encoders (`OneHotEncoder`), imputation statistics, conformal calibration quantiles, and optimal decision thresholds (`Youden's J`, F1, or Equal Opportunity thresholds) **must** be fit strictly on training or dedicated validation splits—never on holdout test folds.
2. **No Selection on the Test Split:** Model, optimizer, hyperparameter and early-stopping choices use validation data or cross-validation only. Report test metrics with bootstrap confidence intervals (`evaluation/cv_utils.py`).
3. **Deterministic Seeding:** Every stochastic routine must accept and respect a `--seed` parameter (default `42`) across NumPy, PyTorch, and scikit-learn splits.
4. **Immutable Clinical Constraints:** Counterfactual recourse and feature perturbation modules must strictly enforce immutability on non-modifiable patient attributes (e.g., `age`, `sex`) and verify zero immutable violation (`0.000000`).

---

## 3. Running Tests & Quality Checks

Before opening a Pull Request, ensure that the entire test suite and linter pass locally:

```bash
# Run the full pytest suite
pytest tests/ -v --tb=short

# Run Ruff linter
ruff check .

# Smoke-test the end-to-end pipeline (uses a GPU if present, otherwise the CPU)
heart-pipeline --epochs 10 --cv-folds 3 --cv-repeats 1 --output-dir scratch_artifacts
```

> **Note:** `heart-pipeline` no longer runs `pytest` before every pipeline run (CI does that on each push). Pass `--run-tests` to run it first.

---

## 4. Pull Request Process

1. **Create a Feature Branch:** Branch off `main` using a descriptive prefix (`feat/`, `fix/`, `exp/`, `docs/`):
   ```bash
   git checkout -b feat/conformal-adaptive-calibration
   ```
2. **Add Unit Tests:** Any new model architecture, loss function, calibration method, or fairness metric must include corresponding unit tests in `tests/`.
3. **Update Documentation & Artifacts:** If your change modifies empirical benchmarks or CLI flags, update `README.md` and run `python update_thesis_results.py --task all` where applicable.
4. **Submit PR:** Fill out the `.github/pull_request_template.md` checklist completely. Ensure all GitHub Actions CI checks (`Python 3.10 - 3.13`) pass.

---

## 5. Code of Conduct & Clinical Disclaimer

This software is developed strictly for academic research and methodological benchmarking in healthcare AI. It is **not** a certified medical device and must not be used for direct clinical diagnosis or patient management without regulatory clearance and prospective clinical validation.

