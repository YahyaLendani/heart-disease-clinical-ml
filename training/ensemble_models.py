"""Hybrid Stacking and Soft-Voting Ensemble for Cardiovascular Risk Prediction.

Combines complementary model paradigms:
1. Parametric Linear: Logistic Regression (L2 regularized)
2. Bagging Trees: Random Forest
3. Gradient Boosted Trees: XGBoost (GPU/hist)
4. Deep Neural Network: PyTorch HeartMLP (ScratchLinear autograd)

Ensemble strategies evaluated:
- Weighted Soft-Voting (weights optimised on Out-Of-Fold probabilities)
- Stacking Classifier (Meta-Learner trained on Out-Of-Fold probabilities)

Decision thresholds are tuned on out-of-fold training predictions, never on
the test split, and every test metric carries a bootstrap 95% CI. The
fold-by-model fits are independent, so they run in parallel processes.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean_data import build_preprocessing_pipeline
from evaluation.clinical_metrics import compute_clinical_metrics
from evaluation.cv_utils import add_bootstrap_cis, resolve_n_jobs, to_markdown_table
from models import HeartMLP, add_cardiovascular_features
from project_paths import resolve_data_path, resolve_output_dir
from training.torch_utils import iterate_minibatches, single_thread


class PyTorchMLPClassifier(BaseEstimator, ClassifierMixin):
    """Scikit-learn compatible wrapper for the PyTorch HeartMLP."""

    def __init__(
        self,
        hidden_dims: tuple[int, int] = (64, 32),
        dropout: float = 0.25,
        epochs: int = 40,
        batch_size: int = 32,
        lr: float = 0.001,
        weight_decay: float = 1e-4,
        device: str = "auto",
        random_state: int = 42,
    ) -> None:
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.random_state = random_state

        self.model_ = None
        self.classes_ = np.array([0, 1])

    def _get_device(self) -> torch.device:
        if self.device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cpu")

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> PyTorchMLPClassifier:
        torch.manual_seed(self.random_state)
        dev = self._get_device()

        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.float32)

        X_t = torch.tensor(X_np, dtype=torch.float32, device=dev)
        y_t = torch.tensor(y_np, dtype=torch.float32, device=dev)
        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        self.model_ = HeartMLP(
            input_dim=X_np.shape[1],
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(dev)

        positives = float(y_np.sum())
        negatives = float(len(y_np) - positives)
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32, device=dev)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.model_.train()
        for _ in range(self.epochs):
            # Data is already on the device; batches are tensor slices.
            for xb, yb in iterate_minibatches(X_t, y_t, self.batch_size, shuffle=True, generator=generator):
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()

        return self

    @torch.inference_mode()
    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        dev = self._get_device()
        self.model_.eval()
        X_np = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X_np).to(dev)
        logits = self.model_(X_t)
        probs_pos = torch.sigmoid(logits).cpu().numpy()
        probs_pos = np.clip(probs_pos, 1e-7, 1 - 1e-7)
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


def _fit_and_predict(model_proto, X_fit, y_fit, X_pred):
    """Fit a fresh copy of ``model_proto``; return (fitted model, P(y=1) on X_pred).

    One PyTorch thread in every case, so results do not depend on ``n_jobs``.
    """
    with single_thread():
        model = copy.deepcopy(model_proto)
        model.fit(X_fit, y_fit)
        proba = model.predict_proba(X_pred)[:, 1] if X_pred is not None else None
    return model, proba


def generate_oof_and_fit_base_models(
    base_specs: list[tuple[str, any]],
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42,
    n_jobs: int = 1,
) -> tuple[np.ndarray, list[tuple[str, any]]]:
    """Compute leak-free Out-Of-Fold (OOF) probabilities and fit full base models once.

    The ``folds x models`` fits plus the final full-data fits are independent,
    so they are dispatched together to ``n_jobs`` worker processes.
    """
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    folds = list(skf.split(X, y))
    oof_preds = np.zeros((len(X), len(base_specs)), dtype=np.float64)

    jobs = []
    for train_idx, val_idx in folds:
        for _, model_proto in base_specs:
            jobs.append((model_proto, X.iloc[train_idx], y.iloc[train_idx], X.iloc[val_idx]))
    for _, model_proto in base_specs:
        jobs.append((model_proto, X, y, None))

    workers = min(resolve_n_jobs(n_jobs), len(jobs))
    if workers > 1:
        from joblib import Parallel, delayed

        outputs = Parallel(n_jobs=workers)(delayed(_fit_and_predict)(*job) for job in jobs)
    else:
        outputs = [_fit_and_predict(*job) for job in jobs]

    n_models = len(base_specs)
    for fold_pos, (_, val_idx) in enumerate(folds):
        for m_idx in range(n_models):
            oof_preds[val_idx, m_idx] = outputs[fold_pos * n_models + m_idx][1]

    final_outputs = outputs[len(folds) * n_models:]
    fitted_base_models = [(name, final_outputs[m_idx][0]) for m_idx, (name, _) in enumerate(base_specs)]
    return oof_preds, fitted_base_models


class SoftVotingEnsemble:
    """Combines predicted probabilities via simplex-optimized or custom weights."""

    def __init__(
        self,
        models: list[tuple[str, any]],
        weights: list[float] | np.ndarray | None = None,
        metric: str = "roc_auc",
    ) -> None:
        self.models = models
        self.metric = metric
        if weights is None:
            self.weights = np.ones(len(models), dtype=np.float64) / len(models)
        else:
            w = np.asarray(weights, dtype=np.float64)
            w = np.clip(w, 0.0, None)
            self.weights = w / w.sum() if w.sum() > 0 else np.ones(len(models), dtype=np.float64) / len(models)

    def fit_weights(
        self,
        oof_probas: np.ndarray,
        y_true: np.ndarray | pd.Series,
        metric: str | None = None,
        maxiter: int = 1000,
    ) -> SoftVotingEnsemble:
        """Optimize ensemble weights on the probability simplex via Nelder-Mead."""
        target_metric = (metric or self.metric).lower()
        y_arr = np.asarray(y_true, dtype=int).ravel()
        P = np.clip(np.asarray(oof_probas, dtype=np.float64), 1e-7, 1.0 - 1e-7)
        n_models = P.shape[1]

        def _objective(theta: np.ndarray) -> float:
            w = softmax(theta)
            blended = np.clip(P @ w, 1e-7, 1.0 - 1e-7)
            if target_metric == "roc_auc":
                return -float(roc_auc_score(y_arr, blended)) + 1e-3 * float(log_loss(y_arr, blended))
            if target_metric == "logloss":
                return float(log_loss(y_arr, blended))
            if target_metric == "brier":
                return float(brier_score_loss(y_arr, blended))
            raise ValueError(f"Unsupported optimization metric: {target_metric}")

        theta0 = np.zeros(n_models, dtype=np.float64)
        res = minimize(
            _objective,
            theta0,
            method="Nelder-Mead",
            options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-6, "adaptive": True},
        )
        self.weights = softmax(res.x)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probas = np.zeros((len(X), 2), dtype=np.float64)
        for weight, (_, model) in zip(self.weights, self.models, strict=True):
            if weight > 0:
                probas += weight * model.predict_proba(X)
        return probas

    def predict(self, X: pd.DataFrame | np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class StackingEnsemble:
    """Two-stage stacking classifier using Out-Of-Fold probabilities for meta-learning."""

    def __init__(self, base_models: list[tuple[str, any]], random_state: int = 42) -> None:
        self.base_models = base_models
        self.random_state = random_state
        self.meta_learner = LogisticRegression(C=1.0, random_state=random_state)
        self.fitted_base_models = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv_folds: int = 5,
        precomputed_oof: np.ndarray | None = None,
        precomputed_fitted_models: list[tuple[str, any]] | None = None,
    ) -> StackingEnsemble:
        if precomputed_oof is not None and precomputed_fitted_models is not None:
            oof_preds = precomputed_oof
            self.fitted_base_models = precomputed_fitted_models
        else:
            oof_preds, self.fitted_base_models = generate_oof_and_fit_base_models(
                self.base_models, X, y, cv_folds=cv_folds, random_state=self.random_state
            )

        self.meta_learner.fit(oof_preds, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        meta_features = np.column_stack([
            m.predict_proba(X)[:, 1] for _, m in self.fitted_base_models
        ])
        return self.meta_learner.predict_proba(meta_features)

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


def evaluate_predictions(
    name: str,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    y_tune: np.ndarray,
    y_tune_proba: np.ndarray,
    threshold: float = 0.5,
    seed: int = 42,
) -> dict:
    """Test metrics; thresholds are tuned on (y_tune, y_tune_proba), i.e. OOF training data."""
    y_pred = (y_proba >= threshold).astype(int)
    clinical = compute_clinical_metrics(y_true, y_proba, y_val_true=y_tune, y_val_prob=y_tune_proba)
    row = {
        "model": name,
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_f1": f1_score(y_true, y_pred),
        "test_roc_auc": roc_auc_score(y_true, y_proba),
        "test_pr_auc": average_precision_score(y_true, y_proba),
        "test_brier_score": brier_score_loss(y_true, y_proba),
        "optimal_threshold_f1": clinical["optimal_threshold_f1"],
        "f1_at_optimal_threshold": clinical["f1_at_optimal"],
        "optimal_threshold_youden": clinical["optimal_threshold_youden"],
        "youden_j": clinical["youden_j"],
    }
    add_bootstrap_cis(
        row, y_true, y_proba, seed=seed,
        name_map={"brier": "test_brier_score"},
    )
    return row


def run_ensemble_benchmark(
    dataset_path: str = "cleaned_heart_disease_combined.csv",
    use_feature_engineering: bool = True,
    cv_folds: int = 5,
    seed: int = 42,
    n_jobs: int = -1,
    output_dir: str = "artifacts",
) -> tuple[pd.DataFrame, dict]:
    print("=" * 70)
    print("HYBRID CLINICAL ENSEMBLE BENCHMARK (PyTorch MLP + XGBoost + RF + LogReg)")
    print("=" * 70)

    effective_jobs = resolve_n_jobs(n_jobs)
    path = resolve_data_path(dataset_path)

    df = pd.read_csv(path)
    if use_feature_engineering:
        print("Applying cardiology domain feature engineering (RPP, ischemia, chronotropic reserve)...")
        df = add_cardiovascular_features(df)

    drop_cols = [c for c in ["diagnosis", "disease_code", "target", "site"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["target"].astype(int)

    print(f"Cohort size: {len(df)} patients | Feature dimension: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    pre_scaled, _, _ = build_preprocessing_pipeline(X_train, scale_numeric=True)
    pre_unscaled, _, _ = build_preprocessing_pipeline(X_train, scale_numeric=False)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    base_specs = [
        (
            "Logistic Regression",
            Pipeline([("pre", pre_scaled), ("clf", LogisticRegression(max_iter=2000, random_state=seed))]),
        ),
        (
            "Random Forest",
            Pipeline([
                ("pre", pre_unscaled),
                ("clf", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=seed, n_jobs=1)),
            ]),
        ),
        (
            "XGBoost",
            Pipeline([
                ("pre", pre_unscaled),
                ("clf", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed, n_jobs=1, eval_metric="logloss")),
            ]),
        ),
        (
            "PyTorch MLP",
            Pipeline([
                ("pre", pre_scaled),
                ("clf", PyTorchMLPClassifier(epochs=50, hidden_dims=(64, 32), dropout=0.25, lr=0.001, device=device_str, random_state=seed)),
            ]),
        ),
    ]

    print(f"\nGenerating {cv_folds}-fold OOF predictions and fitting base models ({effective_jobs} parallel workers)...")
    # A GPU MLP must stay in the main process.
    oof_jobs = 1 if device_str == "cuda" else effective_jobs
    oof_preds, trained_base = generate_oof_and_fit_base_models(
        base_specs, X_train, y_train, cv_folds=cv_folds, random_state=seed, n_jobs=oof_jobs
    )
    y_train_np = y_train.values

    results = []
    test_probas: dict[str, np.ndarray] = {}
    for m_idx, (name, pipe) in enumerate(trained_base):
        proba = pipe.predict_proba(X_test)[:, 1]
        test_probas[f"Ensemble base: {name}"] = proba
        row = evaluate_predictions(name, y_test.values, proba, y_train_np, oof_preds[:, m_idx], seed=seed)
        results.append(row)
        print(
            f"  -> {name:<20} | Accuracy: {row['test_accuracy']:.4f} | F1: {row['test_f1']:.4f} | "
            f"ROC-AUC: {row['test_roc_auc']:.4f} | PR-AUC: {row['test_pr_auc']:.4f}"
        )

    # 1. Simplex-Optimized Soft-Voting Ensemble (weights learned strictly on OOF predictions)
    soft_voting = SoftVotingEnsemble(trained_base, metric="roc_auc")
    soft_voting.fit_weights(oof_preds, y_train.values, metric="roc_auc")
    weight_str = ", ".join(
        f"{name}: {w:.3f}" for (name, _), w in zip(trained_base, soft_voting.weights, strict=True)
    )
    print(f"\n[Nelder-Mead Simplex Weights] {weight_str}")

    sv_proba = soft_voting.predict_proba(X_test)[:, 1]
    test_probas["Hybrid Soft-Voting Ensemble"] = sv_proba
    sv_oof = np.clip(oof_preds @ soft_voting.weights, 0.0, 1.0)
    sv_row = evaluate_predictions("Hybrid Soft-Voting Ensemble", y_test.values, sv_proba, y_train_np, sv_oof, seed=seed)
    results.append(sv_row)
    print(
        f"[Hybrid Soft-Voting] Accuracy: {sv_row['test_accuracy']:.4f} | "
        f"F1: {sv_row['test_f1']:.4f} | ROC-AUC: {sv_row['test_roc_auc']:.4f} | "
        f"PR-AUC: {sv_row['test_pr_auc']:.4f}"
    )

    # 2. Stacking Ensemble (reuses precomputed OOF matrix and fitted base models)
    print("\nFitting Stacking Classifier (Two-Stage Meta-Learning)...")
    stacking = StackingEnsemble(base_specs, random_state=seed)
    stacking.fit(
        X_train,
        y_train,
        cv_folds=cv_folds,
        precomputed_oof=oof_preds,
        precomputed_fitted_models=trained_base,
    )
    st_proba = stacking.predict_proba(X_test)[:, 1]
    test_probas["Hybrid Stacking Ensemble (Ours)"] = st_proba
    # Out-of-fold meta-learner predictions give a leak-free set for threshold tuning.
    st_oof = cross_val_predict(
        LogisticRegression(C=1.0, random_state=seed), oof_preds, y_train_np,
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed), method="predict_proba",
    )[:, 1]
    st_row = evaluate_predictions("Hybrid Stacking Ensemble (Ours)", y_test.values, st_proba, y_train_np, st_oof, seed=seed)
    results.append(st_row)
    print(
        f"[Hybrid Stacking]    Accuracy: {st_row['test_accuracy']:.4f} | "
        f"F1: {st_row['test_f1']:.4f} | ROC-AUC: {st_row['test_roc_auc']:.4f} | "
        f"PR-AUC: {st_row['test_pr_auc']:.4f}"
    )

    res_df = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False).reset_index(drop=True)

    out_p = resolve_output_dir(output_dir)
    from evaluation.model_comparison import save_test_predictions

    save_test_predictions(out_p, "ensemble", dataset_path, X_test.index, y_test, test_probas)
    csv_file = out_p / "ensemble_comparison.csv"
    md_file = out_p / "ensemble_comparison.md"

    res_df.to_csv(csv_file, index=False)
    md_file.write_text(to_markdown_table(res_df), encoding="utf-8")
    print(f"\nSaved ensemble results to:\n  - {csv_file}\n  - {md_file}")

    return res_df, {"soft_voting": sv_row, "stacking": st_row}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run hybrid ensemble benchmark.")
    parser.add_argument("--dataset", type=str, default="cleaned_heart_disease_combined.csv")
    parser.add_argument("--no-fe", action="store_true", help="Disable clinical feature engineering.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel CPU workers (-1 = all cores).")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args(argv)

    run_ensemble_benchmark(
        dataset_path=args.dataset,
        use_feature_engineering=not args.no_fe,
        cv_folds=args.cv_folds,
        seed=args.seed,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
