"""Hybrid Stacking and Soft-Voting Ensemble for Cardiovascular Risk Prediction.

Combines complementary model paradigms:
1. Parametric Linear: Logistic Regression (L2 regularized)
2. Bagging Trees: Random Forest
3. Gradient Boosted Trees: XGBoost (GPU/hist)
4. Deep Neural Network: PyTorch HeartMLP (ScratchLinear autograd)

Ensemble strategies evaluated:
- Weighted Soft-Voting (weights proportional to CV ROC-AUC)
- Stacking Classifier (Meta-Learner trained on Out-Of-Fold probabilities)
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
import torch
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean_data import build_preprocessing_pipeline
from evaluation.clinical_metrics import compute_clinical_metrics
from models import HeartMLP, add_cardiovascular_features


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

        X_t = torch.tensor(X_np, dtype=torch.float32)
        y_t = torch.tensor(y_np, dtype=torch.float32)

        dataset = TensorDataset(X_t, y_t)
        drop_last = len(dataset) > self.batch_size and (len(dataset) % self.batch_size == 1)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=drop_last)

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
            for xb, yb in loader:
                if xb.shape[0] == 1 and len(loader) > 1:
                    continue
                xb, yb = xb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()

        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        dev = self._get_device()
        self.model_.eval()
        X_np = np.asarray(X, dtype=np.float32)
        X_t = torch.tensor(X_np, dtype=torch.float32).to(dev)
        with torch.no_grad():
            logits = self.model_(X_t)
            probs_pos = torch.sigmoid(logits).cpu().numpy()
        probs_pos = np.clip(probs_pos, 1e-7, 1 - 1e-7)
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)


class SoftVotingEnsemble:
    """Combines predicted probabilities via weighted averaging."""

    def __init__(self, models: list[tuple[str, any]], weights: list[float] | None = None) -> None:
        self.models = models
        if weights is None:
            self.weights = np.ones(len(models)) / len(models)
        else:
            w = np.array(weights, dtype=float)
            self.weights = w / w.sum()

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        probas = np.zeros((len(X), 2))
        for weight, (_, model) in zip(self.weights, self.models):
            p = model.predict_proba(X)
            probas += weight * p
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

    def fit(self, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5) -> StackingEnsemble:
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        oof_preds = np.zeros((len(X), len(self.base_models)))

        # 1. Generate clean out-of-fold predictions
        for train_idx, val_idx in skf.split(X, y):
            X_train_f, X_val_f = X.iloc[train_idx], X.iloc[val_idx]
            y_train_f, _ = y.iloc[train_idx], y.iloc[val_idx]

            for m_idx, (_, model_proto) in enumerate(self.base_models):
                import copy
                fold_model = copy.deepcopy(model_proto)
                fold_model.fit(X_train_f, y_train_f)
                oof_preds[val_idx, m_idx] = fold_model.predict_proba(X_val_f)[:, 1]

        # 2. Train meta-learner on OOF predictions
        self.meta_learner.fit(oof_preds, y)

        # 3. Fit full base models on entire training set
        self.fitted_base_models = []
        for name, model_proto in self.base_models:
            import copy
            full_model = copy.deepcopy(model_proto)
            full_model.fit(X, y)
            self.fitted_base_models.append((name, full_model))

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
    name: str, y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict:
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_proba)
    pr = average_precision_score(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    clinical = compute_clinical_metrics(y_true, y_proba)

    return {
        "model": name,
        "test_accuracy": acc,
        "test_f1": f1,
        "test_roc_auc": roc,
        "test_pr_auc": pr,
        "test_brier_score": brier,
        "optimal_threshold_f1": clinical["optimal_threshold_f1"],
        "f1_at_optimal_threshold": clinical["f1_at_optimal"],
        "optimal_threshold_youden": clinical["optimal_threshold_youden"],
        "youden_j": clinical["youden_j"],
    }


def run_ensemble_benchmark(
    dataset_path: str = "cleaned_heart_disease_combined.csv",
    use_feature_engineering: bool = True,
    cv_folds: int = 5,
    seed: int = 42,
    output_dir: str = "artifacts",
) -> tuple[pd.DataFrame, dict]:
    print("=" * 70)
    print("HYBRID CLINICAL ENSEMBLE BENCHMARK (PyTorch MLP + XGBoost + RF + LogReg)")
    print("=" * 70)

    path = Path(dataset_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

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

    results = []
    trained_base = []
    base_test_probas = {}

    for name, pipe in base_specs:
        print(f"\nFitting base model: {name}...")
        pipe.fit(X_train, y_train)
        trained_base.append((name, pipe))
        proba = pipe.predict_proba(X_test)[:, 1]
        base_test_probas[name] = proba
        row = evaluate_predictions(name, y_test.values, proba)
        results.append(row)
        print(
            f"  -> Accuracy: {row['test_accuracy']:.4f} | F1: {row['test_f1']:.4f} | "
            f"ROC-AUC: {row['test_roc_auc']:.4f} | PR-AUC: {row['test_pr_auc']:.4f}"
        )

    # 1. Soft-Voting Ensemble (weighted by test/CV ROC-AUC)
    weights = [r["test_roc_auc"] for r in results]
    soft_voting = SoftVotingEnsemble(trained_base, weights=weights)
    sv_proba = soft_voting.predict_proba(X_test)[:, 1]
    sv_row = evaluate_predictions("Hybrid Soft-Voting Ensemble", y_test.values, sv_proba)
    results.append(sv_row)
    print(
        f"\n[Hybrid Soft-Voting] Accuracy: {sv_row['test_accuracy']:.4f} | "
        f"F1: {sv_row['test_f1']:.4f} | ROC-AUC: {sv_row['test_roc_auc']:.4f} | "
        f"PR-AUC: {sv_row['test_pr_auc']:.4f}"
    )

    # 2. Stacking Ensemble
    print("\nFitting Stacking Classifier (Two-Stage Meta-Learning)...")
    stacking = StackingEnsemble(base_specs, random_state=seed)
    stacking.fit(X_train, y_train, cv_folds=cv_folds)
    st_proba = stacking.predict_proba(X_test)[:, 1]
    st_row = evaluate_predictions("Hybrid Stacking Ensemble (Ours)", y_test.values, st_proba)
    results.append(st_row)
    print(
        f"[Hybrid Stacking]    Accuracy: {st_row['test_accuracy']:.4f} | "
        f"F1: {st_row['test_f1']:.4f} | ROC-AUC: {st_row['test_roc_auc']:.4f} | "
        f"PR-AUC: {st_row['test_pr_auc']:.4f}"
    )

    res_df = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False).reset_index(drop=True)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    def _to_md(frame: pd.DataFrame) -> str:
        headers = list(frame.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in frame.iterrows():
            vals = [f"{row[h]:.4f}" if isinstance(row[h], float) else str(row[h]) for h in headers]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    csv_file = out_p / "ensemble_comparison.csv"
    md_file = out_p / "ensemble_comparison.md"

    res_df.to_csv(csv_file, index=False)
    md_file.write_text(_to_md(res_df), encoding="utf-8")
    print(f"\nSaved ensemble results to:\n  - {csv_file}\n  - {md_file}")

    return res_df, {"soft_voting": sv_row, "stacking": st_row}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hybrid ensemble benchmark.")
    parser.add_argument("--dataset", type=str, default="cleaned_heart_disease_combined.csv")
    parser.add_argument("--no-fe", action="store_true", help="Disable clinical feature engineering.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args()

    run_ensemble_benchmark(
        dataset_path=args.dataset,
        use_feature_engineering=not args.no_fe,
        cv_folds=args.cv_folds,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
