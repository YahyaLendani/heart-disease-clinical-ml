"""PyTorch MLP training script for MIMIC-IV hospital mortality prediction.

Features:
- Leakage-free preprocessing (ColumnTransformer + SimpleImputer + StandardScaler + OneHotEncoder)
- Patient-level splitting (GroupShuffleSplit on subject_id for train/val/test)
- Patient-level cross-validation (StratifiedGroupKFold on subject_id); inside each
  fold a separate patient-grouped split is used for early stopping, so the fold
  that is scored never chooses the stopping epoch
- Bootstrap 95% CIs on holdout metrics (the demo test split holds only a few deaths)
- Imbalance-aware loss functions (Focal Loss and Weighted BCE)
- Comprehensive clinical metrics (PR-AUC, ROC-AUC, Brier score, Youden's J, F1 at optimal threshold)
- Calibration and PR curves generation
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean_data import build_preprocessing_pipeline
from evaluation.clinical_metrics import (
    compute_clinical_metrics,
    plot_calibration_curves,
    plot_precision_recall_curves,
)
from evaluation.cv_utils import add_bootstrap_cis
from models import HeartMLP
from project_paths import resolve_data_path, resolve_output_dir
from training.device_utils import DEVICE_CHOICES, resolve_torch_device
from training.torch_utils import (
    clone_state,
    iterate_minibatches,
    predict_logits,
    set_seed,
    to_tensor,
)


class BinaryFocalLossWithLogits(nn.Module):
    """Binary Focal Loss with Logits for severe class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * (1.0 - p_t).clamp(min=1e-8).pow(self.gamma)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PyTorch MLP on MIMIC hospital mortality.")
    parser.add_argument("--input", type=str, default="mimic_hospital_mortality.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden-dims", nargs=2, type=int, default=[64, 32])
    parser.add_argument("--loss", choices=["bce_weighted", "focal", "bce"], default="bce_weighted")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if "hospital_expire_flag" not in df.columns:
        raise ValueError("Target 'hospital_expire_flag' missing from dataset.")

    y = pd.to_numeric(df["hospital_expire_flag"], errors="coerce").fillna(0).astype(int)
    subject_ids = df["subject_id"] if "subject_id" in df.columns else pd.Series(df.index, index=df.index)

    drop_cols = ["hospital_expire_flag", "hadm_id", "subject_id"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    return X, y, subject_ids


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    subject_ids: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    n_splits_test = max(2, int(round(1.0 / test_size)))
    sgkf_test = StratifiedGroupKFold(n_splits=n_splits_test, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf_test.split(X, y, groups=subject_ids))

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = subject_ids.iloc[train_val_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    relative_val = val_size / (1.0 - test_size)
    n_splits_val = max(2, int(round(1.0 / relative_val)))
    sgkf_val = StratifiedGroupKFold(n_splits=n_splits_val, shuffle=True, random_state=seed)
    train_idx, val_idx = next(sgkf_val.split(X_train_val, y_train_val, groups=groups_train_val))

    X_train = X_train_val.iloc[train_idx]
    y_train = y_train_val.iloc[train_idx]
    groups_train = groups_train_val.iloc[train_idx]

    X_val = X_train_val.iloc[val_idx]
    y_val = y_train_val.iloc[val_idx]

    # train_idx/val_idx index the train+val subset, test_idx indexes the full
    # data, so compare subject IDs rather than mixing the two position spaces.
    test_subjects = set(subject_ids.iloc[test_idx])
    val_subjects = set(groups_train_val.iloc[val_idx])
    assert set(groups_train.values).isdisjoint(test_subjects), (
        "CRITICAL LEAKAGE: subject_id overlap between train and test!"
    )
    assert val_subjects.isdisjoint(test_subjects), (
        "CRITICAL LEAKAGE: subject_id overlap between val and test!"
    )
    assert set(groups_train.values).isdisjoint(val_subjects), (
        "CRITICAL LEAKAGE: subject_id overlap between train and val!"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train


def get_loss_criterion(loss_type: str, y_train: pd.Series, device: torch.device, alpha: float = 0.25, gamma: float = 2.0) -> nn.Module:
    positives = max(int(y_train.sum()), 1)
    negatives = int(len(y_train) - positives)
    pos_weight_val = negatives / positives

    if loss_type == "bce_weighted":
        pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32, device=device)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_type == "focal":
        return BinaryFocalLossWithLogits(alpha=alpha, gamma=gamma)
    else:
        return nn.BCEWithLogitsLoss()


def train_epoch(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    generator: torch.Generator,
) -> float:
    model.train()
    total_loss = torch.zeros((), device=X.device)
    total_samples = 0
    for X_batch, y_batch in iterate_minibatches(X, y, batch_size, shuffle=True, generator=generator):
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.detach() * len(y_batch)
        total_samples += len(y_batch)
    return float(total_loss.item()) / max(total_samples, 1)


def evaluate_model(
    model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion: nn.Module
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """Full-batch evaluation: (loss, ROC-AUC, PR-AUC, probabilities, labels)."""
    logits = predict_logits(model, X)
    val_loss = float(criterion(logits, y).item())
    y_prob = torch.sigmoid(logits).cpu().numpy()
    y_true = y.cpu().numpy()
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5
    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = float(np.mean(y_true))
    return val_loss, roc_auc, pr_auc, y_prob, y_true


def _fit_fold(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_stop: pd.DataFrame,
    y_stop: pd.Series,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
):
    """Train with early stopping on (X_stop, y_stop); return the model and fitted preprocessor."""
    generator = set_seed(seed)
    preprocessor, _, _ = build_preprocessing_pipeline(X_fit, scale_numeric=True)
    X_fit_t = to_tensor(preprocessor.fit_transform(X_fit), device)
    X_stop_t = to_tensor(preprocessor.transform(X_stop), device)
    y_fit_t = to_tensor(y_fit.values, device)
    y_stop_t = to_tensor(y_stop.values, device)

    model = HeartMLP(input_dim=X_fit_t.shape[1], hidden_dims=tuple(args.hidden_dims), dropout=args.dropout).to(device)
    criterion = get_loss_criterion(args.loss, y_fit, device, args.focal_alpha, args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_pr_auc = -1.0
    best_state = None
    patience_counter = 0
    for _ in range(args.epochs):
        train_epoch(model, X_fit_t, y_fit_t, optimizer, criterion, args.batch_size, generator)
        _, _, pr_auc, _, _ = evaluate_model(model, X_stop_t, y_stop_t, criterion)
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_state = clone_state(model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, preprocessor, criterion


def cross_validate_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    cv_metrics: dict[str, list[float]] = {
        "accuracy": [], "f1": [], "recall": [], "precision": [], "roc_auc": [], "pr_auc": [], "brier": [],
    }
    repeats = max(1, getattr(args, "cv_repeats", 1))
    print(f"\n--- Running {args.cv_folds}-Fold x {repeats} Stratified Group CV for MIMIC MLP ---")
    for repeat in range(repeats):
        sgkf = StratifiedGroupKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed + repeat)
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_train, y_train, groups=groups_train), 1):
            X_tr, y_tr, g_tr = X_train.iloc[train_idx], y_train.iloc[train_idx], groups_train.iloc[train_idx]
            X_v, y_v = X_train.iloc[val_idx], y_train.iloc[val_idx]

            # Early stopping gets its own patient-grouped split of the training fold.
            inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=args.seed + 100 * repeat + fold)
            fit_idx, stop_idx = next(inner.split(X_tr, y_tr, groups=g_tr))
            fold_seed = args.seed + 1000 * repeat + fold
            model, preprocessor, criterion = _fit_fold(
                X_tr.iloc[fit_idx], y_tr.iloc[fit_idx], X_tr.iloc[stop_idx], y_tr.iloc[stop_idx],
                args, device, fold_seed,
            )

            X_v_t = to_tensor(preprocessor.transform(X_v), device)
            _, roc_auc, pr_auc, y_prob, y_true = evaluate_model(model, X_v_t, to_tensor(y_v.values, device), criterion)
            y_pred = (y_prob >= 0.5).astype(int)

            cv_metrics["accuracy"].append(accuracy_score(y_true, y_pred))
            cv_metrics["f1"].append(f1_score(y_true, y_pred, zero_division=0))
            cv_metrics["recall"].append(recall_score(y_true, y_pred, zero_division=0))
            cv_metrics["precision"].append(precision_score(y_true, y_pred, zero_division=0))
            cv_metrics["roc_auc"].append(roc_auc)
            cv_metrics["pr_auc"].append(pr_auc)
            cv_metrics["brier"].append(brier_score_loss(y_true, y_prob))

    names = {"f1": "f1_expired", "recall": "recall_expired", "precision": "precision_expired"}
    summary: dict[str, float] = {}
    for metric, values in cv_metrics.items():
        key = names.get(metric, metric)
        summary[f"cv_{key}_mean"] = float(np.mean(values))
        summary[f"cv_{key}_std"] = float(np.std(values))
    summary["cv_n_folds_total"] = len(cv_metrics["accuracy"])
    return summary


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    generator = set_seed(args.seed)
    device = resolve_torch_device(args.device)
    print(f"Using device: {device}")

    input_path = resolve_data_path(args.input)
    X, y, subject_ids = load_data(input_path)

    X_train, X_val, X_test, y_train, y_val, y_test, groups_train = split_data(
        X, y, subject_ids, test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )
    print(f"Data split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

    # Preprocessing pipeline fit on train
    preprocessor, _, _ = build_preprocessing_pipeline(X_train, scale_numeric=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)

    input_dim = X_train_proc.shape[1]
    print(f"Input feature dimension after preprocessing: {input_dim}")

    X_train_t, y_train_t = to_tensor(X_train_proc, device), to_tensor(y_train.values, device)
    X_val_t, y_val_t = to_tensor(X_val_proc, device), to_tensor(y_val.values, device)
    X_test_t, y_test_t = to_tensor(X_test_proc, device), to_tensor(y_test.values, device)

    # Cross-validation
    cv_summary = cross_validate_mlp(X_train, y_train, groups_train, args, device)
    print("\n=== MIMIC MLP Cross-Validation Results ===")
    for k, v in cv_summary.items():
        print(f"  {k}: {v:.4f}")

    # Holdout Training
    print("\n--- Training Holdout MIMIC MLP ---")
    generator = set_seed(args.seed)
    model = HeartMLP(input_dim=input_dim, hidden_dims=tuple(args.hidden_dims), dropout=args.dropout).to(device)
    criterion = get_loss_criterion(args.loss, y_train, device, args.focal_alpha, args.focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, X_train_t, y_train_t, optimizer, criterion, args.batch_size, generator)
        v_loss, v_roc, v_pr, _, _ = evaluate_model(model, X_val_t, y_val_t, criterion)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": v_loss,
            "val_roc_auc": v_roc,
            "val_pr_auc": v_pr,
        })

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            best_state = clone_state(model)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch} (best val loss: {best_val_loss:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Evaluate on validation split for out-of-sample threshold tuning
    _, _, _, y_val_prob, y_val_true = evaluate_model(model, X_val_t, y_val_t, criterion)

    # Test set evaluation
    t_loss, t_roc, t_pr, y_test_prob, y_test_true = evaluate_model(model, X_test_t, y_test_t, criterion)
    y_test_pred = (y_test_prob >= 0.5).astype(int)

    clinical = compute_clinical_metrics(
        pd.Series(y_test_true), y_test_prob, y_val_true=y_val_true, y_val_prob=y_val_prob
    )

    holdout_results = {
        "model": "PyTorch MLP (Ours)",
        "test_events": int(np.sum(y_test_true)),
        "test_accuracy": float(accuracy_score(y_test_true, y_test_pred)),
        "test_f1_expired": float(f1_score(y_test_true, y_test_pred, zero_division=0)),
        "test_recall_expired": float(recall_score(y_test_true, y_test_pred, zero_division=0)),
        "test_precision_expired": float(precision_score(y_test_true, y_test_pred, zero_division=0)),
        "test_roc_auc": float(t_roc),
        "test_pr_auc": float(t_pr),
        "test_brier_score": float(clinical["brier_score"]),
        "optimal_threshold_f1": float(clinical["optimal_threshold_f1"]),
        "f1_at_optimal_threshold": float(clinical["f1_at_optimal"]),
        "optimal_threshold_youden": float(clinical["optimal_threshold_youden"]),
        "youden_j": float(clinical["youden_j"]),
    }
    add_bootstrap_cis(
        holdout_results, y_test_true, y_test_prob, seed=args.seed,
        name_map={
            "accuracy": "test_accuracy",
            "f1": "test_f1_expired",
            "roc_auc": "test_roc_auc",
            "pr_auc": "test_pr_auc",
            "brier": "test_brier_score",
        },
    )
    holdout_results.update(cv_summary)

    print("\n=== Holdout Test Performance (PyTorch MLP) ===")
    for k, v in holdout_results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # Output artifacts
    output_dir = resolve_output_dir(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "mimic_mlp.pt")
    pd.DataFrame(history).to_csv(output_dir / "training_history_mimic_mlp.csv", index=False)

    # Plot curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    hist_df = pd.DataFrame(history)
    ax1.plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss")
    ax1.plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("MIMIC MLP Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(hist_df["epoch"], hist_df["val_roc_auc"], label="Val ROC-AUC")
    ax2.plot(hist_df["epoch"], hist_df["val_pr_auc"], label="Val PR-AUC")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Metric")
    ax2.set_title("Validation Discrimination")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    curves_path = figures_dir / "mimic_mlp_curves.png"
    fig.savefig(curves_path, dpi=150)
    plt.close(fig)
    print(f"Saved MLP curves: {curves_path}")

    # Calibration & PR Curves
    plot_calibration_curves(pd.Series(y_test_true), {"PyTorch MLP": y_test_prob}, "mimic_mlp", figures_dir)
    plot_precision_recall_curves(pd.Series(y_test_true), {"PyTorch MLP": y_test_prob}, "mimic_mlp", figures_dir)

    # Append to mimic_baseline_comparison.csv if it exists
    csv_path = output_dir / "mimic_baseline_comparison.csv"
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        existing_df = existing_df[existing_df["model"] != "PyTorch MLP (Ours)"]
        combined_df = pd.concat([existing_df, pd.DataFrame([holdout_results])], ignore_index=True)
        combined_df = combined_df.sort_values(
            by=["cv_pr_auc_mean", "cv_roc_auc_mean", "test_pr_auc"],
            ascending=False,
        )
        combined_df.to_csv(csv_path, index=False)
        print(f"Updated {csv_path} with PyTorch MLP results.")


if __name__ == "__main__":
    main()
