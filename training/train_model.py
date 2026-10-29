"""PyTorch HeartMLP training for the heart-disease cohort (SGD vs Adam).

Evaluation protocol
-------------------
* 80/20 stratified train+val / test split, then an inner train/val split.
* Repeated stratified k-fold CV on the train+val part. Inside every fold a
  separate early-stopping split is carved out of the training portion, so
  the fold used for scoring is never used to pick the stopping epoch.
* When both optimizers run, the reported "best optimizer" is chosen by
  validation loss, never by test accuracy.
* Thresholds are tuned on the validation split and applied to the test split;
  test metrics carry stratified bootstrap 95% confidence intervals.

Speed: the data stays in tensors (no ``DataLoader``), each epoch runs one
validation forward pass, and CV folds run in parallel processes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
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
from evaluation.cv_utils import add_bootstrap_cis, binary_metrics, resolve_n_jobs
from models import HeartMLP
from project_paths import PRIMARY_HEART_DATASET, resolve_data_path, resolve_output_dir
from training.device_utils import DEVICE_CHOICES, resolve_torch_device
from training.torch_utils import (
    clone_state,
    iterate_minibatches,
    predict_logits,
    set_seed,
    single_thread,
    to_tensor,
)

__all__ = ["set_seed", "train_one_optimizer", "main"]

TEST_METRIC_COLUMNS = {
    "accuracy": "test_accuracy",
    "f1": "test_f1",
    "roc_auc": "test_roc_auc",
    "pr_auc": "test_pr_auc",
    "brier": "test_brier_score",
}


def smooth_binary_targets(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return targets
    return targets * (1 - smoothing) + 0.5 * smoothing


def make_loss(loss_name: str, y_train_tensor: torch.Tensor) -> nn.Module:
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()

    if loss_name == "bce_weighted":
        positives = y_train_tensor.sum().item()
        negatives = len(y_train_tensor) - positives
        pos_weight = torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    raise ValueError(f"Unsupported loss: {loss_name}")


def make_optimizer(name: str, model: nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def save_training_curves(history_rows: list[dict], output_path: Path, optimizer_name: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping training-curve visualization.")
        return

    history = pd.DataFrame(history_rows)
    if history.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{optimizer_name.upper()} Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title(f"{optimizer_name.upper()} Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves: {output_path}")


def fit_with_early_stopping(
    optimizer_name: str,
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    criterion: nn.Module,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    batch_size: int,
    generator: torch.Generator,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.0,
    min_delta: float = 0.0,
    verbose: bool = True,
) -> dict:
    """Train with early stopping on (X_val, y_val); restore the best weights."""
    optimizer = make_optimizer(optimizer_name, model, lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6
    )

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    best_state = clone_state(model)
    wait = 0
    history_rows: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=X_train.device)
        n_batches = 0
        for xb, yb in iterate_minibatches(
            X_train, y_train, batch_size, shuffle=True, drop_last=True, generator=generator
        ):
            optimizer.zero_grad()
            loss = criterion(model(xb), smooth_binary_targets(yb, label_smoothing))
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
            loss_sum += loss.detach()
            n_batches += 1

        train_loss = float(loss_sum.item()) / max(n_batches, 1)
        # One forward pass gives both validation loss and accuracy.
        val_logits = predict_logits(model, X_val)
        val_loss = float(criterion(val_logits, y_val).item())
        val_accuracy = float(((val_logits >= 0).float() == y_val).float().mean().item())
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": current_lr,
            }
        )
        if verbose:
            print(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | LR: {current_lr:.6f}"
            )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = clone_state(model)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return {
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "best_state": best_state,
        "history": history_rows,
    }


def predict_proba(model: nn.Module, X: torch.Tensor) -> np.ndarray:
    return torch.sigmoid(predict_logits(model, X)).cpu().numpy().ravel()


def train_one_optimizer(
    optimizer_name: str,
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    X_test: torch.Tensor,
    y_test: pd.Series,
    criterion: nn.Module,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    batch_size: int,
    generator: torch.Generator,
    checkpoint_path: Path | None = None,
    confusion_matrix_path: Path | None = None,
    history_path: Path | None = None,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.0,
    min_delta: float = 0.0,
    curve_path: Path | None = None,
    seed: int = 42,
) -> dict:
    """Train on the holdout split, then report test metrics with val-tuned thresholds."""
    print(f"\n=== Training with {optimizer_name.upper()} ===")
    fit = fit_with_early_stopping(
        optimizer_name, model, X_train, y_train, X_val, y_val, criterion,
        lr, weight_decay, epochs, patience, batch_size, generator,
        grad_clip=grad_clip, label_smoothing=label_smoothing, min_delta=min_delta,
    )
    print(
        f"Best validation checkpoint for {optimizer_name.upper()}: epoch={fit['best_epoch']}, "
        f"val_loss={fit['best_val_loss']:.4f}, val_accuracy={fit['best_val_accuracy']:.4f}"
    )

    y_prob = predict_proba(model, X_test)
    y_val_prob = predict_proba(model, X_val)
    y_val_np = y_val.cpu().numpy().astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    base = binary_metrics(y_test, y_prob)
    clinical = compute_clinical_metrics(y_test, y_prob, y_val_true=y_val_np, y_val_prob=y_val_prob)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    result = {
        "optimizer": optimizer_name,
        "best_val_loss": fit["best_val_loss"],
        "best_val_accuracy": fit["best_val_accuracy"],
        **{column: base[metric] for metric, column in TEST_METRIC_COLUMNS.items()},
        "optimal_threshold_f1": clinical["optimal_threshold_f1"],
        "f1_at_optimal_threshold": clinical["f1_at_optimal"],
        "optimal_threshold_youden": clinical["optimal_threshold_youden"],
        "youden_j": clinical["youden_j"],
        "best_epoch": fit["best_epoch"],
    }
    add_bootstrap_cis(result, y_test, y_prob, seed=seed, name_map=TEST_METRIC_COLUMNS)

    name = optimizer_name.upper()
    print(f"\n{name} Test Accuracy: {result['test_accuracy']:.4f}")
    print(f"{name} Test F1: {result['test_f1']:.4f}")
    print(
        f"{name} Test ROC-AUC: {result['test_roc_auc']:.4f} "
        f"(95% CI {result['test_roc_auc_ci_low']:.4f}-{result['test_roc_auc_ci_high']:.4f})"
    )
    print(f"{name} Test PR-AUC: {result['test_pr_auc']:.4f}")
    print(f"{name} Test Brier Score: {result['test_brier_score']:.4f}")
    print(
        f"{name} Validation-tuned threshold (F1): {result['optimal_threshold_f1']:.3f} "
        f"(test F1={result['f1_at_optimal_threshold']:.4f})"
    )
    print(classification_report(y_test, y_pred, target_names=["healthy", "sick"]))
    cm_df = pd.DataFrame(cm, index=["actual_healthy", "actual_sick"], columns=["pred_healthy", "pred_sick"])
    print("Confusion Matrix:")
    print(cm_df.to_string())

    if checkpoint_path is not None:
        torch.save(
            {"optimizer": optimizer_name, "best_val_loss": fit["best_val_loss"], "model_state_dict": fit["best_state"]},
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")
    if confusion_matrix_path is not None:
        cm_df.to_csv(confusion_matrix_path)
        print(f"Saved confusion matrix: {confusion_matrix_path}")
    if history_path is not None:
        pd.DataFrame(fit["history"]).to_csv(history_path, index=False)
        print(f"Saved training history: {history_path}")
    if curve_path is not None:
        save_training_curves(fit["history"], curve_path, optimizer_name)

    result["y_prob"] = y_prob
    result["model_state_dict"] = fit["best_state"]
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train heart-disease classifier with PyTorch.")
    parser.add_argument("--optimizer", choices=["sgd", "adam", "both"], default="both")
    parser.add_argument("--loss", choices=["bce", "bce_weighted"], default="bce")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden-dims", nargs=2, type=int, default=[64, 32])
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--save-confusion", action="store_true")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--save-run-metrics", action="store_true")
    parser.add_argument("--comparison-file", type=str, default="artifacts/separate_model_runs.csv")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3, help="Repeats of stratified k-fold CV.")
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="Parallel CV fold processes (default 1: serial is faster for this small MLP; results do not depend on it).",
    )
    parser.add_argument("--dataset", type=str, default=PRIMARY_HEART_DATASET, help="Cleaned dataset CSV (default: 918-patient 4-center cohort).")
    parser.add_argument("--run-xai", action="store_true", help="Run SHAP explainability analysis after training.")
    return parser


def append_run_metrics(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(output_path, index=False)


def _run_cv_fold(
    fold_id: int,
    train_idx: np.ndarray,
    score_idx: np.ndarray,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    optimizer_name: str,
    hparams: dict,
    seed: int,
    device_name: str,
) -> dict:
    """Train one CV fold. Module-level so it can run in a worker process."""
    with single_thread():
        return _train_cv_fold(fold_id, train_idx, score_idx, X_full, y_full, optimizer_name, hparams, seed, device_name)


def _train_cv_fold(
    fold_id: int,
    train_idx: np.ndarray,
    score_idx: np.ndarray,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    optimizer_name: str,
    hparams: dict,
    seed: int,
    device_name: str,
) -> dict:
    fold_seed = seed + 1000 * fold_id
    generator = set_seed(fold_seed)
    device = torch.device(device_name)

    X_fit_all, y_fit_all = X_full.iloc[train_idx], y_full.iloc[train_idx]
    X_score, y_score = X_full.iloc[score_idx], y_full.iloc[score_idx]
    # Early stopping uses its own split of the training portion.
    X_fit, X_stop, y_fit, y_stop = train_test_split(
        X_fit_all, y_fit_all, test_size=hparams["val_size"], random_state=fold_seed, stratify=y_fit_all
    )

    preprocessor, _, _ = build_preprocessing_pipeline(X_fit, scale_numeric=True)
    X_fit_t = to_tensor(preprocessor.fit_transform(X_fit), device)
    X_stop_t = to_tensor(preprocessor.transform(X_stop), device)
    X_score_t = to_tensor(preprocessor.transform(X_score), device)
    y_fit_t = to_tensor(y_fit.values, device)
    y_stop_t = to_tensor(y_stop.values, device)

    model = HeartMLP(input_dim=X_fit_t.shape[1], hidden_dims=hparams["hidden_dims"], dropout=hparams["dropout"]).to(device)
    criterion = make_loss(hparams["loss"], y_fit_t.cpu()).to(device)
    fit = fit_with_early_stopping(
        optimizer_name, model, X_fit_t, y_fit_t, X_stop_t, y_stop_t, criterion,
        hparams["lr"], hparams["weight_decay"], hparams["epochs"], hparams["patience"],
        hparams["batch_size"], generator,
        grad_clip=hparams["grad_clip"], label_smoothing=hparams["label_smoothing"],
        min_delta=hparams["min_delta"], verbose=False,
    )
    metrics = binary_metrics(y_score, predict_proba(model, X_score_t))
    return {
        "optimizer": optimizer_name,
        "fold": fold_id,
        "best_epoch": fit["best_epoch"],
        "best_val_loss": fit["best_val_loss"],
        "best_val_accuracy": fit["best_val_accuracy"],
        "cv_accuracy": metrics["accuracy"],
        "cv_f1": metrics["f1"],
        "cv_roc_auc": metrics["roc_auc"],
        "cv_pr_auc": metrics["pr_auc"],
        "cv_brier_score": metrics["brier"],
    }


def run_mlp_cross_validation(
    optimizer_name: str,
    args: argparse.Namespace,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    device: torch.device,
    hidden_dims: tuple[int, int],
) -> tuple[list[dict], dict]:
    repeats = max(1, args.cv_repeats)
    print(f"\n=== {args.cv_folds}-Fold x {repeats} Cross-Validation ({optimizer_name.upper()}) ===")
    splitter = RepeatedStratifiedKFold(n_splits=args.cv_folds, n_repeats=repeats, random_state=args.seed)
    splits = list(splitter.split(X_full, y_full))
    hparams = {
        "hidden_dims": hidden_dims,
        "dropout": args.dropout,
        "loss": args.loss,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "grad_clip": args.grad_clip,
        "label_smoothing": args.label_smoothing,
        "min_delta": args.min_delta,
        "val_size": args.val_size,
    }

    # Folds run serially by default: each fold trains in about a second, so
    # starting worker processes (each importing PyTorch) costs more than it saves.
    # GPU runs always stay in one process.
    workers = 1 if device.type == "cuda" else min(resolve_n_jobs(args.n_jobs), len(splits))
    job_args = [
        (fold_id, tr, sc, X_full, y_full, optimizer_name, hparams, args.seed, device.type)
        for fold_id, (tr, sc) in enumerate(splits, start=1)
    ]
    if workers > 1:
        from joblib import Parallel, delayed

        fold_rows = Parallel(n_jobs=workers)(delayed(_run_cv_fold)(*job) for job in job_args)
    else:
        fold_rows = [_run_cv_fold(*job) for job in job_args]

    for row in fold_rows:
        print(
            f"Fold {row['fold']}/{len(splits)} | Accuracy: {row['cv_accuracy']:.4f} | "
            f"F1: {row['cv_f1']:.4f} | ROC-AUC: {row['cv_roc_auc']:.4f}"
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = {"optimizer": optimizer_name, "cv_folds": args.cv_folds, "cv_repeats": repeats}
    for metric, column in (
        ("accuracy", "cv_accuracy"),
        ("f1", "cv_f1"),
        ("roc_auc", "cv_roc_auc"),
        ("pr_auc", "cv_pr_auc"),
        ("brier", "cv_brier_score"),
    ):
        summary[f"cv_{metric}_mean"] = float(np.nanmean(fold_df[column]))
        summary[f"cv_{metric}_std"] = float(np.nanstd(fold_df[column]))

    print(
        f"CV Summary ({optimizer_name.upper()}): "
        f"Accuracy {summary['cv_accuracy_mean']:.4f} ± {summary['cv_accuracy_std']:.4f}, "
        f"F1 {summary['cv_f1_mean']:.4f} ± {summary['cv_f1_std']:.4f}, "
        f"ROC-AUC {summary['cv_roc_auc_mean']:.4f} ± {summary['cv_roc_auc_std']:.4f}, "
        f"PR-AUC {summary['cv_pr_auc_mean']:.4f} ± {summary['cv_pr_auc_std']:.4f}, "
        f"Brier {summary['cv_brier_mean']:.4f} ± {summary['cv_brier_std']:.4f}"
    )
    return fold_rows, summary


def load_heart_dataset(dataset: str) -> tuple[pd.DataFrame, pd.Series, Path]:
    cleaned_path = resolve_data_path(dataset)
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"{cleaned_path.name} not found. Run clean_data.py or prepare_heart_combined.py first."
        )
    df = pd.read_csv(cleaned_path)
    if "target" in df.columns:
        df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype(int)
    elif "diagnosis" in df.columns:
        df["target"] = (df["diagnosis"].astype(str).str.lower() == "sick").astype(int)
    else:
        raise ValueError("Dataset must contain either 'target' or 'diagnosis' column.")
    drop_cols = [col for col in ["diagnosis", "disease_code", "target", "site"] if col in df.columns]
    return df.drop(columns=drop_cols), df["target"], cleaned_path


def main(argv: list[str] | None = None) -> list[dict]:
    args = build_arg_parser().parse_args(argv)

    if not 0 < args.val_size < 0.5:
        raise ValueError("--val-size must be between 0 and 0.5.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")
    if any(dim <= 0 for dim in args.hidden_dims):
        raise ValueError("--hidden-dims values must be positive integers.")
    hidden_dims = (args.hidden_dims[0], args.hidden_dims[1])

    X, y, cleaned_path = load_heart_dataset(args.dataset)
    print(f"Loaded cleaned dataset: {cleaned_path.name} ({len(X)} rows)")

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    min_class_count = int(y_train_full.value_counts().min())
    if args.cv_folds > min_class_count:
        raise ValueError(
            f"--cv-folds ({args.cv_folds}) cannot exceed minority class count ({min_class_count}) in the training split."
        )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=args.val_size, random_state=args.seed, stratify=y_train_full
    )

    preprocessor, numeric_features, _ = build_preprocessing_pipeline(X, scale_numeric=True)
    print(f"Feature scaling: StandardScaler on {numeric_features}")

    device = resolve_torch_device(args.device)
    X_train_np = preprocessor.fit_transform(X_train)
    X_test_np = preprocessor.transform(X_test)
    X_train_t = to_tensor(X_train_np, device)
    X_val_t = to_tensor(preprocessor.transform(X_val), device)
    X_test_t = to_tensor(X_test_np, device)
    y_train_t = to_tensor(y_train.values, device)
    y_val_t = to_tensor(y_val.values, device)

    criterion = make_loss(args.loss, y_train_t.cpu()).to(device)
    print(f"Device: {device.type}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loss function: {criterion.__class__.__name__}")
    print(f"Hidden dimensions: {hidden_dims}")

    optimizers_to_run = ["sgd", "adam"] if args.optimizer == "both" else [args.optimizer]
    results: list[dict] = []
    cv_summaries: list[dict] = []
    output_dir = resolve_output_dir(args.output_dir)

    for optimizer_name in optimizers_to_run:
        cv_fold_rows, cv_summary = run_mlp_cross_validation(
            optimizer_name=optimizer_name,
            args=args,
            X_full=X_train_full,
            y_full=y_train_full,
            device=device,
            hidden_dims=hidden_dims,
        )
        cv_fold_path = output_dir / f"mlp_cv_fold_metrics_{optimizer_name}.csv"
        pd.DataFrame(cv_fold_rows).to_csv(cv_fold_path, index=False)
        print(f"Saved CV fold metrics: {cv_fold_path}")
        cv_summaries.append(cv_summary)

        generator = set_seed(args.seed)
        model = HeartMLP(input_dim=X_train_t.shape[1], hidden_dims=hidden_dims, dropout=args.dropout).to(device)
        result = train_one_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            X_train=X_train_t,
            y_train=y_train_t,
            X_val=X_val_t,
            y_val=y_val_t,
            X_test=X_test_t,
            y_test=y_test,
            criterion=criterion,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            generator=generator,
            checkpoint_path=output_dir / f"heart_mlp_{optimizer_name}.pt" if args.save_checkpoint else None,
            confusion_matrix_path=output_dir / f"confusion_matrix_{optimizer_name}.csv" if args.save_confusion else None,
            history_path=output_dir / f"training_history_{optimizer_name}.csv",
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
            min_delta=args.min_delta,
            curve_path=output_dir / f"training_curves_{optimizer_name}.png",
            seed=args.seed,
        )
        result.update({k: v for k, v in cv_summary.items() if k.startswith("cv_")})
        results.append(result)

    cv_summary_path = output_dir / "mlp_cv_summary.csv"
    pd.DataFrame(cv_summaries).to_csv(cv_summary_path, index=False)
    print(f"Saved CV summary metrics: {cv_summary_path}")

    holdout_columns = [
        "optimizer", "best_epoch", "best_val_loss", "best_val_accuracy",
        *TEST_METRIC_COLUMNS.values(),
        *[f"{c}_ci_{side}" for c in TEST_METRIC_COLUMNS.values() for side in ("low", "high")],
        "optimal_threshold_f1", "f1_at_optimal_threshold", "optimal_threshold_youden", "youden_j",
    ]
    holdout_path = output_dir / "mlp_holdout_summary.csv"
    pd.DataFrame([{k: r[k] for k in holdout_columns} for r in results]).to_csv(holdout_path, index=False)
    print(f"Saved holdout summary metrics: {holdout_path}")

    print("\n=== Optimizer Comparison ===")
    for item in results:
        print(
            f"{item['optimizer'].upper()} | Best Epoch: {item['best_epoch']} | "
            f"Best Val Loss: {item['best_val_loss']:.4f} | Best Val Acc: {item['best_val_accuracy']:.4f} | "
            f"Test Accuracy: {item['test_accuracy']:.4f}"
        )

    # Model selection uses validation data only; the test set is reported, not consulted.
    best = min(results, key=lambda x: x["best_val_loss"])
    print(f"Best optimizer on this run (lowest validation loss): {best['optimizer'].upper()}")

    figures_dir = output_dir / "figures"
    model_probas = {f"MLP_{item['optimizer'].upper()}": item["y_prob"] for item in results}
    if model_probas:
        from evaluation.model_comparison import save_test_predictions

        save_test_predictions(output_dir, "mlp", args.dataset, X_test.index, y_test, model_probas)
        plot_calibration_curves(y_test, model_probas, "heart_mlp", figures_dir)
        plot_precision_recall_curves(y_test, model_probas, "heart_mlp", figures_dir)

    if args.run_xai:
        try:
            from evaluation.xai_analysis import generate_shap_mlp, save_xai_summary_csv

            best_model = HeartMLP(input_dim=X_train_t.shape[1], hidden_dims=hidden_dims, dropout=args.dropout).to(device)
            best_model.load_state_dict(best["model_state_dict"])
            print(f"\n=== SHAP Analysis: MLP {best['optimizer'].upper()} (Heart) ===")
            mlp_shap_result = generate_shap_mlp(
                model=best_model,
                X_train_np=X_train_np,
                X_test_np=X_test_np,
                feature_names=list(preprocessor.get_feature_names_out()),
                dataset_name="heart",
                model_name=f"mlp_{best['optimizer']}",
                output_dir=output_dir,
            )
            save_xai_summary_csv([mlp_shap_result], output_dir)
        except Exception as xai_error:
            print(f"Warning: SHAP analysis failed: {xai_error}")
            import traceback

            traceback.print_exc()

    if args.save_run_metrics:
        comparison_path = PROJECT_ROOT / args.comparison_file
        rows = [
            {
                "source": "pytorch_mlp",
                "model": f"MLP_{item['optimizer'].upper()}",
                "test_accuracy": item["test_accuracy"],
                "test_f1": item["test_f1"],
                "test_roc_auc": item["test_roc_auc"],
                "test_pr_auc": item["test_pr_auc"],
                "test_brier_score": item["test_brier_score"],
                "best_val_accuracy": item["best_val_accuracy"],
                "best_epoch": item["best_epoch"],
            }
            for item in results
        ]
        append_run_metrics(comparison_path, rows)
        print(f"Saved/updated separate run metrics: {comparison_path}")

    return results


if __name__ == "__main__":
    main()
