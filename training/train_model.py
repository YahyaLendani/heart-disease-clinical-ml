from pathlib import Path
import argparse
from copy import deepcopy
import random
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from clean_data import build_preprocessing_pipeline
    from evaluation.clinical_metrics import (
        compute_clinical_metrics,
        plot_calibration_curves,
        plot_precision_recall_curves,
    )
    from models import HeartMLP
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from clean_data import build_preprocessing_pipeline
    from evaluation.clinical_metrics import (
        compute_clinical_metrics,
        plot_calibration_curves,
        plot_precision_recall_curves,
    )
    from models import HeartMLP


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def evaluate_loss(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(loss.item())
    return float(sum(losses) / max(len(losses), 1))


def evaluate_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    all_preds: list[int] = []
    all_true: list[int] = []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int().cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(yb.int().cpu().tolist())
    return accuracy_score(all_true, all_preds)


def predict_binary(model: nn.Module, data_loader: DataLoader, device: torch.device) -> list[int]:
    model.eval()
    all_predictions: list[int] = []
    with torch.no_grad():
        for xb, _ in data_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int().cpu().tolist()
            all_predictions.extend(preds)
    return all_predictions


def predict_probabilities(model: nn.Module, data_loader: DataLoader, device: torch.device) -> list[float]:
    model.eval()
    all_probabilities: list[float] = []
    with torch.no_grad():
        for xb, _ in data_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().tolist()
            all_probabilities.extend([float(value) for value in probs])
    return all_probabilities


def save_training_curves(history_rows: list[dict], output_path: Path, optimizer_name: str) -> None:
    try:
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


def train_one_optimizer(
    optimizer_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    y_test: pd.Series,
    criterion: nn.Module,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device: torch.device,
    checkpoint_path: Path | None = None,
    confusion_matrix_path: Path | None = None,
    history_path: Path | None = None,
    grad_clip: float = 1.0,
    label_smoothing: float = 0.0,
    min_delta: float = 0.0,
    curve_path: Path | None = None,
) -> dict:
    if optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
    )

    best_val_loss = float("inf")
    best_val_accuracy = 0.0
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    wait = 0
    history_rows: list[dict] = []

    print(f"\n=== Training with {optimizer_name.upper()} ===")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            yb_train = smooth_binary_targets(yb, label_smoothing)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb_train)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = float(sum(epoch_losses) / max(len(epoch_losses), 1))
        val_loss = evaluate_loss(model, val_loader, criterion, device)
        val_accuracy = evaluate_accuracy(model, val_loader, device)
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

        print(
            f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f} | LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    print(
        f"Best validation checkpoint for {optimizer_name.upper()}: "
        f"epoch={best_epoch}, val_loss={best_val_loss:.4f}, val_accuracy={best_val_accuracy:.4f}"
    )

    model.load_state_dict(best_state)
    y_pred = predict_binary(model, test_loader, device)
    y_prob = predict_probabilities(model, test_loader, device)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    try:
        roc_auc = roc_auc_score(y_test, y_prob)
    except ValueError:
        roc_auc = float("nan")

    # Clinical metrics
    try:
        clinical = compute_clinical_metrics(y_test, y_prob)
    except Exception:
        clinical = {
            "pr_auc": float("nan"),
            "brier_score": float("nan"),
            "optimal_threshold_f1": 0.5,
            "f1_at_optimal": float("nan"),
            "optimal_threshold_youden": 0.5,
            "youden_j": float("nan"),
        }

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"\n{optimizer_name.upper()} Test Accuracy: {accuracy:.4f}")
    print(f"{optimizer_name.upper()} Test F1: {f1:.4f}")
    print(f"{optimizer_name.upper()} Test ROC-AUC: {roc_auc:.4f}")
    print(f"{optimizer_name.upper()} Test PR-AUC: {clinical['pr_auc']:.4f}")
    print(f"{optimizer_name.upper()} Test Brier Score: {clinical['brier_score']:.4f}")
    print(
        f"{optimizer_name.upper()} Optimal threshold (F1): {clinical['optimal_threshold_f1']:.3f} "
        f"(F1={clinical['f1_at_optimal']:.4f})"
    )
    print(classification_report(y_test, y_pred, target_names=["healthy", "sick"]))
    cm_df = pd.DataFrame(cm, index=["actual_healthy", "actual_sick"], columns=["pred_healthy", "pred_sick"])
    print("Confusion Matrix:")
    print(cm_df.to_string())

    if checkpoint_path is not None:
        torch.save(
            {
                "optimizer": optimizer_name,
                "best_val_loss": best_val_loss,
                "model_state_dict": best_state,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")

    if confusion_matrix_path is not None:
        cm_df.to_csv(confusion_matrix_path)
        print(f"Saved confusion matrix: {confusion_matrix_path}")

    if history_path is not None:
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        print(f"Saved training history: {history_path}")

    if curve_path is not None:
        save_training_curves(history_rows, curve_path, optimizer_name)

    return {
        "optimizer": optimizer_name,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": best_val_accuracy,
        "test_accuracy": accuracy,
        "test_f1": f1,
        "test_roc_auc": roc_auc,
        "test_pr_auc": clinical["pr_auc"],
        "test_brier_score": clinical["brier_score"],
        "optimal_threshold_f1": clinical["optimal_threshold_f1"],
        "f1_at_optimal_threshold": clinical["f1_at_optimal"],
        "best_epoch": best_epoch,
        "y_prob": y_prob,
        "model_state_dict": best_state,
    }


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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--save-run-metrics", action="store_true")
    parser.add_argument("--comparison-file", type=str, default="artifacts/separate_model_runs.csv")
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--dataset", type=str, default="cleaned_heart_disease.csv", help="Path to cleaned dataset CSV file.")
    parser.add_argument("--run-xai", action="store_true", help="Run SHAP explainability analysis after training.")
    return parser


def resolve_torch_device(requested_device: str) -> torch.device:
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA-compatible GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def append_run_metrics(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(output_path, index=False)


def run_mlp_cross_validation(
    optimizer_name: str,
    args: argparse.Namespace,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    categorical_features: list[str],
    numeric_features: list[str],
    device: torch.device,
    hidden_dims: tuple[int, int],
) -> tuple[list[dict], dict]:
    print(f"\n=== {args.cv_folds}-Fold Cross-Validation ({optimizer_name.upper()}) ===")
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    fold_rows: list[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full), start=1):
        X_train_fold = X_full.iloc[train_idx]
        X_val_fold = X_full.iloc[val_idx]
        y_train_fold = y_full.iloc[train_idx]
        y_val_fold = y_full.iloc[val_idx]

        preprocessor, _, _ = build_preprocessing_pipeline(X_train_fold, scale_numeric=True)

        X_train_np = preprocessor.fit_transform(X_train_fold)
        X_val_np = preprocessor.transform(X_val_fold)

        X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_fold.values, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val_fold.values, dtype=torch.float32)

        pin_memory = device.type == "cuda"
        train_loader = DataLoader(
            TensorDataset(X_train_tensor, y_train_tensor),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            TensorDataset(X_val_tensor, y_val_tensor),
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=pin_memory,
        )

        model = HeartMLP(input_dim=X_train_tensor.shape[1], hidden_dims=hidden_dims, dropout=args.dropout).to(device)
        criterion = make_loss(args.loss, y_train_tensor).to(device)

        result = train_one_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=val_loader,
            y_test=y_val_fold,
            criterion=criterion,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
            checkpoint_path=None,
            confusion_matrix_path=None,
            history_path=None,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
            min_delta=args.min_delta,
            curve_path=None,
        )

        fold_row = {
            "optimizer": optimizer_name,
            "fold": fold_idx,
            "best_epoch": result["best_epoch"],
            "best_val_loss": result["best_val_loss"],
            "best_val_accuracy": result["best_val_accuracy"],
            "cv_accuracy": result["test_accuracy"],
            "cv_f1": result["test_f1"],
            "cv_roc_auc": result["test_roc_auc"],
            "cv_pr_auc": result["test_pr_auc"],
            "cv_brier_score": result["test_brier_score"],
        }
        fold_rows.append(fold_row)
        print(
            f"Fold {fold_idx}/{args.cv_folds} | "
            f"Accuracy: {fold_row['cv_accuracy']:.4f} | "
            f"F1: {fold_row['cv_f1']:.4f} | "
            f"ROC-AUC: {fold_row['cv_roc_auc']:.4f}"
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "optimizer": optimizer_name,
        "cv_folds": args.cv_folds,
        "cv_accuracy_mean": float(np.nanmean(fold_df["cv_accuracy"])),
        "cv_accuracy_std": float(np.nanstd(fold_df["cv_accuracy"])),
        "cv_f1_mean": float(np.nanmean(fold_df["cv_f1"])),
        "cv_f1_std": float(np.nanstd(fold_df["cv_f1"])),
        "cv_roc_auc_mean": float(np.nanmean(fold_df["cv_roc_auc"])),
        "cv_roc_auc_std": float(np.nanstd(fold_df["cv_roc_auc"])),
        "cv_pr_auc_mean": float(np.nanmean(fold_df["cv_pr_auc"])),
        "cv_pr_auc_std": float(np.nanstd(fold_df["cv_pr_auc"])),
        "cv_brier_mean": float(np.nanmean(fold_df["cv_brier_score"])),
        "cv_brier_std": float(np.nanstd(fold_df["cv_brier_score"])),
    }

    print(
        f"CV Summary ({optimizer_name.upper()}): "
        f"Accuracy {summary['cv_accuracy_mean']:.4f} ± {summary['cv_accuracy_std']:.4f}, "
        f"F1 {summary['cv_f1_mean']:.4f} ± {summary['cv_f1_std']:.4f}, "
        f"ROC-AUC {summary['cv_roc_auc_mean']:.4f} ± {summary['cv_roc_auc_std']:.4f}, "
        f"PR-AUC {summary['cv_pr_auc_mean']:.4f} ± {summary['cv_pr_auc_std']:.4f}, "
        f"Brier {summary['cv_brier_mean']:.4f} ± {summary['cv_brier_std']:.4f}"
    )

    return fold_rows, summary


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    if not 0 < args.val_size < 0.5:
        raise ValueError("--val-size must be between 0 and 0.5.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")
    if any(dim <= 0 for dim in args.hidden_dims):
        raise ValueError("--hidden-dims values must be positive integers.")
    hidden_dims = (args.hidden_dims[0], args.hidden_dims[1])

    cleaned_path = Path(args.dataset)
    if not cleaned_path.is_absolute():
        cleaned_path = PROJECT_ROOT / cleaned_path
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
    print(f"Loaded cleaned dataset: {cleaned_path.name} ({len(df)} rows)")

    drop_cols = [col for col in ["diagnosis", "disease_code", "target", "site"] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["target"]

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

    preprocessor, numeric_features, categorical_features = build_preprocessing_pipeline(
        X, scale_numeric=True
    )
    print(f"Feature scaling: StandardScaler on {numeric_features}")

    X_train_np = preprocessor.fit_transform(X_train)
    X_val_np = preprocessor.transform(X_val)
    X_test_np = preprocessor.transform(X_test)

    X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_np, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)
    device = resolve_torch_device(args.device)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        TensorDataset(X_train_tensor, y_train_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_tensor, y_val_tensor),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_tensor, torch.tensor(y_test.values, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=pin_memory,
    )

    criterion = make_loss(args.loss, y_train_tensor).to(device)
    print(f"Device: {device.type}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loss function: {criterion.__class__.__name__}")
    print(f"Gradient clipping max norm: {args.grad_clip}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Early stopping min delta: {args.min_delta}")
    print(f"Hidden dimensions: {hidden_dims}")

    optimizers_to_run = ["sgd", "adam"] if args.optimizer == "both" else [args.optimizer]
    results: list[dict] = []
    cv_summaries: list[dict] = []
    output_dir_arg = Path(args.output_dir)
    output_dir = output_dir_arg if output_dir_arg.is_absolute() else PROJECT_ROOT / output_dir_arg
    output_dir.mkdir(parents=True, exist_ok=True)

    for optimizer_name in optimizers_to_run:
        cv_fold_rows, cv_summary = run_mlp_cross_validation(
            optimizer_name=optimizer_name,
            args=args,
            X_full=X_train_full,
            y_full=y_train_full,
            categorical_features=categorical_features,
            numeric_features=numeric_features,
            device=device,
            hidden_dims=hidden_dims,
        )
        cv_fold_path = output_dir / f"mlp_cv_fold_metrics_{optimizer_name}.csv"
        pd.DataFrame(cv_fold_rows).to_csv(cv_fold_path, index=False)
        print(f"Saved CV fold metrics: {cv_fold_path}")
        cv_summaries.append(cv_summary)

        model = HeartMLP(input_dim=X_train_tensor.shape[1], hidden_dims=hidden_dims, dropout=args.dropout).to(device)
        checkpoint_path = output_dir / f"heart_mlp_{optimizer_name}.pt" if args.save_checkpoint else None
        confusion_matrix_path = output_dir / f"confusion_matrix_{optimizer_name}.csv" if args.save_confusion else None
        history_path = output_dir / f"training_history_{optimizer_name}.csv"
        curve_path = output_dir / f"training_curves_{optimizer_name}.png"
        result = train_one_optimizer(
            optimizer_name=optimizer_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            y_test=y_test,
            criterion=criterion,
            lr=args.lr,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            patience=args.patience,
            device=device,
            checkpoint_path=checkpoint_path,
            confusion_matrix_path=confusion_matrix_path,
            history_path=history_path,
            grad_clip=args.grad_clip,
            label_smoothing=args.label_smoothing,
            min_delta=args.min_delta,
            curve_path=curve_path,
        )
        results.append(result)

    cv_summary_path = output_dir / "mlp_cv_summary.csv"
    pd.DataFrame(cv_summaries).to_csv(cv_summary_path, index=False)
    print(f"Saved CV summary metrics: {cv_summary_path}")

    print("\n=== Optimizer Comparison ===")
    for item in results:
        print(
            f"{item['optimizer'].upper()} | Best Epoch: {item['best_epoch']} | "
            f"Best Val Loss: {item['best_val_loss']:.4f} | Best Val Acc: {item['best_val_accuracy']:.4f} | "
            f"Test Accuracy: {item['test_accuracy']:.4f}"
        )

    best = max(results, key=lambda x: x["test_accuracy"])
    print(f"Best optimizer on this run: {best['optimizer'].upper()}")

    # --- Calibration & PR Curves ---
    figures_dir = output_dir / "figures"
    model_probas: dict[str, list[float]] = {}
    for item in results:
        if "y_prob" in item:
            model_probas[f"MLP_{item['optimizer'].upper()}"] = item["y_prob"]
    if model_probas:
        plot_calibration_curves(y_test, model_probas, "heart_mlp", figures_dir)
        plot_precision_recall_curves(y_test, model_probas, "heart_mlp", figures_dir)

    # --- SHAP Explainability ---
    if args.run_xai:
        try:
            try:
                from evaluation.xai_analysis import generate_shap_mlp, save_xai_summary_csv
            except ModuleNotFoundError:
                sys.path.append(str(PROJECT_ROOT))
                from evaluation.xai_analysis import generate_shap_mlp, save_xai_summary_csv

            best_optimizer_name = best["optimizer"]
            checkpoint_file = output_dir / f"heart_mlp_{best_optimizer_name}.pt"

            # Rebuild model from best optimizer and load best state
            best_model = HeartMLP(
                input_dim=X_train_tensor.shape[1],
                hidden_dims=hidden_dims,
                dropout=args.dropout,
            ).to(device)
            if "model_state_dict" in best and best["model_state_dict"] is not None:
                best_model.load_state_dict(best["model_state_dict"])
            elif checkpoint_file.exists():
                try:
                    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=True)
                    best_model.load_state_dict(checkpoint["model_state_dict"])
                except Exception as load_err:
                    print(f"Warning: could not load checkpoint ({load_err})")
            else:
                print(f"Warning: checkpoint not found at {checkpoint_file}")

            feature_names = list(preprocessor.get_feature_names_out())

            print(f"\n=== SHAP Analysis: MLP {best_optimizer_name.upper()} (Heart) ===")
            mlp_shap_result = generate_shap_mlp(
                model=best_model,
                X_train_np=X_train_np,
                X_test_np=X_test_np,
                feature_names=feature_names,
                dataset_name="heart",
                model_name=f"mlp_{best_optimizer_name}",
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


if __name__ == "__main__":
    main()