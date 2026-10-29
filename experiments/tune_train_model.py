"""Grid search over HeartMLP training hyperparameters.

Configurations are ranked by repeated-CV ROC-AUC (then validation loss).
Test accuracy is recorded for reference but never used to choose a winner,
since choosing on the test set would make the reported test score optimistic.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def save_learning_rate_plot(results: pd.DataFrame, output_path: Path) -> None:
    """Plot learning rate against the selection metric (repeated-CV ROC-AUC)."""
    metric = "cv_roc_auc_mean"
    if "learning_rate" not in results.columns or metric not in results.columns:
        return

    plot_df = results.dropna(subset=["learning_rate", metric]).copy()
    if plot_df.empty:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping learning-rate plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    groups = plot_df.groupby("optimizer") if "optimizer" in plot_df.columns else [("model", plot_df)]
    for optimizer_name, group in groups:
        # Several dropout/clip settings share a learning rate: plot the best one per rate.
        best = group.groupby("learning_rate", as_index=False)[metric].max().sort_values("learning_rate")
        ax.plot(best["learning_rate"], best[metric], marker="o", label=str(optimizer_name).upper())

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Best CV ROC-AUC")
    ax.set_title("Learning Rate vs Cross-Validated ROC-AUC")
    ax.set_ylim(0.5, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved learning-rate plot: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune PyTorch heart model hyperparameters.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--loss", choices=["bce", "bce_weighted"], default="bce_weighted")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizers", nargs="+", default=["adam", "sgd"])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.2, 0.3])
    parser.add_argument("--grad-clips", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--label-smoothings", nargs="+", type=float, default=[0.0, 0.02])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="cleaned_heart_disease.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    return parser


def run_trial(
    optimizer: str,
    learning_rate: float,
    dropout: float,
    grad_clip: float,
    label_smoothing: float,
    args: argparse.Namespace,
    trial_output_dir: Path,
) -> dict:
    from training.train_model import main as train_main

    argv = [
        "--optimizer", optimizer,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--loss", args.loss,
        "--batch-size", str(args.batch_size),
        "--lr", str(learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--dropout", str(dropout),
        "--grad-clip", str(grad_clip),
        "--label-smoothing", str(label_smoothing),
        "--seed", str(args.seed),
        "--cv-folds", str(args.cv_folds),
        "--cv-repeats", str(args.cv_repeats),
        "--dataset", args.dataset,
        "--output-dir", str(trial_output_dir),
    ]
    row = {
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "grad_clip": grad_clip,
        "label_smoothing": label_smoothing,
        "status": "ok",
        "cv_roc_auc_mean": None,
        "cv_roc_auc_std": None,
        "best_val_loss": None,
        "val_accuracy": None,
        "test_accuracy": None,
    }
    try:
        result = train_main(argv)[0]
    except Exception as error:  # keep the grid going; record the failure
        row["status"] = f"failed: {error}"
        return row

    row.update(
        {
            "cv_roc_auc_mean": result["cv_roc_auc_mean"],
            "cv_roc_auc_std": result["cv_roc_auc_std"],
            "best_val_loss": result["best_val_loss"],
            "val_accuracy": result["best_val_accuracy"],
            "test_accuracy": result["test_accuracy"],
        }
    )
    return row


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    learning_rates = args.learning_rates if args.learning_rates else [args.lr]

    trials = list(
        itertools.product(
            args.optimizers,
            learning_rates,
            args.dropouts,
            args.grad_clips,
            args.label_smoothings,
        )
    )
    print(f"Total tuning trials: {len(trials)}")

    rows: list[dict] = []
    for idx, (optimizer, learning_rate, dropout, grad_clip, label_smoothing) in enumerate(trials, start=1):
        print(
            f"[{idx}/{len(trials)}] optimizer={optimizer}, lr={learning_rate}, dropout={dropout}, "
            f"grad_clip={grad_clip}, label_smoothing={label_smoothing}"
        )
        rows.append(
            run_trial(
                optimizer,
                learning_rate,
                dropout,
                grad_clip,
                label_smoothing,
                args,
                output_dir / "tuning_runs" / f"trial_{idx:03d}",
            )
        )

    results = pd.DataFrame(rows)
    ok_results = results[results["status"] == "ok"].copy()

    if ok_results.empty:
        results_path = output_dir / "tuning_results.csv"
        results.to_csv(results_path, index=False)
        raise RuntimeError(f"No successful tuning runs. See {results_path}")

    ranked = ok_results.sort_values(
        by=["cv_roc_auc_mean", "best_val_loss"], ascending=[False, True]
    ).reset_index(drop=True)

    results_path = output_dir / "tuning_results.csv"
    ranked.to_csv(results_path, index=False)

    lr_plot_path = output_dir / "learning_rate_vs_accuracy.png"
    save_learning_rate_plot(ranked, lr_plot_path)

    best = ranked.iloc[0].to_dict()
    best_path = output_dir / "best_hyperparameters.json"
    with open(best_path, "w", encoding="utf-8") as file:
        json.dump(best, file, indent=2)

    print("\n=== Best Configuration ===")
    for key, value in best.items():
        print(f"{key}: {value}")

    print(f"Saved tuning results: {results_path}")
    print(f"Saved best config: {best_path}")


if __name__ == "__main__":
    main()
