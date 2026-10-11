import argparse
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


def save_learning_rate_plot(results: pd.DataFrame, output_path: Path) -> None:
    if "learning_rate" not in results.columns or "test_accuracy" not in results.columns:
        return

    plot_df = results.dropna(subset=["learning_rate", "test_accuracy"]).copy()
    if plot_df.empty:
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping learning-rate plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    if "optimizer" in plot_df.columns:
        for optimizer_name, group in plot_df.groupby("optimizer"):
            group = group.sort_values("learning_rate")
            ax.plot(
                group["learning_rate"],
                group["test_accuracy"],
                marker="o",
                label=str(optimizer_name).upper(),
            )
    else:
        group = plot_df.sort_values("learning_rate")
        ax.plot(group["learning_rate"], group["test_accuracy"], marker="o", label="Model")

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Learning Rate vs Test Accuracy")
    ax.set_ylim(0, 1)
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
    parser.add_argument("--output-dir", type=str, default="artifacts")
    return parser


def parse_metrics(stdout: str, optimizer: str) -> tuple[float, float] | None:
    pattern = re.compile(
        rf"{optimizer.upper()} \| Best Epoch: .*?\| Best Val Loss: .*?\| Best Val Acc: ([0-9.]+) \| Test Accuracy: ([0-9.]+)"
    )
    match = pattern.search(stdout)
    if not match:
        return None
    return float(match.group(1)), float(match.group(2))


def run_trial(
    train_script: Path,
    optimizer: str,
    learning_rate: float,
    dropout: float,
    grad_clip: float,
    label_smoothing: float,
    args: argparse.Namespace,
) -> dict:
    command = [
        sys.executable,
        str(train_script),
        "--optimizer",
        optimizer,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--loss",
        args.loss,
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(learning_rate),
        "--weight-decay",
        str(args.weight_decay),
        "--dropout",
        str(dropout),
        "--grad-clip",
        str(grad_clip),
        "--label-smoothing",
        str(label_smoothing),
        "--seed",
        str(args.seed),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    parsed = parse_metrics(result.stdout, optimizer)

    row = {
        "optimizer": optimizer,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "grad_clip": grad_clip,
        "label_smoothing": label_smoothing,
        "exit_code": result.returncode,
        "val_accuracy": None,
        "test_accuracy": None,
    }

    if parsed is not None:
        row["val_accuracy"] = parsed[0]
        row["test_accuracy"] = parsed[1]

    if result.returncode != 0:
        row["stderr_tail"] = result.stderr[-1000:]

    return row


def main() -> None:
    args = build_arg_parser().parse_args()
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    train_script = project_root / "training" / "train_model.py"
    if not train_script.exists():
        raise FileNotFoundError("train_model.py not found.")

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
                train_script,
                optimizer,
                learning_rate,
                dropout,
                grad_clip,
                label_smoothing,
                args,
            )
        )

    results = pd.DataFrame(rows)
    ok_results = results.dropna(subset=["test_accuracy", "val_accuracy"]).copy()

    if ok_results.empty:
        results_path = output_dir / "tuning_results.csv"
        results.to_csv(results_path, index=False)
        raise RuntimeError(f"No successful tuning runs. See {results_path}")

    ranked = ok_results.sort_values(by=["test_accuracy", "val_accuracy"], ascending=False).reset_index(drop=True)

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
