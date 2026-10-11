import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze overfitting across small/large MLP architectures.")
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["bce", "bce_weighted"], default="bce")
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["small:32,16", "base:64,32", "large:128,64"],
        help="Architecture list in label:h1,h2 format.",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts")
    return parser


def parse_architectures(items: list[str]) -> list[tuple[str, tuple[int, int]]]:
    parsed: list[tuple[str, tuple[int, int]]] = []
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid architecture '{item}'. Use label:h1,h2 format.")
        label, dims_str = item.split(":", maxsplit=1)
        dims_parts = [part.strip() for part in dims_str.split(",")]
        if len(dims_parts) != 2:
            raise ValueError(f"Invalid architecture '{item}'. Exactly two hidden dims are required.")
        try:
            hidden_1, hidden_2 = int(dims_parts[0]), int(dims_parts[1])
        except ValueError as exc:
            raise ValueError(f"Invalid hidden dims in '{item}'.") from exc
        if hidden_1 <= 0 or hidden_2 <= 0:
            raise ValueError(f"Hidden dims must be positive in '{item}'.")
        parsed.append((label.strip(), (hidden_1, hidden_2)))
    return parsed


def parse_eval_metrics(stdout: str, optimizer: str) -> tuple[float, float, int] | None:
    pattern = re.compile(
        rf"{optimizer.upper()} \| Best Epoch: ([0-9]+) \| Best Val Loss: [0-9.]+ \| Best Val Acc: ([0-9.]+) \| Test Accuracy: ([0-9.]+)"
    )
    match = pattern.search(stdout)
    if not match:
        return None
    return float(match.group(2)), float(match.group(3)), int(match.group(1))


def run_one_architecture(
    train_script: Path,
    output_root: Path,
    label: str,
    hidden_dims: tuple[int, int],
    args: argparse.Namespace,
) -> dict:
    run_output_dir = output_root / f"overfit_{label}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(train_script),
        "--optimizer",
        args.optimizer,
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--dropout",
        str(args.dropout),
        "--weight-decay",
        str(args.weight_decay),
        "--loss",
        args.loss,
        "--cv-folds",
        str(args.cv_folds),
        "--val-size",
        str(args.val_size),
        "--seed",
        str(args.seed),
        "--hidden-dims",
        str(hidden_dims[0]),
        str(hidden_dims[1]),
        "--output-dir",
        str(run_output_dir),
    ]

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    parsed = parse_eval_metrics(result.stdout, args.optimizer)

    row: dict = {
        "label": label,
        "hidden_dims": f"{hidden_dims[0]},{hidden_dims[1]}",
        "parameters_scale": hidden_dims[0] * hidden_dims[1],
        "exit_code": result.returncode,
        "best_epoch": None,
        "best_val_accuracy": None,
        "test_accuracy": None,
        "best_train_loss": None,
        "best_val_loss": None,
        "loss_gap_at_best": None,
        "final_train_loss": None,
        "final_val_loss": None,
        "loss_gap_final": None,
        "generalization_gap": None,
        "cv_accuracy_mean": None,
        "cv_accuracy_std": None,
    }

    if parsed is not None:
        best_val_accuracy, test_accuracy, best_epoch = parsed
        row["best_epoch"] = best_epoch
        row["best_val_accuracy"] = best_val_accuracy
        row["test_accuracy"] = test_accuracy
        row["generalization_gap"] = best_val_accuracy - test_accuracy

    history_path = run_output_dir / f"training_history_{args.optimizer}.csv"
    if history_path.exists():
        history = pd.read_csv(history_path)
        if not history.empty and "val_loss" in history.columns and "train_loss" in history.columns:
            best_idx = int(history["val_loss"].idxmin())
            row["best_train_loss"] = float(history.loc[best_idx, "train_loss"])
            row["best_val_loss"] = float(history.loc[best_idx, "val_loss"])
            row["loss_gap_at_best"] = row["best_val_loss"] - row["best_train_loss"]
            row["final_train_loss"] = float(history.iloc[-1]["train_loss"])
            row["final_val_loss"] = float(history.iloc[-1]["val_loss"])
            row["loss_gap_final"] = row["final_val_loss"] - row["final_train_loss"]

    cv_summary_path = run_output_dir / "mlp_cv_summary.csv"
    if cv_summary_path.exists():
        cv_summary = pd.read_csv(cv_summary_path)
        if not cv_summary.empty:
            row["cv_accuracy_mean"] = float(cv_summary.iloc[0]["cv_accuracy_mean"])
            row["cv_accuracy_std"] = float(cv_summary.iloc[0]["cv_accuracy_std"])

    if result.returncode != 0:
        row["stderr_tail"] = result.stderr[-1000:]

    return row


def save_markdown_table(df: pd.DataFrame, output_path: Path) -> None:
    lines = [
        "# Overfitting Analysis by Network Size",
        "",
        "Sorted by generalization gap (Best Val Acc - Test Acc, higher usually indicates stronger overfitting).",
        "",
        "| Model | Hidden Dims | Best Val Acc | Test Acc | Gen Gap (Val-Test) | Best Loss Gap | Final Loss Gap | CV Acc Mean ± Std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for _, row in df.iterrows():
        cv_cell = "-"
        if pd.notna(row.get("cv_accuracy_mean")) and pd.notna(row.get("cv_accuracy_std")):
            cv_cell = f"{row['cv_accuracy_mean']:.4f} ± {row['cv_accuracy_std']:.4f}"
        lines.append(
            "| "
            f"{row['label']} | {row['hidden_dims']} | "
            f"{row['best_val_accuracy']:.4f} | {row['test_accuracy']:.4f} | "
            f"{row['generalization_gap']:.4f} | {row['loss_gap_at_best']:.4f} | "
            f"{row['loss_gap_final']:.4f} | {cv_cell} |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_arg_parser().parse_args()
    base_dir = Path(__file__).resolve().parent
    project_root = base_dir.parent
    train_script = project_root / "training" / "train_model.py"
    if not train_script.exists():
        raise FileNotFoundError("train_model.py not found.")

    architectures = parse_architectures(args.architectures)
    output_root = project_root / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    print(f"Running overfitting analysis on {len(architectures)} architectures...")
    for index, (label, hidden_dims) in enumerate(architectures, start=1):
        print(f"[{index}/{len(architectures)}] {label} ({hidden_dims[0]},{hidden_dims[1]})")
        rows.append(run_one_architecture(train_script, output_root, label, hidden_dims, args))

    results = pd.DataFrame(rows)
    valid = results[results["exit_code"] == 0].copy()
    if valid.empty:
        results_path = output_root / "overfitting_by_network_size.csv"
        results.to_csv(results_path, index=False)
        raise RuntimeError(f"All architecture runs failed. See {results_path}")

    ranked = valid.sort_values(by=["generalization_gap", "loss_gap_at_best"], ascending=False).reset_index(drop=True)
    results_path = output_root / "overfitting_by_network_size.csv"
    markdown_path = output_root / "overfitting_by_network_size.md"
    ranked.to_csv(results_path, index=False)
    save_markdown_table(ranked, markdown_path)

    print("\n=== Overfitting Ranking (higher generalization gap => more overfitting tendency) ===")
    for _, row in ranked.iterrows():
        print(
            f"{row['label']}: hidden={row['hidden_dims']}, "
            f"best_val_acc={row['best_val_accuracy']:.4f}, test_acc={row['test_accuracy']:.4f}, "
            f"gen_gap={row['generalization_gap']:.4f}, final_loss_gap={row['loss_gap_final']:.4f}"
        )

    print(f"Saved analysis CSV: {results_path}")
    print(f"Saved analysis Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
