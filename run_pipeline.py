import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name: str, extra_args: list[str] | None = None) -> None:
    script_path = Path(__file__).resolve().parent / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Required script not found: {script_name}")

    print(f"\n>>> Running {script_name}")
    command = [sys.executable, str(script_path)]
    if extra_args:
        command.extend(extra_args)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed with exit code {result.returncode}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run clean -> EDA -> train pipeline.")
    parser.add_argument("--optimizer", choices=["sgd", "adam", "both"], default="both")
    parser.add_argument("--loss", choices=["bce", "bce_weighted"], default="bce")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--save-confusion", action="store_true")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--run-baselines", action="store_true")
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=25)
    parser.add_argument("--xgb-eval-size", type=float, default=0.2)
    parser.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility across splits and models.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cleaned_heart_disease.csv",
        help="Heart dataset CSV path to use (e.g. cleaned_heart_disease_combined.csv).",
    )
    parser.add_argument("--update-thesis", action="store_true")
    parser.add_argument("--update-thesis-all", action="store_true")
    parser.add_argument("--run-xai", action="store_true", help="Run SHAP explainability analysis after training.")
    parser.add_argument("--run-advancements", action="store_true", help="Run all 4 new clinical advancements: features, ensemble, LOCO, and fairness.")
    parser.add_argument("--run-ensemble", action="store_true", help="Run hybrid ensemble benchmark (MLP + XGBoost + RF + LogReg).")
    parser.add_argument("--run-loco", action="store_true", help="Run Leave-One-Center-Out (LOCO) cross-hospital generalization.")
    parser.add_argument("--run-fairness", action="store_true", help="Run demographic fairness and disparity analysis.")
    parser.add_argument("--run-reading-list", action="store_true", help="Run full AI Thesis reading list suite (TabTransformer, GNN, Multimodal, Calibration, Deep Ensembles, Causal, Fairness).")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest validation before running the pipeline.")
    return parser


def run_tests() -> None:
    """Run the pytest suite. Abort pipeline if any test fails."""
    print("\n>>> Running test suite (pytest)")
    tests_dir = Path(__file__).resolve().parent / "tests"
    if not tests_dir.exists():
        raise FileNotFoundError(f"Tests directory not found: {tests_dir}")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(tests_dir), "-v", "--tb=short"],
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Test suite failed with exit code {result.returncode}. "
            "Pipeline aborted — fix failing tests before proceeding."
        )


def main() -> None:
    args = build_arg_parser().parse_args()

    if not args.skip_tests:
        run_tests()
    else:
        print("\n>>> Skipping test suite (--skip-tests)")

    print("\nStarting full pipeline: clean -> EDA -> train")
    if args.dataset == "cleaned_heart_disease.csv":
        run_script("clean_data.py")
        run_script("eda.py")
    else:
        print(f"\n>>> Using custom/expanded dataset: {args.dataset}")

    train_args = [
        "--optimizer",
        args.optimizer,
        "--loss",
        args.loss,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--device",
        args.device,
        "--weight-decay",
        str(args.weight_decay),
        "--dropout",
        str(args.dropout),
        "--patience",
        str(args.patience),
        "--val-size",
        str(args.val_size),
        "--cv-folds",
        str(args.cv_folds),
        "--seed",
        str(args.seed),
        "--dataset",
        args.dataset,
        "--output-dir",
        args.output_dir,
    ]

    if args.save_checkpoint:
        train_args.append("--save-checkpoint")
    if args.save_confusion:
        train_args.append("--save-confusion")
    if args.run_xai:
        train_args.append("--run-xai")

    run_script("training/train_model.py", train_args)

    if args.run_baselines:
        baseline_args = [
            "--xgb-early-stopping-rounds",
            str(args.xgb_early_stopping_rounds),
            "--xgb-eval-size",
            str(args.xgb_eval_size),
            "--xgb-device",
            args.xgb_device,
            "--cv-folds",
            str(args.cv_folds),
            "--seed",
            str(args.seed),
            "--dataset",
            args.dataset,
        ]
        if args.run_xai:
            baseline_args.append("--run-xai")
        run_script("training/baseline_models.py", baseline_args)

    if args.run_advancements or args.run_ensemble:
        ensemble_args = [
            "--dataset",
            args.dataset,
            "--cv-folds",
            str(args.cv_folds),
            "--seed",
            str(args.seed),
            "--output-dir",
            args.output_dir,
        ]
        run_script("training/ensemble_models.py", ensemble_args)

    if args.run_advancements or args.run_loco:
        loco_args = [
            "--dataset",
            args.dataset,
            "--seed",
            str(args.seed),
            "--output-dir",
            args.output_dir,
        ]
        run_script("experiments/cross_center_generalization.py", loco_args)

    if args.run_advancements or args.run_fairness:
        fairness_args = [
            "--dataset",
            args.dataset,
            "--seed",
            str(args.seed),
            "--output-dir",
            args.output_dir,
        ]
        run_script("evaluation/fairness_analysis.py", fairness_args)

    if args.run_reading_list:
        reading_list_args = [
            "--heart-csv",
            args.dataset,
            "--output-dir",
            args.output_dir,
        ]
        run_script("experiments/run_reading_list_suite.py", reading_list_args)

    if args.update_thesis:
        run_script("update_thesis_results.py", ["--task", "heart", "--heart-dataset", args.dataset])

    if args.update_thesis_all:
        run_script("update_thesis_results.py", ["--task", "all", "--heart-dataset", args.dataset])

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
