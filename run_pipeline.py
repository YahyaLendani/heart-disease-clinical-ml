"""Single command-line entry point for the whole project (installed as ``heart-pipeline``).

Two ways to use it:

* ``heart-pipeline [options]`` runs the end-to-end pipeline
  (clean -> EDA -> MLP training, plus any optional stages you switch on).
* ``heart-pipeline <step> [step options]`` runs one step, e.g.
  ``heart-pipeline baselines --cv-repeats 5`` or ``heart-pipeline loco --help``.

Every stage runs in this Python process (no subprocess per stage), so heavy
libraries are imported once and only by the stages that need them.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEVICE_CHOICES = ("auto", "cpu", "cuda")
MULTI_CENTER_DATASET = "cleaned_heart_disease_combined.csv"

# step name -> (module with a ``main(argv)`` function, one-line description)
STEPS: dict[str, tuple[str, str]] = {
    "clean": ("clean_data", "Clean the raw Cleveland heart data into data/processed."),
    "prepare-combined": ("prepare_heart_combined", "Build the 4-center combined heart cohort."),
    "prepare-mimic": ("mimic_prepare_dataset", "Build the MIMIC-IV mortality dataset from the demo extract."),
    "eda": ("eda", "Exploratory data analysis tables."),
    "train-mlp": ("training.train_model", "PyTorch HeartMLP (SGD vs Adam) with repeated CV."),
    "baselines": ("training.baseline_models", "Logistic Regression, Random Forest and XGBoost baselines."),
    "ensemble": ("training.ensemble_models", "Soft-voting and stacking ensembles."),
    "loco": ("experiments.cross_center_generalization", "Leave-one-center-out generalisation."),
    "fairness": ("evaluation.fairness_analysis", "Sex and age subgroup fairness analysis."),
    "reading-list": ("experiments.run_reading_list_suite", "TabTransformer, GNNs, calibration, conformal, causal, recourse."),
    "mimic-baselines": ("training.mimic_train_baselines", "MIMIC-IV mortality baselines."),
    "mimic-mlp": ("training.mimic_train_mlp", "MIMIC-IV mortality MLP."),
    "tune": ("experiments.tune_train_model", "HeartMLP hyperparameter grid search (ranked by CV)."),
    "overfit": ("experiments.analyze_overfitting_network_size", "Overfitting vs network size."),
    "compare-runs": ("evaluation.compare_separate_runs", "Compare separately trained runs."),
    "compare-models": ("evaluation.model_comparison", "Paired DeLong tests between all models on the shared test split."),
    "update-thesis": ("evaluation.update_thesis_results", "Write result tables into thesis_draft.md."),
    "test": ("", "Run the pytest suite."),
}


def run_step(step: str, argv: list[str] | None = None) -> object:
    """Run one pipeline step in-process and report how long it took."""
    module_name, _ = STEPS[step]
    argv = list(argv or [])
    print(f"\n>>> {step} {' '.join(argv)}".rstrip())
    start = time.perf_counter()
    if step == "test":
        result = run_tests(argv)
    else:
        result = importlib.import_module(module_name).main(argv)
    print(f"<<< {step} finished in {time.perf_counter() - start:.1f}s")
    return result


def run_tests(extra_args: list[str] | None = None) -> None:
    """Run the pytest suite; raise if any test fails."""
    import pytest

    code = pytest.main([str(PROJECT_ROOT / "tests"), "-q", *(extra_args or [])])
    if code != 0:
        raise RuntimeError(f"Test suite failed with exit code {code}.")


def build_arg_parser() -> argparse.ArgumentParser:
    step_lines = "\n".join(f"  {name:<17}{desc}" for name, (_, desc) in STEPS.items())
    parser = argparse.ArgumentParser(
        description="Run clean -> EDA -> train pipeline, or a single step.",
        epilog=f"Single steps (heart-pipeline <step> --help for options):\n{step_lines}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--optimizer", choices=["sgd", "adam", "both"], default="both")
    parser.add_argument("--loss", choices=["bce", "bce_weighted"], default="bce")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
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
    parser.add_argument("--xgb-device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3, help="Repeats of stratified k-fold CV.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers (-1 = all cores).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility across splits and models.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=MULTI_CENTER_DATASET,
        help="Heart dataset CSV (default: the 918-patient 4-center cohort; "
        "cleaned_heart_disease.csv for Cleveland only).",
    )
    parser.add_argument("--update-thesis", action="store_true")
    parser.add_argument("--update-thesis-all", action="store_true")
    parser.add_argument("--run-xai", action="store_true", help="Run SHAP explainability analysis after training.")
    parser.add_argument("--run-advancements", action="store_true", help="Run ensemble, LOCO and fairness stages.")
    parser.add_argument("--run-ensemble", action="store_true", help="Run hybrid ensemble benchmark (MLP + XGBoost + RF + LogReg).")
    parser.add_argument("--run-loco", action="store_true", help="Run Leave-One-Center-Out (LOCO) cross-hospital generalization.")
    parser.add_argument("--run-fairness", action="store_true", help="Run demographic fairness and disparity analysis.")
    parser.add_argument("--run-reading-list", action="store_true", help="Run full AI Thesis reading list suite.")
    parser.add_argument("--run-tests", action="store_true", help="Run pytest before the pipeline (CI already runs it).")
    parser.add_argument("--skip-tests", action="store_true", help=argparse.SUPPRESS)  # kept for old command lines
    return parser


def run_full_pipeline(args: argparse.Namespace) -> None:
    if args.run_tests and not args.skip_tests:
        run_step("test")

    print("\nStarting full pipeline: prepare data -> EDA -> train")
    dataset_name = Path(args.dataset).name
    if dataset_name == "cleaned_heart_disease.csv":
        run_step("clean")
    elif dataset_name == MULTI_CENTER_DATASET:
        run_step("prepare-combined")  # rebuilt from data/raw (downloads only if missing)
    else:
        print(f"\n>>> Using custom dataset as-is: {args.dataset}")
    run_step("eda", ["--dataset", args.dataset, "--output-dir", args.output_dir])

    common = ["--seed", str(args.seed), "--dataset", args.dataset, "--output-dir", args.output_dir]
    train_args = [
        "--optimizer", args.optimizer,
        "--loss", args.loss,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--device", args.device,
        "--weight-decay", str(args.weight_decay),
        "--dropout", str(args.dropout),
        "--patience", str(args.patience),
        "--val-size", str(args.val_size),
        "--cv-folds", str(args.cv_folds),
        "--cv-repeats", str(args.cv_repeats),
        *common,
    ]
    if args.save_checkpoint:
        train_args.append("--save-checkpoint")
    if args.save_confusion:
        train_args.append("--save-confusion")
    if args.run_xai:
        train_args.append("--run-xai")
    run_step("train-mlp", train_args)

    if args.run_baselines:
        baseline_args = [
            "--xgb-early-stopping-rounds", str(args.xgb_early_stopping_rounds),
            "--xgb-eval-size", str(args.xgb_eval_size),
            "--xgb-device", args.xgb_device,
            "--cv-folds", str(args.cv_folds),
            "--cv-repeats", str(args.cv_repeats),
            "--n-jobs", str(args.n_jobs),
            *common,
        ]
        if args.run_xai:
            baseline_args.append("--run-xai")
        run_step("baselines", baseline_args)

    if args.run_advancements or args.run_ensemble:
        run_step("ensemble", ["--cv-folds", str(args.cv_folds), "--n-jobs", str(args.n_jobs), *common])

    # LOCO needs hospital sites, and the fairness audit needs subgroup sizes that only
    # the 918-patient multi-center cohort provides, so both always use that cohort.
    multi_center = ["--seed", str(args.seed), "--dataset", MULTI_CENTER_DATASET, "--output-dir", args.output_dir]
    if args.run_advancements or args.run_loco:
        run_step("loco", ["--n-jobs", str(args.n_jobs), *multi_center])

    if args.run_advancements or args.run_fairness:
        run_step("fairness", ["--n-jobs", str(args.n_jobs), *multi_center])

    if args.run_reading_list:
        run_step(
            "reading-list",
            [
                "--heart-csv", args.dataset,
                "--output-dir", args.output_dir,
                "--seed", str(args.seed),
                "--device", args.device,
                "--cv-folds", str(args.cv_folds),
                "--cv-repeats", str(args.cv_repeats),
                "--n-jobs", str(args.n_jobs),
            ],
        )

    # Paired DeLong tests across every model that scored the shared test split.
    run_step("compare-models", ["--output-dir", args.output_dir, "--dataset", args.dataset])

    if args.update_thesis:
        run_step("update-thesis", ["--task", "heart", "--heart-dataset", args.dataset])
    if args.update_thesis_all:
        run_step("update-thesis", ["--task", "all", "--heart-dataset", args.dataset])

    print("\nPipeline completed successfully.")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in STEPS:
        run_step(argv[0], argv[1:])
        return
    run_full_pipeline(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    main()
