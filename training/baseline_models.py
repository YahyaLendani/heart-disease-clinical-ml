"""Classical baselines (Logistic Regression, Random Forest, XGBoost) for the heart cohort.

Evaluation protocol
-------------------
* 60/20/20 stratified train / validation / test split.
* Hyperparameters chosen by repeated stratified k-fold CV on the training split.
  The CV scores of the chosen configuration are reused for reporting, so no
  configuration is cross-validated twice.
* Decision thresholds (F1-optimal, Youden-optimal) are tuned on the validation
  split and only *applied* to the test split.
* Every test metric carries a stratified bootstrap 95% confidence interval.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean_data import build_preprocessing_pipeline
from evaluation.clinical_metrics import (
    compute_clinical_metrics,
    plot_calibration_curves,
    plot_decision_curve,
    plot_precision_recall_curves,
)
from evaluation.cv_utils import (
    add_bootstrap_cis,
    binary_metrics,
    multi_metric_scorer,
    summarize_cv_scores,
    to_markdown_table,
)
from project_paths import PRIMARY_HEART_DATASET, resolve_data_path, resolve_output_dir
from training.device_utils import (
    DEVICE_CHOICES,
    align_xgb_inference_device,
    cv_n_jobs_for,
    resolve_xgb_device,
)

# Tree models use one thread each: on a few hundred rows thread start-up costs
# more than it saves. Parallelism happens across CV folds instead.
RF_BASE_PARAMS = {"n_estimators": 300, "random_state": 42, "n_jobs": 1}
RF_PARAM_GRID = {
    "max_depth": [None, 6],
    "min_samples_leaf": [1, 5],
    "max_features": ["sqrt", 0.5],
}

XGB_BASE_PARAMS = {
    "n_estimators": 300,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": 1,
}
XGB_PARAM_GRID = {
    "max_depth": [2, 4],
    "learning_rate": [0.03, 0.1],
    "min_child_weight": [1, 5],
}


def expand_grid(base: dict, grid: dict) -> list[dict]:
    """All combinations of ``grid`` merged into ``base`` (8 configurations per model here)."""
    from sklearn.model_selection import ParameterGrid

    return [{**base, **combo} for combo in ParameterGrid(grid)]


TEST_METRIC_COLUMNS = {
    "accuracy": "test_accuracy",
    "f1": "test_f1",
    "roc_auc": "test_roc_auc",
    "pr_auc": "test_pr_auc",
    "brier": "test_brier_score",
}


def load_data(dataset_path: str = PRIMARY_HEART_DATASET) -> tuple[pd.DataFrame, pd.Series]:
    cleaned_path = resolve_data_path(dataset_path)
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
    X = df.drop(columns=drop_cols)
    y = df["target"]
    return X, y


def build_preprocessors(X: pd.DataFrame) -> tuple:
    """Scaled and unscaled preprocessing pipelines (imputation is fit per fold)."""
    preprocess_scaled, _, _ = build_preprocessing_pipeline(X, scale_numeric=True)
    preprocess_unscaled, _, _ = build_preprocessing_pipeline(X, scale_numeric=False)
    return preprocess_scaled, preprocess_unscaled


def holdout_result(
    name: str,
    y_test: pd.Series,
    y_proba: np.ndarray,
    y_val: pd.Series,
    y_val_proba: np.ndarray,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Test metrics at 0.5, validation-tuned thresholds, and bootstrap CIs."""
    base = binary_metrics(y_test, y_proba)
    clinical = compute_clinical_metrics(y_test, y_proba, y_val_true=y_val, y_val_prob=y_val_proba)
    row = {
        "model": name,
        **{column: base[metric] for metric, column in TEST_METRIC_COLUMNS.items()},
        "optimal_threshold_f1": clinical["optimal_threshold_f1"],
        "f1_at_optimal_threshold": clinical["f1_at_optimal"],
        "optimal_threshold_youden": clinical["optimal_threshold_youden"],
        "youden_j": clinical["youden_j"],
    }
    add_bootstrap_cis(row, y_test, y_proba, seed=seed, name_map=TEST_METRIC_COLUMNS)

    if verbose:
        print(f"\n=== {name} ===")
        print(
            f"Accuracy: {row['test_accuracy']:.4f} | F1: {row['test_f1']:.4f} | "
            f"ROC-AUC: {row['test_roc_auc']:.4f} "
            f"[{row['test_roc_auc_ci_low']:.4f}, {row['test_roc_auc_ci_high']:.4f}] | "
            f"PR-AUC: {row['test_pr_auc']:.4f} | Brier: {row['test_brier_score']:.4f}"
        )
        print(
            f"Validation-tuned threshold (F1): {row['optimal_threshold_f1']:.3f} "
            f"-> test F1 {row['f1_at_optimal_threshold']:.4f} | "
            f"Validation-tuned threshold (Youden): {row['optimal_threshold_youden']:.3f} "
            f"-> test J {row['youden_j']:.4f}"
        )
        print(
            classification_report(
                y_test, (np.asarray(y_proba) >= 0.5).astype(int), target_names=["healthy", "sick"]
            )
        )
    return row


def evaluate_model(
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    seed: int = 42,
) -> tuple[dict, np.ndarray]:
    align_xgb_inference_device(model)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_val_proba = model.predict_proba(X_val)[:, 1]
    return holdout_result(name, y_test, y_proba, y_val, y_val_proba, seed=seed), y_proba


def cross_validate_model(
    name: str,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    seed: int = 42,
    n_jobs: int = -1,
    cv_repeats: int = 1,
) -> dict:
    """Repeated stratified k-fold CV; each fold predicts once and scores every metric."""
    cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=seed)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=multi_metric_scorer(before_predict=align_xgb_inference_device),
        n_jobs=cv_n_jobs_for(model, n_jobs),
    )
    summary = summarize_cv_scores(scores)
    summary["cv_n_folds_total"] = cv_folds * cv_repeats

    print(
        f"CV ({cv_folds}-fold x {cv_repeats}) for {name}: "
        f"Accuracy={summary['cv_accuracy_mean']:.4f}±{summary['cv_accuracy_std']:.4f}, "
        f"F1={summary['cv_f1_mean']:.4f}±{summary['cv_f1_std']:.4f}, "
        f"ROC-AUC={summary['cv_roc_auc_mean']:.4f}±{summary['cv_roc_auc_std']:.4f}, "
        f"PR-AUC={summary['cv_pr_auc_mean']:.4f}±{summary['cv_pr_auc_std']:.4f}, "
        f"Brier={summary['cv_brier_mean']:.4f}±{summary['cv_brier_std']:.4f}"
    )
    return summary


def select_best_params(
    model_name: str,
    preprocess: ColumnTransformer,
    model_cls,
    candidates: list[dict],
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    seed: int = 42,
    n_jobs: int = -1,
    cv_repeats: int = 1,
) -> tuple[dict, dict, pd.DataFrame]:
    """Pick the candidate configuration with the highest mean CV ROC-AUC.

    Returns ``(selected_params, selected_cv_scores, search_table)``. The CV
    scores of the winner are reused for reporting (no second CV run). Because
    the same folds are used to select and to report, the winner's CV score is
    slightly optimistic; the untouched holdout split is the unbiased check.
    """
    print(f"\nSelecting {model_name} hyperparameters: {len(candidates)} configurations, {cv_folds}-fold x {cv_repeats} CV")

    rows = []
    best_params, best_scores = None, None
    for idx, params in enumerate(candidates, start=1):
        pipeline = Pipeline(steps=[("preprocess", preprocess), ("classifier", model_cls(**params))])
        scores = cross_validate_model(
            f"{model_name} config {idx}", pipeline, X, y, cv_folds,
            seed=seed, n_jobs=n_jobs, cv_repeats=cv_repeats,
        )
        tunable = {k: v for k, v in params.items() if k not in ("random_state", "n_jobs", "device", "tree_method")}
        rows.append({"config": idx, **{f"param_{k}": v for k, v in tunable.items()}, **scores})
        if best_scores is None or scores["cv_roc_auc_mean"] > best_scores["cv_roc_auc_mean"]:
            best_params, best_scores = params, scores

    table = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False).reset_index(drop=True)
    print(f"Selected {model_name}: config {int(table.iloc[0]['config'])} (CV ROC-AUC {best_scores['cv_roc_auc_mean']:.4f})")
    return best_params, best_scores, table


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=25)
    parser.add_argument("--xgb-eval-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3, help="Repeats of stratified k-fold CV.")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--update-thesis", action="store_true")
    parser.add_argument("--xgb-device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--model", choices=["all", "logreg", "rf", "xgb"], default="all")
    parser.add_argument("--dataset", type=str, default=PRIMARY_HEART_DATASET, help="Dataset CSV (default: 918-patient 4-center cohort).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility across splits and models.")
    parser.add_argument("--save-run-metrics", action="store_true")
    parser.add_argument("--comparison-file", type=str, default="artifacts/separate_model_runs.csv")
    parser.add_argument("--run-xai", action="store_true", help="Run SHAP explainability analysis after training.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="CPU workers for parallel CV folds (-1 = all cores).")
    return parser


def append_run_metrics(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if output_path.exists():
        existing = pd.read_csv(output_path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(output_path, index=False)


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")
    if not (0 < val_size < 1):
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    relative_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_val,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def save_xgb_learning_curve(evals_result: dict, output_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping XGBoost learning-curve visualization.")
        return

    if not evals_result:
        return

    metric_values = None
    for metrics in evals_result.values():
        if "logloss" in metrics:
            metric_values = metrics["logloss"]
            break

    if metric_values is None:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(metric_values) + 1), metric_values, label="Validation Logloss")
    ax.set_title("XGBoost Early-Stopping Curve")
    ax.set_xlabel("Boosting Round")
    ax.set_ylabel("Logloss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved XGBoost learning curve: {output_path}")


def write_comparison_tables(summary: pd.DataFrame, output_dir: Path, stem: str) -> tuple[Path, Path]:
    csv_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    summary.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown_table(summary), encoding="utf-8")
    return csv_path, md_path


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = build_arg_parser().parse_args(argv)
    X, y = load_data(args.dataset)
    xgb_device = resolve_xgb_device(args.xgb_device)
    print(f"XGBoost device: {xgb_device}")

    run_logreg = args.model in {"all", "logreg"}
    run_rf = args.model in {"all", "rf"}
    run_xgb = args.model in {"all", "xgb"}

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X, y, test_size=args.test_size, val_size=args.val_size, random_state=args.seed
    )
    print(f"Data split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    cv_kwargs = {"seed": args.seed, "n_jobs": args.n_jobs, "cv_repeats": args.cv_repeats}

    preprocess_scaled, preprocess_unscaled = build_preprocessors(X)
    output_dir = resolve_output_dir(args.output_dir)

    models: list[tuple[str, Pipeline]] = []
    cv_lookup: dict[str, dict] = {}

    if run_logreg:
        logreg = Pipeline(
            steps=[
                ("preprocess", preprocess_scaled),
                ("classifier", LogisticRegression(max_iter=2000, random_state=args.seed)),
            ]
        )
        print(f"\n=== Cross-Validation ({args.cv_folds}-fold x {args.cv_repeats}, Stratified) ===")
        cv_lookup["Logistic Regression"] = cross_validate_model(
            "Logistic Regression", logreg, X_train, y_train, args.cv_folds, **cv_kwargs
        )
        models.append(("Logistic Regression", logreg))

    if run_rf:
        selected_rf_params, cv_lookup["Random Forest"], rf_search = select_best_params(
            model_name="Random Forest",
            preprocess=preprocess_unscaled,
            model_cls=RandomForestClassifier,
            candidates=expand_grid({**RF_BASE_PARAMS, "random_state": args.seed}, RF_PARAM_GRID),
            X=X_train,
            y=y_train,
            cv_folds=args.cv_folds,
            **cv_kwargs,
        )
        rf_search.to_csv(output_dir / "hyperparameter_search_random_forest.csv", index=False)
        models.append(
            (
                "Random Forest",
                Pipeline(
                    steps=[
                        ("preprocess", preprocess_unscaled),
                        ("classifier", RandomForestClassifier(**selected_rf_params)),
                    ]
                ),
            )
        )

    selected_xgb_params = None
    if run_xgb:
        xgb_extra = {"tree_method": "hist", "device": xgb_device, "random_state": args.seed}
        selected_xgb_params, cv_lookup["XGBoost"], xgb_search = select_best_params(
            model_name="XGBoost",
            preprocess=preprocess_unscaled,
            model_cls=XGBClassifier,
            candidates=expand_grid({**XGB_BASE_PARAMS, **xgb_extra}, XGB_PARAM_GRID),
            X=X_train,
            y=y_train,
            cv_folds=args.cv_folds,
            **cv_kwargs,
        )
        xgb_search.to_csv(output_dir / "hyperparameter_search_xgboost.csv", index=False)

    results: list[dict] = []
    model_probas: dict[str, np.ndarray] = {}
    for name, model in models:
        model.fit(X_train, y_train)
        model_result, y_proba = evaluate_model(name, model, X_test, y_test, X_val, y_val, seed=args.seed)
        model_result.update(cv_lookup[name])
        results.append(model_result)
        model_probas[name] = y_proba

    xgb_model = None
    if run_xgb and selected_xgb_params is not None:
        print(
            f"\nXGBoost early stopping: enabled (rounds={args.xgb_early_stopping_rounds}, "
            f"validation_set=explicit val split)"
        )
        xgb_preprocessor = preprocess_unscaled
        X_train_xgb_np = xgb_preprocessor.fit_transform(X_train)
        X_val_xgb_np = xgb_preprocessor.transform(X_val)
        X_test_xgb_np = xgb_preprocessor.transform(X_test)

        xgb_model = XGBClassifier(
            **selected_xgb_params,
            early_stopping_rounds=args.xgb_early_stopping_rounds,
        )
        xgb_model.fit(X_train_xgb_np, y_train, eval_set=[(X_val_xgb_np, y_val)], verbose=False)
        save_xgb_learning_curve(xgb_model.evals_result(), output_dir / "xgb_learning_curve.png")

        align_xgb_inference_device(xgb_model)
        y_proba_xgb = xgb_model.predict_proba(X_test_xgb_np)[:, 1]
        y_val_proba_xgb = xgb_model.predict_proba(X_val_xgb_np)[:, 1]
        print(f"Best iteration (early stopping): {xgb_model.best_iteration}")
        xgb_row = holdout_result("XGBoost", y_test, y_proba_xgb, y_val, y_val_proba_xgb, seed=args.seed)
        xgb_row.update(cv_lookup["XGBoost"])
        results.append(xgb_row)
        model_probas["XGBoost"] = y_proba_xgb

    # --- SHAP Explainability ---
    if args.run_xai:
        try:
            from evaluation.xai_analysis import generate_shap_tree, save_xai_summary_csv

            xai_results: list[dict] = []
            fitted_pipelines = dict(models)

            if "Random Forest" in fitted_pipelines:
                print("\n=== SHAP Analysis: Random Forest (Heart) ===")
                xai_results.append(
                    generate_shap_tree(
                        model_pipeline=fitted_pipelines["Random Forest"],
                        X_train=X_train,
                        X_test=X_test,
                        feature_names=None,
                        dataset_name="heart",
                        model_name="random_forest",
                        output_dir=output_dir,
                    )
                )

            if xgb_model is not None:
                # Preprocessor was already fit during XGBoost training
                xgb_pipeline_for_shap = Pipeline(
                    steps=[("preprocess", preprocess_unscaled), ("classifier", xgb_model)]
                )
                print("\n=== SHAP Analysis: XGBoost (Heart) ===")
                xai_results.append(
                    generate_shap_tree(
                        model_pipeline=xgb_pipeline_for_shap,
                        X_train=X_train,
                        X_test=X_test,
                        feature_names=None,
                        dataset_name="heart",
                        model_name="xgboost",
                        output_dir=output_dir,
                    )
                )

            if xai_results:
                save_xai_summary_csv(xai_results, output_dir)

        except Exception as xai_error:
            print(f"Warning: SHAP analysis failed: {xai_error}")
            import traceback

            traceback.print_exc()

    if model_probas:
        from evaluation.model_comparison import save_test_predictions

        save_test_predictions(output_dir, "baselines", args.dataset, X_test.index, y_test, model_probas)

    # --- Calibration & PR Curves ---
    figures_dir = output_dir / "figures"
    if model_probas:
        plot_calibration_curves(y_test, model_probas, "heart", figures_dir)
        plot_precision_recall_curves(y_test, model_probas, "heart", figures_dir)
        plot_decision_curve(y_test, model_probas, "heart", figures_dir)

    summary = pd.DataFrame(results).sort_values(
        by=["cv_roc_auc_mean", "test_roc_auc", "test_f1"],
        ascending=False,
    )

    print("\n=== Baseline Model Comparison ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    csv_path, md_path = write_comparison_tables(summary, output_dir, "baseline_comparison")
    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison Markdown: {md_path}")

    if args.save_run_metrics:
        comparison_path = resolve_output_dir(Path(args.comparison_file).parent) / Path(args.comparison_file).name
        rows = [
            {
                "source": "heart_baseline",
                "model": row["model"],
                "test_accuracy": row["test_accuracy"],
                "test_f1": row["test_f1"],
                "test_roc_auc": row["test_roc_auc"],
                "best_val_accuracy": row.get("cv_accuracy_mean", float("nan")),
                "best_epoch": float("nan"),
            }
            for _, row in summary.iterrows()
        ]
        append_run_metrics(comparison_path, rows)
        print(f"Saved/updated separate run metrics: {comparison_path}")

    if args.update_thesis:
        from evaluation import update_thesis_results

        update_thesis_results.main(
            [
                "--task",
                "heart",
                "--heart-dataset",
                args.dataset,
                "--heart-comparison",
                str(Path(args.output_dir) / "baseline_comparison.csv"),
            ]
        )

    return summary


if __name__ == "__main__":
    main()
