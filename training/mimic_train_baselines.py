"""Classical baselines for MIMIC-IV in-hospital mortality.

The bundled MIMIC-IV demo extract is very small (275 admissions, 15 deaths),
so a 20% test split holds only a handful of deaths. Results therefore carry
bootstrap confidence intervals, CV is repeated, and the script warns when the
test split has too few events for a stable ranking.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean_data import build_preprocessing_pipeline
from evaluation.clinical_metrics import (
    compute_clinical_metrics,
    plot_calibration_curves,
    plot_precision_recall_curves,
)
from evaluation.cv_utils import (
    add_bootstrap_cis,
    multi_metric_scorer,
    summarize_cv_scores,
    to_markdown_table,
)
from project_paths import cohort_label, resolve_data_path, resolve_output_dir
from training.device_utils import (
    DEVICE_CHOICES,
    align_xgb_inference_device,
    cv_n_jobs_for,
    resolve_xgb_device,
)

CV_METRICS = ("accuracy", "f1", "recall", "precision", "roc_auc", "pr_auc", "brier")
MIN_TEST_EVENTS_FOR_RANKING = 10


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline models on MIMIC hospital mortality dataset.")
    parser.add_argument("--input", type=str, default="mimic_hospital_mortality.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--update-thesis", action="store_true")
    parser.add_argument("--cv-repeats", type=int, default=3, help="Repeats of (group) stratified k-fold CV.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--xgb-device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--run-xai", action="store_true", help="Run SHAP explainability analysis after training.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of CPU workers for parallel CV (-1 = all cores).")
    return parser


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}. Run mimic_prepare_dataset.py first.")

    df = pd.read_csv(path)
    y = pd.to_numeric(df["hospital_expire_flag"], errors="coerce").fillna(0).astype(int)
    subject_ids = df["subject_id"] if "subject_id" in df.columns else None
    X = df.drop(columns=["hospital_expire_flag", "subject_id", "hadm_id"], errors="ignore")
    return X, y, subject_ids


def make_preprocessor(X: pd.DataFrame):
    """Build preprocessing pipeline with imputation for MIMIC data.

    Uses the centralized build_preprocessing_pipeline() factory from
    clean_data.py so that imputation is computed per-fold.
    """
    preprocessor, _, _ = build_preprocessing_pipeline(X, scale_numeric=True)
    return preprocessor


def evaluate_holdout(
    name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    seed: int = 42,
) -> tuple[dict, np.ndarray]:
    align_xgb_inference_device(model)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    recall_expired = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    precision_expired = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    # Clinical metrics: tune threshold on validation set when available to prevent test-set threshold leakage
    if X_val is not None and y_val is not None:
        y_val_prob = model.predict_proba(X_val)[:, 1]
        clinical = compute_clinical_metrics(y_test, y_prob, y_val_true=y_val, y_val_prob=y_val_prob)
    else:
        clinical = compute_clinical_metrics(y_test, y_prob)

    print(f"\n=== {name} ===")
    print(
        f"Accuracy: {accuracy:.4f} | F1(expired): {f1:.4f} | "
        f"Recall(expired): {recall_expired:.4f} | Precision(expired): {precision_expired:.4f} | "
        f"ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f} | Brier: {clinical['brier_score']:.4f}"
    )
    print(
        f"Optimal threshold (F1): {clinical['optimal_threshold_f1']:.3f} "
        f"(F1={clinical['f1_at_optimal']:.4f}) | "
        f"Optimal threshold (Youden): {clinical['optimal_threshold_youden']:.3f} "
        f"(J={clinical['youden_j']:.4f})"
    )
    print(classification_report(y_test, y_pred, target_names=["survived", "expired"], zero_division=0))

    row = {
        "model": name,
        "test_events": int(np.sum(y_test)),
        "test_accuracy": accuracy,
        "test_f1_expired": f1,
        "test_recall_expired": recall_expired,
        "test_precision_expired": precision_expired,
        "test_roc_auc": roc_auc,
        "test_pr_auc": pr_auc,
        "test_brier_score": clinical["brier_score"],
        "optimal_threshold_f1": clinical["optimal_threshold_f1"],
        "f1_at_optimal_threshold": clinical["f1_at_optimal"],
        "optimal_threshold_youden": clinical["optimal_threshold_youden"],
        "youden_j": clinical["youden_j"],
    }
    add_bootstrap_cis(
        row,
        y_test,
        y_prob,
        seed=seed,
        name_map={
            "accuracy": "test_accuracy",
            "f1": "test_f1_expired",
            "roc_auc": "test_roc_auc",
            "pr_auc": "test_pr_auc",
            "brier": "test_brier_score",
        },
    )
    print(
        f"95% bootstrap CI -> ROC-AUC [{row['test_roc_auc_ci_low']:.3f}, {row['test_roc_auc_ci_high']:.3f}] | "
        f"PR-AUC [{row['test_pr_auc_ci_low']:.3f}, {row['test_pr_auc_ci_high']:.3f}]"
    )
    return row, y_prob


def evaluate_cv(
    name: str,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int,
    groups: pd.Series | None = None,
    n_jobs: int = -1,
    repeats: int = 1,
    seed: int = 42,
) -> dict:
    """Repeated (group-aware) stratified k-fold CV with one prediction per fold."""
    scorer = multi_metric_scorer(extended=True, before_predict=align_xgb_inference_device)
    cv_jobs = cv_n_jobs_for(model, n_jobs)
    collected: dict[str, list[float]] = {}
    for repeat in range(repeats):
        if groups is not None:
            cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        else:
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + repeat)
        scores = cross_validate(model, X, y, groups=groups, cv=cv, scoring=scorer, n_jobs=cv_jobs)
        for key, values in scores.items():
            if key.startswith("test_"):
                collected.setdefault(key, []).extend(np.asarray(values, dtype=float).tolist())

    summary = summarize_cv_scores(collected, metrics=CV_METRICS)
    # Keep the historical column names used by the thesis updater.
    renamed = {}
    for key, value in summary.items():
        for old, new in (("_f1_", "_f1_expired_"), ("_recall_", "_recall_expired_"), ("_precision_", "_precision_expired_")):
            if old in key:
                key = key.replace(old, new)
        renamed[key] = value
    renamed["cv_n_folds_total"] = folds * repeats
    return renamed


def split_train_val_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    val_size: float,
    random_state: int = 42,
    groups: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series | None]:
    """Split data into train/val/test sets.

    When *groups* is provided (e.g. subject_id for MIMIC), splitting is
    done using StratifiedGroupKFold so that all admissions belonging to the
    same patient stay in the same partition while maintaining class balance.
    This guarantees zero patient-level data leakage.
    """
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")
    if not (0 < val_size < 1):
        raise ValueError("val_size must be between 0 and 1.")
    if test_size + val_size >= 1:
        raise ValueError("test_size + val_size must be < 1.")

    if groups is not None:
        # Use StratifiedGroupKFold to ensure BOTH patient isolation and minority-class stratification
        n_splits_test = max(2, int(round(1.0 / test_size)))
        sgkf_test = StratifiedGroupKFold(
            n_splits=n_splits_test, shuffle=True, random_state=random_state
        )
        train_val_idx, test_idx = next(sgkf_test.split(X, y, groups=groups))
        X_train_val = X.iloc[train_val_idx]
        X_test = X.iloc[test_idx]
        y_train_val = y.iloc[train_val_idx]
        y_test = y.iloc[test_idx]
        groups_train_val = groups.iloc[train_val_idx]

        relative_val_size = val_size / (1 - test_size)
        n_splits_val = max(2, int(round(1.0 / relative_val_size)))
        sgkf_val = StratifiedGroupKFold(
            n_splits=n_splits_val, shuffle=True, random_state=random_state
        )
        train_idx, val_idx = next(
            sgkf_val.split(X_train_val, y_train_val, groups=groups_train_val)
        )
        X_train = X_train_val.iloc[train_idx]
        X_val = X_train_val.iloc[val_idx]
        y_train = y_train_val.iloc[train_idx]
        y_val = y_train_val.iloc[val_idx]
        groups_train = groups_train_val.iloc[train_idx]

        # Explicit verification against patient leakage. train_idx/val_idx are
        # positions inside the train+val subset, so compare actual subject IDs.
        test_subjects = set(groups.iloc[test_idx])
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
    else:
        # --- Row-level stratified splitting (no group info) ---
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
        groups_train = None

    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train


def main(argv: list[str] | None = None) -> pd.DataFrame:
    args = build_arg_parser().parse_args(argv)
    input_path = resolve_data_path(args.input)
    X, y, subject_ids = load_data(input_path)
    xgb_device = resolve_xgb_device(args.xgb_device)
    print(f"XGBoost device: {xgb_device}")

    X_train, X_val, X_test, y_train, y_val, y_test, groups_train = split_train_val_test(
        X,
        y,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        groups=subject_ids,
    )
    print(
        f"Data split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    test_events = int(y_test.sum())
    print(f"Deaths per split -> train: {int(y_train.sum())}, val: {int(y_val.sum())}, test: {test_events}")
    if test_events < MIN_TEST_EVENTS_FOR_RANKING:
        print(
            f"WARNING: the test split contains only {test_events} deaths. Holdout ROC-AUC/PR-AUC are "
            "unstable at this size; compare models on the repeated-CV means and the bootstrap CIs, "
            "not on point estimates."
        )

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / max(positives, 1)
    print(
        f"Class imbalance (train): positives={positives}, negatives={negatives}, "
        f"scale_pos_weight={scale_pos_weight:.2f}"
    )

    preprocessor = make_preprocessor(X_train)

    models: list[tuple[str, Pipeline]] = [
        (
            "Logistic Regression",
            Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=2000,
                            random_state=args.seed,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        ),
        (
            "Random Forest",
            Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    (
                        "classifier",
                        RandomForestClassifier(
                            n_estimators=300,
                            max_depth=5,
                            min_samples_split=5,
                            random_state=args.seed,
                            class_weight="balanced_subsample",
                        ),
                    ),
                ]
            ),
        ),
        (
            "XGBoost",
            Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    (
                        "classifier",
                        XGBClassifier(
                            n_estimators=200,
                            max_depth=3,
                            learning_rate=0.05,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            eval_metric="logloss",
                            random_state=args.seed,
                            scale_pos_weight=scale_pos_weight,
                            tree_method="hist",
                            device=xgb_device,
                        ),
                    ),
                ]
            ),
        ),
    ]

    results: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}
    model_probas: dict[str, np.ndarray] = {}
    print(f"\n=== MIMIC Cross-Validation ({args.cv_folds}-fold) ===")
    for name, model in models:
        cv_result = evaluate_cv(
            name, model, X_train, y_train, args.cv_folds, groups=groups_train,
            n_jobs=args.n_jobs, repeats=args.cv_repeats, seed=args.seed,
        )
        print(
            f"{name}: Accuracy={cv_result['cv_accuracy_mean']:.4f}±{cv_result['cv_accuracy_std']:.4f}, "
            f"F1(expired)={cv_result['cv_f1_expired_mean']:.4f}±{cv_result['cv_f1_expired_std']:.4f}, "
            f"Recall(expired)={cv_result['cv_recall_expired_mean']:.4f}±{cv_result['cv_recall_expired_std']:.4f}, "
            f"ROC-AUC={cv_result['cv_roc_auc_mean']:.4f}±{cv_result['cv_roc_auc_std']:.4f}, "
            f"PR-AUC={cv_result['cv_pr_auc_mean']:.4f}±{cv_result['cv_pr_auc_std']:.4f}, "
            f"Brier={cv_result['cv_brier_mean']:.4f}±{cv_result['cv_brier_std']:.4f}"
        )

        model.fit(X_train, y_train)
        holdout_result, y_proba = evaluate_holdout(
            name, model, X_test, y_test, X_val=X_val, y_val=y_val, seed=args.seed
        )
        holdout_result.update(cv_result)
        results.append(holdout_result)
        fitted_models[name] = model
        model_probas[name] = y_proba

    summary = pd.DataFrame(results).sort_values(
        by=["cv_pr_auc_mean", "cv_roc_auc_mean", "test_pr_auc"],
        ascending=False,
    )

    print("\n=== MIMIC Baseline Comparison ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    output_dir = resolve_output_dir(args.output_dir)

    csv_path = output_dir / "mimic_baseline_comparison.csv"
    md_path = output_dir / "mimic_baseline_comparison.md"

    summary.to_csv(csv_path, index=False)

    md_path.write_text(to_markdown_table(summary), encoding="utf-8")

    # --- Calibration & PR Curves ---
    figures_dir = output_dir / "figures"
    if model_probas:
        plot_calibration_curves(y_test, model_probas, "mimic", figures_dir, cohort=cohort_label(args.input))
        plot_precision_recall_curves(y_test, model_probas, "mimic", figures_dir, cohort=cohort_label(args.input))

    print(f"Saved MIMIC comparison CSV: {csv_path}")
    print(f"Saved MIMIC comparison Markdown: {md_path}")

    # --- SHAP Explainability ---
    if args.run_xai:
        try:
            from evaluation.xai_analysis import generate_shap_tree, save_xai_summary_csv

            xai_output_dir = output_dir
            xai_results: list[dict] = []

            for model_label, shap_name in [("XGBoost", "xgboost"), ("Random Forest", "random_forest")]:
                if model_label in fitted_models:
                    print(f"\n=== SHAP Analysis: {model_label} (MIMIC) ===")
                    result = generate_shap_tree(
                        model_pipeline=fitted_models[model_label],
                        X_train=X_train,
                        X_test=X_test,
                        feature_names=None,
                        dataset_name="mimic",
                        model_name=shap_name,
                        output_dir=xai_output_dir,
                    )
                    xai_results.append(result)

            if xai_results:
                save_xai_summary_csv(xai_results, xai_output_dir)

        except Exception as xai_error:
            print(f"Warning: SHAP analysis failed: {xai_error}")
            import traceback
            traceback.print_exc()

    if args.update_thesis:
        from evaluation import update_thesis_results

        update_thesis_results.main(
            [
                "--task",
                "mimic",
                "--mimic-dataset",
                args.input,
                "--mimic-comparison",
                str(Path(args.output_dir) / "mimic_baseline_comparison.csv"),
            ]
        )

    return summary


if __name__ == "__main__":
    main()
