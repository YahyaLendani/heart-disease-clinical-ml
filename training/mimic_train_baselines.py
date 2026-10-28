import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from xgboost.core import XGBoostError
import torch

try:
    from clean_data import build_preprocessing_pipeline
    from evaluation.clinical_metrics import (
        compute_clinical_metrics,
        plot_calibration_curves,
        plot_precision_recall_curves,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from clean_data import build_preprocessing_pipeline
    from evaluation.clinical_metrics import (
        compute_clinical_metrics,
        plot_calibration_curves,
        plot_precision_recall_curves,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline models on MIMIC hospital mortality dataset.")
    parser.add_argument("--input", type=str, default="mimic_hospital_mortality.csv")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--update-thesis", action="store_true")
    parser.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--run-xai", action="store_true", help="Run SHAP explainability analysis after training.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of CPU workers for parallel CV (-1 = all cores).")
    return parser


def resolve_xgb_device(requested_device: str) -> str:
    if requested_device == "cpu":
        return "cpu"

    if requested_device == "auto" and not torch.cuda.is_available():
        return "cpu"

    try:
        test_model = XGBClassifier(
            n_estimators=1,
            max_depth=1,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cuda",
            n_jobs=1,
        )
        X_test_gpu = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
        y_test_gpu = np.array([0, 1, 0, 1], dtype=np.int32)
        test_model.fit(X_test_gpu, y_test_gpu, verbose=False)
        return "cuda"
    except (XGBoostError, ValueError) as error:
        if requested_device == "cuda":
            raise RuntimeError(
                "CUDA was requested for XGBoost, but GPU execution is unavailable. "
                "Ensure your XGBoost build supports CUDA."
            ) from error
        print("XGBoost GPU not available; falling back to CPU.")
        return "cpu"


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
) -> tuple[dict, np.ndarray]:
    align_xgb_inference_device(model)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

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

    return {
        "model": name,
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
    }, y_prob


def evaluate_cv(
    name: str,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    folds: int,
    groups: pd.Series | None = None,
    n_jobs: int = -1,
) -> dict:
    def _score_accuracy(estimator, X_batch, y_batch):
        align_xgb_inference_device(estimator)
        return accuracy_score(y_batch, estimator.predict(X_batch))

    def _score_f1(estimator, X_batch, y_batch):
        align_xgb_inference_device(estimator)
        return f1_score(y_batch, estimator.predict(X_batch), zero_division=0)

    def _score_recall(estimator, X_batch, y_batch):
        align_xgb_inference_device(estimator)
        return recall_score(y_batch, estimator.predict(X_batch), zero_division=0)

    def _score_precision(estimator, X_batch, y_batch):
        align_xgb_inference_device(estimator)
        return precision_score(y_batch, estimator.predict(X_batch), zero_division=0)

    def _score_roc_auc(estimator, X_batch, y_batch):
        align_xgb_inference_device(estimator)
        return roc_auc_score(y_batch, estimator.predict_proba(X_batch)[:, 1])

    def _score_average_precision(estimator, X_batch, y_batch):
        align_xgb_inference_device(estimator)
        return average_precision_score(y_batch, estimator.predict_proba(X_batch)[:, 1])

    def _score_brier(estimator, X_batch, y_batch):
        align_xgb_inference_device(estimator)
        return brier_score_loss(y_batch, estimator.predict_proba(X_batch)[:, 1])

    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=42)
    else:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)

    clf = model.named_steps.get("classifier") if isinstance(model, Pipeline) else model
    cv_jobs = 1 if getattr(clf, "n_jobs", 1) not in (None, 1) else (os.cpu_count() or 1 if n_jobs == -1 else n_jobs)

    scores = cross_validate(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring={
            "accuracy": _score_accuracy,
            "f1": _score_f1,
            "recall": _score_recall,
            "precision": _score_precision,
            "roc_auc": _score_roc_auc,
            "average_precision": _score_average_precision,
            "brier": _score_brier,
        },
        n_jobs=cv_jobs,
    )

    return {
        "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
        "cv_accuracy_std": float(scores["test_accuracy"].std()),
        "cv_f1_expired_mean": float(scores["test_f1"].mean()),
        "cv_f1_expired_std": float(scores["test_f1"].std()),
        "cv_recall_expired_mean": float(scores["test_recall"].mean()),
        "cv_recall_expired_std": float(scores["test_recall"].std()),
        "cv_precision_expired_mean": float(scores["test_precision"].mean()),
        "cv_precision_expired_std": float(scores["test_precision"].std()),
        "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
        "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
        "cv_pr_auc_mean": float(scores["test_average_precision"].mean()),
        "cv_pr_auc_std": float(scores["test_average_precision"].std()),
        "cv_brier_mean": float(scores["test_brier"].mean()),
        "cv_brier_std": float(scores["test_brier"].std()),
    }


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

        # Explicit verification against patient leakage
        assert set(groups.iloc[train_idx]).isdisjoint(set(groups.iloc[test_idx])), (
            "CRITICAL LEAKAGE: subject_id overlap between train and test!"
        )
        assert set(groups.iloc[val_idx]).isdisjoint(set(groups.iloc[test_idx])), (
            "CRITICAL LEAKAGE: subject_id overlap between val and test!"
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


def align_xgb_inference_device(model) -> None:
    classifier = model
    if isinstance(model, Pipeline):
        classifier = model.named_steps.get("classifier")

    if not isinstance(classifier, XGBClassifier):
        return

    xgb_params = classifier.get_xgb_params()
    if xgb_params.get("device") != "cuda":
        return

    try:
        classifier.get_booster().set_param({"device": "cpu"})
    except (XGBoostError, ValueError):
        return


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path_arg = Path(args.input)
    input_path = input_path_arg if input_path_arg.is_absolute() else PROJECT_ROOT / input_path_arg
    X, y, subject_ids = load_data(input_path)
    xgb_device = resolve_xgb_device(args.xgb_device)
    print(f"XGBoost device: {xgb_device}")

    X_train, X_val, X_test, y_train, y_val, y_test, groups_train = split_train_val_test(
        X,
        y,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=42,
        groups=subject_ids,
    )
    print(
        f"Data split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
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
                            random_state=42,
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
                            random_state=42,
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
                            random_state=42,
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
        cv_result = evaluate_cv(name, model, X_train, y_train, args.cv_folds, groups=groups_train, n_jobs=args.n_jobs)
        print(
            f"{name}: Accuracy={cv_result['cv_accuracy_mean']:.4f}±{cv_result['cv_accuracy_std']:.4f}, "
            f"F1(expired)={cv_result['cv_f1_expired_mean']:.4f}±{cv_result['cv_f1_expired_std']:.4f}, "
            f"Recall(expired)={cv_result['cv_recall_expired_mean']:.4f}±{cv_result['cv_recall_expired_std']:.4f}, "
            f"ROC-AUC={cv_result['cv_roc_auc_mean']:.4f}±{cv_result['cv_roc_auc_std']:.4f}, "
            f"PR-AUC={cv_result['cv_pr_auc_mean']:.4f}±{cv_result['cv_pr_auc_std']:.4f}, "
            f"Brier={cv_result['cv_brier_mean']:.4f}±{cv_result['cv_brier_std']:.4f}"
        )

        model.fit(X_train, y_train)
        holdout_result, y_proba = evaluate_holdout(name, model, X_test, y_test, X_val=X_val, y_val=y_val)
        holdout_result.update(cv_result)
        results.append(holdout_result)
        fitted_models[name] = model
        model_probas[name] = y_proba

    summary = pd.DataFrame(results).sort_values(
        by=["test_pr_auc", "test_f1_expired", "test_recall_expired", "test_roc_auc"],
        ascending=False,
    )

    print("\n=== MIMIC Baseline Comparison ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "mimic_baseline_comparison.csv"
    md_path = output_dir / "mimic_baseline_comparison.md"

    summary.to_csv(csv_path, index=False)

    display = summary.copy()
    for col in display.columns:
        if col != "model":
            display[col] = display[col].map(lambda value: f"{value:.4f}")

    headers = display.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    # --- Calibration & PR Curves ---
    figures_dir = PROJECT_ROOT / args.output_dir / "figures"
    if model_probas:
        plot_calibration_curves(y_test, model_probas, "mimic", figures_dir)
        plot_precision_recall_curves(y_test, model_probas, "mimic", figures_dir)

    print(f"Saved MIMIC comparison CSV: {csv_path}")
    print(f"Saved MIMIC comparison Markdown: {md_path}")

    # --- SHAP Explainability ---
    if args.run_xai:
        try:
            try:
                from evaluation.xai_analysis import generate_shap_tree, save_xai_summary_csv
            except ModuleNotFoundError:
                sys.path.append(str(PROJECT_ROOT))
                from evaluation.xai_analysis import generate_shap_tree, save_xai_summary_csv

            xai_output_dir = PROJECT_ROOT / args.output_dir
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
        updater_path = PROJECT_ROOT / "evaluation" / "update_thesis_results.py"
        if not updater_path.exists():
            raise FileNotFoundError("update_thesis_results.py not found.")

        update_command = [
            sys.executable,
            str(updater_path),
            "--dataset",
            args.input,
            "--comparison",
            str(Path(args.output_dir) / "mimic_baseline_comparison.csv"),
        ]
        result = subprocess.run(update_command, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"update_thesis_results.py failed with exit code {result.returncode}"
            )


if __name__ == "__main__":
    main()
