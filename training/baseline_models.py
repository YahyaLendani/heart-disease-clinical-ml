import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
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
        plot_decision_curve,
    )
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from clean_data import build_preprocessing_pipeline
    from evaluation.clinical_metrics import (
        compute_clinical_metrics,
        plot_calibration_curves,
        plot_precision_recall_curves,
        plot_decision_curve,
    )


PROJECT_ROOT = Path(__file__).resolve().parent.parent


RF_BASELINE_PARAMS = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "random_state": 42,
    "n_jobs": -1,
}

RF_CANDIDATE_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "min_samples_split": 5,
    "random_state": 42,
    "n_jobs": -1,
}

XGB_BASELINE_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

XGB_CANDIDATE_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}


def load_data(dataset_path: str = "cleaned_heart_disease.csv") -> tuple[pd.DataFrame, pd.Series]:
    cleaned_path = Path(dataset_path)
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

    drop_cols = [col for col in ["diagnosis", "disease_code", "target", "site"] if col in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["target"]
    return X, y


def build_preprocessors(X: pd.DataFrame) -> tuple:
    """Build scaled and unscaled preprocessing pipelines with imputation.

    Uses the centralized build_preprocessing_pipeline() factory from
    clean_data.py so that imputation (median/mode) is computed per-fold
    rather than globally, preventing data leakage.
    """
    preprocess_scaled, _, _ = build_preprocessing_pipeline(X, scale_numeric=True)
    preprocess_unscaled, _, _ = build_preprocessing_pipeline(X, scale_numeric=False)
    return preprocess_scaled, preprocess_unscaled


def evaluate_model(name: str, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[dict, np.ndarray]:
    align_xgb_inference_device(model)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    # Clinical metrics
    clinical = compute_clinical_metrics(y_test, y_proba)

    print(f"\n=== {name} ===")
    print(
        f"Accuracy: {accuracy:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | "
        f"PR-AUC: {clinical['pr_auc']:.4f} | Brier: {clinical['brier_score']:.4f}"
    )
    print(
        f"Optimal threshold (F1): {clinical['optimal_threshold_f1']:.3f} "
        f"(F1={clinical['f1_at_optimal']:.4f}) | "
        f"Optimal threshold (Youden): {clinical['optimal_threshold_youden']:.3f} "
        f"(J={clinical['youden_j']:.4f})"
    )
    print(classification_report(y_test, y_pred, target_names=["healthy", "sick"]))

    return {
        "model": name,
        "test_accuracy": accuracy,
        "test_f1": f1,
        "test_roc_auc": roc_auc,
        "test_pr_auc": clinical["pr_auc"],
        "test_brier_score": clinical["brier_score"],
        "optimal_threshold_f1": clinical["optimal_threshold_f1"],
        "f1_at_optimal_threshold": clinical["f1_at_optimal"],
        "optimal_threshold_youden": clinical["optimal_threshold_youden"],
        "youden_j": clinical["youden_j"],
    }, y_proba


def _resolve_cv_n_jobs(model: Pipeline, requested_n_jobs: int) -> int:
    """Avoid nested oversubscription and CUDA multiprocessing contention."""
    clf = model.named_steps.get("classifier") if isinstance(model, Pipeline) else model
    if isinstance(clf, XGBClassifier) and clf.get_xgb_params().get("device") == "cuda":
        return 1  # Keep GPU execution in main process
    clf_jobs = getattr(clf, "n_jobs", 1)
    if clf_jobs is not None and clf_jobs != 1:
        return 1  # Estimator already parallelizes internally across all cores
    return requested_n_jobs


def cross_validate_model(
    name: str,
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    seed: int = 42,
    n_jobs: int = -1,
) -> dict:
    def _score_accuracy(estimator, X_batch, y_batch):
        align_xgb_inference_device(estimator)
        return accuracy_score(y_batch, estimator.predict(X_batch))

    def _score_f1(estimator, X_batch, y_batch):
        align_xgb_inference_device(estimator)
        return f1_score(y_batch, estimator.predict(X_batch))

    def _score_roc_auc(estimator, X_batch, y_batch):
        align_xgb_inference_device(estimator)
        return roc_auc_score(y_batch, estimator.predict_proba(X_batch)[:, 1])

    def _score_pr_auc(estimator, X_batch, y_batch):
        align_xgb_inference_device(estimator)
        return average_precision_score(y_batch, estimator.predict_proba(X_batch)[:, 1])

    def _score_brier(estimator, X_batch, y_batch):
        align_xgb_inference_device(estimator)
        return brier_score_loss(y_batch, estimator.predict_proba(X_batch)[:, 1])

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cv_jobs = _resolve_cv_n_jobs(model, n_jobs)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "accuracy": _score_accuracy,
            "f1": _score_f1,
            "roc_auc": _score_roc_auc,
            "pr_auc": _score_pr_auc,
            "brier": _score_brier,
        },
        n_jobs=cv_jobs,
    )

    acc_mean = float(np.mean(scores["test_accuracy"]))
    acc_std = float(np.std(scores["test_accuracy"]))
    f1_mean = float(np.mean(scores["test_f1"]))
    f1_std = float(np.std(scores["test_f1"]))
    roc_mean = float(np.mean(scores["test_roc_auc"]))
    roc_std = float(np.std(scores["test_roc_auc"]))
    pr_auc_mean = float(np.mean(scores["test_pr_auc"]))
    pr_auc_std = float(np.std(scores["test_pr_auc"]))
    brier_mean = float(np.mean(scores["test_brier"]))
    brier_std = float(np.std(scores["test_brier"]))

    print(
        f"CV ({cv_folds}-fold) for {name}: "
        f"Accuracy={acc_mean:.4f}±{acc_std:.4f}, F1={f1_mean:.4f}±{f1_std:.4f}, "
        f"ROC-AUC={roc_mean:.4f}±{roc_std:.4f}, PR-AUC={pr_auc_mean:.4f}±{pr_auc_std:.4f}, "
        f"Brier={brier_mean:.4f}±{brier_std:.4f}"
    )

    return {
        "cv_accuracy_mean": acc_mean,
        "cv_accuracy_std": acc_std,
        "cv_f1_mean": f1_mean,
        "cv_f1_std": f1_std,
        "cv_roc_auc_mean": roc_mean,
        "cv_roc_auc_std": roc_std,
        "cv_pr_auc_mean": pr_auc_mean,
        "cv_pr_auc_std": pr_auc_std,
        "cv_brier_mean": brier_mean,
        "cv_brier_std": brier_std,
    }


def select_best_params(
    model_name: str,
    preprocess: ColumnTransformer,
    model_cls,
    baseline_params: dict,
    candidate_params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    seed: int = 42,
) -> dict:
    print(f"\nSelecting {model_name} hyperparameters via {cv_folds}-fold CV")

    baseline_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", model_cls(**baseline_params)),
        ]
    )
    candidate_model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", model_cls(**candidate_params)),
        ]
    )

    baseline_scores = cross_validate_model(
        f"{model_name} (baseline)", baseline_model, X, y, cv_folds, seed=seed
    )
    candidate_scores = cross_validate_model(
        f"{model_name} (candidate)", candidate_model, X, y, cv_folds, seed=seed
    )

    if candidate_scores["cv_roc_auc_mean"] > baseline_scores["cv_roc_auc_mean"]:
        print(f"Selected {model_name}: candidate params")
        return candidate_params

    print(f"Selected {model_name}: baseline params")
    return baseline_params


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train baseline models.")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--xgb-early-stopping-rounds", type=int, default=25)
    parser.add_argument("--xgb-eval-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--update-thesis", action="store_true")
    parser.add_argument("--xgb-device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model", choices=["all", "logreg", "rf", "xgb"], default="all")
    parser.add_argument("--dataset", type=str, default="cleaned_heart_disease.csv", help="Dataset CSV path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility across splits and models.")
    parser.add_argument("--save-run-metrics", action="store_true")
    parser.add_argument("--comparison-file", type=str, default="artifacts/separate_model_runs.csv")
    parser.add_argument("--run-xai", action="store_true", help="Run SHAP explainability analysis after training.")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of CPU workers for parallel CV and grid search (-1 = all cores).")
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


def main() -> None:
    args = build_arg_parser().parse_args()
    X, y = load_data(args.dataset)
    xgb_device = resolve_xgb_device(args.xgb_device)
    print(f"XGBoost device: {xgb_device}")

    run_logreg = args.model in {"all", "logreg"}
    run_rf = args.model in {"all", "rf"}
    run_xgb = args.model in {"all", "xgb"}

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_val_test(
        X,
        y,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )
    print(
        f"Data split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )

    preprocess_scaled, preprocess_unscaled = build_preprocessors(X)

    selected_rf_params = None
    if run_rf:
        rf_baseline = {**RF_BASELINE_PARAMS, "random_state": args.seed}
        rf_candidate = {**RF_CANDIDATE_PARAMS, "random_state": args.seed}
        selected_rf_params = select_best_params(
            model_name="Random Forest",
            preprocess=preprocess_unscaled,
            model_cls=RandomForestClassifier,
            baseline_params=rf_baseline,
            candidate_params=rf_candidate,
            X=X_train,
            y=y_train,
            cv_folds=args.cv_folds,
            seed=args.seed,
        )

    selected_xgb_params = None
    if run_xgb:
        selected_xgb_params = select_best_params(
            model_name="XGBoost",
            preprocess=preprocess_unscaled,
            model_cls=XGBClassifier,
            baseline_params={
                **XGB_BASELINE_PARAMS,
                "tree_method": "hist",
                "device": xgb_device,
                "random_state": args.seed,
            },
            candidate_params={
                **XGB_CANDIDATE_PARAMS,
                "tree_method": "hist",
                "device": xgb_device,
                "random_state": args.seed,
            },
            X=X_train,
            y=y_train,
            cv_folds=args.cv_folds,
            seed=args.seed,
        )

    models: list[tuple[str, Pipeline]] = []
    if run_logreg:
        models.append(
            (
                "Logistic Regression",
                Pipeline(
                    steps=[
                        ("preprocess", preprocess_scaled),
                        (
                            "classifier",
                            LogisticRegression(max_iter=2000, random_state=args.seed),
                        ),
                    ]
                ),
            )
        )
    if run_rf and selected_rf_params is not None:
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

    results: list[dict] = []
    cv_models: list[tuple[str, Pipeline]] = []
    if run_logreg:
        cv_models.append(
            (
                "Logistic Regression",
                Pipeline(
                    steps=[
                        ("preprocess", preprocess_scaled),
                        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
                    ]
                ),
            )
        )
    if run_rf and selected_rf_params is not None:
        cv_models.append(
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
    if run_xgb and selected_xgb_params is not None:
        cv_models.append(
            (
                "XGBoost",
                Pipeline(
                    steps=[
                        ("preprocess", preprocess_unscaled),
                        (
                            "classifier",
                            XGBClassifier(**selected_xgb_params),
                        ),
                    ]
                ),
            )
        )

    print(f"\n=== Cross-Validation ({args.cv_folds}-fold, Stratified) ===")
    cv_lookup: dict[str, dict] = {}
    for name, cv_model in cv_models:
        cv_lookup[name] = cross_validate_model(
            name, cv_model, X_train, y_train, args.cv_folds, seed=args.seed, n_jobs=args.n_jobs
        )

    model_probas: dict[str, np.ndarray] = {}
    for name, model in models:
        model.fit(X_train, y_train)
        model_result, y_proba = evaluate_model(name, model, X_test, y_test)
        model_result.update(cv_lookup[name])
        results.append(model_result)
        model_probas[name] = y_proba

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
        save_xgb_learning_curve(
            xgb_model.evals_result(),
            PROJECT_ROOT / args.output_dir / "xgb_learning_curve.png",
        )

        align_xgb_inference_device(xgb_model)
        y_pred_xgb = xgb_model.predict(X_test_xgb_np)
        y_proba_xgb = xgb_model.predict_proba(X_test_xgb_np)[:, 1]

        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        f1_xgb = f1_score(y_test, y_pred_xgb)
        roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)

        # Clinical metrics for XGBoost
        xgb_clinical = compute_clinical_metrics(y_test, y_proba_xgb)

        print("\n=== XGBoost ===")
        print(
            f"Accuracy: {accuracy_xgb:.4f} | F1: {f1_xgb:.4f} | ROC-AUC: {roc_auc_xgb:.4f} | "
            f"PR-AUC: {xgb_clinical['pr_auc']:.4f} | Brier: {xgb_clinical['brier_score']:.4f}"
        )
        print(
            f"Optimal threshold (F1): {xgb_clinical['optimal_threshold_f1']:.3f} "
            f"(F1={xgb_clinical['f1_at_optimal']:.4f}) | "
            f"Optimal threshold (Youden): {xgb_clinical['optimal_threshold_youden']:.3f} "
            f"(J={xgb_clinical['youden_j']:.4f})"
        )
        print(classification_report(y_test, y_pred_xgb, target_names=["healthy", "sick"]))
        print(f"Best iteration (early stopping): {xgb_model.best_iteration}")

        model_probas["XGBoost"] = y_proba_xgb

        results.append(
            {
                "model": "XGBoost",
                "test_accuracy": accuracy_xgb,
                "test_f1": f1_xgb,
                "test_roc_auc": roc_auc_xgb,
                "test_pr_auc": xgb_clinical["pr_auc"],
                "test_brier_score": xgb_clinical["brier_score"],
                "optimal_threshold_f1": xgb_clinical["optimal_threshold_f1"],
                "f1_at_optimal_threshold": xgb_clinical["f1_at_optimal"],
                "optimal_threshold_youden": xgb_clinical["optimal_threshold_youden"],
                "youden_j": xgb_clinical["youden_j"],
                **cv_lookup["XGBoost"],
            }
        )

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

            # Collect fitted pipelines for SHAP analysis
            fitted_pipelines: dict[str, Pipeline] = {}
            for name, model in models:
                fitted_pipelines[name] = model

            if "Random Forest" in fitted_pipelines:
                print("\n=== SHAP Analysis: Random Forest (Heart) ===")
                rf_result = generate_shap_tree(
                    model_pipeline=fitted_pipelines["Random Forest"],
                    X_train=X_train,
                    X_test=X_test,
                    feature_names=None,
                    dataset_name="heart",
                    model_name="random_forest",
                    output_dir=xai_output_dir,
                )
                xai_results.append(rf_result)

            if run_xgb and selected_xgb_params is not None:
                # Build a fitted XGBoost pipeline for SHAP
                xgb_pipeline_for_shap = Pipeline(
                    steps=[
                        ("preprocess", preprocess_unscaled),
                        ("classifier", xgb_model),
                    ]
                )
                # Preprocessor was already fit during XGBoost training
                print("\n=== SHAP Analysis: XGBoost (Heart) ===")
                xgb_shap_result = generate_shap_tree(
                    model_pipeline=xgb_pipeline_for_shap,
                    X_train=X_train,
                    X_test=X_test,
                    feature_names=None,
                    dataset_name="heart",
                    model_name="xgboost",
                    output_dir=xai_output_dir,
                )
                xai_results.append(xgb_shap_result)

            if xai_results:
                save_xai_summary_csv(xai_results, xai_output_dir)

        except Exception as xai_error:
            print(f"Warning: SHAP analysis failed: {xai_error}")
            import traceback
            traceback.print_exc()

    # --- Calibration & PR Curves ---
    figures_dir = PROJECT_ROOT / args.output_dir / "figures"
    if model_probas:
        plot_calibration_curves(y_test, model_probas, "heart", figures_dir)
        plot_precision_recall_curves(y_test, model_probas, "heart", figures_dir)
        plot_decision_curve(y_test, model_probas, "heart", figures_dir)

    summary = pd.DataFrame(results).sort_values(
        by=["test_roc_auc", "test_f1", "test_accuracy"],
        ascending=False,
    )

    print("\n=== Baseline Model Comparison ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "baseline_comparison.csv"
    md_path = output_dir / "baseline_comparison.md"

    summary.to_csv(csv_path, index=False)

    summary_rounded = summary.copy()
    for col in [
        "test_accuracy",
        "test_f1",
        "test_roc_auc",
        "test_pr_auc",
        "test_brier_score",
        "optimal_threshold_f1",
        "f1_at_optimal_threshold",
        "optimal_threshold_youden",
        "youden_j",
        "cv_accuracy_mean",
        "cv_accuracy_std",
        "cv_f1_mean",
        "cv_f1_std",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
        "cv_pr_auc_mean",
        "cv_pr_auc_std",
        "cv_brier_mean",
        "cv_brier_std",
    ]:
        summary_rounded[col] = summary_rounded[col].map(lambda x: f"{x:.4f}")

    headers = summary_rounded.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in summary_rounded.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved comparison CSV: {csv_path}")
    print(f"Saved comparison Markdown: {md_path}")

    if args.save_run_metrics:
        comparison_path = PROJECT_ROOT / args.comparison_file
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
        updater_path = PROJECT_ROOT / "evaluation" / "update_thesis_results.py"
        if not updater_path.exists():
            raise FileNotFoundError("update_thesis_results.py not found.")

        update_command = [
            sys.executable,
            str(updater_path),
            "--task",
            "heart",
            "--dataset",
            "cleaned_heart_disease.csv",
            "--comparison",
            str(Path(args.output_dir) / "baseline_comparison.csv"),
        ]
        result = subprocess.run(update_command, check=False)
        if result.returncode != 0:
            raise RuntimeError(
                f"update_thesis_results.py failed with exit code {result.returncode}"
            )


if __name__ == "__main__":
    main()
