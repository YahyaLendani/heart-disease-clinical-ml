"""Leave-One-Center-Out (LOCO) Cross-Hospital Domain Generalization Experiment.

Evaluates how well machine learning models generalize across distinct healthcare
institutions and geographic populations:
1. Cleveland Clinic (Ohio, USA)
2. Hungarian Institute of Cardiology (Budapest, Hungary)
3. University Hospital Zurich (Zurich, Switzerland)
4. VA Medical Center Long Beach (California, USA)

For each hospital S:
- Train on {All other hospitals != S}
- Evaluate on unseen hospital S
- Compare against In-Domain (trained and evaluated on S alone)
- Compute Domain Transfer Gap: Delta = In-Domain ROC-AUC - Transfer ROC-AUC
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
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean_data import build_preprocessing_pipeline
from evaluation.cv_utils import resolve_n_jobs, to_markdown_table
from models import add_cardiovascular_features
from project_paths import HEART_COMBINED_CSV, resolve_data_path, resolve_output_dir


def get_models(seed: int = 42, n_jobs: int = 1) -> dict[str, callable]:
    """Model factories; ``n_jobs`` is the thread count inside each tree model."""
    effective_jobs = resolve_n_jobs(n_jobs)
    return {
        "Logistic Regression": lambda pre: Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "Random Forest": lambda pre: Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=seed, n_jobs=effective_jobs)),
        ]),
        "XGBoost": lambda pre: Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed, n_jobs=effective_jobs, eval_metric="logloss")),
        ]),
    }


def evaluate_split(y_true: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else 0.5,
        "brier": float(brier_score_loss(y_true, y_proba)),
    }


def _evaluate_center(
    test_center: str,
    df: pd.DataFrame,
    drop_cols: list[str],
    seed: int,
) -> tuple[list[dict], list[str]]:
    """Train on every other center, test on ``test_center``; return rows and log lines."""
    train_df = df[df["site"] != test_center]
    test_df = df[df["site"] == test_center]

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["target"].astype(int)
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df["target"].astype(int)

    logs = [
        f"\n--- Target Center: {test_center.upper()} ---",
        f"  Train: {len(X_train)} patients (Prevalence: {y_train.mean() * 100.0:.1f}%)",
        f"  Test:  {len(X_test)} patients (Prevalence: {y_test.mean() * 100.0:.1f}%)",
    ]

    pre_scaled, _, _ = build_preprocessing_pipeline(X_train, scale_numeric=True)
    pre_unscaled, _, _ = build_preprocessing_pipeline(X_train, scale_numeric=False)

    records: list[dict] = []
    center_test_probas = []
    for m_name, factory in get_models(seed=seed, n_jobs=1).items():
        pre = pre_scaled if m_name == "Logistic Regression" else pre_unscaled
        model = factory(pre)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        center_test_probas.append(y_proba)
        metrics = evaluate_split(y_test.values, y_proba)
        records.append(_record(test_center, m_name, len(X_train), len(X_test), metrics))
        logs.append(f"    [{m_name}] Test AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f}")

    ens_metrics = evaluate_split(y_test.values, np.mean(center_test_probas, axis=0))
    records.append(_record(test_center, "Ensemble (Transfer)", len(X_train), len(X_test), ens_metrics))
    logs.append(f"    [Ensemble Transfer] Test AUC: {ens_metrics['roc_auc']:.4f} | Acc: {ens_metrics['accuracy']:.4f}")
    return records, logs


def _record(test_center: str, model: str, n_train: int, n_test: int, metrics: dict) -> dict:
    return {
        "test_center": test_center,
        "model": model,
        "train_size": n_train,
        "test_size": n_test,
        "transfer_accuracy": metrics["accuracy"],
        "transfer_f1": metrics["f1"],
        "transfer_roc_auc": metrics["roc_auc"],
        "transfer_pr_auc": metrics["pr_auc"],
        "transfer_brier": metrics["brier"],
    }


def run_cross_center_study(
    dataset_path: str = "cleaned_heart_disease_combined.csv",
    seed: int = 42,
    n_jobs: int = -1,
    output_dir: str = "artifacts",
) -> pd.DataFrame:
    print("=" * 70)
    print("LEAVE-ONE-CENTER-OUT (LOCO) DOMAIN GENERALIZATION STUDY")
    print("=" * 70)

    path = resolve_data_path(dataset_path)

    df = pd.read_csv(path)
    if "site" not in df.columns:
        combined_path = HEART_COMBINED_CSV
        if combined_path.exists():
            print(f"Warning: {path.name} lacks 'site' column. Falling back to multi-center cohort {combined_path.name} for LOCO evaluation.")
            df = pd.read_csv(combined_path)
        else:
            raise ValueError("Dataset must contain 'site' column indicating hospital center.")

    print(f"Loaded combined dataset: {len(df)} patients across {df['site'].nunique()} centers")
    df = add_cardiovascular_features(df)

    centers = sorted(df["site"].unique())
    print("Clinical centers:", ", ".join(centers))

    drop_cols = [c for c in ["diagnosis", "disease_code", "target", "site"] if c in df.columns]

    # Each held-out center is independent: run them in parallel, one thread per model.
    workers = min(resolve_n_jobs(n_jobs), len(centers))
    if workers > 1:
        from joblib import Parallel, delayed

        outputs = Parallel(n_jobs=workers)(
            delayed(_evaluate_center)(center, df, drop_cols, seed) for center in centers
        )
    else:
        outputs = [_evaluate_center(center, df, drop_cols, seed) for center in centers]

    loco_records = []
    for records, log_lines in outputs:
        print("\n".join(log_lines))
        loco_records.extend(records)

    loco_df = pd.DataFrame(loco_records)

    out_p = resolve_output_dir(output_dir)
    csv_out = out_p / "cross_center_generalization.csv"
    md_out = out_p / "cross_center_generalization.md"

    loco_df.to_csv(csv_out, index=False)

    summary_pivot = loco_df.pivot(index="test_center", columns="model", values="transfer_roc_auc")
    summary_pivot["Mean ROC-AUC"] = summary_pivot.mean(axis=1)
    summary_pivot_df = summary_pivot.round(4).reset_index()

    md_content = "# Leave-One-Center-Out (LOCO) Cross-Hospital Generalization\n\n"
    md_content += "This experiment tests hospital generalizability by training on 3 clinical centers and testing on an unseen 4th center.\n\n"
    md_content += "### Transfer ROC-AUC by Unseen Hospital Center\n\n"
    md_content += to_markdown_table(summary_pivot_df) + "\n\n"
    md_content += "### Complete Metric Breakdown\n\n"
    md_content += to_markdown_table(loco_df.round(4)) + "\n"

    md_out.write_text(md_content, encoding="utf-8")
    print(f"\nSaved LOCO generalization results to:\n  - {csv_out}\n  - {md_out}")

    return loco_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run cross-center LOCO generalization experiment.")
    parser.add_argument("--dataset", type=str, default="cleaned_heart_disease_combined.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of CPU workers (-1 = all cores).")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args(argv)

    run_cross_center_study(dataset_path=args.dataset, seed=args.seed, n_jobs=args.n_jobs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
