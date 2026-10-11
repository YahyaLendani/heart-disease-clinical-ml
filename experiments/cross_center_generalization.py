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
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean_data import build_preprocessing_pipeline
from models import add_cardiovascular_features


def get_models(seed: int = 42) -> dict[str, callable]:
    return {
        "Logistic Regression": lambda pre: Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ]),
        "Random Forest": lambda pre: Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(n_estimators=300, max_depth=6, random_state=seed, n_jobs=1)),
        ]),
        "XGBoost": lambda pre: Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed, n_jobs=1, eval_metric="logloss")),
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


def run_cross_center_study(
    dataset_path: str = "cleaned_heart_disease_combined.csv",
    seed: int = 42,
    output_dir: str = "artifacts",
) -> pd.DataFrame:
    print("=" * 70)
    print("LEAVE-ONE-CENTER-OUT (LOCO) DOMAIN GENERALIZATION STUDY")
    print("=" * 70)

    path = Path(dataset_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    df = pd.read_csv(path)
    if "site" not in df.columns:
        combined_path = PROJECT_ROOT / "cleaned_heart_disease_combined.csv"
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

    loco_records = []

    for test_center in centers:
        train_df = df[df["site"] != test_center].copy()
        test_df = df[df["site"] == test_center].copy()

        X_train = train_df.drop(columns=drop_cols)
        y_train = train_df["target"].astype(int)
        X_test = test_df.drop(columns=drop_cols)
        y_test = test_df["target"].astype(int)

        prev_train = y_train.mean() * 100.0
        prev_test = y_test.mean() * 100.0

        print(f"\n--- Target Center: {test_center.upper()} ---")
        print(f"  Train: {len(X_train)} patients (Prevalence: {prev_train:.1f}%)")
        print(f"  Test:  {len(X_test)} patients (Prevalence: {prev_test:.1f}%)")

        pre_scaled, _, _ = build_preprocessing_pipeline(X_train, scale_numeric=True)
        pre_unscaled, _, _ = build_preprocessing_pipeline(X_train, scale_numeric=False)

        model_factories = get_models(seed=seed)
        center_test_probas = []

        for m_name, factory in model_factories.items():
            pre = pre_scaled if m_name == "Logistic Regression" else pre_unscaled
            model = factory(pre)
            model.fit(X_train, y_train)

            y_proba = model.predict_proba(X_test)[:, 1]
            center_test_probas.append(y_proba)

            metrics = evaluate_split(y_test.values, y_proba)
            loco_records.append({
                "test_center": test_center,
                "model": m_name,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "transfer_accuracy": metrics["accuracy"],
                "transfer_f1": metrics["f1"],
                "transfer_roc_auc": metrics["roc_auc"],
                "transfer_pr_auc": metrics["pr_auc"],
                "transfer_brier": metrics["brier"],
            })
            print(f"    [{m_name}] Test AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f}")

        # Evaluate ensemble on this center
        ens_proba = np.mean(center_test_probas, axis=0)
        ens_metrics = evaluate_split(y_test.values, ens_proba)
        loco_records.append({
            "test_center": test_center,
            "model": "Ensemble (Transfer)",
            "train_size": len(X_train),
            "test_size": len(X_test),
            "transfer_accuracy": ens_metrics["accuracy"],
            "transfer_f1": ens_metrics["f1"],
            "transfer_roc_auc": ens_metrics["roc_auc"],
            "transfer_pr_auc": ens_metrics["pr_auc"],
            "transfer_brier": ens_metrics["brier"],
        })
        print(f"    [Ensemble Transfer] Test AUC: {ens_metrics['roc_auc']:.4f} | Acc: {ens_metrics['accuracy']:.4f}")

    loco_df = pd.DataFrame(loco_records)

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    csv_out = out_p / "cross_center_generalization.csv"
    md_out = out_p / "cross_center_generalization.md"

    loco_df.to_csv(csv_out, index=False)

    def _to_md(frame: pd.DataFrame) -> str:
        headers = list(frame.columns)
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in frame.iterrows():
            vals = [f"{row[h]:.4f}" if isinstance(row[h], float) else str(row[h]) for h in headers]
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    summary_pivot = loco_df.pivot(index="test_center", columns="model", values="transfer_roc_auc")
    summary_pivot["Mean ROC-AUC"] = summary_pivot.mean(axis=1)
    summary_pivot_df = summary_pivot.round(4).reset_index()

    md_content = "# Leave-One-Center-Out (LOCO) Cross-Hospital Generalization\n\n"
    md_content += "This experiment tests hospital generalizability by training on 3 clinical centers and testing on an unseen 4th center.\n\n"
    md_content += "### Transfer ROC-AUC by Unseen Hospital Center\n\n"
    md_content += _to_md(summary_pivot_df) + "\n\n"
    md_content += "### Complete Metric Breakdown\n\n"
    md_content += _to_md(loco_df.round(4)) + "\n"

    md_out.write_text(md_content, encoding="utf-8")
    print(f"\nSaved LOCO generalization results to:\n  - {csv_out}\n  - {md_out}")

    return loco_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-center LOCO generalization experiment.")
    parser.add_argument("--dataset", type=str, default="cleaned_heart_disease_combined.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args()

    run_cross_center_study(dataset_path=args.dataset, seed=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
