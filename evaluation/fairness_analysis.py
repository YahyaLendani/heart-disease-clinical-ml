"""Algorithmic Fairness and Demographic Subgroup Analysis in Cardiovascular Prediction.

Evaluates model performance across demographic subgroups to detect disparities
in clinical diagnostic accuracy:
1. Biological Sex: Female (sex=0) vs Male (sex=1)
2. Age Brackets:
   - Younger: < 50 years
   - Middle-aged: 50 - 64 years
   - Senior: >= 65 years

Clinical Fairness Metrics computed:
- True Positive Rate (TPR / Sensitivity / Recall): Did the model catch sick patients equally?
- True Negative Rate (TNR / Specificity): Did it correctly identify healthy patients?
- False Negative Rate (FNR = 1 - TPR): Missed disease risk
- False Positive Rate (FPR = 1 - TNR): Unnecessary cardiac workup risk
- Equal Opportunity Difference: |TPR_group1 - TPR_group2|
- Disparate Impact Ratio: Selection Rate Ratio (PR_unprivileged / PR_privileged)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clean_data import build_preprocessing_pipeline
from evaluation.cv_utils import resolve_n_jobs, to_markdown_table
from models import add_cardiovascular_features
from project_paths import resolve_data_path, resolve_output_dir


def compute_subgroup_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    if len(y_true) == 0:
        return {}

    n_total = len(y_true)
    prevalence = float(y_true.mean() * 100.0)
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc = float(roc_auc_score(y_true, y_proba)) if (y_proba is not None and len(np.unique(y_true)) > 1) else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tpr = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0  # Sensitivity / Recall
    tnr = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0  # Specificity
    fnr = 1.0 - tpr
    fpr = 1.0 - tnr
    selection_rate = float((tp + fp) / n_total)

    return {
        "n_samples": n_total,
        "prevalence_pct": prevalence,
        "accuracy": acc,
        "f1": f1,
        "roc_auc": roc,
        "sensitivity_tpr": tpr,
        "specificity_tnr": tnr,
        "false_negative_rate": fnr,
        "false_positive_rate": fpr,
        "selection_rate": selection_rate,
    }


def plot_fairness_figures(fairness_df: pd.DataFrame, output_path: Path) -> None:
    """Sensitivity/specificity and missed-disease rate by subgroup (horizontal bars, n in labels)."""
    try:
        import matplotlib.pyplot as plt

        from evaluation.plot_style import (
            apply_style,
            finish_axes,
            legend_above,
            palette,
            series_color,
            subtitle,
        )
    except ImportError:
        print("matplotlib not available; skipping fairness plot.")
        return

    plot_df = fairness_df[fairness_df["subgroup"] != "All Patients"].copy()
    if plot_df.empty:
        return

    apply_style()
    labels = [
        f"{name.split(' (')[0].replace(' Patients', '')}  (n={int(n)})"
        for name, n in zip(plot_df["subgroup"], plot_df["n_samples"], strict=True)
    ]
    y = np.arange(len(plot_df))[::-1]
    height = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)

    axes[0].barh(y + height / 2, plot_df["sensitivity_tpr"] * 100, height, color=series_color(0), label="Sensitivity")
    axes[0].barh(y - height / 2, plot_df["specificity_tnr"] * 100, height, color=series_color(1), label="Specificity")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_xlim(0, 100)
    axes[0].set_xlabel("Percent of patients")
    axes[0].set_title("Sensitivity and specificity")
    subtitle(axes[0], "XGBoost, test split of the 4-center cohort, threshold 0.5", legend_row=True)
    legend_above(axes[0])
    axes[0].grid(axis="y", visible=False)
    finish_axes(axes[0])

    fnr = plot_df["false_negative_rate"] * 100
    axes[1].barh(y, fnr, 0.5, color=series_color(6))
    for yi, value in zip(y, fnr, strict=True):
        axes[1].annotate(f"{value:.1f}%", xy=(value, yi), xytext=(4, 0), textcoords="offset points",
                         va="center", fontsize=9, color=palette()["ink_secondary"])
    axes[1].set_xlim(0, max(50, float(fnr.max()) + 10))
    axes[1].set_xlabel("Percent of sick patients missed")
    axes[1].set_title("Missed-disease (false negative) rate")
    subtitle(axes[1], "Small groups give imprecise rates", legend_row=True)
    axes[1].grid(axis="y", visible=False)
    finish_axes(axes[1])

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved fairness plot: {output_path}")


def run_fairness_study(
    dataset_path: str = "cleaned_heart_disease_combined.csv",
    seed: int = 42,
    n_jobs: int = -1,
    output_dir: str = "artifacts",
) -> tuple[pd.DataFrame, dict]:
    print("=" * 70)
    print("ALGORITHMIC FAIRNESS & CLINICAL DISPARITY STUDY (Sex & Age)")
    print("=" * 70)

    effective_jobs = resolve_n_jobs(n_jobs)
    path = resolve_data_path(dataset_path)

    df = pd.read_csv(path)
    print(f"Loaded dataset: {len(df)} patients")

    df = add_cardiovascular_features(df)

    drop_cols = [c for c in ["diagnosis", "disease_code", "target", "site"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["target"].astype(int)

    # Retain demographics for stratified evaluation
    test_size = 0.2
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=test_size, random_state=seed, stratify=y
    )

    pre_unscaled, _, _ = build_preprocessing_pipeline(X_train, scale_numeric=False)
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        random_state=seed,
        n_jobs=effective_jobs,
        eval_metric="logloss",
    )
    from sklearn.pipeline import Pipeline
    model = Pipeline([("pre", pre_unscaled), ("clf", clf)])
    model.fit(X_train, y_train)

    y_test_proba = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_proba >= 0.5).astype(int)

    test_subgroups = {}

    # Overall
    test_subgroups["All Patients"] = np.ones(len(df_test), dtype=bool)

    # Sex Subgroups
    if "sex" in df_test.columns:
        test_subgroups["Female Patients (sex=0)"] = (df_test["sex"] == 0).values
        test_subgroups["Male Patients (sex=1)"] = (df_test["sex"] == 1).values

    # Age Subgroups
    if "age" in df_test.columns:
        test_subgroups["Younger (<50 yrs)"] = (df_test["age"] < 50).values
        test_subgroups["Middle-aged (50-64 yrs)"] = ((df_test["age"] >= 50) & (df_test["age"] < 65)).values
        test_subgroups["Senior (>=65 yrs)"] = (df_test["age"] >= 65).values

    records = []
    for sub_name, mask in test_subgroups.items():
        if mask.sum() == 0:
            continue
        sub_y_true = y_test.values[mask]
        sub_y_pred = y_test_pred[mask]
        sub_y_proba = y_test_proba[mask]
        metrics = compute_subgroup_metrics(sub_y_true, sub_y_pred, sub_y_proba)
        records.append({"subgroup": sub_name, **metrics})

    fairness_df = pd.DataFrame(records)

    # Disparity calculations
    f_metrics = {}
    if "Female Patients (sex=0)" in test_subgroups and "Male Patients (sex=1)" in test_subgroups:
        row_f = fairness_df[fairness_df["subgroup"] == "Female Patients (sex=0)"].iloc[0]
        row_m = fairness_df[fairness_df["subgroup"] == "Male Patients (sex=1)"].iloc[0]

        equal_opp_diff = abs(row_m["sensitivity_tpr"] - row_f["sensitivity_tpr"])
        disparate_impact = (row_f["selection_rate"] / max(row_m["selection_rate"], 1e-6))

        f_metrics["equal_opportunity_diff_sex"] = equal_opp_diff
        f_metrics["disparate_impact_ratio_sex"] = disparate_impact
        f_metrics["female_fnr"] = row_f["false_negative_rate"]
        f_metrics["male_fnr"] = row_m["false_negative_rate"]

        print("\n--- Sex Disparity Summary ---")
        print(f"Equal Opportunity Difference (Sensitivity Gap): {equal_opp_diff:.4f}")
        print(f"Disparate Impact Ratio (Female / Male Selection Rate): {disparate_impact:.4f}")
        print(f"Female Missed Disease Rate (FNR): {row_f['false_negative_rate']*100:.1f}%")
        print(f"Male Missed Disease Rate (FNR):   {row_m['false_negative_rate']*100:.1f}%")

    out_p = resolve_output_dir(output_dir)
    csv_out = out_p / "fairness_analysis.csv"
    md_out = out_p / "fairness_analysis.md"
    fig_out = out_p / "figures" / "fairness_disparities.png"

    fairness_df.to_csv(csv_out, index=False)
    plot_fairness_figures(fairness_df, fig_out)

    md_content = "# Demographic Fairness and Clinical Disparity Analysis\n\n"
    md_content += "Evaluates whether the clinical risk model demonstrates parity across biological sex and age brackets.\n\n"
    md_content += "### Subgroup Performance Table\n\n"
    md_content += to_markdown_table(fairness_df.round(4)) + "\n\n"
    if f_metrics:
        md_content += "### Key Demographic Fairness Findings\n\n"
        md_content += f"- **Equal Opportunity Difference (Sex):** `{f_metrics['equal_opportunity_diff_sex']:.4f}`\n"
        md_content += f"- **Disparate Impact Ratio (Female vs. Male):** `{f_metrics['disparate_impact_ratio_sex']:.4f}`\n"
        md_content += f"- **Missed Disease Rate (FNR):** Female = `{f_metrics['female_fnr']*100:.1f}%`, Male = `{f_metrics['male_fnr']*100:.1f}%`\n"

    md_out.write_text(md_content, encoding="utf-8")
    print(f"\nSaved fairness analysis to:\n  - {csv_out}\n  - {md_out}")

    return fairness_df, f_metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run demographic fairness study.")
    parser.add_argument("--dataset", type=str, default="cleaned_heart_disease_combined.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of CPU workers (-1 = all cores).")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args(argv)

    run_fairness_study(dataset_path=args.dataset, seed=args.seed, n_jobs=args.n_jobs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
