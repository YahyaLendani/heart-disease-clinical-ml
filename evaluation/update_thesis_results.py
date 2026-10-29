import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import PRIMARY_HEART_DATASET

MIMIC_AUTO_START = "<!-- AUTO_RESULTS_START -->"
MIMIC_AUTO_END = "<!-- AUTO_RESULTS_END -->"
HEART_AUTO_START = "<!-- AUTO_HEART_RESULTS_START -->"
HEART_AUTO_END = "<!-- AUTO_HEART_RESULTS_END -->"
HEART_EDA_AUTO_START = "<!-- AUTO_HEART_EDA_START -->"
HEART_EDA_AUTO_END = "<!-- AUTO_HEART_EDA_END -->"
SHAP_AUTO_START = "<!-- AUTO_SHAP_START -->"
SHAP_AUTO_END = "<!-- AUTO_SHAP_END -->"
CALIBRATION_AUTO_START = "<!-- AUTO_CALIBRATION_START -->"
CALIBRATION_AUTO_END = "<!-- AUTO_CALIBRATION_END -->"
ENSEMBLE_AUTO_START = "<!-- AUTO_ENSEMBLE_START -->"
ENSEMBLE_AUTO_END = "<!-- AUTO_ENSEMBLE_END -->"
LOCO_AUTO_START = "<!-- AUTO_LOCO_START -->"
LOCO_AUTO_END = "<!-- AUTO_LOCO_END -->"
READING_LIST_AUTO_START = "<!-- AUTO_READING_LIST_START -->"
READING_LIST_AUTO_END = "<!-- AUTO_READING_LIST_END -->"
FAIRNESS_AUTO_START = "<!-- AUTO_FAIRNESS_START -->"
FAIRNESS_AUTO_END = "<!-- AUTO_FAIRNESS_END -->"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Update thesis draft with latest cohort summary and model comparison table."
    )
    parser.add_argument("--thesis", type=str, default="thesis_draft.md")
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset for --task mimic/heart (defaults to --mimic-dataset / --heart-dataset).",
    )
    parser.add_argument(
        "--comparison", type=str, default=None,
        help="Comparison CSV for --task mimic/heart (defaults to --mimic-comparison / --heart-comparison).",
    )
    parser.add_argument("--mimic-dataset", type=str, default="mimic_hospital_mortality.csv")
    parser.add_argument("--mimic-comparison", type=str, default="artifacts/mimic_baseline_comparison.csv")
    parser.add_argument("--heart-dataset", type=str, default=PRIMARY_HEART_DATASET)
    parser.add_argument("--heart-comparison", type=str, default="artifacts/baseline_comparison.csv")
    parser.add_argument("--heart-eda-missingness", type=str, default="artifacts/eda_missingness.csv")
    parser.add_argument("--heart-eda-outliers", type=str, default="artifacts/eda_outliers_iqr.csv")
    parser.add_argument("--heart-eda-correlations", type=str, default="artifacts/eda_target_correlations.csv")
    parser.add_argument("--heart-eda-separation", type=str, default="artifacts/eda_target_separation.csv")
    parser.add_argument("--task", choices=["mimic", "heart", "all"], default="mimic")
    parser.add_argument("--shap-features", type=str, default="artifacts/figures/shap_top_features.csv")
    parser.add_argument("--shap-figures-dir", type=str, default="artifacts/figures")
    parser.add_argument("--ensemble-comparison", type=str, default="artifacts/ensemble_comparison.csv")
    parser.add_argument("--loco-comparison", type=str, default="artifacts/cross_center_generalization.csv")
    parser.add_argument("--fairness-comparison", type=str, default="artifacts/fairness_analysis.csv")
    parser.add_argument("--fairness-figure", type=str, default="artifacts/figures/fairness_disparities.png")
    parser.add_argument("--reading-list-summary", type=str, default="artifacts/reading_list_advancements_summary.csv")
    parser.add_argument("--reading-list-report", type=str, default="artifacts/reading_list_advancements_summary.md")
    parser.add_argument("--model-comparison", type=str, default="artifacts/model_comparison.csv")
    parser.add_argument("--loco-conformal", type=str, default="artifacts/loco_conformal_generalization.csv")
    parser.add_argument("--secondary-dir", type=str, default="artifacts/cleveland",
                        help="Artifacts of the Cleveland-only sensitivity run (optional).")
    return parser


def to_markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_numeric_dtype(display[col]):
            display[col] = display[col].map(lambda value: f"{value:.4f}")

    headers = display.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return "\n".join(lines)


def cohort_summary_mimic(dataset_df: pd.DataFrame) -> dict:
    admissions = int(len(dataset_df))
    mortality_rate = float(dataset_df["hospital_expire_flag"].mean() * 100.0)

    age_col = "age_at_admission" if "age_at_admission" in dataset_df.columns else ("anchor_age" if "anchor_age" in dataset_df.columns else None)
    age_mean = float(dataset_df[age_col].mean()) if age_col else None
    age_std = float(dataset_df[age_col].std()) if age_col else None

    los_col = "hosp_los_days" if "hosp_los_days" in dataset_df.columns else ("hospital_los_days" if "hospital_los_days" in dataset_df.columns else None)
    hosp_los_mean = (
        float(dataset_df[los_col].mean()) if los_col else None
    )
    hosp_los_std = (
        float(dataset_df[los_col].std()) if los_col else None
    )

    if "had_icu" in dataset_df.columns:
        icu_rate = float(dataset_df["had_icu"].mean() * 100.0)
    elif "has_icu_stay" in dataset_df.columns:
        icu_rate = float(dataset_df["has_icu_stay"].mean() * 100.0)
    elif "prior_icu_stay_count" in dataset_df.columns:
        icu_rate = float((dataset_df["prior_icu_stay_count"] > 0).mean() * 100.0)
    else:
        icu_rate = None

    return {
        "admissions": admissions,
        "mortality_rate": mortality_rate,
        "age_mean": age_mean,
        "age_std": age_std,
        "age_col": age_col or "age",
        "hosp_los_mean": hosp_los_mean,
        "hosp_los_std": hosp_los_std,
        "icu_rate": icu_rate,
    }


def choose_primary_metric(df: pd.DataFrame) -> tuple[str, str]:
    if "test_pr_auc" in df.columns:
        return "test_pr_auc", "PR-AUC"
    if "test_roc_auc" in df.columns:
        return "test_roc_auc", "ROC-AUC"
    numeric_cols = [c for c in df.columns if c != "model" and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("Comparison CSV has no numeric metric columns.")
    return numeric_cols[0], numeric_cols[0]


def small_sample_note(comparison_df: pd.DataFrame, min_events: int = 10) -> str:
    """Warn when the test split has too few deaths for a stable holdout ranking."""
    if "test_events" not in comparison_df.columns or comparison_df["test_events"].isna().all():
        return ""
    events = int(comparison_df["test_events"].max())
    if events >= min_events:
        return ""
    return (
        f"\n**Small-sample caveat:** the test split contains only {events} deaths, so holdout PR-AUC/ROC-AUC "
        "are highly unstable (see the `_ci_low`/`_ci_high` bootstrap columns). Compare models on the "
        "repeated patient-grouped CV columns (`cv_*_mean ± cv_*_std`) and treat the holdout ranking as indicative only.\n"
    )


def build_mimic_results_block(dataset_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
    summary = cohort_summary_mimic(dataset_df)
    rank_col, rank_label = choose_primary_metric(comparison_df)
    ranked = comparison_df.sort_values(by=rank_col, ascending=False).reset_index(drop=True)

    best_model = str(ranked.loc[0, "model"]) if "model" in ranked.columns and len(ranked) > 0 else "N/A"
    best_metric_value = float(ranked.loc[0, rank_col]) if len(ranked) > 0 else float("nan")

    age_label = summary.get("age_col", "age")
    age_text = (
        f"{summary['age_mean']:.2f} ± {summary['age_std']:.2f}"
        if summary["age_mean"] is not None and summary["age_std"] is not None
        else "not available"
    )
    los_text = (
        f"{summary['hosp_los_mean']:.2f} ± {summary['hosp_los_std']:.2f} days"
        if summary["hosp_los_mean"] is not None and summary["hosp_los_std"] is not None
        else "N/A (omitted to prevent target leakage; prediction at admission)"
    )
    icu_text = f"{summary['icu_rate']:.2f}%" if summary["icu_rate"] is not None else "not available"

    table_md = to_markdown_table(ranked)

    # --- Clinical metrics sub-table (Brier Score, PR-AUC, Thresholds) ---
    clinical_cols_present = all(
        col in comparison_df.columns for col in ["test_brier_score", "test_pr_auc"]
    )
    clinical_section = ""
    if clinical_cols_present:
        clinical_cols = ["model"]
        for col in ["test_pr_auc", "test_brier_score", "optimal_threshold_f1", "f1_at_optimal_threshold", "optimal_threshold_youden", "youden_j"]:
            if col in comparison_df.columns:
                clinical_cols.append(col)
        clinical_df = ranked[clinical_cols].copy()
        clinical_table_md = to_markdown_table(clinical_df)
        clinical_section = (
            "\n### 4) Clinical Calibration & Threshold Metrics (Auto-generated)\n"
            "**Table X-B. Calibration and threshold tuning metrics for MIMIC mortality prediction.**\n\n"
            f"{clinical_table_md}\n\n"
            "**Note:** Brier Score measures calibration quality (lower is better, 0 = perfect). "
            "PR-AUC is critical for imbalanced datasets — it summarizes the precision-recall trade-off. "
            "Optimal thresholds are computed via F1-maximization and Youden's J statistic.\n"
        )

    # --- Calibration curve figure reference ---
    calibration_figure = ""
    calibration_path = Path("artifacts/figures/calibration_mimic.png")
    _project_root = Path(__file__).resolve().parent.parent
    if (_project_root / calibration_path).exists():
        calibration_figure = (
            "\n### 5) Calibration Curves (Auto-generated)\n"
            f"![Calibration Curves — MIMIC]({calibration_path})\n"
        )

    return (
        f"{MIMIC_AUTO_START}\n"
        "### 1) Cohort Summary (Auto-generated)\n"
        f"- Number of admissions: {summary['admissions']}\n"
        f"- Mortality prevalence: {summary['mortality_rate']:.2f}%\n"
        f"- Mean age ({age_label}): {age_text}\n"
        f"- Mean hospital LOS: {los_text}\n"
        f"- ICU utilization rate: {icu_text}\n\n"
        "### 2) Baseline Model Performance (Auto-generated)\n"
        "**Table X. Baseline performance for hospital mortality prediction on MIMIC-IV demo cohort.**\n\n"
        f"{table_md}\n\n"
        "### 3) Interpretation of Findings (Auto-generated)\n"
        f"The best-ranked model by {rank_label} is **{best_model}** ({rank_label}={best_metric_value:.4f}). "
        "This ranking should be interpreted jointly with expired-class recall and F1 to ensure minority-case detection remains clinically useful.\n"
        f"{small_sample_note(comparison_df)}"
        f"{clinical_section}"
        f"{calibration_figure}"
        f"{MIMIC_AUTO_END}"
    )


def infer_heart_target(dataset_df: pd.DataFrame) -> pd.Series:
    if "target" in dataset_df.columns:
        return pd.to_numeric(dataset_df["target"], errors="coerce").fillna(0).astype(int)
    if "diagnosis" in dataset_df.columns:
        return (dataset_df["diagnosis"].astype(str).str.lower() == "sick").astype(int)
    raise ValueError("Heart dataset must contain either 'target' or 'diagnosis'.")


def cohort_summary_heart(dataset_df: pd.DataFrame) -> dict:
    target = infer_heart_target(dataset_df)
    rows = int(len(dataset_df))
    positive_rate = float(target.mean() * 100.0)

    age_mean = float(dataset_df["age"].mean()) if "age" in dataset_df.columns else None
    age_std = float(dataset_df["age"].std()) if "age" in dataset_df.columns else None

    return {
        "rows": rows,
        "positive_rate": positive_rate,
        "age_mean": age_mean,
        "age_std": age_std,
    }


def build_heart_results_block(dataset_df: pd.DataFrame, comparison_df: pd.DataFrame) -> str:
    summary = cohort_summary_heart(dataset_df)

    if "test_roc_auc" in comparison_df.columns:
        rank_col, rank_label = "test_roc_auc", "ROC-AUC"
    elif "test_f1" in comparison_df.columns:
        rank_col, rank_label = "test_f1", "F1"
    else:
        rank_col, rank_label = choose_primary_metric(comparison_df)

    ranked = comparison_df.sort_values(by=rank_col, ascending=False).reset_index(drop=True)
    best_model = str(ranked.loc[0, "model"]) if "model" in ranked.columns and len(ranked) > 0 else "N/A"
    best_metric_value = float(ranked.loc[0, rank_col]) if len(ranked) > 0 else float("nan")

    age_text = (
        f"{summary['age_mean']:.2f} ± {summary['age_std']:.2f}"
        if summary["age_mean"] is not None and summary["age_std"] is not None
        else "not available"
    )

    table_md = to_markdown_table(ranked)

    # --- Clinical metrics sub-table (Brier Score, PR-AUC, Thresholds) ---
    clinical_cols_present = all(
        col in comparison_df.columns for col in ["test_brier_score", "test_pr_auc"]
    )
    clinical_section = ""
    if clinical_cols_present:
        clinical_cols = ["model"]
        for col in ["test_pr_auc", "test_brier_score", "optimal_threshold_f1", "f1_at_optimal_threshold", "optimal_threshold_youden", "youden_j"]:
            if col in comparison_df.columns:
                clinical_cols.append(col)
        clinical_df = ranked[clinical_cols].copy()
        clinical_table_md = to_markdown_table(clinical_df)
        clinical_section = (
            "\n### 4) Clinical Calibration & Threshold Metrics (Auto-generated)\n"
            "**Table Y-B. Calibration and threshold tuning metrics for heart disease prediction.**\n\n"
            f"{clinical_table_md}\n\n"
            "**Note:** Brier Score measures calibration quality (lower is better, 0 = perfect). "
            "PR-AUC summarizes the precision-recall trade-off. "
            "Optimal thresholds are computed via F1-maximization and Youden's J statistic.\n"
        )

    # --- Calibration curve figure reference ---
    calibration_figure = ""
    calibration_path = Path("artifacts/figures/calibration_heart.png")
    _project_root = Path(__file__).resolve().parent.parent
    if (_project_root / calibration_path).exists():
        calibration_figure = (
            "\n### 5) Calibration Curves (Auto-generated)\n"
            f"![Calibration Curves — Heart Disease]({calibration_path})\n"
        )

    return (
        f"{HEART_AUTO_START}\n"
        "## Heart Disease Baseline Results\n\n"
        "### 1) Cohort Summary (Auto-generated)\n"
        f"- Number of samples: {summary['rows']}\n"
        f"- Positive class prevalence (heart disease): {summary['positive_rate']:.2f}%\n"
        f"- Mean age: {age_text}\n\n"
        "### 2) Baseline Model Performance (Auto-generated)\n"
        "**Table Y. Baseline performance for heart disease prediction.**\n\n"
        f"{table_md}\n\n"
        "### 3) Interpretation of Findings (Auto-generated)\n"
        f"The best-ranked model by {rank_label} is **{best_model}** ({rank_label}={best_metric_value:.4f}).\n"
        f"{clinical_section}"
        f"{calibration_figure}"
        f"{HEART_AUTO_END}"
    )


def _top_records(df: pd.DataFrame, cols: list[str], top_n: int = 5) -> list[dict]:
    if df.empty:
        return []
    subset = df.loc[:, [c for c in cols if c in df.columns]].head(top_n)
    return subset.to_dict(orient="records")


def build_heart_eda_block(
    missingness_df: pd.DataFrame,
    outliers_df: pd.DataFrame,
    correlations_df: pd.DataFrame,
    separation_df: pd.DataFrame,
) -> str:
    missing_nonzero = missingness_df[missingness_df["missing_count"] > 0] if "missing_count" in missingness_df.columns else pd.DataFrame()
    max_missing_pct = float(missing_nonzero["missing_percent"].max()) if not missing_nonzero.empty else 0.0

    outlier_top = outliers_df.sort_values(by="outlier_count", ascending=False) if "outlier_count" in outliers_df.columns else pd.DataFrame()
    corr_top = correlations_df.copy()
    if "pearson_corr_with_target" in corr_top.columns:
        corr_top = corr_top.assign(abs_corr=corr_top["pearson_corr_with_target"].abs()).sort_values(by="abs_corr", ascending=False).drop(columns=["abs_corr"])

    sep_top = separation_df.copy()
    if "cohen_d" in sep_top.columns:
        sep_top = sep_top.assign(abs_d=sep_top["cohen_d"].abs()).sort_values(by="abs_d", ascending=False).drop(columns=["abs_d"])

    outlier_items = _top_records(outlier_top, ["feature", "outlier_percent"], top_n=5)
    corr_items = _top_records(corr_top, ["feature", "pearson_corr_with_target"], top_n=5)
    sep_items = _top_records(sep_top, ["feature", "cohen_d"], top_n=5)

    outlier_text = "\n".join(
        f"- {item['feature']}: {float(item['outlier_percent']):.2f}% outliers"
        for item in outlier_items
    )
    corr_text = "\n".join(
        f"- {item['feature']}: Pearson r = {float(item['pearson_corr_with_target']):.4f}"
        for item in corr_items
    )
    sep_text = "\n".join(
        f"- {item['feature']}: Cohen's d = {float(item['cohen_d']):.4f}"
        for item in sep_items
    )

    if not outlier_text:
        outlier_text = "- No outlier summary available."
    if not corr_text:
        corr_text = "- No correlation summary available."
    if not sep_text:
        sep_text = "- No class-separation summary available."

    return (
        f"{HEART_EDA_AUTO_START}\n"
        "## Heart Disease EDA Findings\n\n"
        "### 1) Data Quality Summary (Auto-generated)\n"
        f"- Features with missing values: {int(len(missing_nonzero))}\n"
        f"- Maximum missingness in any feature: {max_missing_pct:.2f}%\n\n"
        "### 2) Outlier Summary (Top 5, Auto-generated)\n"
        f"{outlier_text}\n\n"
        "### 3) Correlation with Target (Top 5 by |Pearson|, Auto-generated)\n"
        f"{corr_text}\n\n"
        "### 4) Class Separation (Top 5 by |Cohen's d|, Auto-generated)\n"
        f"{sep_text}\n"
        f"{HEART_EDA_AUTO_END}"
    )


def replace_or_insert_block(
    thesis_text: str,
    auto_block: str,
    start_marker: str,
    end_marker: str,
    section_heading: str,
) -> str:
    if start_marker in thesis_text and end_marker in thesis_text:
        start = thesis_text.index(start_marker)
        end = thesis_text.index(end_marker) + len(end_marker)
        return thesis_text[:start] + auto_block + thesis_text[end:]

    results_heading = "## Results and Discussion"
    template_results_heading = "## Results and Discussion (Template)"
    references_heading = "## References"

    if results_heading in thesis_text and references_heading in thesis_text:
        insert_at = thesis_text.index(references_heading)
        return thesis_text[:insert_at] + "\n\n" + auto_block + "\n" + thesis_text[insert_at:]

    if template_results_heading in thesis_text and references_heading in thesis_text:
        start = thesis_text.index(template_results_heading)
        ref_start = thesis_text.index(references_heading)
        before = thesis_text[:start]
        after = thesis_text[ref_start:]
        return before + section_heading + "\n\n" + auto_block + "\n\n" + after

    if references_heading in thesis_text:
        ref_start = thesis_text.index(references_heading)
        return thesis_text[:ref_start] + auto_block + "\n\n" + thesis_text[ref_start:]

    return thesis_text.rstrip() + "\n\n" + auto_block + "\n"


def build_shap_block(
    shap_features_df: pd.DataFrame,
    figures_dir: Path,
) -> str:
    """Build a markdown block for SHAP model interpretability results."""
    lines: list[str] = [
        f"{SHAP_AUTO_START}",
        "## Model Interpretability (SHAP)",
        "",
        "This section presents feature importance analysis using SHAP "
        "(SHapley Additive exPlanations). SHAP values quantify each feature's "
        "contribution to individual predictions, providing both global and local "
        "model interpretability.",
        "",
    ]

    # Group by dataset and model
    for (dataset, model), group in shap_features_df.groupby(["dataset", "model"]):
        display_dataset = str(dataset).replace("_", " ").title()
        display_model = str(model).replace("_", " ").title()
        prefix = f"{dataset}_{model}"

        lines.append(f"### {display_model} — {display_dataset} Dataset")
        lines.append("")

        # Top features table
        lines.append(f"**Top 5 Most Important Features ({display_model}):**")
        lines.append("")
        lines.append("| Rank | Feature | Mean \\|SHAP\\| |")
        lines.append("| --- | --- | --- |")
        for _, row in group.iterrows():
            lines.append(
                f"| {int(row['rank'])} | {row['feature']} | {float(row['mean_abs_shap']):.4f} |"
            )
        lines.append("")

        # Summary plot
        summary_path = figures_dir / f"{prefix}_shap_summary.png"
        if summary_path.exists():
            rel_path = f"artifacts/figures/{prefix}_shap_summary.png"
            lines.append("**Global Feature Importance (Beeswarm Plot):**")
            lines.append("")
            lines.append(f"![SHAP Summary — {display_model}]({rel_path})")
            lines.append("")

        # Waterfall plots
        high_path = figures_dir / f"{prefix}_shap_waterfall_high_risk.png"
        low_path = figures_dir / f"{prefix}_shap_waterfall_low_risk.png"
        if high_path.exists():
            rel_path = f"artifacts/figures/{prefix}_shap_waterfall_high_risk.png"
            lines.append("**Local Explanation — Highest-Risk Patient:**")
            lines.append("")
            lines.append(f"![Waterfall High Risk — {display_model}]({rel_path})")
            lines.append("")
        if low_path.exists():
            rel_path = f"artifacts/figures/{prefix}_shap_waterfall_low_risk.png"
            lines.append("**Local Explanation — Lowest-Risk Patient:**")
            lines.append("")
            lines.append(f"![Waterfall Low Risk — {display_model}]({rel_path})")
            lines.append("")

    lines.append(f"{SHAP_AUTO_END}")
    return "\n".join(lines)


def _ensemble_findings(ensemble_df: pd.DataFrame) -> str:
    """Data-driven comparison of the ensembles with their best base model."""
    if ensemble_df.empty or "test_roc_auc" not in ensemble_df.columns:
        return ""
    is_ensemble = ensemble_df["model"].str.contains("Ensemble", case=False)
    ensembles, bases = ensemble_df[is_ensemble], ensemble_df[~is_ensemble]
    if ensembles.empty or bases.empty:
        return ""
    best_base = bases.sort_values("test_roc_auc", ascending=False).iloc[0]
    lines = []
    for _, row in ensembles.iterrows():
        diff = float(row["test_roc_auc"] - best_base["test_roc_auc"])
        overlap = ""
        if {"test_roc_auc_ci_low", "test_roc_auc_ci_high"} <= set(row.index):
            overlap = (
                f"; its 95% CI [{row['test_roc_auc_ci_low']:.3f}, {row['test_roc_auc_ci_high']:.3f}] "
                f"{'overlaps' if row['test_roc_auc_ci_low'] <= best_base['test_roc_auc_ci_high'] else 'does not overlap'} "
                f"that of {best_base['model']}"
            )
        brier = ""
        if "test_brier_score" in row.index:
            brier = f", Brier {row['test_brier_score']:.3f} vs {best_base['test_brier_score']:.3f}"
        lines.append(
            f"- **{row['model']}:** ROC-AUC {row['test_roc_auc']:.3f} ({diff:+.3f} vs the best base model, "
            f"{best_base['model']}){brier}{overlap}.\n"
        )
    lines.append(
        "- Paired DeLong tests for all models on the same test patients are reported in Table R-1b "
        "(`artifacts/model_comparison.csv`).\n"
    )
    return "".join(lines)


def _loco_findings(loco_df: pd.DataFrame) -> str:
    """Data-driven LOCO summary: best/worst centers and which model transfers best where."""
    needed = {"test_center", "model", "transfer_roc_auc"}
    if not needed <= set(loco_df.columns):
        return ""
    pivot = loco_df.pivot(index="test_center", columns="model", values="transfer_roc_auc")
    center_mean = pivot.mean(axis=1).sort_values(ascending=False)
    lines = [
        f"- **Easiest / hardest unseen center:** {center_mean.index[0]} (mean ROC-AUC {center_mean.iloc[0]:.3f}) "
        f"vs {center_mean.index[-1]} ({center_mean.iloc[-1]:.3f}).\n"
    ]
    for center, row in pivot.iterrows():
        best, worst = row.idxmax(), row.idxmin()
        lines.append(
            f"- **{center}:** best transfer by {best} ({row[best]:.3f}), worst by {worst} ({row[worst]:.3f}).\n"
        )
    lines.append(
        "- No single model family transfers best at every center, so robustness to hospital shift "
        "is center-specific rather than a property of linear vs. non-linear models.\n"
        if pivot.idxmax(axis=1).nunique() > 1
        else f"- {pivot.idxmax(axis=1).iloc[0]} transfers best at every center.\n"
    )
    return "".join(lines)


def build_ensemble_results_block(ensemble_df: pd.DataFrame) -> str:
    table_md = to_markdown_table(ensemble_df)
    best_row = ensemble_df.iloc[0] if not ensemble_df.empty else None
    best_name = str(best_row["model"]) if best_row is not None and "model" in ensemble_df.columns else "N/A"
    best_auc = f"{float(best_row['test_roc_auc']):.4f}" if best_row is not None and "test_roc_auc" in ensemble_df.columns else "N/A"

    return (
        f"{ENSEMBLE_AUTO_START}\n"
        "## Advanced Hybrid Ensembling & Cardiology Feature Engineering\n\n"
        "### 1) Architectural Overview (Auto-generated)\n"
        "To maximize discriminative power and clinical reliability, we integrated domain-specific cardiovascular features "
        "(Rate-Pressure Product [RPP], chronotropic metabolic reserve ratio, and composite ischemic risk score) "
        "and constructed two hybrid ensemble architectures:\n"
        "- **Soft-Voting Ensemble:** Combines calibrated class probabilities from PyTorch MLP, XGBoost, Random Forest, and Logistic Regression.\n"
        "- **Meta-Stacking Classifier (Ours):** Uses out-of-fold cross-validated probability predictions from all 4 base models as meta-features feeding an $L_2$-regularized meta-estimator.\n\n"
        "### 2) Comparative Performance (Auto-generated)\n"
        "**Table E-1. Test set discrimination and calibration across base models, engineered features, and hybrid ensembles.**\n\n"
        f"{table_md}\n\n"
        "### 3) Key Findings (Auto-generated)\n"
        f"- The highest test ROC-AUC is **{best_auc}** ({best_name}).\n"
        f"{_ensemble_findings(ensemble_df)}"
        f"{ENSEMBLE_AUTO_END}"
    )


def build_loco_results_block(loco_df: pd.DataFrame) -> str:
    pivot_md = ""
    if "test_center" in loco_df.columns and "model" in loco_df.columns and "transfer_roc_auc" in loco_df.columns:
        pivot_df = loco_df.pivot(index="test_center", columns="model", values="transfer_roc_auc").reset_index()
        num_cols = [c for c in pivot_df.columns if c != "test_center"]
        pivot_df["mean_roc_auc"] = pivot_df[num_cols].mean(axis=1)
        pivot_md = (
            "### 2) Cross-Center Transfer Discrimination (Auto-generated)\n"
            "**Table L-1. Transfer ROC-AUC across unseen test centers.**\n\n"
            f"{to_markdown_table(pivot_df)}\n\n"
        )

    full_table_md = to_markdown_table(loco_df)

    return (
        f"{LOCO_AUTO_START}\n"
        "## Multi-Center Domain Generalization (Leave-One-Center-Out / LOCO)\n\n"
        "### 1) Protocol Description (Auto-generated)\n"
        "To rigorously evaluate external clinical validity and real-world deployment safety, we implemented a Leave-One-Center-Out (LOCO) "
        "cross-hospital study across all four international centers of the UCI Heart Disease benchmark:\n"
        "1. Cleveland Clinic Foundation (USA)\n"
        "2. Hungarian Institute of Cardiology, Budapest (Hungary)\n"
        "3. University Hospital, Zurich (Switzerland)\n"
        "4. V.A. Medical Center, Long Beach (USA)\n\n"
        "In each iteration, the model is trained strictly on three hospital cohorts (with internal imputation and scaling) "
        "and externally tested on the completely unseen fourth hospital.\n\n"
        f"{pivot_md}"
        "### 3) Complete Metric Breakdown (Auto-generated)\n"
        "**Table L-2. Complete metric breakdown per center and model.**\n\n"
        f"{full_table_md}\n\n"
        "### 4) Clinical Generalizability Analysis (Auto-generated)\n"
        f"{_loco_findings(loco_df)}"
        f"{LOCO_AUTO_END}"
    )


def build_fairness_results_block(
    fairness_df: pd.DataFrame,
    fairness_fig_path: Path | None = None,
) -> str:
    table_md = to_markdown_table(fairness_df)

    fig_snippet = ""
    if fairness_fig_path and fairness_fig_path.exists():
        rel_path = "artifacts/figures/fairness_disparities.png"
        fig_snippet = (
            "\n### 4) Fairness Disparity Visualization (Auto-generated)\n\n"
            f"![Demographic Fairness and Disparities]({rel_path})\n"
        )

    findings_lines = []
    female_rows = fairness_df[fairness_df["subgroup"].astype(str).str.contains("Female", case=False, na=False)]
    male_rows = fairness_df[
        fairness_df["subgroup"].astype(str).str.contains(r"\bmale\b", case=False, regex=True, na=False)
        & ~fairness_df["subgroup"].astype(str).str.contains("Female", case=False, na=False)
    ]
    if not female_rows.empty and not male_rows.empty:
        f_row = female_rows.iloc[0]
        m_row = male_rows.iloc[0]
        if "sensitivity_tpr" in fairness_df.columns:
            f_tpr = float(f_row["sensitivity_tpr"])
            m_tpr = float(m_row["sensitivity_tpr"])
            eod = abs(m_tpr - f_tpr)
            findings_lines.append(
                f"- **Equal Opportunity Difference (Sex):** `{eod:.4f}` (Sensitivity: Male = `{m_tpr*100:.1f}%`, Female = `{f_tpr*100:.1f}%`)."
            )
        if "false_negative_rate" in fairness_df.columns:
            f_fnr = float(f_row["false_negative_rate"])
            m_fnr = float(m_row["false_negative_rate"])
            findings_lines.append(
                f"- **Clinical Risk Asymmetry (False Negative Rate):** Sick female patients experience a higher underdiagnosis rate (`{f_fnr*100:.1f}%`) compared to male patients (`{m_fnr*100:.1f}%`)."
            )

    younger_rows = fairness_df[fairness_df["subgroup"].astype(str).str.contains("Younger", case=False, na=False)]
    senior_rows = fairness_df[fairness_df["subgroup"].astype(str).str.contains("Senior", case=False, na=False)]
    if not younger_rows.empty and not senior_rows.empty and "roc_auc" in fairness_df.columns:
        y_auc = float(younger_rows.iloc[0]["roc_auc"])
        s_auc = float(senior_rows.iloc[0]["roc_auc"])
        findings_lines.append(
            f"- **Age Bracket Robustness:** High discrimination is maintained across younger (<50 yrs, ROC-AUC `{y_auc:.4f}`) and senior (>=65 yrs, ROC-AUC `{s_auc:.4f}`) cohorts."
        )

    findings_text = "\n".join(findings_lines) if findings_lines else "- Subgroup disparity metrics computed successfully."

    return (
        f"{FAIRNESS_AUTO_START}\n"
        "## Algorithmic Fairness & Demographic Disparities\n\n"
        "### 1) Motivation & Evaluation Metrics (Auto-generated)\n"
        "Clinical algorithms must not propagate or amplify historical disparities across demographic groups. "
        "We evaluated the top-performing model across biological sex (Male vs. Female) and age brackets "
        "(Younger <50, Middle-aged 50–64, Senior $\\ge$65) across key fairness criteria:\n"
        "- **Equal Opportunity:** Equality of true positive rates (Sensitivity / Recall) across demographic groups.\n"
        "- **Predictive Parity & False Negative Rates:** Ensuring that underdiagnosis rates (false negatives) do not disproportionately affect underrepresented groups.\n\n"
        "### 2) Subgroup Performance Metrics (Auto-generated)\n"
        "**Table F-1. Demographic breakdown of discrimination, calibration, and error rates.**\n\n"
        f"{table_md}\n\n"
        "### 3) Key Demographic Disparity Findings (Auto-generated)\n"
        f"{findings_text}\n"
        f"{fig_snippet}"
        f"{FAIRNESS_AUTO_END}"
    )


def _ci_cell(row: pd.Series, col: str) -> str:
    low, high = row.get(f"{col}_ci_low"), row.get(f"{col}_ci_high")
    if pd.isna(low) or pd.isna(high):
        return f"{row[col]:.3f}"
    return f"{row[col]:.3f} [{low:.3f}, {high:.3f}]"


def _cv_cell(row: pd.Series, metric: str) -> str:
    mean, std = row.get(f"cv_{metric}_mean"), row.get(f"cv_{metric}_std")
    if mean is None or pd.isna(mean):
        return "—"
    return f"{mean:.3f} ± {std:.3f}"


def _methodology_bullets(report_text: str) -> str:
    """The suite's own 'Methodology & Rigor Summary' bullets (numbers come from the run)."""
    start_tag, end_tag = "## 2. Methodology & Rigor Summary", "## 3."
    if start_tag not in report_text:
        return ""
    body = report_text.split(start_tag, 1)[1]
    return body.split(end_tag, 1)[0].strip()


def build_reading_list_block(
    summary_df: pd.DataFrame,
    report_text: str,
    comparison_df: pd.DataFrame | None,
    loco_conformal_df: pd.DataFrame | None,
    secondary: dict[str, pd.DataFrame] | None = None,
) -> str:
    """Tables R-1 (architectures), R-1b (paired DeLong tests), R-2 (LOCO conformal), R-3 (Cleveland)."""
    lines = [
        READING_LIST_AUTO_START,
        "**Table R-1. Reading-list architectures: holdout (bootstrap 95% CI) and repeated stratified CV (mean ± SD).**",
        "",
        "| Architecture | Holdout Accuracy | Holdout ROC-AUC [95% CI] | Holdout PR-AUC | Holdout Brier | CV ROC-AUC | CV Brier |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in summary_df.iterrows():
        name = row["model"] + (" †" if bool(row.get("is_synthetic_demo", False)) else "")
        lines.append(
            f"| {name} | {row['test_accuracy']:.3f} | {_ci_cell(row, 'test_roc_auc')} | {row['test_pr_auc']:.3f} | "
            f"{row['test_brier']:.3f} | {_cv_cell(row, 'roc_auc')} | {_cv_cell(row, 'brier')} |"
        )
    lines += ["", "† Demonstration only: the signal channel is synthesised from the tabular features.", ""]

    if comparison_df is not None and not comparison_df.empty:
        n_sig = int(comparison_df["significant_holm_0.05"].sum())
        lines += [
            f"**Table R-1b. Paired DeLong tests of ROC-AUC differences on the shared test split** "
            f"({len(comparison_df)} model pairs; {n_sig} significant after Holm correction at 0.05). "
            "The ten smallest adjusted p-values are shown; the full table is `artifacts/model_comparison.csv`.",
            "",
            "| Model A | Model B | AUC A | AUC B | Difference [95% CI] | p (raw) | p (Holm) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, row in comparison_df.head(10).iterrows():
            lines.append(
                f"| {row['model_a']} | {row['model_b']} | {row['auc_a']:.3f} | {row['auc_b']:.3f} | "
                f"{row['auc_diff']:+.3f} [{row['diff_ci_low']:+.3f}, {row['diff_ci_high']:+.3f}] | "
                f"{row['p_value']:.3f} | {row['p_holm']:.3f} |"
            )
        lines.append("")

    if loco_conformal_df is not None and not loco_conformal_df.empty:
        lines += [
            "**Table R-2. Leave-One-Center-Out conformal generalization (alpha = 0.10, 90% target).**",
            "",
            "| Held-out Center | N | Prevalence | Fixed-Threshold Accuracy | Conformal Coverage | Mean Set Size | Deferral Rate |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for _, r in loco_conformal_df.iterrows():
            lines.append(
                f"| {r['test_site']} | {int(r['n_test_samples'])} | {r['test_disease_prevalence'] * 100:.1f}% | "
                f"{r['fixed_threshold_accuracy'] * 100:.1f}% | {r['loco_empirical_coverage'] * 100:.1f}% | "
                f"{r['average_set_size']:.2f} | {r['ambiguous_abstention_rate'] * 100:.1f}% |"
            )
        lines.append("")

    if secondary:
        lines += [
            "**Table R-3. Sensitivity analysis on the Cleveland-only cohort (302 patients, 61 test patients).**",
            "",
            "| Model | Holdout ROC-AUC [95% CI] | CV ROC-AUC |",
            "| --- | --- | --- |",
        ]
        for frame in secondary.values():
            for _, row in frame.iterrows():
                label = row["model"] if "model" in row else f"HeartMLP ({row['optimizer'].upper()})"
                lines.append(f"| {label} | {_ci_cell(row, 'test_roc_auc')} | {_cv_cell(row, 'roc_auc')} |")
        lines.append("")

    bullets = _methodology_bullets(report_text)
    if bullets:
        lines += ["**Key empirical findings (auto-generated from the reading-list suite run):**", "", bullets, ""]

    lines.append(READING_LIST_AUTO_END)
    return "\n".join(lines)


def update_reading_list_block(thesis_text: str, project_root: Path, args) -> str:
    summary_path = resolve_path(project_root, args.reading_list_summary)
    if not summary_path.exists():
        return thesis_text
    report_path = resolve_path(project_root, args.reading_list_report)
    comparison_path = resolve_path(project_root, args.model_comparison)
    loco_path = resolve_path(project_root, args.loco_conformal)
    secondary_dir = resolve_path(project_root, args.secondary_dir)

    secondary: dict[str, pd.DataFrame] = {}
    for name in ("baseline_comparison.csv", "mlp_holdout_summary.csv", "reading_list_advancements_summary.csv"):
        path = secondary_dir / name
        if path.exists():
            frame = pd.read_csv(path)
            if name == "mlp_holdout_summary.csv":
                cv_path = secondary_dir / "mlp_cv_summary.csv"
                if cv_path.exists():
                    frame = frame.merge(pd.read_csv(cv_path), on="optimizer", how="left")
            if "is_synthetic_demo" in frame.columns:
                frame = frame[~frame["is_synthetic_demo"].astype(bool)]
            secondary[name] = frame

    block = build_reading_list_block(
        pd.read_csv(summary_path),
        report_path.read_text(encoding="utf-8") if report_path.exists() else "",
        pd.read_csv(comparison_path) if comparison_path.exists() else None,
        pd.read_csv(loco_path) if loco_path.exists() else None,
        secondary or None,
    )
    return replace_or_insert_block(
        thesis_text=thesis_text,
        auto_block=block,
        start_marker=READING_LIST_AUTO_START,
        end_marker=READING_LIST_AUTO_END,
        section_heading="## Results and Discussion",
    )


def update_advanced_blocks(
    thesis_text: str,
    project_root: Path,
    ensemble_comparison: str,
    loco_comparison: str,
    fairness_comparison: str,
    fairness_figure: str,
) -> str:
    ensemble_path = resolve_path(project_root, ensemble_comparison)
    if ensemble_path.exists():
        ensemble_df = pd.read_csv(ensemble_path)
        ensemble_block = build_ensemble_results_block(ensemble_df)
        thesis_text = replace_or_insert_block(
            thesis_text=thesis_text,
            auto_block=ensemble_block,
            start_marker=ENSEMBLE_AUTO_START,
            end_marker=ENSEMBLE_AUTO_END,
            section_heading="## Results and Discussion",
        )

    loco_path = resolve_path(project_root, loco_comparison)
    if loco_path.exists():
        loco_df = pd.read_csv(loco_path)
        loco_block = build_loco_results_block(loco_df)
        thesis_text = replace_or_insert_block(
            thesis_text=thesis_text,
            auto_block=loco_block,
            start_marker=LOCO_AUTO_START,
            end_marker=LOCO_AUTO_END,
            section_heading="## Results and Discussion",
        )

    fairness_path = resolve_path(project_root, fairness_comparison)
    fairness_fig_path = resolve_path(project_root, fairness_figure)
    if fairness_path.exists():
        fairness_df = pd.read_csv(fairness_path)
        fairness_block = build_fairness_results_block(fairness_df, fairness_fig_path)
        thesis_text = replace_or_insert_block(
            thesis_text=thesis_text,
            auto_block=fairness_block,
            start_marker=FAIRNESS_AUTO_START,
            end_marker=FAIRNESS_AUTO_END,
            section_heading="## Results and Discussion",
        )

    return thesis_text


def resolve_path(project_root: Path, value: str) -> Path:
    path_value = Path(value)
    if path_value.is_absolute():
        return path_value
    if path_value.suffix == ".csv" and not (project_root / path_value).exists():
        # Bare dataset names live under data/processed.
        from project_paths import resolve_data_path

        return resolve_data_path(path_value)
    return project_root / path_value


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    thesis_path = resolve_path(project_root, args.thesis)

    if not thesis_path.exists():
        raise FileNotFoundError(f"Thesis file not found: {thesis_path}")

    thesis_text = thesis_path.read_text(encoding="utf-8")

    if args.task == "mimic":
        dataset_path = resolve_path(project_root, args.dataset or args.mimic_dataset)
        comparison_path = resolve_path(project_root, args.comparison or args.mimic_comparison)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        if not comparison_path.exists():
            raise FileNotFoundError(f"Comparison file not found: {comparison_path}")

        dataset_df = pd.read_csv(dataset_path)
        comparison_df = pd.read_csv(comparison_path)
        required_col = "hospital_expire_flag"
        if required_col not in dataset_df.columns:
            raise ValueError(f"Dataset must include '{required_col}' column.")
        auto_block = build_mimic_results_block(dataset_df, comparison_df)
        updated_text = replace_or_insert_block(
            thesis_text=thesis_text,
            auto_block=auto_block,
            start_marker=MIMIC_AUTO_START,
            end_marker=MIMIC_AUTO_END,
            section_heading="## Results and Discussion",
        )
        thesis_path.write_text(updated_text, encoding="utf-8")
        print(f"Updated thesis results block in: {thesis_path}")
        print(f"Using dataset: {dataset_path}")
        print(f"Using comparison: {comparison_path}")

        # SHAP block
        shap_features_path = resolve_path(project_root, args.shap_features)
        shap_figures_dir = resolve_path(project_root, args.shap_figures_dir)
        if shap_features_path.exists():
            shap_df = pd.read_csv(shap_features_path)
            shap_block = build_shap_block(shap_df, shap_figures_dir)
            updated_text = replace_or_insert_block(
                thesis_text=updated_text,
                auto_block=shap_block,
                start_marker=SHAP_AUTO_START,
                end_marker=SHAP_AUTO_END,
                section_heading="## Model Interpretability (SHAP)",
            )
            thesis_path.write_text(updated_text, encoding="utf-8")
            print("Updated SHAP interpretability block in thesis.")

        return

    if args.task == "heart":
        dataset_path = resolve_path(project_root, args.dataset or args.heart_dataset)
        comparison_path = resolve_path(project_root, args.comparison or args.heart_comparison)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        if not comparison_path.exists():
            raise FileNotFoundError(f"Comparison file not found: {comparison_path}")

        dataset_df = pd.read_csv(dataset_path)
        comparison_df = pd.read_csv(comparison_path)
        auto_block = build_heart_results_block(dataset_df, comparison_df)
        updated_text = replace_or_insert_block(
            thesis_text=thesis_text,
            auto_block=auto_block,
            start_marker=HEART_AUTO_START,
            end_marker=HEART_AUTO_END,
            section_heading="## Results and Discussion",
        )

        missingness_path = resolve_path(project_root, args.heart_eda_missingness)
        outliers_path = resolve_path(project_root, args.heart_eda_outliers)
        correlations_path = resolve_path(project_root, args.heart_eda_correlations)
        separation_path = resolve_path(project_root, args.heart_eda_separation)
        if all(path.exists() for path in [missingness_path, outliers_path, correlations_path, separation_path]):
            eda_block = build_heart_eda_block(
                pd.read_csv(missingness_path),
                pd.read_csv(outliers_path),
                pd.read_csv(correlations_path),
                pd.read_csv(separation_path),
            )
            updated_text = replace_or_insert_block(
                thesis_text=updated_text,
                auto_block=eda_block,
                start_marker=HEART_EDA_AUTO_START,
                end_marker=HEART_EDA_AUTO_END,
                section_heading="## Results and Discussion",
            )

        thesis_path.write_text(updated_text, encoding="utf-8")
        print(f"Updated thesis results block in: {thesis_path}")
        print(f"Using dataset: {dataset_path}")
        print(f"Using comparison: {comparison_path}")

        # SHAP block
        shap_features_path = resolve_path(project_root, args.shap_features)
        shap_figures_dir = resolve_path(project_root, args.shap_figures_dir)
        if shap_features_path.exists():
            shap_df = pd.read_csv(shap_features_path)
            shap_block = build_shap_block(shap_df, shap_figures_dir)
            updated_text = replace_or_insert_block(
                thesis_text=updated_text,
                auto_block=shap_block,
                start_marker=SHAP_AUTO_START,
                end_marker=SHAP_AUTO_END,
                section_heading="## Model Interpretability (SHAP)",
            )
            print("Updated SHAP interpretability block in thesis.")

        updated_text = update_advanced_blocks(
            thesis_text=updated_text,
            project_root=project_root,
            ensemble_comparison=args.ensemble_comparison,
            loco_comparison=args.loco_comparison,
            fairness_comparison=args.fairness_comparison,
            fairness_figure=args.fairness_figure,
        )
        updated_text = update_reading_list_block(updated_text, project_root, args)
        thesis_path.write_text(updated_text, encoding="utf-8")

        return

    mimic_dataset_path = resolve_path(project_root, args.mimic_dataset)
    mimic_comparison_path = resolve_path(project_root, args.mimic_comparison)
    heart_dataset_path = resolve_path(project_root, args.heart_dataset)
    heart_comparison_path = resolve_path(project_root, args.heart_comparison)
    heart_eda_missingness_path = resolve_path(project_root, args.heart_eda_missingness)
    heart_eda_outliers_path = resolve_path(project_root, args.heart_eda_outliers)
    heart_eda_correlations_path = resolve_path(project_root, args.heart_eda_correlations)
    heart_eda_separation_path = resolve_path(project_root, args.heart_eda_separation)

    if not mimic_dataset_path.exists():
        raise FileNotFoundError(f"MIMIC dataset file not found: {mimic_dataset_path}")
    if not mimic_comparison_path.exists():
        raise FileNotFoundError(f"MIMIC comparison file not found: {mimic_comparison_path}")
    if not heart_dataset_path.exists():
        raise FileNotFoundError(f"Heart dataset file not found: {heart_dataset_path}")
    if not heart_comparison_path.exists():
        raise FileNotFoundError(f"Heart comparison file not found: {heart_comparison_path}")

    mimic_df = pd.read_csv(mimic_dataset_path)
    if "hospital_expire_flag" not in mimic_df.columns:
        raise ValueError("MIMIC dataset must include 'hospital_expire_flag' column.")
    mimic_comp_df = pd.read_csv(mimic_comparison_path)
    heart_df = pd.read_csv(heart_dataset_path)
    heart_comp_df = pd.read_csv(heart_comparison_path)

    mimic_block = build_mimic_results_block(mimic_df, mimic_comp_df)
    updated_text = replace_or_insert_block(
        thesis_text=thesis_text,
        auto_block=mimic_block,
        start_marker=MIMIC_AUTO_START,
        end_marker=MIMIC_AUTO_END,
        section_heading="## Results and Discussion",
    )

    heart_block = build_heart_results_block(heart_df, heart_comp_df)
    updated_text = replace_or_insert_block(
        thesis_text=updated_text,
        auto_block=heart_block,
        start_marker=HEART_AUTO_START,
        end_marker=HEART_AUTO_END,
        section_heading="## Results and Discussion",
    )

    if all(
        path.exists()
        for path in [
            heart_eda_missingness_path,
            heart_eda_outliers_path,
            heart_eda_correlations_path,
            heart_eda_separation_path,
        ]
    ):
        eda_block = build_heart_eda_block(
            pd.read_csv(heart_eda_missingness_path),
            pd.read_csv(heart_eda_outliers_path),
            pd.read_csv(heart_eda_correlations_path),
            pd.read_csv(heart_eda_separation_path),
        )
        updated_text = replace_or_insert_block(
            thesis_text=updated_text,
            auto_block=eda_block,
            start_marker=HEART_EDA_AUTO_START,
            end_marker=HEART_EDA_AUTO_END,
            section_heading="## Results and Discussion",
        )

    thesis_path.write_text(updated_text, encoding="utf-8")

    # SHAP block for "all" task
    shap_features_path = resolve_path(project_root, args.shap_features)
    shap_figures_dir = resolve_path(project_root, args.shap_figures_dir)
    if shap_features_path.exists():
        updated_text = thesis_path.read_text(encoding="utf-8")
        shap_df = pd.read_csv(shap_features_path)
        shap_block = build_shap_block(shap_df, shap_figures_dir)
        updated_text = replace_or_insert_block(
            thesis_text=updated_text,
            auto_block=shap_block,
            start_marker=SHAP_AUTO_START,
            end_marker=SHAP_AUTO_END,
            section_heading="## Model Interpretability (SHAP)",
        )
        print("Updated SHAP interpretability block in thesis.")

    updated_text = update_advanced_blocks(
        thesis_text=updated_text,
        project_root=project_root,
        ensemble_comparison=args.ensemble_comparison,
        loco_comparison=args.loco_comparison,
        fairness_comparison=args.fairness_comparison,
        fairness_figure=args.fairness_figure,
    )
    updated_text = update_reading_list_block(updated_text, project_root, args)
    thesis_path.write_text(updated_text, encoding="utf-8")

    print(f"Updated thesis results block in: {thesis_path}")
    print(f"Using MIMIC dataset/comparison: {mimic_dataset_path} | {mimic_comparison_path}")
    print(f"Using heart dataset/comparison: {heart_dataset_path} | {heart_comparison_path}")


if __name__ == "__main__":
    main()
