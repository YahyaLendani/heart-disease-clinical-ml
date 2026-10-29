from pathlib import Path

import numpy as np
import pandas as pd


def _compute_missingness(df: pd.DataFrame) -> pd.DataFrame:
    missing_counts = df.isna().sum()
    missing_pct = (missing_counts / len(df) * 100.0).round(2)
    table = pd.DataFrame(
        {
            "feature": missing_counts.index,
            "missing_count": missing_counts.values,
            "missing_percent": missing_pct.values,
        }
    )
    return table.sort_values(by=["missing_count", "feature"], ascending=[False, True]).reset_index(drop=True)


def _compute_outlier_counts(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            outlier_count = 0
        else:
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_count = int(((series < lower) | (series > upper)).sum())
        rows.append(
            {
                "feature": col,
                "outlier_count": outlier_count,
                "outlier_percent": round(outlier_count / max(len(df), 1) * 100.0, 2),
            }
        )
    return pd.DataFrame(rows).sort_values(by="outlier_count", ascending=False).reset_index(drop=True)


def _compute_target_group_stats(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby("target")[numeric_cols]
    stats = grouped.agg(["mean", "median", "std", "min", "max"])
    stats.columns = [f"{col}_{agg}" for col, agg in stats.columns]
    return stats.round(4)


def _compute_target_separation(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    healthy = df[df["target"] == 0]
    sick = df[df["target"] == 1]

    for col in numeric_cols:
        a = pd.to_numeric(healthy[col], errors="coerce").dropna()
        b = pd.to_numeric(sick[col], errors="coerce").dropna()
        if len(a) == 0 or len(b) == 0:
            continue

        mean_healthy = float(a.mean())
        mean_sick = float(b.mean())
        diff = mean_sick - mean_healthy

        var_healthy = float(a.var(ddof=1)) if len(a) > 1 else 0.0
        var_sick = float(b.var(ddof=1)) if len(b) > 1 else 0.0
        pooled_sd = np.sqrt(((len(a) - 1) * var_healthy + (len(b) - 1) * var_sick) / max(len(a) + len(b) - 2, 1))
        effect_size = diff / pooled_sd if pooled_sd > 0 else 0.0

        rows.append(
            {
                "feature": col,
                "mean_target_0": round(mean_healthy, 4),
                "mean_target_1": round(mean_sick, 4),
                "mean_diff_1_minus_0": round(diff, 4),
                "cohen_d": round(effect_size, 4),
                "abs_cohen_d": round(abs(effect_size), 4),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(by="abs_cohen_d", ascending=False)
        .drop(columns=["abs_cohen_d"])
        .reset_index(drop=True)
    )


def _compute_target_correlations(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    target_series = pd.to_numeric(df["target"], errors="coerce")
    for col in numeric_cols:
        if col == "target":
            continue
        feature_series = pd.to_numeric(df[col], errors="coerce")
        pearson_corr = feature_series.corr(target_series, method="pearson")
        spearman_corr = feature_series.corr(target_series, method="spearman")
        rows.append(
            {
                "feature": col,
                "pearson_corr_with_target": round(float(pearson_corr), 4) if pd.notna(pearson_corr) else 0.0,
                "spearman_corr_with_target": round(float(spearman_corr), 4) if pd.notna(spearman_corr) else 0.0,
            }
        )
    corr_df = pd.DataFrame(rows)
    if corr_df.empty:
        return corr_df
    corr_df["abs_pearson"] = corr_df["pearson_corr_with_target"].abs()
    corr_df = corr_df.sort_values(by="abs_pearson", ascending=False).drop(columns=["abs_pearson"])
    return corr_df.reset_index(drop=True)


def run_eda(df: pd.DataFrame, output_dir: Path) -> None:
    print("\n=== EDA: Dataset Overview ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    print("\n=== EDA: Missing Values ===")
    missing_table = _compute_missingness(df)
    missing_nonzero = missing_table[missing_table["missing_count"] > 0]
    if missing_nonzero.empty:
        print("No missing values found.")
    else:
        print(missing_nonzero.to_string(index=False))

    print("\n=== EDA: Data Types ===")
    print(df.dtypes.to_string())

    print("\n=== EDA: Target Distribution ===")
    target_counts = df["target"].value_counts().sort_index()
    target_ratio = (target_counts / len(df) * 100).round(2)
    target_summary = pd.DataFrame(
        {
            "count": target_counts,
            "percent": target_ratio,
        }
    )
    print(target_summary.to_string())

    numeric_cols = df.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

    print("\n=== EDA: Numeric Summary ===")
    print(df[numeric_cols].describe().round(2).to_string())

    print("\n=== EDA: Outlier Counts (IQR Rule) ===")
    outlier_table = _compute_outlier_counts(df, numeric_cols)
    print(outlier_table.head(10).to_string(index=False))

    print("\n=== EDA: Category Levels ===")
    for col in categorical_cols:
        top_values = df[col].value_counts().head(5)
        print(f"\n{col} (top 5):")
        print(top_values.to_string())

    print("\n=== EDA: Numeric Means by Target ===")
    means_by_target = df.groupby("target")[numeric_cols].mean().round(2)
    print(means_by_target.to_string())

    print("\n=== EDA: Target-Stratified Stats (mean/median/std/min/max) ===")
    target_group_stats = _compute_target_group_stats(df, numeric_cols)
    print(target_group_stats.to_string())

    print("\n=== EDA: Correlation with Target (Top 10 by |Pearson|) ===")
    target_corr = _compute_target_correlations(df, numeric_cols)
    if target_corr.empty:
        print("No numeric features available for correlation analysis.")
    else:
        print(target_corr.head(10).to_string(index=False))

    print("\n=== EDA: Class Separation (Top 10 by |Cohen's d|) ===")
    separation = _compute_target_separation(df, numeric_cols)
    if separation.empty:
        print("No numeric features available for class-separation analysis.")
    else:
        print(separation.head(10).to_string(index=False))

    output_dir.mkdir(parents=True, exist_ok=True)
    missing_table.to_csv(output_dir / "eda_missingness.csv", index=False)
    outlier_table.to_csv(output_dir / "eda_outliers_iqr.csv", index=False)
    target_group_stats.to_csv(output_dir / "eda_target_group_stats.csv")
    target_corr.to_csv(output_dir / "eda_target_correlations.csv", index=False)
    separation.to_csv(output_dir / "eda_target_separation.csv", index=False)
    print(f"\nSaved EDA artifacts to: {output_dir}")


def main(argv: list[str] | None = None) -> None:
    import argparse

    from project_paths import PRIMARY_HEART_DATASET, resolve_data_path, resolve_output_dir

    parser = argparse.ArgumentParser(description="Exploratory data analysis of the cleaned heart cohort.")
    parser.add_argument("--dataset", type=str, default=PRIMARY_HEART_DATASET)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args(argv)

    cleaned_path = resolve_data_path(args.dataset)
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
    print(f"Loaded cleaned dataset: {cleaned_path.name}")

    run_eda(df, resolve_output_dir(args.output_dir))


if __name__ == "__main__":
    main()
