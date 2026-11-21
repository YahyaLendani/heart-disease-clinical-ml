"""Prepare Large-Scale Public Cardiovascular Disease Cohort (OpenML ID 45547, N=70,000).

Fetches the 70,000-patient Cardiovascular Disease dataset via scikit-learn's
``fetch_openml(data_id=45547)``, enforces strict clinical physiological validity
bounds (eliminating negative/corrupted blood pressure sensor readings), removes
exact duplicate patient records to prevent train/test leakage, and engineers
domain-specific cardiovascular features:

- ``age_years``: Patient age converted from days to years (``age / 365.25``)
- ``bmi``: Body Mass Index (``weight / (height / 100)^2``)
- ``pulse_pressure``: Systolic minus Diastolic BP (``ap_hi - ap_lo``)
- ``mean_arterial_pressure``: ``ap_lo + (ap_hi - ap_lo) / 3``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml

from project_paths import PROCESSED_DATA_DIR


def prepare_large_cvd_cohort(
    output_path: str | Path = PROCESSED_DATA_DIR / "cleaned_cardio_70k.csv",
) -> pd.DataFrame:
    """Download, clean, deduplicate, and feature-engineer the 70k CVD cohort."""
    print("Fetching OpenML Cardiovascular Disease Dataset (ID=45547, N=70,000)...")
    ds = fetch_openml(data_id=45547, as_frame=True, parser="auto")
    df = ds.frame.copy()
    initial_n = len(df)

    # Convert all columns to numeric cleanly
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna().reset_index(drop=True)

    # Strict physiological validity bounds (remove corrupted BP/anthropometric sensor entries)
    valid_mask = (
        df["ap_hi"].between(60, 250)
        & df["ap_lo"].between(40, 160)
        & (df["ap_hi"] > df["ap_lo"])
        & df["height"].between(120, 230)
        & df["weight"].between(30, 200)
    )
    df = df.loc[valid_mask].copy()
    after_bounds_n = len(df)

    # Remove exact duplicate records to guarantee zero cross-split leakage
    df = df.drop_duplicates().reset_index(drop=True)
    after_dedup_n = len(df)

    # Clinical feature engineering
    df["age_years"] = (df["age"] / 365.25).round(2)
    df["bmi"] = (df["weight"] / ((df["height"] / 100.0) ** 2)).round(2)
    df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
    df["mean_arterial_pressure"] = (df["ap_lo"] + df["pulse_pressure"] / 3.0).round(2)

    # Normalize binary/ordinal encodings cleanly
    df["sex"] = (df["gender"] - 1).astype(int)  # 0/1
    df["cholesterol"] = (df["cholesterol"] - 1).astype(int)  # 0/1/2
    df["gluc"] = (df["gluc"] - 1).astype(int)  # 0/1/2
    df["smoke"] = df["smoke"].astype(int)
    df["alco"] = df["alco"].astype(int)
    df["active"] = df["active"].astype(int)
    df["target"] = df["cardio"].astype(int)

    ordered_cols = [
        "age_years",
        "sex",
        "height",
        "weight",
        "bmi",
        "ap_hi",
        "ap_lo",
        "pulse_pressure",
        "mean_arterial_pressure",
        "cholesterol",
        "gluc",
        "smoke",
        "alco",
        "active",
        "target",
    ]
    df = df[ordered_cols]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    pos_rate = float(np.mean(df["target"])) * 100.0
    print(
        f"Cleaned Large CVD Cohort -> Initial: {initial_n:,} | "
        f"After physiological bounds: {after_bounds_n:,} | "
        f"After deduplication: {after_dedup_n:,} ({pos_rate:.1f}% CVD+)"
    )
    print(f"Saved cleaned dataset to: {out.resolve()}")
    return df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare 70,000-patient OpenML CVD dataset.")
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED_DATA_DIR / "cleaned_cardio_70k.csv"),
    )
    args = parser.parse_args(argv)
    prepare_large_cvd_cohort(args.output)


if __name__ == "__main__":
    main()
