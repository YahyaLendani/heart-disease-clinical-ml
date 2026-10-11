import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


DEFAULT_MIMIC_ROOT = Path("c:/Users/yahya/Downloads/mimic-iv-clinical-database-demo-2.2")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build hospital mortality modeling dataset from MIMIC-IV demo.")
    parser.add_argument("--mimic-root", type=str, default=str(DEFAULT_MIMIC_ROOT))
    parser.add_argument("--output", type=str, default="mimic_hospital_mortality.csv")
    return parser


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def build_dataset(mimic_root: Path) -> pd.DataFrame:
    admissions = _safe_read_csv(mimic_root / "hosp" / "admissions.csv")
    patients = _safe_read_csv(mimic_root / "hosp" / "patients.csv")
    icustays = _safe_read_csv(mimic_root / "icu" / "icustays.csv")
    diagnoses = _safe_read_csv(mimic_root / "hosp" / "diagnoses_icd.csv")

    admissions = admissions.copy()
    patients = patients.copy()
    icustays = icustays.copy()
    diagnoses = diagnoses.copy()

    for time_col in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]:
        if time_col in admissions.columns:
            admissions[time_col] = pd.to_datetime(admissions[time_col], errors="coerce")

    admissions["hospital_los_days"] = (
        admissions["dischtime"] - admissions["admittime"]
    ).dt.total_seconds() / (24 * 3600)

    admissions["ed_los_hours"] = (
        admissions["edouttime"] - admissions["edregtime"]
    ).dt.total_seconds() / 3600

    admissions["hospital_expire_flag"] = pd.to_numeric(
        admissions["hospital_expire_flag"], errors="coerce"
    ).fillna(0).astype(int)

    patients = patients[["subject_id", "gender", "anchor_age"]].copy()
    patients["anchor_age"] = pd.to_numeric(patients["anchor_age"], errors="coerce")

    icu_agg = (
        icustays.groupby("hadm_id", as_index=False)
        .agg(
            icu_stay_count=("stay_id", "nunique"),
            icu_total_los_days=("los", "sum"),
            first_careunit=("first_careunit", "first"),
        )
    )
    icu_agg["has_icu_stay"] = 1

    diagnoses["icd_code"] = diagnoses["icd_code"].astype(str).str.upper().str.strip()
    diagnoses["icd_version"] = pd.to_numeric(diagnoses["icd_version"], errors="coerce")

    diagnoses["is_circulatory"] = np.where(
        (diagnoses["icd_version"] == 10) & diagnoses["icd_code"].str.startswith("I"),
        1,
        np.where(
            (diagnoses["icd_version"] == 9)
            & diagnoses["icd_code"].str.match(r"^([0-3][0-9]{2}|4[0-5][0-9])", na=False),
            1,
            0,
        ),
    )

    diagnoses["is_diabetes"] = np.where(
        (diagnoses["icd_version"] == 10) & diagnoses["icd_code"].str.startswith("E1"),
        1,
        np.where(
            (diagnoses["icd_version"] == 9) & diagnoses["icd_code"].str.startswith("250"),
            1,
            0,
        ),
    )

    diagnoses["is_ckd"] = np.where(
        (diagnoses["icd_version"] == 10) & diagnoses["icd_code"].str.startswith("N18"),
        1,
        np.where(
            (diagnoses["icd_version"] == 9) & diagnoses["icd_code"].str.startswith("585"),
            1,
            0,
        ),
    )

    diag_agg = (
        diagnoses.groupby("hadm_id", as_index=False)
        .agg(
            diag_count=("icd_code", "count"),
            unique_diag_count=("icd_code", "nunique"),
            has_circulatory=("is_circulatory", "max"),
            has_diabetes=("is_diabetes", "max"),
            has_ckd=("is_ckd", "max"),
        )
    )

    model_df = admissions.merge(patients, on="subject_id", how="left")
    model_df = model_df.merge(icu_agg, on="hadm_id", how="left")
    model_df = model_df.merge(diag_agg, on="hadm_id", how="left")

    model_df["has_icu_stay"] = model_df["has_icu_stay"].fillna(0).astype(int)
    model_df["icu_stay_count"] = model_df["icu_stay_count"].fillna(0)
    model_df["icu_total_los_days"] = model_df["icu_total_los_days"].fillna(0.0)

    for col in ["diag_count", "unique_diag_count", "has_circulatory", "has_diabetes", "has_ckd"]:
        model_df[col] = model_df[col].fillna(0)

    features = [
        "subject_id",
        "hadm_id",
        "hospital_expire_flag",
        "anchor_age",
        "gender",
        "admission_type",
        "admission_location",
        "insurance",
        "marital_status",
        "race",
        "hospital_los_days",
        "ed_los_hours",
        "has_icu_stay",
        "icu_stay_count",
        "icu_total_los_days",
        "first_careunit",
        "diag_count",
        "unique_diag_count",
        "has_circulatory",
        "has_diabetes",
        "has_ckd",
    ]

    model_df = model_df[features].copy()

    model_df = model_df.replace("?", pd.NA)
    for col in model_df.columns:
        if is_numeric_dtype(model_df[col]):
            model_df[col] = model_df[col].fillna(model_df[col].median())
        else:
            mode_value = model_df[col].mode(dropna=True)
            fill_value = mode_value.iloc[0] if not mode_value.empty else "UNKNOWN"
            model_df[col] = model_df[col].fillna(fill_value)

    model_df = model_df.drop_duplicates(subset=["hadm_id"]).reset_index(drop=True)
    return model_df


def main() -> None:
    args = build_arg_parser().parse_args()
    mimic_root = Path(args.mimic_root)
    output_path = Path(__file__).with_name(args.output)

    dataset = build_dataset(mimic_root)
    dataset.to_csv(output_path, index=False)

    mortality_rate = dataset["hospital_expire_flag"].mean() * 100
    print(f"MIMIC root: {mimic_root}")
    print(f"Rows: {len(dataset)}")
    print(f"Hospital mortality rate: {mortality_rate:.2f}%")
    print(f"Saved: {output_path.name}")


if __name__ == "__main__":
    main()
