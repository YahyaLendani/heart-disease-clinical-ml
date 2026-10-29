"""Build an admission-level hospital mortality dataset from the MIMIC-IV demo.

Prediction time (t0) = hospital admission time (``admittime``).

Default feature set ``admission`` uses ONLY information available at t0:
  - demographics (age at admission, gender, race, marital status, insurance)
  - admission context (admission type / location, ED visit and ED length of stay,
    which precede the inpatient admission)
  - history from the patient's PRIOR admissions (count, recency, prior ICU stays,
    prior diagnosis burden and comorbidity flags)

The legacy ``discharge`` feature set reproduces the original dataset, which
includes variables only known at discharge (hospital LOS, ICU LOS/count, and
diagnoses coded for the current admission). Those variables leak the outcome:
for patients who die, ``dischtime`` is effectively the time of death. Keep it
only for a leakage sensitivity analysis, never as the primary model.

No global imputation is performed here: missing values are left as NaN and are
imputed inside the training pipelines (fit on training folds only).
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from project_paths import PROCESSED_DATA_DIR, RAW_DATA_DIR

# Set MIMIC_IV_ROOT or pass --mimic-root to point at the extracted MIMIC-IV demo.
DEFAULT_MIMIC_ROOT = Path(
    os.environ.get("MIMIC_IV_ROOT", RAW_DATA_DIR / "mimic-iv-clinical-database-demo-2.2")
)

ID_AND_TARGET = ["subject_id", "hadm_id", "hospital_expire_flag"]

ADMISSION_FEATURES = [
    "age_at_admission",
    "gender",
    "admission_type",
    "admission_location",
    "insurance",
    "marital_status",
    "race",
    "has_ed_visit",
    "ed_los_hours",
    "prior_admission_count",
    "days_since_last_discharge",
    "prior_icu_stay_count",
    "prior_diag_count",
    "prior_circulatory",
    "prior_diabetes",
    "prior_ckd",
]

# Variables only known after the outcome window closes (discharge or death).
LEAKY_DISCHARGE_FEATURES = [
    "hospital_los_days",
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

DISCHARGE_FEATURES = [
    "anchor_age",
    "gender",
    "admission_type",
    "admission_location",
    "insurance",
    "marital_status",
    "race",
    "ed_los_hours",
    *LEAKY_DISCHARGE_FEATURES,
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build hospital mortality modeling dataset from MIMIC-IV demo.")
    parser.add_argument("--mimic-root", type=str, default=str(DEFAULT_MIMIC_ROOT))
    parser.add_argument(
        "--feature-set",
        choices=["admission", "discharge"],
        default="admission",
        help="'admission' (leakage-free, default) or 'discharge' (legacy, leaky; sensitivity analysis only).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV name. Defaults to mimic_hospital_mortality.csv (admission) "
        "or mimic_hospital_mortality_discharge.csv (discharge).",
    )
    return parser


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def _flag_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    diagnoses = diagnoses.copy()
    diagnoses["icd_code"] = diagnoses["icd_code"].astype(str).str.upper().str.strip()
    diagnoses["icd_version"] = pd.to_numeric(diagnoses["icd_version"], errors="coerce")
    v9 = diagnoses["icd_version"] == 9
    v10 = diagnoses["icd_version"] == 10
    code = diagnoses["icd_code"]

    # ICD-9 circulatory system = 390-459; ICD-10 = chapter I.
    icd9_num = pd.to_numeric(code.str[:3], errors="coerce")
    diagnoses["is_circulatory"] = ((v10 & code.str.startswith("I")) | (v9 & icd9_num.between(390, 459))).astype(int)
    diagnoses["is_diabetes"] = ((v10 & code.str.match(r"^E1[0-4]")) | (v9 & code.str.startswith("250"))).astype(int)
    diagnoses["is_ckd"] = ((v10 & code.str.startswith("N18")) | (v9 & code.str.startswith("585"))).astype(int)
    return diagnoses


def _prior_history_features(admissions: pd.DataFrame, icustays: pd.DataFrame, diagnoses: pd.DataFrame) -> pd.DataFrame:
    """For each admission, aggregate information from the SAME patient's earlier admissions.

    An earlier admission counts only if it was discharged before the current admittime,
    so nothing from the current stay (or the future) can enter the features.
    """
    adm = admissions[["subject_id", "hadm_id", "admittime", "dischtime"]]
    pairs = adm.merge(adm, on="subject_id", suffixes=("", "_prior"))
    pairs = pairs[pairs["dischtime_prior"] < pairs["admittime"]]

    icu_per_adm = icustays.groupby("hadm_id").size().rename("icu_n").reset_index()
    diag_per_adm = (
        diagnoses.groupby("hadm_id")
        .agg(
            diag_n=("icd_code", "count"),
            circ=("is_circulatory", "max"),
            diab=("is_diabetes", "max"),
            ckd=("is_ckd", "max"),
        )
        .reset_index()
    )
    pairs = pairs.merge(icu_per_adm, left_on="hadm_id_prior", right_on="hadm_id", how="left", suffixes=("", "_icu"))
    pairs = pairs.merge(diag_per_adm, left_on="hadm_id_prior", right_on="hadm_id", how="left", suffixes=("", "_diag"))
    pairs["gap_days"] = (pairs["admittime"] - pairs["dischtime_prior"]).dt.total_seconds() / 86400.0

    hist = (
        pairs.groupby("hadm_id")
        .agg(
            prior_admission_count=("hadm_id_prior", "nunique"),
            days_since_last_discharge=("gap_days", "min"),
            prior_icu_stay_count=("icu_n", "sum"),
            prior_diag_count=("diag_n", "sum"),
            prior_circulatory=("circ", "max"),
            prior_diabetes=("diab", "max"),
            prior_ckd=("ckd", "max"),
        )
        .reset_index()
    )

    out = admissions[["hadm_id"]].merge(hist, on="hadm_id", how="left")
    zero_fill = ["prior_admission_count", "prior_icu_stay_count", "prior_diag_count",
                 "prior_circulatory", "prior_diabetes", "prior_ckd"]
    out[zero_fill] = out[zero_fill].fillna(0)
    # days_since_last_discharge stays NaN for first admissions (imputed in-pipeline).
    return out


def build_dataset(mimic_root: Path, feature_set: str = "admission") -> pd.DataFrame:
    admissions = _safe_read_csv(mimic_root / "hosp" / "admissions.csv")
    patients = _safe_read_csv(mimic_root / "hosp" / "patients.csv")
    icustays = _safe_read_csv(mimic_root / "icu" / "icustays.csv")
    diagnoses = _flag_diagnoses(_safe_read_csv(mimic_root / "hosp" / "diagnoses_icd.csv"))

    for time_col in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]:
        if time_col in admissions.columns:
            admissions[time_col] = pd.to_datetime(admissions[time_col], errors="coerce")

    admissions["hospital_expire_flag"] = (
        pd.to_numeric(admissions["hospital_expire_flag"], errors="coerce").fillna(0).astype(int)
    )
    admissions["ed_los_hours"] = (admissions["edouttime"] - admissions["edregtime"]).dt.total_seconds() / 3600
    admissions["has_ed_visit"] = admissions["edregtime"].notna().astype(int)

    patients = patients[["subject_id", "gender", "anchor_age", "anchor_year"]].copy()
    model_df = admissions.merge(patients, on="subject_id", how="left")
    # MIMIC-IV convention: age at admission = anchor_age + (admission year - anchor_year).
    model_df["age_at_admission"] = model_df["anchor_age"] + (model_df["admittime"].dt.year - model_df["anchor_year"])

    if feature_set == "admission":
        model_df = model_df.merge(_prior_history_features(admissions, icustays, diagnoses), on="hadm_id", how="left")
        features = ADMISSION_FEATURES
    elif feature_set == "discharge":
        model_df["hospital_los_days"] = (model_df["dischtime"] - model_df["admittime"]).dt.total_seconds() / 86400
        icu_agg = icustays.groupby("hadm_id", as_index=False).agg(
            icu_stay_count=("stay_id", "nunique"),
            icu_total_los_days=("los", "sum"),
            first_careunit=("first_careunit", "first"),
        )
        icu_agg["has_icu_stay"] = 1
        diag_agg = diagnoses.groupby("hadm_id", as_index=False).agg(
            diag_count=("icd_code", "count"),
            unique_diag_count=("icd_code", "nunique"),
            has_circulatory=("is_circulatory", "max"),
            has_diabetes=("is_diabetes", "max"),
            has_ckd=("is_ckd", "max"),
        )
        model_df = model_df.merge(icu_agg, on="hadm_id", how="left").merge(diag_agg, on="hadm_id", how="left")
        fill_zero = ["has_icu_stay", "icu_stay_count", "icu_total_los_days", "diag_count",
                     "unique_diag_count", "has_circulatory", "has_diabetes", "has_ckd"]
        model_df[fill_zero] = model_df[fill_zero].fillna(0)
        features = DISCHARGE_FEATURES
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    model_df = model_df[ID_AND_TARGET + features].replace("?", np.nan)
    return model_df.drop_duplicates(subset=["hadm_id"]).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    mimic_root = Path(args.mimic_root)
    default_name = (
        "mimic_hospital_mortality.csv" if args.feature_set == "admission" else "mimic_hospital_mortality_discharge.csv"
    )
    output_path = PROCESSED_DATA_DIR / (args.output or default_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(mimic_root, feature_set=args.feature_set)
    dataset.to_csv(output_path, index=False)

    print(f"MIMIC root: {mimic_root}")
    print(f"Feature set: {args.feature_set}")
    if args.feature_set == "discharge":
        print("WARNING: discharge feature set contains post-outcome variables (target leakage).")
    print(f"Rows: {len(dataset)} | Deaths: {int(dataset['hospital_expire_flag'].sum())}")
    print(f"Hospital mortality rate: {dataset['hospital_expire_flag'].mean() * 100:.2f}%")
    print(f"Saved: {output_path.name}")


if __name__ == "__main__":
    main()
