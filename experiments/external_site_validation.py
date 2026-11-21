"""External Validation & Local Hospital Site Recalibration Proof (calibrate-site).

Evaluates the Stage-1 Clinical CDS v3 bundle (trained on OpenML 45547, N=68,589) under
real distribution shift — both zero-shot and after 2-second local hospital recalibration
via ``clinical_cds.calibrate_site.recalibrate_bundle_for_site``:

1. **Cross-Dataset Hospital Transfer (OpenML 68.6k -> UCI 4-Center Consortium, N=918):**
   Maps unseen international hospital referrals (Cleveland, Budapest, Zurich, Long Beach VA)
   into the Stage-1 screening interface (age, sex, resting SBP, cholesterol tier, fasting
   glucose tier) to quantify zero-shot prevalence/covariate shift and local recalibration gain.
2. **Real Subpopulation Clinic Shift (OpenML Holdout Test Set, N=10,289):**
   Simulates two realistic specialty clinics with strongly shifted base rates:
   - *Geriatric Cardiology Clinic* (patients aged >= 58, ~68% CVD prevalence)
   - *Preventive Wellness Clinic* (patients aged < 50, ~28% CVD prevalence)

Outputs:
- ``artifacts/external_site_validation.json``
- ``artifacts/external_site_validation.md``
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from clinical_cds import ClinicalInferenceEngine, PatientClinicalInput, recalibrate_bundle_for_site
from clinical_cds.bundle import DEFAULT_BUNDLE_PATH
from models import compute_ece
from project_paths import ARTIFACTS_DIR, resolve_data_path


def _evaluate_df(
    engine: ClinicalInferenceEngine, df: pd.DataFrame, alpha: float = 0.10
) -> dict[str, float | int]:
    """Run batch Stage-1 inference on a dataframe and compute AUC, ECE, Brier, and conformal coverage."""
    patients = [
        PatientClinicalInput(
            age_years=float(r["age_years"]),
            sex=int(r["sex"]),
            height_cm=float(r["height"]),
            weight_kg=float(r["weight"]),
            systolic_bp=float(r["ap_hi"]),
            diastolic_bp=float(r["ap_lo"]),
            cholesterol_level=int(r["cholesterol"]) + 1,
            glucose_level=int(r["gluc"]) + 1,
            smoke=int(r["smoke"]),
            alcohol=int(r["alco"]),
            active=int(r["active"]),
            conformal_alpha=alpha,
        )
        for _, r in df.iterrows()
    ]
    responses = engine.predict_batch(patients)
    probs = np.asarray([resp.calibrated_risk_probability for resp in responses], dtype=np.float64)
    y_true = df["target"].values.astype(int)
    covered = [int(int(y) in resp.conformal_prediction_set) for y, resp in zip(y_true, responses, strict=True)]
    deferred = [int(resp.triage_decision == "DEFER_TO_CLINICIAN") for resp in responses]

    return {
        "n_eval": int(len(df)),
        "prevalence": round(float(np.mean(y_true)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "brier": round(float(brier_score_loss(y_true, probs)), 4),
        "ece": round(float(compute_ece(probs, y_true)), 4),
        "conformal_90_coverage": round(float(np.mean(covered)), 4),
        "deferral_rate": round(float(np.mean(deferred)), 4),
    }


def _map_uci_to_screening_df(uci_df: pd.DataFrame) -> pd.DataFrame:
    """Map UCI 4-center cardiology patients into Stage-1 screening vitals (conservative anthropometric defaults)."""
    df = uci_df.dropna(subset=["target", "age", "sex"]).copy()
    sbp = pd.to_numeric(df["trestbps"], errors="coerce").replace(0, np.nan).fillna(132.0).clip(90.0, 220.0)
    dbp = (sbp * 0.62).clip(55.0, 125.0)
    dbp = np.minimum(dbp, sbp - 15.0)
    chol_raw = pd.to_numeric(df["chol"], errors="coerce").replace(0, np.nan).fillna(235.0)
    chol_tier = np.where(chol_raw < 200.0, 0, np.where(chol_raw < 240.0, 1, 2))
    fbs = pd.to_numeric(df["fbs"], errors="coerce").fillna(0).astype(int)
    gluc_tier = np.where(fbs == 1, 2, 0)
    sex = pd.to_numeric(df["sex"], errors="coerce").fillna(1).astype(int)
    height = np.where(sex == 1, 174.0, 162.0)
    weight = np.where(sex == 1, 80.0, 67.0)
    bmi = weight / ((height / 100.0) ** 2)
    pp = sbp - dbp
    map_bp = dbp + pp / 3.0

    return pd.DataFrame({
        "age_years": pd.to_numeric(df["age"], errors="coerce").clip(20.0, 95.0).values,
        "sex": sex.values,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "ap_hi": sbp.values,
        "ap_lo": dbp.values,
        "pulse_pressure": pp.values,
        "mean_arterial_pressure": map_bp.values,
        "cholesterol": chol_tier,
        "gluc": gluc_tier,
        "smoke": np.zeros(len(df), dtype=int),
        "alco": np.zeros(len(df), dtype=int),
        "active": np.ones(len(df), dtype=int),
        "target": df["target"].astype(int).values,
    })


def run_external_validation(
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
    openml_csv: str | Path = "cleaned_cardio_70k.csv",
    uci_csv: str | Path = "cleaned_heart_disease_combined.csv",
    output_dir: str | Path = ARTIFACTS_DIR,
    seed: int = 42,
) -> dict[str, Any]:
    """Run zero-shot vs local site-recalibrated shift benchmark across 3 real cohorts."""
    base_engine = ClinicalInferenceEngine.from_bundle(bundle_path, device="cpu")

    uci_raw = pd.read_csv(resolve_data_path(uci_csv))
    uci_mapped = _map_uci_to_screening_df(uci_raw)

    openml_df = pd.read_csv(resolve_data_path(openml_csv))
    # Recreate the untouched 15% holdout test split from seed=42
    _, openml_test = train_test_split(
        openml_df, test_size=0.15, random_state=seed, stratify=openml_df["target"]
    )
    # Apply the same physiological input guardrails enforced by PatientClinicalInput
    openml_test = openml_test[
        (openml_test["ap_hi"] > openml_test["ap_lo"] + 5.0)
        & (openml_test["ap_hi"].between(60.0, 250.0))
        & (openml_test["ap_lo"].between(40.0, 160.0))
        & (openml_test["bmi"].between(12.0, 70.0))
    ].reset_index(drop=True)
    geriatric_clinic = openml_test[openml_test["age_years"] >= 58.0].reset_index(drop=True)
    wellness_clinic = openml_test[openml_test["age_years"] < 50.0].reset_index(drop=True)

    scenarios: list[tuple[str, str, pd.DataFrame]] = [
        ("UCI_4Center_External", "UCI 4-Center Hospital Consortium (External Dataset)", uci_mapped),
        ("Geriatric_Cardiology_Clinic", "Geriatric Cardiology Clinic (Age >= 58, High Prevalence)", geriatric_clinic),
        ("Preventive_Wellness_Clinic", "Preventive Wellness Clinic (Age < 50, Low Prevalence)", wellness_clinic),
    ]

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for site_id, label, site_df in scenarios:
            cal_df, eval_df = train_test_split(
                site_df, test_size=0.50, random_state=seed, stratify=site_df["target"]
            )
            cal_df = cal_df.reset_index(drop=True)
            eval_df = eval_df.reset_index(drop=True)

            zero_shot = _evaluate_df(base_engine, eval_df, alpha=0.10)

            cal_csv = tmp_path / f"{site_id}_cal.csv"
            cal_bundle = tmp_path / f"{site_id}_recal.pt"
            cal_df.to_csv(cal_csv, index=False)
            recalibrate_bundle_for_site(
                local_csv=cal_csv,
                site_name=site_id,
                base_bundle_path=bundle_path,
                output_path=cal_bundle,
                device="cpu",
            )
            recal_engine = ClinicalInferenceEngine.from_bundle(cal_bundle, device="cpu")
            after_recal = _evaluate_df(recal_engine, eval_df, alpha=0.10)

            results.append({
                "site_id": site_id,
                "label": label,
                "n_calibration": int(len(cal_df)),
                "zero_shot": zero_shot,
                "recalibrated": after_recal,
                "ece_delta": round(float(after_recal["ece"] - zero_shot["ece"]), 4),
                "coverage_delta": round(
                    float(after_recal["conformal_90_coverage"] - zero_shot["conformal_90_coverage"]), 4
                ),
            })

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"bundle_version": base_engine.bundle.model_version, "scenarios": results}
    (out_dir / "external_site_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# External Site Transfer & Local Recalibration Benchmark (`calibrate-site`)",
        "",
        f"Base bundle: `{base_engine.bundle.model_version}` (OpenML 45547, N=68,589).",
        "Each target clinic is split 50/50 into a local calibration slice and an untouched local holdout slice.",
        "",
        "| Target Clinic / Cohort | Eval N | Prevalence | Zero-Shot ECE | Recalibrated ECE | Zero-Shot 90% Cov | Recalibrated 90% Cov | ROC-AUC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in results:
        zs, rc = r["zero_shot"], r["recalibrated"]
        md_lines.append(
            f"| **{r['label']}** | {zs['n_eval']} | {zs['prevalence']*100:.1f}% | "
            f"{zs['ece']:.4f} | **{rc['ece']:.4f}** | "
            f"{zs['conformal_90_coverage']*100:.1f}% | **{rc['conformal_90_coverage']*100:.1f}%** | "
            f"{rc['roc_auc']:.4f} |"
        )
    (out_dir / "external_site_validation.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("\n".join(md_lines))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="External validation & local site recalibration proof.")
    parser.add_argument("--bundle", type=str, default=str(DEFAULT_BUNDLE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(ARTIFACTS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    run_external_validation(bundle_path=args.bundle, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
