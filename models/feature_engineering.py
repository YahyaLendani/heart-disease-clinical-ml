"""Clinical feature engineering for cardiovascular risk prediction.

Cardiology-informed features:
1. Rate-Pressure Product (RPP):
   RPP = (trestbps * thalach) / 100
   Clinical index of myocardial oxygen consumption (MVO2) and workload.

2. Chronotropic Reserve Ratio (% of age-predicted maximum heart rate):
   Max_HR_predicted = 220 - age
   HR_reserve_ratio = thalach / Max_HR_predicted
   Inability to reach 85% of age-predicted max HR (chronotropic incompetence)
   is an established predictor of coronary disease and mortality.

3. Exercise Ischemia Burden:
   ischemia_burden = oldpeak * exang
   Joint measure of ST depression magnitude multiplied by exercise angina symptom.

4. Vessel-Ischemia Severity:
   vessel_ischemia = ca * oldpeak
   Combines anatomical fluoroscopy vessel occlusion with electrical ischemia.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_cardiovascular_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute and append domain-specific cardiology features without data leakage.

    Missing values in source features propagate as NaN and are imputed
    downstream inside per-fold pipelines.
    """
    out = df.copy()

    # 1. Rate-Pressure Product (RPP)
    if "trestbps" in out.columns and "thalach" in out.columns:
        out["rpp"] = (out["trestbps"] * out["thalach"]) / 100.0

    # 2. Chronotropic Reserve Ratio
    if "age" in out.columns and "thalach" in out.columns:
        predicted_max_hr = 220.0 - out["age"]
        # Protect against div by zero or negative predicted max HR
        predicted_max_hr = np.where(predicted_max_hr <= 0, 1.0, predicted_max_hr)
        out["chronotropic_ratio"] = out["thalach"] / predicted_max_hr

    # 3. Exercise Ischemia Burden
    if "oldpeak" in out.columns and "exang" in out.columns:
        out["ischemia_burden"] = out["oldpeak"] * out["exang"]

    # 4. Vessel-Ischemia Severity
    if "ca" in out.columns and "oldpeak" in out.columns:
        out["vessel_ischemia"] = out["ca"] * out["oldpeak"]

    return out
