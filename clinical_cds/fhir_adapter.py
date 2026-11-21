"""HL7 FHIR R4 interoperability adapter (Bundle -> PatientClinicalInput -> RiskAssessment).

Supports standard LOINC observation codes used across hospital EHRs (Epic, Cerner,
SMART on FHIR):
- ``8302-2``  Body height (cm)
- ``29463-7`` Body weight (kg)
- ``8480-6``  Systolic blood pressure (mmHg) — standalone or inside panel ``85354-9``
- ``8462-4``  Diastolic blood pressure (mmHg) — standalone or inside panel ``85354-9``
- ``2093-3``  Total cholesterol (mg/dL -> tier 1: <200, 2: 200-239, 3: >=240)
- ``2339-0``  Glucose (mg/dL -> tier 1: <100, 2: 100-125, 3: >=126)
- ``72166-2`` Tobacco smoking status (0/1)
- ``74013-4`` Alcohol intake (0/1)
- ``68516-4`` Physical activity (0/1)
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from .schemas import ClinicalPredictionResponse, PatientClinicalInput

LOINC_HEIGHT = "8302-2"
LOINC_WEIGHT = "29463-7"
LOINC_SBP = "8480-6"
LOINC_DBP = "8462-4"
LOINC_BP_PANEL = "85354-9"
LOINC_CHOL = "2093-3"
LOINC_GLUC = "2339-0"
LOINC_SMOKE = "72166-2"
LOINC_ALCO = "74013-4"
LOINC_ACTIVE = "68516-4"


def _extract_loinc_codes(code_obj: dict[str, Any] | None) -> set[str]:
    if not code_obj:
        return set()
    return {
        str(c.get("code"))
        for c in code_obj.get("coding", [])
        if c.get("code") is not None
    }


def _age_from_birthdate(birth_date_str: str) -> float:
    bdate = datetime.strptime(birth_date_str[:10], "%Y-%m-%d").date()
    today = date.today()
    return round((today - bdate).days / 365.25, 2)


def _cholesterol_mg_dl_to_tier(val: float) -> int:
    if val in (1, 2, 3):
        return int(val)
    if val < 200.0:
        return 1
    if val < 240.0:
        return 2
    return 3


def _glucose_mg_dl_to_tier(val: float) -> int:
    if val in (1, 2, 3):
        return int(val)
    if val < 100.0:
        return 1
    if val < 126.0:
        return 2
    return 3


def fhir_bundle_to_patient_input(fhir_bundle: dict[str, Any]) -> PatientClinicalInput:
    """Parse an HL7 FHIR R4 Bundle (Patient + Observations) into PatientClinicalInput."""
    if fhir_bundle.get("resourceType") != "Bundle":
        raise ValueError("Expected a FHIR R4 resource with resourceType == 'Bundle'.")

    entries = fhir_bundle.get("entry", [])
    extracted: dict[str, Any] = {"smoke": 0, "alcohol": 0, "active": 1}

    for entry in entries:
        res = entry.get("resource", {})
        rtype = res.get("resourceType")

        if rtype == "Patient":
            if res.get("id"):
                extracted["patient_pseudo_id"] = str(res["id"])
            gender = str(res.get("gender", "")).lower()
            if gender == "male":
                extracted["sex"] = 1
            elif gender == "female":
                extracted["sex"] = 0
            if res.get("birthDate"):
                extracted["age_years"] = _age_from_birthdate(str(res["birthDate"]))

        elif rtype == "Observation":
            codes = _extract_loinc_codes(res.get("code"))
            val_qty = res.get("valueQuantity", {}).get("value")
            val_int = res.get("valueInteger")
            val_bool = res.get("valueBoolean")

            # Handle composite Blood Pressure Panel (LOINC 85354-9)
            if LOINC_BP_PANEL in codes or "component" in res:
                for comp in res.get("component", []):
                    c_codes = _extract_loinc_codes(comp.get("code"))
                    c_val = comp.get("valueQuantity", {}).get("value")
                    if c_val is not None:
                        if LOINC_SBP in c_codes:
                            extracted["systolic_bp"] = float(c_val)
                        elif LOINC_DBP in c_codes:
                            extracted["diastolic_bp"] = float(c_val)

            if val_qty is not None:
                val_f = float(val_qty)
                if LOINC_HEIGHT in codes:
                    extracted["height_cm"] = val_f
                elif LOINC_WEIGHT in codes:
                    extracted["weight_kg"] = val_f
                elif LOINC_SBP in codes:
                    extracted["systolic_bp"] = val_f
                elif LOINC_DBP in codes:
                    extracted["diastolic_bp"] = val_f
                elif LOINC_CHOL in codes:
                    extracted["cholesterol_level"] = _cholesterol_mg_dl_to_tier(val_f)
                elif LOINC_GLUC in codes:
                    extracted["glucose_level"] = _glucose_mg_dl_to_tier(val_f)

            if val_int is not None or val_bool is not None:
                flag = int(bool(val_bool)) if val_bool is not None else int(val_int)
                if LOINC_SMOKE in codes:
                    extracted["smoke"] = 1 if flag else 0
                elif LOINC_ALCO in codes:
                    extracted["alcohol"] = 1 if flag else 0
                elif LOINC_ACTIVE in codes:
                    extracted["active"] = 1 if flag else 0

    required = [
        "age_years",
        "sex",
        "height_cm",
        "weight_kg",
        "systolic_bp",
        "diastolic_bp",
        "cholesterol_level",
        "glucose_level",
    ]
    missing = [k for k in required if k not in extracted]
    if missing:
        raise ValueError(f"FHIR Bundle missing required clinical observations/demographics: {missing}")

    return PatientClinicalInput(**extracted)


def prediction_to_fhir_risk_assessment(
    prediction: ClinicalPredictionResponse,
    subject_reference: str | None = None,
) -> dict[str, Any]:
    """Serialize a ClinicalPredictionResponse into a valid HL7 FHIR R4 RiskAssessment."""
    qualitative_map = {
        "LOW_RISK": ("low", "Low Cardiovascular Risk"),
        "HIGH_RISK": ("high", "High Cardiovascular Risk"),
        "DEFER_TO_CLINICIAN": ("indeterminate", "Indeterminate — Defer to Clinician Review"),
    }
    q_code, q_display = qualitative_map[prediction.triage_decision]
    top_feats = list(prediction.dual_explanation.biomarker_attentions.items())[:5]
    feat_summary = ", ".join(f"{k}={v*100:.1f}%" for k, v in top_feats)

    resource: dict[str, Any] = {
        "resourceType": "RiskAssessment",
        "id": prediction.request_id,
        "status": "final",
        "occurrenceDateTime": prediction.timestamp_utc,
        "method": {
            "text": f"GraphAugmentedClinicalTransformer ({prediction.model_version})",
        },
        "prediction": [
            {
                "outcome": {
                    "coding": [
                        {
                            "system": "http://snomed.info/sct",
                            "code": "49601007",
                            "display": "Disorder of cardiovascular system",
                        }
                    ],
                    "text": "Cardiovascular Disease Risk",
                },
                "probabilityDecimal": round(prediction.calibrated_risk_probability, 4),
                "qualitativeRisk": {
                    "coding": [
                        {
                            "system": "http://terminology.hl7.org/CodeSystem/risk-probability",
                            "code": q_code,
                            "display": q_display,
                        }
                    ],
                    "text": q_display,
                },
                "rationale": (
                    f"Conformal set={prediction.conformal_prediction_set} "
                    f"(nominal {int((1 - prediction.conformal_alpha) * 100)}%); "
                    f"OOD={prediction.ood_report.is_ood}; "
                    f"Graph gate={prediction.dual_explanation.graph_reliance_gate:.2f}; "
                    f"Top biomarkers: {feat_summary}."
                ),
            }
        ],
        "note": [
            {"text": prediction.triage_reason},
            {"text": prediction.clinical_disclaimer},
        ],
        "extension": [
            {
                "url": "https://heart-disease-clinical-ml.org/fhir/StructureDefinition/conformal-triage",
                "valueString": prediction.triage_decision,
            },
            {
                "url": "https://heart-disease-clinical-ml.org/fhir/StructureDefinition/bundle-sha256",
                "valueString": prediction.bundle_sha256,
            },
            {
                "url": "https://heart-disease-clinical-ml.org/fhir/StructureDefinition/ood-flag",
                "valueBoolean": prediction.ood_report.is_ood,
            },
        ],
    }

    if subject_reference:
        resource["subject"] = {"reference": subject_reference}
    if prediction.actionable_recourse:
        resource["mitigation"] = "; ".join(
            f"{r.intervention} (projected risk {r.projected_risk_probability*100:.1f}%, "
            f"-{r.absolute_risk_reduction*100:.1f}%)"
            for r in prediction.actionable_recourse[:3]
        )
    return resource
