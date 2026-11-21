"""FastAPI Clinical Decision Support Service + Interactive Bedside Clinician UI.

Endpoints:
- ``GET  /``                      Interactive Bedside Clinician Decision Support UI
- ``GET  /favicon.ico``           Inline SVG heart-pulse favicon (eliminates 404 noise)
- ``GET  /v1/health``             Readiness probe + bundle SHA-256 checksum
- ``GET  /v1/model-card``         Transparent clinical Model Card (metrics & subgroup audit)
- ``GET  /v1/drift``              Live audit telemetry & OOD/deferral drift monitor
- ``POST /v1/predict``            Validated JSON clinical risk + conformal triage + explanations
- ``POST /v1/predict/with-fhir``  Returns both ClinicalPredictionResponse and FHIR R4 RiskAssessment
- ``POST /v1/predict/batch``      High-throughput batched clinical inference
- ``POST /v1/predict/fhir``       HL7 FHIR R4 Bundle -> FHIR R4 RiskAssessment
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from project_paths import ARTIFACTS_DIR

from .bundle import DEFAULT_BUNDLE_PATH
from .engine import ClinicalInferenceEngine
from .fhir_adapter import fhir_bundle_to_patient_input, prediction_to_fhir_risk_assessment
from .schemas import ClinicalPredictionResponse, PatientClinicalInput

STATIC_UI_PATH = Path(__file__).resolve().parent / "static" / "index.html"
FAVICON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'>"
    "<rect width='32' height='32' rx='8' fill='#0f5c6e'/>"
    "<path d='M5 17h5l2.5-6 4 12 3-7 2 1h5.5' fill='none' stroke='#fff' "
    "stroke-width='2.4' stroke-linecap='round' stroke-linejoin='round'/>"
    "</svg>"
)


def create_app(
    engine: ClinicalInferenceEngine | None = None,
    bundle_path: str | Path | None = None,
    audit_log_path: str | Path | None = None,
) -> FastAPI:
    """Create the FastAPI Clinical Decision Support application."""
    app = FastAPI(
        title="CardioGraph CDS — Graph-Augmented Clinical Transformer API",
        version="1.1.0",
        description=(
            "Fail-safe clinical decision support API with physiological guardrails, "
            "sex-stratified Mondrian conformal triage, OOD detection, dual biomarker + "
            "cohort-neighbor interpretability, actionable recourse, live drift telemetry, "
            "and HL7 FHIR R4 interoperability."
        ),
    )

    state_engine: dict[str, ClinicalInferenceEngine | None] = {"engine": engine}

    def _get_engine() -> ClinicalInferenceEngine:
        if state_engine["engine"] is None:
            resolved_bundle = Path(
                bundle_path or os.environ.get("CDS_BUNDLE_PATH", str(DEFAULT_BUNDLE_PATH))
            )
            resolved_audit = Path(
                audit_log_path
                or os.environ.get("CDS_AUDIT_LOG_PATH", str(ARTIFACTS_DIR / "cds_audit.jsonl"))
            )
            state_engine["engine"] = ClinicalInferenceEngine.from_bundle(
                bundle_path=resolved_bundle,
                device=os.environ.get("CDS_DEVICE", "cpu"),
                audit_log_path=resolved_audit,
            )
        return state_engine["engine"]

    @app.get("/", response_class=HTMLResponse, tags=["Clinician UI"])
    def clinician_ui() -> HTMLResponse:
        return HTMLResponse(content=STATIC_UI_PATH.read_text(encoding="utf-8"))

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(content=FAVICON_SVG, media_type="image/svg+xml")

    @app.get("/v1/health", tags=["Operations"])
    def health_check() -> dict[str, Any]:
        eng = _get_engine()
        return {
            "status": "healthy",
            "model_version": eng.bundle.model_version,
            "bundle_sha256": eng.bundle.bundle_sha256,
            "device": str(eng.device),
            "cohort_bank_size": int(eng.bundle.H_tab_bank.size(0)),
            "k_neighbors": eng.bundle.k_neighbors,
        }

    @app.get("/v1/model-card", tags=["Governance"])
    def get_model_card() -> dict[str, Any]:
        eng = _get_engine()
        return {
            **eng.bundle.model_card,
            "bundle_sha256": eng.bundle.bundle_sha256,
        }

    @app.get("/v1/drift", tags=["Operations"])
    def get_drift_telemetry(window: int = 1000) -> dict[str, Any]:
        """Compute rolling clinical safety, deferral, OOD, and latency telemetry from audit logs."""
        eng = _get_engine()
        log_path = eng.audit_log_path
        if log_path is None or not log_path.exists():
            return {
                "total_requests": 0,
                "window_size": window,
                "mean_risk": 0.0,
                "deferral_rate": 0.0,
                "ood_rate": 0.0,
                "mean_latency_ms": 0.0,
                "triage_counts": {},
            }

        lines = log_path.read_text(encoding="utf-8").strip().splitlines()
        recent = [json.loads(line) for line in lines[-max(1, window) :] if line.strip()]
        n = len(recent)
        if n == 0:
            return {
                "total_requests": 0,
                "window_size": window,
                "mean_risk": 0.0,
                "deferral_rate": 0.0,
                "ood_rate": 0.0,
                "mean_latency_ms": 0.0,
                "triage_counts": {},
            }

        triage_counts = Counter(r["triage_decision"] for r in recent)
        return {
            "total_requests": n,
            "window_size": window,
            "mean_risk": round(sum(float(r["calibrated_risk_probability"]) for r in recent) / n, 4),
            "deferral_rate": round(triage_counts.get("DEFER_TO_CLINICIAN", 0) / n, 4),
            "ood_rate": round(sum(1 for r in recent if r.get("is_ood")) / n, 4),
            "mean_latency_ms": round(sum(float(r["latency_ms"]) for r in recent) / n, 2),
            "triage_counts": dict(triage_counts),
        }

    @app.post("/v1/predict", response_model=ClinicalPredictionResponse, tags=["Clinical Inference"])
    def predict_single(patient: PatientClinicalInput) -> ClinicalPredictionResponse:
        return _get_engine().predict(patient)

    @app.post("/v1/predict/with-fhir", tags=["Clinical Inference"])
    def predict_with_fhir(patient: PatientClinicalInput) -> dict[str, Any]:
        pred = _get_engine().predict(patient)
        subject_ref = f"Patient/{patient.patient_pseudo_id}" if patient.patient_pseudo_id else None
        return {
            "prediction": pred.model_dump(),
            "fhir_risk_assessment": prediction_to_fhir_risk_assessment(
                pred, subject_reference=subject_ref
            ),
        }

    @app.post(
        "/v1/predict/batch",
        response_model=list[ClinicalPredictionResponse],
        tags=["Clinical Inference"],
    )
    def predict_batch(patients: list[PatientClinicalInput]) -> list[ClinicalPredictionResponse]:
        if len(patients) > 512:
            raise HTTPException(status_code=400, detail="Maximum batch size is 512 patients.")
        return _get_engine().predict_batch(patients)

    @app.post("/v1/predict/fhir", tags=["HL7 FHIR R4"])
    def predict_fhir(fhir_bundle: dict[str, Any]) -> dict[str, Any]:
        try:
            patient = fhir_bundle_to_patient_input(fhir_bundle)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        prediction = _get_engine().predict(patient)
        subject_ref = f"Patient/{patient.patient_pseudo_id}" if patient.patient_pseudo_id else None
        return prediction_to_fhir_risk_assessment(prediction, subject_reference=subject_ref)

    return app


app = create_app()
