"""FastAPI Clinical Decision Support Service + Interactive Bedside Clinician UI.

Endpoints:
- ``GET  /``                      Interactive 2-Stage Bedside Clinician Decision Support UI
- ``GET  /favicon.ico``           Inline SVG heart-pulse favicon (eliminates 404 noise)
- ``GET  /v1/health``             Readiness probe + bundle SHA-256 checksum + Stage-2 status
- ``GET  /v1/model-card``         Transparent Stage-1 clinical Model Card (metrics & subgroup audit)
- ``GET  /v1/drift``              Live audit telemetry & OOD/deferral drift monitor
- ``POST /v1/predict``            Stage-1 primary-care screening risk + conformal triage
- ``POST /v1/predict/with-fhir``  Returns both ClinicalPredictionResponse and FHIR R4 RiskAssessment
- ``POST /v1/predict/batch``      High-throughput batched Stage-1 clinical inference
- ``POST /v1/predict/fhir``       HL7 FHIR R4 Bundle -> FHIR R4 RiskAssessment
- ``GET  /v1/workup/health``      Stage-2 Cardiology Workup readiness probe
- ``GET  /v1/workup/model-card``  Stage-2 Cardiology Workup Model Card (N=918 multi-center)
- ``POST /v1/workup/predict``     Stage-2 Cardiology Workup risk + conformal triage + explanations
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response

from project_paths import ARTIFACTS_DIR

from .bundle import DEFAULT_BUNDLE_PATH
from .engine import ClinicalInferenceEngine, compute_psi
from .fhir_adapter import fhir_bundle_to_patient_input, prediction_to_fhir_risk_assessment
from .schemas import CardiologyWorkupInput, ClinicalPredictionResponse, PatientClinicalInput
from .workup import DEFAULT_WORKUP_BUNDLE_PATH, CardiologyWorkupEngine

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
    workup_engine: CardiologyWorkupEngine | None = None,
    workup_bundle_path: str | Path | None = None,
) -> FastAPI:
    """Create the FastAPI Clinical Decision Support application."""
    app = FastAPI(
        title="CardioGraph CDS — 2-Stage Graph-Augmented Clinical Transformer API",
        version="2.0.0",
        description=(
            "Fail-safe 2-stage clinical decision support API combining Stage-1 Primary Care "
            "Screening (68.6k cohort) with Stage-2 Cardiology Workup escalation (N=918 ECG/stress "
            "cohort), featuring PLR+UniMP Graph-Augmented Transformers, 5-fold OOF 3-paradigm "
            "ensembles, sex-stratified Mondrian conformal triage, OOD guardrails, actionable "
            "recourse, live drift telemetry, and HL7 FHIR R4 interoperability."
        ),
    )

    state_engine: dict[str, Any] = {
        "engine": engine,
        "workup_engine": workup_engine,
    }

    def _resolve_workup_bundle_path() -> Path:
        return Path(
            workup_bundle_path
            or os.environ.get("CDS_WORKUP_BUNDLE_PATH", str(DEFAULT_WORKUP_BUNDLE_PATH))
        )

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

    def _get_workup_engine() -> CardiologyWorkupEngine:
        if state_engine["workup_engine"] is None:
            resolved_workup = _resolve_workup_bundle_path()
            if not resolved_workup.exists():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Stage-2 Cardiology Workup bundle is not built yet. "
                        "Run `heart-pipeline build-workup-bundle` to generate it."
                    ),
                )
            state_engine["workup_engine"] = CardiologyWorkupEngine.from_bundle(
                bundle_path=resolved_workup,
                device=os.environ.get("CDS_DEVICE", "cpu"),
            )
        return state_engine["workup_engine"]

    @app.get("/", response_class=HTMLResponse, tags=["Clinician UI"])
    def clinician_ui() -> HTMLResponse:
        return HTMLResponse(content=STATIC_UI_PATH.read_text(encoding="utf-8"))

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(content=FAVICON_SVG, media_type="image/svg+xml")

    @app.get("/v1/health", tags=["Operations"])
    def health_check() -> dict[str, Any]:
        eng = _get_engine()
        w_gat, w_xgb, w_hgb = eng.bundle.ensemble_weights
        workup_avail = (
            state_engine["workup_engine"] is not None or _resolve_workup_bundle_path().exists()
        )
        return {
            "status": "healthy",
            "model_version": eng.bundle.model_version,
            "bundle_sha256": eng.bundle.bundle_sha256,
            "device": str(eng.device),
            "cohort_bank_size": int(eng.bundle.H_tab_bank.size(0)),
            "k_neighbors": eng.bundle.k_neighbors,
            "hybrid_ensemble": eng.bundle.xgb_model is not None,
            "ensemble_gat_weight": round(eng.bundle.ensemble_gat_weight, 4),
            "ensemble_weights": {
                "graph_augmented_transformer": round(float(w_gat), 4),
                "xgboost": round(float(w_xgb), 4),
                "hist_gradient_boosting": round(float(w_hgb), 4),
            },
            "supported_alphas": sorted(eng.bundle.conformal_quantiles.keys()),
            "workup_available": workup_avail,
        }

    @app.get("/v1/model-card", tags=["Governance"])
    def get_model_card() -> dict[str, Any]:
        eng = _get_engine()
        return {
            **eng.bundle.model_card,
            "bundle_sha256": eng.bundle.bundle_sha256,
        }

    @app.get("/v1/workup/health", tags=["Stage-2 Cardiology Workup"])
    def workup_health_check() -> dict[str, Any]:
        weng = _get_workup_engine()
        w_gat, w_xgb, w_rf = weng.bundle.ensemble_weights
        return {
            "status": "healthy",
            "model_version": weng.bundle.model_version,
            "bundle_sha256": weng.bundle.bundle_sha256,
            "device": str(weng.device),
            "cohort_bank_size": int(weng.bundle.H_tab_bank.size(0)),
            "k_neighbors": weng.bundle.k_neighbors,
            "ensemble_weights": {
                "graph_augmented_transformer": round(float(w_gat), 4),
                "xgboost": round(float(w_xgb), 4),
                "random_forest": round(float(w_rf), 4),
            },
            "supported_alphas": sorted(weng.bundle.conformal_quantiles.keys()),
        }

    @app.get("/v1/workup/model-card", tags=["Stage-2 Cardiology Workup"])
    def get_workup_model_card() -> dict[str, Any]:
        weng = _get_workup_engine()
        return {
            **weng.bundle.model_card,
            "bundle_sha256": weng.bundle.bundle_sha256,
        }

    @app.post(
        "/v1/workup/predict",
        response_model=ClinicalPredictionResponse,
        tags=["Stage-2 Cardiology Workup"],
    )
    def predict_workup(patient: CardiologyWorkupInput) -> ClinicalPredictionResponse:
        return _get_workup_engine().predict(patient)

    @app.post("/v1/workup/predict/with-fhir", tags=["Stage-2 Cardiology Workup"])
    def predict_workup_with_fhir(patient: CardiologyWorkupInput) -> dict[str, Any]:
        pred = _get_workup_engine().predict(patient)
        subject_ref = f"Patient/{patient.patient_pseudo_id}" if patient.patient_pseudo_id else None
        return {
            "prediction": pred.model_dump(),
            "fhir_risk_assessment": prediction_to_fhir_risk_assessment(
                pred, subject_reference=subject_ref
            ),
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
                "mean_disagreement": 0.0,
                "mean_latency_ms": 0.0,
                "psi": 0.0,
                "psi_status": "STABLE",
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
                "mean_disagreement": 0.0,
                "mean_latency_ms": 0.0,
                "psi": 0.0,
                "psi_status": "STABLE",
                "triage_counts": {},
            }

        risks = np.asarray([float(r["calibrated_risk_probability"]) for r in recent], dtype=np.float64)
        psi = compute_psi(eng.bundle.val_risk_hist_props, risks, eng.bundle.val_risk_hist_edges)
        if psi < 0.10:
            psi_status = "STABLE"
        elif psi < 0.25:
            psi_status = "MODERATE_DRIFT"
        else:
            psi_status = "SIGNIFICANT_DRIFT"

        triage_counts = Counter(r["triage_decision"] for r in recent)
        return {
            "total_requests": n,
            "window_size": window,
            "mean_risk": round(float(np.mean(risks)), 4),
            "deferral_rate": round(triage_counts.get("DEFER_TO_CLINICIAN", 0) / n, 4),
            "ood_rate": round(sum(1 for r in recent if r.get("is_ood")) / n, 4),
            "mean_disagreement": round(
                sum(float(r.get("model_disagreement", 0.0)) for r in recent) / n, 4
            ),
            "mean_latency_ms": round(sum(float(r["latency_ms"]) for r in recent) / n, 2),
            "psi": round(psi, 4),
            "psi_status": psi_status,
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
