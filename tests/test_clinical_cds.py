"""Unit and integration tests for the Clinical Decision Support (CDS) & FHIR R4 stack."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from clinical_cds import (
    CardiologyWorkupBundle,
    CardiologyWorkupEngine,
    CardiologyWorkupInput,
    ClinicalInferenceEngine,
    ClinicalModelBundle,
    PatientClinicalInput,
    build_clinical_bundle,
    build_workup_bundle,
    compute_psi,
    fhir_bundle_to_patient_input,
    prediction_to_fhir_risk_assessment,
    recalibrate_bundle_for_site,
)
from clinical_cds.api import create_app


@pytest.fixture(scope="module")
def synthetic_cds_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a tiny deterministic v3 ClinicalModelBundle on synthetic cohort for fast testing."""
    tmp_dir = tmp_path_factory.mktemp("cds_bundle")
    rng = np.random.default_rng(42)
    n = 360
    age = rng.uniform(35.0, 75.0, size=n)
    sex = rng.integers(0, 2, size=n)
    height = rng.normal(168.0, 8.0, size=n).clip(145.0, 195.0)
    weight = rng.normal(75.0, 12.0, size=n).clip(48.0, 125.0)
    bmi = weight / ((height / 100.0) ** 2)
    ap_hi = rng.normal(130.0, 18.0, size=n).clip(95.0, 195.0)
    ap_lo = (ap_hi - rng.uniform(30.0, 55.0, size=n)).clip(60.0, 120.0)
    pp = ap_hi - ap_lo
    map_bp = ap_lo + pp / 3.0
    chol = rng.integers(0, 3, size=n)
    gluc = rng.integers(0, 3, size=n)
    smoke = rng.integers(0, 2, size=n)
    alco = rng.integers(0, 2, size=n)
    active = rng.integers(0, 2, size=n)

    logit = 0.08 * (ap_hi - 130.0) + 0.06 * (age - 55.0) + 0.9 * chol + 0.4 * smoke - 0.4 * active
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame({
        "age_years": age,
        "sex": sex,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "ap_hi": ap_hi,
        "ap_lo": ap_lo,
        "pulse_pressure": pp,
        "mean_arterial_pressure": map_bp,
        "cholesterol": chol,
        "gluc": gluc,
        "smoke": smoke,
        "alco": alco,
        "active": active,
        "target": target,
    })
    csv_path = tmp_dir / "synthetic_cvd.csv"
    bundle_path = tmp_dir / "test_cds_bundle.pt"
    df.to_csv(csv_path, index=False)

    build_clinical_bundle(
        dataset_path=csv_path,
        output_path=bundle_path,
        epochs=3,
        batch_size=64,
        k_neighbors=5,
        embed_dim=16,
        n_prototypes=64,
        cv_folds=2,
        seed=42,
    )
    return bundle_path


@pytest.fixture(scope="module")
def synthetic_workup_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Train a tiny deterministic Stage-2 Cardiology Workup bundle for fast testing."""
    tmp_dir = tmp_path_factory.mktemp("workup_bundle")
    rng = np.random.default_rng(7)
    n = 240
    age = rng.uniform(35.0, 77.0, size=n)
    sex = rng.integers(0, 2, size=n)
    cp = rng.integers(1, 5, size=n)
    trestbps = rng.normal(132.0, 17.0, size=n).clip(95.0, 195.0)
    chol = rng.normal(245.0, 48.0, size=n).clip(140.0, 390.0)
    fbs = rng.integers(0, 2, size=n)
    restecg = rng.integers(0, 3, size=n)
    thalach = rng.normal(148.0, 22.0, size=n).clip(85.0, 195.0)
    exang = rng.integers(0, 2, size=n)
    oldpeak = rng.exponential(1.1, size=n).clip(0.0, 5.5)
    slope = rng.integers(1, 4, size=n)
    ca = rng.integers(0, 4, size=n)
    thal = rng.choice([3, 6, 7], size=n)

    logit = (
        0.7 * (cp == 4)
        + 0.8 * exang
        + 0.6 * oldpeak
        + 0.7 * ca
        + 0.8 * (thal == 7)
        - 0.03 * (thalach - 145.0)
        - 1.6
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    target = (rng.uniform(size=n) < prob).astype(int)

    df = pd.DataFrame({
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
        "target": target,
    })
    csv_path = tmp_dir / "synthetic_workup.csv"
    bundle_path = tmp_dir / "test_workup_bundle.pt"
    df.to_csv(csv_path, index=False)

    build_workup_bundle(
        dataset_path=csv_path,
        output_path=bundle_path,
        epochs=6,
        cv_folds=2,
        k_neighbors=5,
        embed_dim=16,
        seed=42,
    )
    return bundle_path


class TestPhysiologicalGuardrails:
    """Verify invalid or corrupted clinical inputs are rejected before inference."""

    def test_rejects_inverted_blood_pressure(self):
        with pytest.raises(ValidationError, match="systolic_bp"):
            PatientClinicalInput(
                age_years=55,
                sex=1,
                height_cm=175,
                weight_kg=80,
                systolic_bp=90,
                diastolic_bp=110,
                cholesterol_level=1,
                glucose_level=1,
            )

    def test_rejects_impossible_bmi(self):
        with pytest.raises(ValidationError, match="BMI"):
            PatientClinicalInput(
                age_years=45,
                sex=0,
                height_cm=220,
                weight_kg=35,
                systolic_bp=120,
                diastolic_bp=75,
                cholesterol_level=1,
                glucose_level=1,
            )


class TestClinicalBundleAndEngine:
    """Verify SHA-256 tamper detection, OOD fail-safe abstention, and immutable recourse."""

    def test_sha256_tamper_detection(self, synthetic_cds_bundle: Path):
        bundle = ClinicalModelBundle.load(synthetic_cds_bundle, device="cpu")
        assert len(bundle.bundle_sha256) == 64

        with pytest.raises(ValueError, match="integrity check failed"):
            ClinicalModelBundle.load(
                synthetic_cds_bundle, device="cpu", expected_sha256="0" * 64
            )

    def test_prediction_structure_and_immutable_recourse(
        self, synthetic_cds_bundle: Path, tmp_path: Path
    ):
        audit_file = tmp_path / "audit.jsonl"
        engine = ClinicalInferenceEngine.from_bundle(
            synthetic_cds_bundle, device="cpu", audit_log_path=audit_file
        )
        high_risk_pt = PatientClinicalInput(
            patient_pseudo_id="PT-HIGH-001",
            age_years=68,
            sex=1,
            height_cm=172,
            weight_kg=95,
            systolic_bp=175,
            diastolic_bp=105,
            cholesterol_level=3,
            glucose_level=2,
            smoke=1,
            alcohol=0,
            active=0,
        )
        resp = engine.predict(high_risk_pt)

        assert 0.0 <= resp.calibrated_risk_probability <= 1.0
        assert resp.triage_decision in ("LOW_RISK", "HIGH_RISK", "DEFER_TO_CLINICIAN")
        assert len(resp.dual_explanation.top_cohort_neighbors) == 5
        assert np.isclose(sum(resp.dual_explanation.biomarker_attentions.values()), 1.0, atol=1e-2)

        # Immutable attributes must never appear in recourse recommendations
        immutable = {"age_years", "sex", "height_cm"}
        for rec in resp.actionable_recourse:
            assert immutable.isdisjoint(set(rec.changes.keys()))
            assert rec.absolute_risk_reduction > 0.0

        # De-identified audit log written without raw pseudo ID
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        logged = json.loads(lines[0])
        assert logged["request_id"] == resp.request_id
        assert "PT-HIGH-001" not in lines[0]

    def test_ood_patient_triggers_fail_safe_deferral(self, synthetic_cds_bundle: Path):
        engine = ClinicalInferenceEngine.from_bundle(synthetic_cds_bundle, device="cpu")
        # Extreme physiologically valid but strongly out-of-distribution profile (108 yrs, 245 mmHg)
        ood_pt = PatientClinicalInput(
            age_years=108,
            sex=0,
            height_cm=140,
            weight_kg=120,
            systolic_bp=245,
            diastolic_bp=150,
            cholesterol_level=3,
            glucose_level=3,
            smoke=1,
            alcohol=1,
            active=0,
        )
        resp = engine.predict(ood_pt)
        assert resp.ood_report.is_ood is True
        assert resp.triage_decision == "DEFER_TO_CLINICIAN"
        assert len(resp.ood_report.reasons) >= 1


class TestFHIRAndFastAPI:
    """Verify HL7 FHIR R4 round-trip interoperability and FastAPI endpoints."""

    def test_fhir_bundle_parsing_and_risk_assessment(self, synthetic_cds_bundle: Path):
        fhir_bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "fhir-pt-999",
                        "gender": "female",
                        "birthDate": "1968-05-15",
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "8302-2"}]},
                        "valueQuantity": {"value": 164, "unit": "cm"},
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "29463-7"}]},
                        "valueQuantity": {"value": 72, "unit": "kg"},
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9"}]},
                        "component": [
                            {
                                "code": {"coding": [{"code": "8480-6"}]},
                                "valueQuantity": {"value": 146},
                            },
                            {
                                "code": {"coding": [{"code": "8462-4"}]},
                                "valueQuantity": {"value": 88},
                            },
                        ],
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"code": "2093-3"}]},
                        "valueQuantity": {"value": 245, "unit": "mg/dL"},
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"code": "2339-0"}]},
                        "valueQuantity": {"value": 98, "unit": "mg/dL"},
                    }
                },
            ],
        }

        pt = fhir_bundle_to_patient_input(fhir_bundle)
        assert pt.sex == 0
        assert pt.cholesterol_level == 3
        assert pt.glucose_level == 1
        assert pt.systolic_bp == 146.0

        engine = ClinicalInferenceEngine.from_bundle(
            synthetic_cds_bundle, device="cpu", audit_log_path=synthetic_cds_bundle.parent / "api_audit.jsonl"
        )
        app = create_app(engine=engine)
        client = TestClient(app)

        assert client.get("/").status_code == 200
        assert client.get("/favicon.ico").status_code == 200
        health = client.get("/v1/health")
        assert health.status_code == 200 and health.json()["status"] == "healthy"
        assert client.get("/v1/model-card").status_code == 200

        fhir_resp = client.post("/v1/predict/fhir", json=fhir_bundle)
        assert fhir_resp.status_code == 200
        ra = fhir_resp.json()
        assert ra["resourceType"] == "RiskAssessment"
        assert ra["subject"]["reference"] == "Patient/fhir-pt-999"
        assert 0.0 <= ra["prediction"][0]["probabilityDecimal"] <= 1.0

        # Combined JSON + FHIR endpoint & live drift telemetry
        combo = client.post("/v1/predict/with-fhir", json=pt.model_dump())
        assert combo.status_code == 200
        combo_body = combo.json()
        assert 0.0 <= combo_body["prediction"]["conformal_low_threshold"] <= 1.0
        assert 0.0 <= combo_body["prediction"]["conformal_high_threshold"] <= 1.0
        assert combo_body["fhir_risk_assessment"]["resourceType"] == "RiskAssessment"

        drift = client.get("/v1/drift")
        assert drift.status_code == 200
        drift_data = drift.json()
        assert drift_data["total_requests"] >= 2
        assert "psi" in drift_data and drift_data["psi_status"] in ("STABLE", "MODERATE_DRIFT", "SIGNIFICANT_DRIFT")

        # Direct JSON prediction & conversion helper sanity check
        pred = engine.predict(pt)
        ra2 = prediction_to_fhir_risk_assessment(pred)
        assert ra2["resourceType"] == "RiskAssessment"


class TestV2OptimizationsAndSiteRecalibration:
    """Verify hybrid ensemble, multi-alpha Mondrian bands, PSI drift, and hospital site recalibration."""

    def test_multialpha_conformal_monotonicity_and_hybrid_ensemble(
        self, synthetic_cds_bundle: Path
    ):
        engine = ClinicalInferenceEngine.from_bundle(synthetic_cds_bundle, device="cpu")
        assert engine.bundle.xgb_model is not None

        base_kwargs = {
            "age_years": 56,
            "sex": 1,
            "height_cm": 174,
            "weight_kg": 84,
            "systolic_bp": 144,
            "diastolic_bp": 90,
            "cholesterol_level": 2,
            "glucose_level": 1,
        }
        r85 = engine.predict(PatientClinicalInput(**base_kwargs, conformal_alpha=0.15))
        r90 = engine.predict(PatientClinicalInput(**base_kwargs, conformal_alpha=0.10))
        r95 = engine.predict(PatientClinicalInput(**base_kwargs, conformal_alpha=0.05))

        assert r90.xgboost_probability is not None
        assert r90.hgb_probability is not None
        assert 0.0 <= r90.model_disagreement <= 1.0
        # Higher nominal coverage (smaller alpha) requires equal or wider deferral band
        band_85 = r85.conformal_high_threshold - r85.conformal_low_threshold
        band_95 = r95.conformal_high_threshold - r95.conformal_low_threshold
        assert band_95 >= band_85 - 1e-6

    def test_hospital_site_recalibration_and_psi(
        self, synthetic_cds_bundle: Path, tmp_path: Path
    ):
        local_csv = synthetic_cds_bundle.parent / "synthetic_cvd.csv"
        out_bundle = tmp_path / "site_zurich_bundle.pt"

        saved_path, card = recalibrate_bundle_for_site(
            local_csv=local_csv,
            site_name="Zurich_Cardiology",
            base_bundle_path=synthetic_cds_bundle,
            output_path=out_bundle,
            device="cpu",
        )
        assert saved_path.exists()
        assert "site_recalibration" in card
        assert card["site_recalibration"]["site_name"] == "Zurich_Cardiology"

        reloaded = ClinicalInferenceEngine.from_bundle(saved_path, device="cpu")
        assert "zurich_cardiology" in reloaded.bundle.model_version

        # PSI on identical distribution must be ~0
        edges = np.linspace(0.0, 1.0, 11)
        expected = np.full(10, 0.1)
        uniform_samples = np.repeat(np.linspace(0.05, 0.95, 10), 10)
        assert compute_psi(expected, uniform_samples, edges) < 1e-6


class TestStage2CardiologyWorkup:
    """Verify Stage-2 Cardiology Workup bundle, engine, recourse, and API endpoints."""

    def test_workup_bundle_engine_and_api(
        self, synthetic_cds_bundle: Path, synthetic_workup_bundle: Path
    ):
        bundle = CardiologyWorkupBundle.load(synthetic_workup_bundle, device="cpu")
        assert len(bundle.bundle_sha256) == 64
        assert abs(sum(bundle.ensemble_weights) - 1.0) < 1e-4

        w_engine = CardiologyWorkupEngine(bundle)
        high_pt = CardiologyWorkupInput(
            patient_pseudo_id="CARD-001",
            age=64,
            sex=1,
            cp=4,
            trestbps=160,
            chol=295,
            fbs=1,
            restecg=2,
            thalach=112,
            exang=1,
            oldpeak=2.8,
            slope=2,
            ca=2,
            thal=7,
            conformal_alpha=0.10,
        )
        resp = w_engine.predict(high_pt)
        assert 0.0 <= resp.calibrated_risk_probability <= 1.0
        assert resp.xgboost_probability is not None
        assert resp.hgb_probability is not None
        assert resp.triage_decision in ("LOW_RISK", "HIGH_RISK", "DEFER_TO_CLINICIAN")
        assert len(resp.dual_explanation.top_cohort_neighbors) == 5

        s1_engine = ClinicalInferenceEngine.from_bundle(synthetic_cds_bundle, device="cpu")
        app = create_app(engine=s1_engine, workup_engine=w_engine)
        client = TestClient(app)

        h1 = client.get("/v1/health").json()
        assert h1["workup_available"] is True
        assert "hist_gradient_boosting" in h1["ensemble_weights"]

        hw = client.get("/v1/workup/health")
        assert hw.status_code == 200
        assert "random_forest" in hw.json()["ensemble_weights"]

        mc = client.get("/v1/workup/model-card")
        assert mc.status_code == 200
        mc_data = mc.json()
        assert "holdout_performance" in mc_data
        assert "mondrian_conformal_by_alpha" in mc_data

        pred_resp = client.post("/v1/workup/predict", json=high_pt.model_dump())
        assert pred_resp.status_code == 200
        assert pred_resp.json()["request_id"].startswith("workup-")

        fhir_resp = client.post("/v1/workup/predict/with-fhir", json=high_pt.model_dump())
        assert fhir_resp.status_code == 200
        assert fhir_resp.json()["fhir_risk_assessment"]["resourceType"] == "RiskAssessment"
        assert fhir_resp.headers["x-content-type-options"] == "nosniff"
        assert fhir_resp.headers["x-frame-options"] == "DENY"

    def test_workup_fhir_intake_audit_metrics_and_api_key_auth(
        self, synthetic_cds_bundle: Path, synthetic_workup_bundle: Path, tmp_path: Path
    ):
        # Verify Stage-2 SHA-256 tamper detection
        with pytest.raises(ValueError, match="integrity check failed"):
            CardiologyWorkupBundle.load(
                synthetic_workup_bundle, device="cpu", expected_sha256="0" * 64
            )

        audit_file = tmp_path / "two_stage_audit.jsonl"
        s1_eng = ClinicalInferenceEngine.from_bundle(
            synthetic_cds_bundle, device="cpu", audit_log_path=audit_file
        )
        w_eng = CardiologyWorkupEngine.from_bundle(
            synthetic_workup_bundle, device="cpu", audit_log_path=audit_file
        )
        app = create_app(engine=s1_eng, workup_engine=w_eng, api_key="hospital-secret-key")
        client = TestClient(app)

        # Public health & Prometheus endpoints stay reachable without auth
        assert client.get("/v1/health").status_code == 200
        assert client.get("/v1/health").json()["auth_enabled"] is True
        assert client.get("/metrics").status_code == 200

        # Protected clinical inference rejects missing/wrong API key
        workup_fhir_bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [
                {
                    "resource": {
                        "resourceType": "Patient",
                        "id": "cardio-pt-42",
                        "gender": "male",
                        "birthDate": "1962-03-10",
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"code": "8480-6"}]},
                        "valueQuantity": {"value": 152},
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"code": "2093-3"}]},
                        "valueQuantity": {"value": 274},
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"code": "8867-4"}]},
                        "valueQuantity": {"value": 122},
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"code": "8633-0"}]},
                        "valueQuantity": {"value": 2.2},
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"code": "89555-7"}]},
                        "valueBoolean": True,
                    }
                },
                {
                    "resource": {
                        "resourceType": "Observation",
                        "code": {"coding": [{"code": "59011-7"}]},
                        "valueInteger": 2,
                    }
                },
            ],
        }
        assert client.post("/v1/workup/predict/fhir", json=workup_fhir_bundle).status_code == 401

        auth_headers = {"Authorization": "Bearer hospital-secret-key"}
        r_fhir = client.post(
            "/v1/workup/predict/fhir", json=workup_fhir_bundle, headers=auth_headers
        )
        assert r_fhir.status_code == 200
        ra = r_fhir.json()
        assert ra["resourceType"] == "RiskAssessment"
        assert ra["subject"]["reference"] == "Patient/cardio-pt-42"

        # Stage-2 audit record logged & filterable in /v1/drift?stage=stage2 and /metrics
        drift_s2 = client.get("/v1/drift?stage=stage2", headers=auth_headers)
        assert drift_s2.status_code == 200
        assert drift_s2.json()["total_requests"] == 1
        assert drift_s2.json()["stage_counts"]["stage2"] == 1

        metrics_txt = client.get("/metrics").text
        assert 'cardiograph_requests_total{stage="stage2"} 1' in metrics_txt

        # Verify SMART on FHIR discovery & launch endpoints + strict zero-pickle weights_only=True loading
        import torch

        assert client.get("/.well-known/smart-configuration").status_code == 200
        assert "stage2_cardiology_workup" in client.get("/.well-known/smart-configuration").json()["fhir_endpoints"]
        assert client.get("/smart/launch?iss=https://sandbox.example/r4&launch=tok123").status_code == 200

        p1 = torch.load(synthetic_cds_bundle, map_location="cpu", weights_only=True)
        p2 = torch.load(synthetic_workup_bundle, map_location="cpu", weights_only=True)
        assert p1.get("safe_serialization") == "weights_only_tensors_json"
        assert p2.get("safe_serialization") == "weights_only_tensors_json"

    def test_silent_shadow_mode_suppresses_clinical_output_and_logs_audit(
        self, synthetic_cds_bundle: Path, synthetic_workup_bundle: Path, tmp_path: Path
    ):
        audit_file = tmp_path / "shadow_audit.jsonl"
        s1_eng = ClinicalInferenceEngine.from_bundle(
            synthetic_cds_bundle, device="cpu", audit_log_path=audit_file
        )
        w_eng = CardiologyWorkupEngine.from_bundle(
            synthetic_workup_bundle, device="cpu", audit_log_path=audit_file
        )
        app = create_app(engine=s1_eng, workup_engine=w_eng, shadow_mode=True)
        client = TestClient(app)

        health = client.get("/v1/health")
        assert health.status_code == 200
        assert health.json()["shadow_mode"] is True
        assert health.headers["x-cds-shadow-mode"] == "true"

        pt1 = PatientClinicalInput(
            patient_pseudo_id="SHADOW-1",
            age_years=60, sex=1, height_cm=175, weight_kg=85,
            systolic_bp=150, diastolic_bp=92, cholesterol_level=2, glucose_level=1,
        )
        r1 = client.post("/v1/shadow/stage1", json=pt1.model_dump())
        assert r1.status_code == 200
        body1 = r1.json()
        assert body1["status"] == "logged_silently"
        assert body1["shadow_mode"] is True
        assert "calibrated_risk_probability" not in body1
        assert "triage_decision" not in body1

        pt2 = CardiologyWorkupInput(
            patient_pseudo_id="SHADOW-2",
            age=62, sex=1, cp=4, trestbps=150, chol=260, fbs=1, restecg=2,
            thalach=125, exang=1, oldpeak=2.0, slope=2, ca=1, thal=7,
        )
        r2 = client.post("/v1/shadow/stage2", json=pt2.model_dump())
        assert r2.status_code == 200
        assert r2.json()["status"] == "logged_silently"
        assert "calibrated_risk_probability" not in r2.json()

        # Both encounters must be silently recorded in de-identified audit log
        lines = audit_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert "SHADOW-1" not in lines[0] and "SHADOW-2" not in lines[1]

    def test_smart_launch_escapes_xss_and_sets_csp(self, synthetic_cds_bundle: Path):
        engine = ClinicalInferenceEngine.from_bundle(synthetic_cds_bundle, device="cpu")
        app = create_app(engine=engine)
        client = TestClient(app)

        payload_iss = 'https://evil.test/" onmouseover="alert(1)'
        resp = client.get(
            "/smart/launch",
            params={"iss": payload_iss, "launch": "<script>alert(1)</script>", "patient": '"bad"'},
        )
        assert resp.status_code == 200
        assert "content-security-policy" in resp.headers
        assert '" onmouseover="' not in resp.text
        assert "<script>alert(1)</script>" not in resp.text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in resp.text

        # Non-http(s) javascript: scheme must fall back to the default safe sandbox URL
        js_resp = client.get("/smart/launch", params={"iss": "javascript:alert(1)"})
        assert js_resp.status_code == 200
        assert "javascript:alert(1)" not in js_resp.text
        assert "launch.smarthealthit.org" in js_resp.text

    def test_hf_space_mode_allows_iframe_framing(self, synthetic_cds_bundle: Path, monkeypatch):
        monkeypatch.setenv("SPACE_ID", "YahyaLendani/cardiograph-cds")
        engine = ClinicalInferenceEngine.from_bundle(synthetic_cds_bundle, device="cpu")
        app = create_app(engine=engine)
        client = TestClient(app)

        resp = client.get("/")
        assert resp.status_code == 200
        assert "x-frame-options" not in resp.headers
        assert "frame-ancestors https://huggingface.co https://*.hf.space" in resp.headers["content-security-policy"]

    def test_weighted_conformal_sets_zero_label_covariate_shift(self):
        from clinical_cds import compute_weighted_conformal_sets

        rng = np.random.default_rng(7)
        X_src = rng.normal(0.0, 1.0, size=(300, 4))
        logits_src = 1.6 * X_src[:, 0] - 0.8 * X_src[:, 1]
        p_src = 1.0 / (1.0 + np.exp(-logits_src))
        y_src = (rng.random(300) < p_src).astype(int)

        X_tgt = rng.normal(0.6, 1.0, size=(120, 4))
        logits_tgt = 1.6 * X_tgt[:, 0] - 0.8 * X_tgt[:, 1]
        p_tgt = 1.0 / (1.0 + np.exp(-logits_tgt))
        y_tgt = (rng.random(120) < p_tgt).astype(int)

        sets, diag = compute_weighted_conformal_sets(
            X_source=X_src,
            cal_probs=p_src,
            cal_y=y_src,
            X_target=X_tgt,
            target_probs=p_tgt,
            alpha=0.10,
            seed=7,
        )
        assert len(sets) == len(y_tgt)
        cov = float(np.mean([int(int(y) in s) for y, s in zip(y_tgt, sets, strict=True)]))
        assert cov >= 0.82
        assert 0.5 < diag["domain_auc"] <= 1.0
        assert diag["effective_sample_size"] > 10.0

    def test_shadow_trial_report_generator_and_esc_score2(
        self, synthetic_cds_bundle: Path, tmp_path: Path
    ):
        import pandas as pd

        from clinical_cds import generate_shadow_trial_report
        from experiments.framingham_benchmark import compute_esc_score2_2021_risk

        audit_file = tmp_path / "shadow_audit.jsonl"
        engine = ClinicalInferenceEngine.from_bundle(
            synthetic_cds_bundle, device="cpu", audit_log_path=audit_file
        )
        outcomes_rows = []
        rng = np.random.default_rng(11)
        for i in range(16):
            high = i % 2 == 1
            pt = PatientClinicalInput(
                age_years=66 if high else 42,
                sex=i % 2,
                height_cm=172,
                weight_kg=92 if high else 66,
                systolic_bp=168 if high else 116,
                diastolic_bp=98 if high else 74,
                cholesterol_level=3 if high else 1,
                glucose_level=2 if high else 1,
                smoke=1 if high else 0,
                active=0 if high else 1,
            )
            pred = engine.predict(pt)
            outcomes_rows.append(
                {
                    "request_id": pred.request_id,
                    "target": 1 if (high or rng.random() < 0.1) else 0,
                    "sex": "male" if pt.sex == 1 else "female",
                }
            )
        outcomes_csv = tmp_path / "outcomes.csv"
        pd.DataFrame(outcomes_rows).to_csv(outcomes_csv, index=False)

        report = generate_shadow_trial_report(
            audit_log_path=audit_file,
            outcomes_csv=outcomes_csv,
            stage="stage1",
            output_dir=tmp_path,
        )
        assert report["n_matched_outcomes"] == 16
        assert "sex" in report["subgroups"]
        assert (tmp_path / "shadow_trial_report.md").exists()
        assert (tmp_path / "shadow_trial_report.json").exists()

        # Verify ESC SCORE2 (2021 Poland High-Risk) monotonicity: older smoking hypertensive > young healthy
        demo_df = pd.DataFrame(
            [
                {"age_years": 42, "sex": 0, "ap_hi": 115, "cholesterol": 0, "gluc": 0, "smoke": 0},
                {"age_years": 68, "sex": 1, "ap_hi": 170, "cholesterol": 2, "gluc": 2, "smoke": 1},
            ]
        )
        s2 = compute_esc_score2_2021_risk(demo_df, region="high")
        assert 0.0 < s2[0] < s2[1] < 1.0
