"""FastAPI Clinical Decision Support Service + Interactive Clinician Web UI.

Endpoints:
- ``GET  /``                  Interactive Clinician Bedside Decision Support UI
- ``GET  /v1/health``         Readiness probe + bundle SHA-256 checksum
- ``GET  /v1/model-card``     Transparent clinical Model Card (metrics & subgroup audit)
- ``POST /v1/predict``        Validated JSON clinical risk + conformal triage + explanations
- ``POST /v1/predict/batch``  High-throughput batched clinical inference
- ``POST /v1/predict/fhir``   HL7 FHIR R4 Bundle -> FHIR R4 RiskAssessment
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from project_paths import ARTIFACTS_DIR

from .bundle import DEFAULT_BUNDLE_PATH
from .engine import ClinicalInferenceEngine
from .fhir_adapter import fhir_bundle_to_patient_input, prediction_to_fhir_risk_assessment
from .schemas import ClinicalPredictionResponse, PatientClinicalInput

CLINICIAN_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Clinical CDS — Graph-Augmented Transformer</title>
  <style>
    :root { --bg:#0f172a; --card:#1e293b; --muted:#94a3b8; --text:#f8fafc; --accent:#38bdf8; --low:#10b981; --high:#ef4444; --defer:#f59e0b; }
    * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { margin:0; background:var(--bg); color:var(--text); padding:24px; }
    .container { max-width:1160px; margin:0 auto; }
    .banner { background:#7c2d12; border:1px solid #ea580c; color:#ffedd5; padding:10px 16px; border-radius:8px; font-size:13px; margin-bottom:20px; }
    h1 { margin:0 0 6px; font-size:24px; }
    .sub { color:var(--muted); margin:0 0 22px; font-size:14px; }
    .grid { display:grid; grid-template-columns: 380px 1fr; gap:20px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card { background:var(--card); border:1px solid #334155; border-radius:12px; padding:20px; }
    .form-grid { display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
    label { display:block; font-size:12px; color:var(--muted); margin-bottom:4px; }
    input, select { width:100%; padding:8px 10px; border-radius:6px; border:1px solid #475569; background:#0f172a; color:var(--text); font-size:14px; }
    button { width:100%; margin-top:16px; padding:12px; border:none; border-radius:8px; background:var(--accent); color:#0f172a; font-weight:700; font-size:15px; cursor:pointer; }
    button:hover { opacity:0.92; }
    .badge { display:inline-block; padding:6px 14px; border-radius:999px; font-weight:700; font-size:13px; letter-spacing:0.4px; }
    .LOW_RISK { background:rgba(16,185,129,0.2); color:#34d399; border:1px solid var(--low); }
    .HIGH_RISK { background:rgba(239,68,68,0.2); color:#f87171; border:1px solid var(--high); }
    .DEFER_TO_CLINICIAN { background:rgba(245,158,11,0.2); color:#fbbf24; border:1px solid var(--defer); }
    .kpi-row { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; margin-bottom:16px; }
    .risk-num { font-size:40px; font-weight:800; }
    .bar-row { display:flex; align-items:center; gap:10px; margin:6px 0; font-size:13px; }
    .bar-label { width:165px; color:#cbd5e1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .bar-track { flex:1; background:#0f172a; height:10px; border-radius:5px; overflow:hidden; }
    .bar-fill { background:var(--accent); height:100%; border-radius:5px; }
    table { width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }
    th, td { text-align:left; padding:7px 8px; border-bottom:1px solid #334155; }
    th { color:var(--muted); font-weight:600; }
    .sec-title { font-size:14px; font-weight:700; color:#e2e8f0; margin:18px 0 8px; text-transform:uppercase; letter-spacing:0.5px; }
    .rec-item { background:#0f172a; border-left:3px solid var(--low); padding:9px 12px; border-radius:6px; margin-bottom:8px; font-size:13px; }
    .err { color:#fca5a5; font-size:13px; margin-top:10px; }
  </style>
</head>
<body>
  <div class="container">
    <div class="banner"><strong>CLINICAL DECISION SUPPORT PROTOTYPE:</strong> For research and clinician-in-the-loop review only. Not FDA-cleared or CE-marked medical software.</div>
    <h1>Graph-Augmented Clinical Transformer — Bedside CDS</h1>
    <p class="sub">Calibrated cardiovascular risk triage with Mondrian conformal abstention, OOD guardrails, dual biomarker + cohort-neighbor attention, and immutable-safe recourse.</p>

    <div class="grid">
      <div class="card">
        <div class="sec-title" style="margin-top:0">Patient Intake</div>
        <div class="form-grid">
          <div><label>Age (years)</label><input id="age_years" type="number" step="0.5" value="58" /></div>
          <div><label>Biological Sex</label><select id="sex"><option value="0">Female (0)</option><option value="1" selected>Male (1)</option></select></div>
          <div><label>Height (cm)</label><input id="height_cm" type="number" value="174" /></div>
          <div><label>Weight (kg)</label><input id="weight_kg" type="number" step="0.5" value="88" /></div>
          <div><label>Systolic BP (mmHg)</label><input id="systolic_bp" type="number" value="152" /></div>
          <div><label>Diastolic BP (mmHg)</label><input id="diastolic_bp" type="number" value="94" /></div>
          <div><label>Cholesterol Tier</label><select id="cholesterol_level"><option value="1">1 - Normal</option><option value="2" selected>2 - Above Normal</option><option value="3">3 - Well Above</option></select></div>
          <div><label>Glucose Tier</label><select id="glucose_level"><option value="1" selected>1 - Normal</option><option value="2">2 - Above Normal</option><option value="3">3 - Well Above</option></select></div>
          <div><label>Smoker</label><select id="smoke"><option value="0" selected>No</option><option value="1">Yes</option></select></div>
          <div><label>Alcohol</label><select id="alcohol"><option value="0" selected>No</option><option value="1">Yes</option></select></div>
          <div><label>Physically Active</label><select id="active"><option value="0" selected>No</option><option value="1">Yes</option></select></div>
        </div>
        <button onclick="assess()">Assess Clinical Risk</button>
        <div id="err" class="err"></div>
      </div>

      <div class="card" id="result">
        <p style="color:var(--muted)">Enter patient vitals on the left and click <strong>Assess Clinical Risk</strong> to view calibrated probability, conformal triage state, biomarker attention, matched historical cohort neighbors, and modifiable recourse options.</p>
      </div>
    </div>
  </div>
  <script>
    async function assess() {
      const errEl = document.getElementById("err");
      errEl.textContent = "";
      const payload = {
        age_years: parseFloat(document.getElementById("age_years").value),
        sex: parseInt(document.getElementById("sex").value),
        height_cm: parseFloat(document.getElementById("height_cm").value),
        weight_kg: parseFloat(document.getElementById("weight_kg").value),
        systolic_bp: parseFloat(document.getElementById("systolic_bp").value),
        diastolic_bp: parseFloat(document.getElementById("diastolic_bp").value),
        cholesterol_level: parseInt(document.getElementById("cholesterol_level").value),
        glucose_level: parseInt(document.getElementById("glucose_level").value),
        smoke: parseInt(document.getElementById("smoke").value),
        alcohol: parseInt(document.getElementById("alcohol").value),
        active: parseInt(document.getElementById("active").value),
      };
      const res = await fetch("/v1/predict", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        errEl.textContent = data.detail ? JSON.stringify(data.detail) : "Validation error";
        return;
      }
      const topFeats = Object.entries(data.dual_explanation.biomarker_attentions).slice(0, 6);
      const maxW = Math.max(...topFeats.map(f => f[1]), 0.01);
      const featBars = topFeats.map(([k, v]) =>
        `<div class="bar-row"><div class="bar-label">${k}</div><div class="bar-track"><div class="bar-fill" style="width:${(v/maxW*100).toFixed(1)}%"></div></div><div style="width:46px;text-align:right">${(v*100).toFixed(1)}%</div></div>`
      ).join("");
      const nbrRows = data.dual_explanation.top_cohort_neighbors.map(n =>
        `<tr><td>#${n.cohort_index}</td><td>${n.historical_outcome}</td><td>${(n.attention_weight*100).toFixed(1)}%</td><td>${n.age_years}y</td><td>${n.systolic_bp} mmHg</td><td>${n.bmi}</td></tr>`
      ).join("");
      const recHtml = data.actionable_recourse.length
        ? data.actionable_recourse.map(r => `<div class="rec-item"><strong>${r.intervention}</strong><br/>Projected risk: <b>${(r.projected_risk_probability*100).toFixed(1)}%</b> (&minus;${(r.absolute_risk_reduction*100).toFixed(1)} pts)</div>`).join("")
        : `<p style="color:var(--muted);font-size:13px">No risk-reducing modifiable recourse needed.</p>`;

      document.getElementById("result").innerHTML = `
        <div class="kpi-row">
          <div>
            <div style="font-size:12px;color:var(--muted)">CALIBRATED CVD RISK</div>
            <div class="risk-num">${(data.calibrated_risk_probability*100).toFixed(1)}%</div>
          </div>
          <div style="text-align:right">
            <span class="badge ${data.triage_decision}">${data.triage_decision}</span>
            <div style="font-size:12px;color:var(--muted);margin-top:6px">Conformal Set: {${data.conformal_prediction_set.join(", ")}} | Gate: ${data.dual_explanation.graph_reliance_gate.toFixed(2)} | ${data.latency_ms} ms</div>
          </div>
        </div>
        <div style="font-size:13px;color:#cbd5e1;margin-bottom:10px">${data.triage_reason}</div>
        ${data.ood_report.is_ood ? `<div class="banner" style="margin:10px 0"><strong>OOD ALERT:</strong> ${data.ood_report.reasons.join(" ")}</div>` : ""}
        <div class="sec-title">Level-1 Biomarker Attention ([CLS] &rarr; Feature)</div>
        ${featBars}
        <div class="sec-title">Level-2 Matched Historical Cohort Neighbors (Inductive GAT)</div>
        <table><thead><tr><th>Cohort Pt</th><th>Outcome</th><th>GAT Attn</th><th>Age</th><th>SBP</th><th>BMI</th></tr></thead><tbody>${nbrRows}</tbody></table>
        <div class="sec-title">Mutable-Only Actionable Recourse (Age/Sex/Height Locked)</div>
        ${recHtml}
      `;
    }
  </script>
</body>
</html>
"""


def create_app(
    engine: ClinicalInferenceEngine | None = None,
    bundle_path: str | Path | None = None,
    audit_log_path: str | Path | None = None,
) -> FastAPI:
    """Create the FastAPI Clinical Decision Support application."""
    app = FastAPI(
        title="Graph-Augmented Clinical Transformer CDS API",
        version="1.0.0",
        description=(
            "Fail-safe clinical decision support API with physiological guardrails, "
            "Mondrian conformal triage, OOD detection, dual biomarker + cohort-neighbor "
            "interpretability, actionable recourse, and HL7 FHIR R4 interoperability."
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
        return HTMLResponse(content=CLINICIAN_UI_HTML)

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

    @app.post("/v1/predict", response_model=ClinicalPredictionResponse, tags=["Clinical Inference"])
    def predict_single(patient: PatientClinicalInput) -> ClinicalPredictionResponse:
        return _get_engine().predict(patient)

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
