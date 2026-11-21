"""Strict Pydantic v2 schemas and physiological guardrails for clinical inference."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

CLINICAL_DISCLAIMER = (
    "RESEARCH & CLINICIAN-IN-THE-LOOP DECISION SUPPORT ONLY. "
    "Not an FDA-cleared or CE-marked medical device. Final diagnostic and "
    "therapeutic decisions must be made by a qualified licensed clinician."
)


class PatientClinicalInput(BaseModel):
    """Physiologically validated cardiovascular patient intake payload.

    Enforces hard clinical plausibility boundaries before data ever reaches
    the neural network (preventing sensor corruption or unit-entry errors).
    """

    patient_pseudo_id: str | None = Field(
        default=None,
        max_length=128,
        description="Optional de-identified / pseudonymized correlation ID (no raw PII).",
    )
    age_years: float = Field(..., ge=18.0, le=110.0, description="Patient age in years.")
    sex: Literal[0, 1] = Field(..., description="Biological sex (0 = Female, 1 = Male).")
    height_cm: float = Field(..., ge=120.0, le=230.0, description="Height in centimeters.")
    weight_kg: float = Field(..., ge=30.0, le=220.0, description="Weight in kilograms.")
    systolic_bp: float = Field(..., ge=60.0, le=250.0, description="Systolic blood pressure (mmHg).")
    diastolic_bp: float = Field(..., ge=40.0, le=160.0, description="Diastolic blood pressure (mmHg).")
    cholesterol_level: Literal[1, 2, 3] = Field(
        ...,
        description="Total cholesterol tier (1 = Normal, 2 = Above normal, 3 = Well above normal).",
    )
    glucose_level: Literal[1, 2, 3] = Field(
        ...,
        description="Fasting glucose tier (1 = Normal, 2 = Above normal, 3 = Well above normal).",
    )
    smoke: Literal[0, 1] = Field(default=0, description="Current tobacco smoker (0 = No, 1 = Yes).")
    alcohol: Literal[0, 1] = Field(default=0, description="Regular alcohol intake (0 = No, 1 = Yes).")
    active: Literal[0, 1] = Field(default=1, description="Physically active lifestyle (0 = No, 1 = Yes).")
    conformal_alpha: Literal[0.05, 0.10, 0.15] = Field(
        default=0.10,
        description="Conformal miscoverage level (0.05 = 95% coverage, 0.10 = 90%, 0.15 = 85%).",
    )

    @model_validator(mode="after")
    def validate_physiological_consistency(self) -> PatientClinicalInput:
        if self.systolic_bp <= self.diastolic_bp + 5.0:
            raise ValueError(
                f"Physiologically invalid blood pressure: systolic_bp ({self.systolic_bp}) "
                f"must exceed diastolic_bp ({self.diastolic_bp}) by more than 5 mmHg."
            )
        bmi = self.weight_kg / ((self.height_cm / 100.0) ** 2)
        if not (12.0 <= bmi <= 70.0):
            raise ValueError(
                f"Derived BMI ({bmi:.1f} kg/m^2) is outside physiologically plausible bounds [12, 70]."
            )
        return self

    @property
    def bmi(self) -> float:
        return round(self.weight_kg / ((self.height_cm / 100.0) ** 2), 2)

    @property
    def pulse_pressure(self) -> float:
        return round(self.systolic_bp - self.diastolic_bp, 2)

    @property
    def mean_arterial_pressure(self) -> float:
        return round(self.diastolic_bp + (self.systolic_bp - self.diastolic_bp) / 3.0, 2)


class OODReport(BaseModel):
    """Out-of-distribution & covariate drift assessment for fail-safe clinical abstention."""

    is_ood: bool
    max_abs_zscore: float
    min_neighbor_cosine_distance: float
    ood_distance_threshold: float
    reasons: list[str] = Field(default_factory=list)


class MatchedCohortNeighbor(BaseModel):
    """Anonymized historical training patient matched via inductive Level-2 GAT."""

    cohort_index: int
    attention_weight: float
    cosine_similarity: float
    historical_outcome: Literal["CVD+", "Healthy"]
    age_years: float
    systolic_bp: float
    bmi: float


class DualClinicalExplanation(BaseModel):
    """Dual-level Graph-Augmented Transformer clinical interpretability output."""

    graph_reliance_gate: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Learned sigmoid gate balancing cohort graph vs individual tabular features.",
    )
    biomarker_attentions: dict[str, float] = Field(
        ...,
        description="Level-1 [CLS]-to-feature attention weights (sum to 1.0, sorted descending).",
    )
    top_cohort_neighbors: list[MatchedCohortNeighbor] = Field(
        default_factory=list,
        description="Level-2 inductive GAT top-k historical cohort neighbors.",
    )


class RecourseRecommendation(BaseModel):
    """Immutable-safe, physiologically valid modifiable intervention recommendation."""

    intervention: str
    changes: dict[str, float | int]
    projected_risk_probability: float
    absolute_risk_reduction: float


class ClinicalPredictionResponse(BaseModel):
    """Full fail-safe clinical decision support response."""

    request_id: str
    timestamp_utc: str
    model_version: str
    bundle_sha256: str
    calibrated_risk_probability: float = Field(..., ge=0.0, le=1.0)
    gat_transformer_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Standalone calibrated Graph-Augmented Transformer probability."
    )
    xgboost_probability: float | None = Field(
        default=None, description="Standalone XGBoost probability when hybrid ensemble is active."
    )
    model_disagreement: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Absolute epistemic disagreement |p_gat - p_xgb| between neural and tree models.",
    )
    triage_decision: Literal["LOW_RISK", "HIGH_RISK", "DEFER_TO_CLINICIAN"]
    triage_reason: str
    conformal_prediction_set: list[int]
    conformal_alpha: float
    conformal_low_threshold: float = Field(
        ...,
        description="Sex-stratified Mondrian threshold below which conformal set is {0} (LOW_RISK).",
    )
    conformal_high_threshold: float = Field(
        ...,
        description="Sex-stratified Mondrian threshold above which conformal set is {1} (HIGH_RISK).",
    )
    ood_report: OODReport
    dual_explanation: DualClinicalExplanation
    actionable_recourse: list[RecourseRecommendation] = Field(default_factory=list)
    latency_ms: float
    clinical_disclaimer: str = CLINICAL_DISCLAIMER
