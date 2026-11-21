"""Clinical Decision Support (CDS) & HL7 FHIR R4 Inference Stack.

Production-aligned inference engine for ``GraphAugmentedClinicalTransformer``:
- Strict physiological input validation (Pydantic v2)
- Versioned, SHA-256-verified model bundles with pre-encoded O(1)-lookup neighbor banks
- Out-of-Distribution (OOD) detection + fail-safe Mondrian Conformal Triage abstention
- Dual-level biomarker + matched cohort neighbor interpretability
- Immutable-safe actionable clinical recourse recommendations
- HL7 FHIR R4 Bundle -> RiskAssessment interoperability
- FastAPI REST service + de-identified JSONL audit logging + clinician web UI
"""

from __future__ import annotations

from .bundle import (
    ClinicalModelBundle,
    build_clinical_bundle,
    engineer_screening_interactions,
)
from .calibrate_site import recalibrate_bundle_for_site
from .engine import ClinicalInferenceEngine, compute_psi
from .fhir_adapter import (
    fhir_bundle_to_patient_input,
    fhir_bundle_to_workup_input,
    prediction_to_fhir_risk_assessment,
)
from .schemas import (
    CardiologyWorkupInput,
    ClinicalPredictionResponse,
    DualClinicalExplanation,
    MatchedCohortNeighbor,
    OODReport,
    PatientClinicalInput,
    RecourseRecommendation,
)
from .workup import (
    CardiologyWorkupBundle,
    CardiologyWorkupEngine,
    build_workup_bundle,
)

__all__ = [
    "ClinicalModelBundle",
    "build_clinical_bundle",
    "engineer_screening_interactions",
    "recalibrate_bundle_for_site",
    "ClinicalInferenceEngine",
    "CardiologyWorkupBundle",
    "CardiologyWorkupEngine",
    "build_workup_bundle",
    "compute_psi",
    "PatientClinicalInput",
    "CardiologyWorkupInput",
    "ClinicalPredictionResponse",
    "OODReport",
    "DualClinicalExplanation",
    "MatchedCohortNeighbor",
    "RecourseRecommendation",
    "fhir_bundle_to_patient_input",
    "fhir_bundle_to_workup_input",
    "prediction_to_fhir_risk_assessment",
]
