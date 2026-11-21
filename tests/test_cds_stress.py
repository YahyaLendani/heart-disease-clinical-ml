"""Thread-safety concurrency stress tests, boundary fuzzing, and checksum verification."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from clinical_cds import (
    CardiologyWorkupEngine,
    CardiologyWorkupInput,
    ClinicalInferenceEngine,
    PatientClinicalInput,
    build_clinical_bundle,
    build_workup_bundle,
)
from clinical_cds.download_bundles import (
    compute_file_sha256,
    parse_sha256sums,
    verify_or_remove,
)


@pytest.fixture(scope="module")
def stress_engines(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[ClinicalInferenceEngine, CardiologyWorkupEngine]:
    """Build compact zero-pickle Stage-1 and Stage-2 engines for concurrent stress testing."""
    tmp_dir = tmp_path_factory.mktemp("stress_cds")
    rng = np.random.default_rng(123)
    n1 = 260
    h = rng.normal(170.0, 8.0, size=n1).clip(145.0, 195.0)
    w = rng.normal(76.0, 12.0, size=n1).clip(50.0, 120.0)
    sbp = rng.normal(132.0, 16.0, size=n1).clip(95.0, 195.0)
    dbp = (sbp - rng.uniform(30.0, 50.0, size=n1)).clip(60.0, 120.0)
    df1 = pd.DataFrame({
        "age_years": rng.uniform(35.0, 75.0, size=n1),
        "sex": rng.integers(0, 2, size=n1),
        "height": h,
        "weight": w,
        "bmi": w / ((h / 100.0) ** 2),
        "ap_hi": sbp,
        "ap_lo": dbp,
        "pulse_pressure": sbp - dbp,
        "mean_arterial_pressure": dbp + (sbp - dbp) / 3.0,
        "cholesterol": rng.integers(0, 3, size=n1),
        "gluc": rng.integers(0, 3, size=n1),
        "smoke": rng.integers(0, 2, size=n1),
        "alco": rng.integers(0, 2, size=n1),
        "active": rng.integers(0, 2, size=n1),
        "target": rng.integers(0, 2, size=n1),
    })
    csv1 = tmp_dir / "s1.csv"
    b1 = tmp_dir / "s1.pt"
    df1.to_csv(csv1, index=False)
    build_clinical_bundle(
        dataset_path=csv1, output_path=b1, epochs=2, batch_size=64,
        k_neighbors=5, embed_dim=16, n_prototypes=48, cv_folds=2, seed=42,
    )

    n2 = 180
    df2 = pd.DataFrame({
        "age": rng.uniform(35.0, 75.0, size=n2),
        "sex": rng.integers(0, 2, size=n2),
        "cp": rng.integers(1, 5, size=n2),
        "trestbps": rng.normal(134.0, 15.0, size=n2).clip(95.0, 190.0),
        "chol": rng.normal(240.0, 40.0, size=n2).clip(140.0, 380.0),
        "fbs": rng.integers(0, 2, size=n2),
        "restecg": rng.integers(0, 3, size=n2),
        "thalach": rng.normal(148.0, 20.0, size=n2).clip(85.0, 195.0),
        "exang": rng.integers(0, 2, size=n2),
        "oldpeak": rng.exponential(1.0, size=n2).clip(0.0, 5.0),
        "slope": rng.integers(1, 4, size=n2),
        "ca": rng.integers(0, 4, size=n2),
        "thal": rng.choice([3, 6, 7], size=n2),
        "target": rng.integers(0, 2, size=n2),
    })
    csv2 = tmp_dir / "s2.csv"
    b2 = tmp_dir / "s2.pt"
    df2.to_csv(csv2, index=False)
    build_workup_bundle(
        dataset_path=csv2, output_path=b2, epochs=4, cv_folds=2,
        k_neighbors=5, embed_dim=16, seed=42,
    )
    return (
        ClinicalInferenceEngine.from_bundle(b1, device="cpu"),
        CardiologyWorkupEngine.from_bundle(b2, device="cpu"),
    )


class TestConcurrentThreadSafetyAndFuzzing:
    """Verify deterministic thread-safety under concurrent traffic and strict boundary rejection."""

    def test_concurrent_multithreaded_inference_determinism(
        self, stress_engines: tuple[ClinicalInferenceEngine, CardiologyWorkupEngine]
    ):
        eng1, eng2 = stress_engines
        pt1 = PatientClinicalInput(
            age_years=58, sex=1, height_cm=174, weight_kg=85,
            systolic_bp=148, diastolic_bp=92, cholesterol_level=2, glucose_level=1,
            smoke=1, alcohol=0, active=0,
        )
        pt2 = CardiologyWorkupInput(
            age=62, sex=1, cp=4, trestbps=155, chol=270, fbs=1, restecg=2,
            thalach=120, exang=1, oldpeak=2.2, slope=2, ca=1, thal=7,
        )
        ref1 = eng1.predict(pt1).calibrated_risk_probability
        ref2 = eng2.predict(pt2).calibrated_risk_probability

        def _worker(i: int) -> tuple[int, float]:
            if i % 2 == 0:
                return (1, eng1.predict(pt1).calibrated_risk_probability)
            return (2, eng2.predict(pt2).calibrated_risk_probability)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(_worker, range(32)))

        for stage, prob in results:
            expected = ref1 if stage == 1 else ref2
            assert abs(prob - expected) < 1e-6

    @pytest.mark.parametrize(
        "bad_overrides",
        [
            {"age_years": 17.0},
            {"age_years": 115.0},
            {"systolic_bp": 55.0},
            {"systolic_bp": 260.0},
            {"diastolic_bp": 35.0},
            {"systolic_bp": 120.0, "diastolic_bp": 116.0},
            {"cholesterol_level": 0},
            {"cholesterol_level": 4},
            {"glucose_level": 4},
            {"conformal_alpha": 0.0},
            {"conformal_alpha": 0.9},
            {"height_cm": 230.0, "weight_kg": 35.0},
        ],
    )
    def test_stage1_physiological_boundary_fuzz_rejections(self, bad_overrides: dict):
        valid = {
            "age_years": 50.0, "sex": 1, "height_cm": 175.0, "weight_kg": 78.0,
            "systolic_bp": 128.0, "diastolic_bp": 80.0, "cholesterol_level": 1,
            "glucose_level": 1, "smoke": 0, "alcohol": 0, "active": 1, "conformal_alpha": 0.10,
        }
        valid.update(bad_overrides)
        with pytest.raises(ValidationError):
            PatientClinicalInput(**valid)

    @pytest.mark.parametrize(
        "bad_overrides",
        [
            {"age": 12.0},
            {"cp": 0},
            {"cp": 5},
            {"trestbps": 40.0},
            {"chol": 30.0},
            {"thalach": 260.0},
            {"oldpeak": 12.0},
            {"slope": 4},
            {"ca": 4},
            {"thal": 5},
        ],
    )
    def test_stage2_workup_boundary_fuzz_rejections(self, bad_overrides: dict):
        valid = {
            "age": 58.0, "sex": 1, "cp": 2, "trestbps": 136.0, "chol": 230.0,
            "fbs": 0, "restecg": 0, "thalach": 150.0, "exang": 0, "oldpeak": 1.0,
            "slope": 2, "ca": 0, "thal": 3, "conformal_alpha": 0.10,
        }
        valid.update(bad_overrides)
        with pytest.raises(ValidationError):
            CardiologyWorkupInput(**valid)

    def test_bundle_checksum_verification_and_tamper_cleanup(self, tmp_path: Path):
        sample = tmp_path / "dummy_bundle.pt"
        sample.write_bytes(b"zero-pickle-bundle-bytes")
        digest = compute_file_sha256(sample)

        sums_file = tmp_path / "SHA256SUMS.txt"
        sums_file.write_text(f"{digest}  dummy_bundle.pt\n", encoding="utf-8")
        parsed = parse_sha256sums(sums_file)
        assert parsed["dummy_bundle.pt"] == digest
        assert verify_or_remove(sample, digest) == digest

        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            verify_or_remove(sample, "0" * 64)
        assert not sample.exists()
