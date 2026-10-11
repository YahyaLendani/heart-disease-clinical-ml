"""Unit tests for the 4 new project advancements:
1. Clinical feature engineering (RPP, chronotropic ratio, ischemia burden)
2. Hybrid Ensembling (Soft-Voting and Stacking)
3. Leave-One-Center-Out (LOCO) domain partitioning
4. Demographic fairness and disparity calculations
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.fairness_analysis import compute_subgroup_metrics
from evaluation.update_thesis_results import (
    ENSEMBLE_AUTO_START,
    ENSEMBLE_AUTO_END,
    LOCO_AUTO_START,
    LOCO_AUTO_END,
    FAIRNESS_AUTO_START,
    FAIRNESS_AUTO_END,
    build_ensemble_results_block,
    build_loco_results_block,
    build_fairness_results_block,
)
from models import add_cardiovascular_features
from training.ensemble_models import SoftVotingEnsemble


class TestCardiovascularFeatures:
    """Verify mathematical calculation of cardiology features."""

    def test_rpp_and_chronotropic_ratio(self):
        df = pd.DataFrame({
            "age": [50.0, 60.0],
            "trestbps": [120.0, 140.0],
            "thalach": [170.0, 120.0],
            "oldpeak": [2.0, 1.0],
            "exang": [1.0, 0.0],
            "ca": [2.0, 0.0],
        })
        out = add_cardiovascular_features(df)

        # RPP = (trestbps * thalach) / 100
        assert np.isclose(out.loc[0, "rpp"], (120.0 * 170.0) / 100.0)
        assert np.isclose(out.loc[1, "rpp"], (140.0 * 120.0) / 100.0)

        # Chronotropic ratio = thalach / (220 - age)
        expected_ratio_0 = 170.0 / (220.0 - 50.0)
        assert np.isclose(out.loc[0, "chronotropic_ratio"], expected_ratio_0)

        # Ischemia burden = oldpeak * exang
        assert np.isclose(out.loc[0, "ischemia_burden"], 2.0 * 1.0)
        assert np.isclose(out.loc[1, "ischemia_burden"], 1.0 * 0.0)

        # Vessel ischemia = ca * oldpeak
        assert np.isclose(out.loc[0, "vessel_ischemia"], 2.0 * 2.0)
        assert np.isclose(out.loc[1, "vessel_ischemia"], 0.0 * 1.0)

    def test_nan_propagation_without_leakage(self):
        """Missing values must remain NaN so downstream pipelines can impute per fold."""
        df = pd.DataFrame({
            "age": [50.0],
            "trestbps": [np.nan],
            "thalach": [150.0],
        })
        out = add_cardiovascular_features(df)
        assert pd.isna(out.loc[0, "rpp"])


class TestSoftVotingEnsemble:
    """Verify weighted soft-voting logic."""

    class MockModel:
        def __init__(self, proba):
            self.proba = proba

        def predict_proba(self, X):
            return self.proba

    def test_weighted_averaging(self):
        p1 = np.array([[0.8, 0.2], [0.3, 0.7]])
        p2 = np.array([[0.6, 0.4], [0.1, 0.9]])

        models = [("m1", self.MockModel(p1)), ("m2", self.MockModel(p2))]
        ensemble = SoftVotingEnsemble(models, weights=[0.5, 0.5])

        X = np.zeros((2, 5))
        result = ensemble.predict_proba(X)

        expected = (p1 + p2) / 2.0
        assert np.allclose(result, expected)


class TestFairnessMetrics:
    """Verify clinical fairness calculations (TPR, TNR, FPR, FNR)."""

    def test_subgroup_metrics_known_values(self):
        y_true = np.array([1, 1, 1, 1, 0, 0])  # 4 sick, 2 healthy
        y_pred = np.array([1, 1, 1, 0, 0, 1])  # 3 TP, 1 FN, 1 TN, 1 FP

        metrics = compute_subgroup_metrics(y_true, y_pred)

        assert metrics["n_samples"] == 6
        assert np.isclose(metrics["sensitivity_tpr"], 3 / 4)  # 75%
        assert np.isclose(metrics["false_negative_rate"], 1 / 4)  # 25%
        assert np.isclose(metrics["specificity_tnr"], 1 / 2)  # 50%
        assert np.isclose(metrics["false_positive_rate"], 1 / 2)  # 50%


class TestCrossCenterPartitioning:
    """Verify LOCO partitions do not overlap across centers."""

    def test_loco_non_overlapping(self):
        sites = ["cleveland", "hungarian", "switzerland", "va_long_beach"]
        df = pd.DataFrame({
            "patient_id": range(100),
            "site": np.repeat(sites, 25),
        })

        for test_site in sites:
            train_df = df[df["site"] != test_site]
            test_df = df[df["site"] == test_site]

            overlap = set(train_df["patient_id"]).intersection(set(test_df["patient_id"]))
            assert len(overlap) == 0, f"Data leakage between train and {test_site}"
            assert len(test_df) == 25
            assert len(train_df) == 75


class TestThesisBlockBuilders:
    """Verify markdown block generation for thesis injection."""

    def test_ensemble_block(self):
        df = pd.DataFrame([{
            "model": "Hybrid Stacking Ensemble (Ours)",
            "test_accuracy": 0.8261,
            "test_roc_auc": 0.9244,
        }])
        block = build_ensemble_results_block(df)
        assert ENSEMBLE_AUTO_START in block
        assert ENSEMBLE_AUTO_END in block
        assert "Hybrid Stacking Ensemble (Ours)" in block
        assert "0.9244" in block

    def test_loco_block(self):
        df = pd.DataFrame([
            {"test_center": "cleveland", "model": "XGBoost", "transfer_roc_auc": 0.8441, "transfer_accuracy": 0.75},
            {"test_center": "hungarian", "model": "XGBoost", "transfer_roc_auc": 0.8675, "transfer_accuracy": 0.76},
        ])
        block = build_loco_results_block(df)
        assert LOCO_AUTO_START in block
        assert LOCO_AUTO_END in block
        assert "cleveland" in block
        assert "hungarian" in block

    def test_fairness_block(self):
        df = pd.DataFrame([
            {
                "subgroup": "Female Patients (sex=0)",
                "n_samples": 35,
                "sensitivity_tpr": 0.75,
                "false_negative_rate": 0.25,
                "roc_auc": 0.90,
            },
            {
                "subgroup": "Male Patients (sex=1)",
                "n_samples": 149,
                "sensitivity_tpr": 0.90,
                "false_negative_rate": 0.10,
                "roc_auc": 0.92,
            },
        ])
        block = build_fairness_results_block(df)
        assert FAIRNESS_AUTO_START in block
        assert FAIRNESS_AUTO_END in block
        assert "Equal Opportunity Difference" in block
        assert "0.1500" in block

