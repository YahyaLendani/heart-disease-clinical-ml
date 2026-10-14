"""Unit tests verifying the mathematical guarantees of the 3 methodological upgrades."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from evaluation.advanced_methodology import (
    ClinicalCounterfactualGenerator,
    SplitConformalClassifier,
    compute_doubly_robust_aipw,
    compute_e_value,
)


class _SimpleLinearClinicalNet(nn.Module):
    """Differentiable clinical risk model for deterministic counterfactual testing."""

    def __init__(self, weights: list[float], bias: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(len(weights), 1)
        with torch.no_grad():
            self.linear.weight.copy_(torch.tensor([weights], dtype=torch.float32))
            self.linear.bias.fill_(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def test_vanderweele_e_value_exact_formula() -> None:
    """Verify VanderWeele & Ding (2017) E-value formula against known analytical values."""
    # For RR = 2.0: E-value = 2 + sqrt(2 * 1) = 3.41421356
    res = compute_e_value(risk_ratio=2.0, ci_bound_closest_to_null=1.5)
    assert np.isclose(res["e_value_point"], 2.0 + np.sqrt(2.0), atol=1e-6)
    assert np.isclose(res["e_value_ci"], 1.5 + np.sqrt(1.5 * 0.5), atol=1e-6)

    # Symmetry for protective effect RR = 0.5 (RR* = 2.0)
    res_prot = compute_e_value(risk_ratio=0.5, ci_bound_closest_to_null=0.8)
    assert np.isclose(res_prot["e_value_point"], 2.0 + np.sqrt(2.0), atol=1e-6)

    # Null-crossing CI must yield E-value_ci == 1.0
    res_null = compute_e_value(risk_ratio=1.8, ci_bound_closest_to_null=0.95)
    assert res_null["e_value_ci"] == 1.0


def test_doubly_robust_aipw_recovers_true_causal_effect_under_confounding() -> None:
    """Verify cross-fitted AIPW removes confounding bias on a synthetic structural causal model."""
    rng = np.random.RandomState(42)
    N = 1200
    # Confounder X ~ N(0, 1) increases both treatment probability and outcome risk
    X = rng.normal(0, 1, size=(N, 3))
    logits_t = 1.1 * X[:, 0] - 0.6 * X[:, 1]
    prob_t = 1.0 / (1.0 + np.exp(-logits_t))
    T = rng.binomial(1, prob_t)

    # True causal effect of T on Y is exactly +0.20, plus strong positive confounding from X[:, 0]
    base_prob_y = 0.30 + 0.18 * (X[:, 0] > 0).astype(float) - 0.08 * (X[:, 1] > 0).astype(float)
    prob_y = np.clip(base_prob_y + 0.20 * T, 0.01, 0.99)
    Y = rng.binomial(1, prob_y).astype(float)

    res = compute_doubly_robust_aipw(X, T, Y, treatment_name="Synthetic ICU", seed=42)

    # Observational difference is severely upward biased (> 0.26) due to X[:, 0]
    assert res["observational_risk_difference"] > 0.26
    # Doubly robust AIPW should recover the true ATE (~0.20) within +/- 0.06
    assert abs(res["causal_ate_aipw"] - 0.20) < 0.06
    assert res["aipw_ci_lower"] < res["causal_ate_aipw"] < res["aipw_ci_upper"]
    assert res["e_value_point"] > 1.2


def test_split_and_mondrian_conformal_finite_sample_coverage() -> None:
    """Verify marginal and Mondrian conformal prediction achieve >= 1 - alpha coverage."""
    rng = np.random.RandomState(123)
    N_cal, N_test = 600, 1000
    y_cal = rng.binomial(1, 0.25, size=N_cal)
    y_test = rng.binomial(1, 0.25, size=N_test)

    # Simulated noisy probability predictions
    prob_cal = np.clip(0.2 + 0.55 * y_cal + rng.normal(0, 0.2, size=N_cal), 0.01, 0.99)
    prob_test = np.clip(0.2 + 0.55 * y_test + rng.normal(0, 0.2, size=N_test), 0.01, 0.99)

    conformal = SplitConformalClassifier(alpha=0.10, mondrian=True).fit(y_cal, prob_cal)
    metrics = conformal.evaluate(y_test, prob_test)

    # Finite-sample coverage guarantee (~0.90 within sampling fluctuation)
    assert metrics["empirical_coverage"] >= 0.86
    assert metrics["positive_class_coverage"] >= 0.85
    assert metrics["negative_class_coverage"] >= 0.85
    assert 0.0 <= metrics["average_set_size"] <= 2.0
    assert np.isclose(
        metrics["certifiable_low_risk_rate"]
        + metrics["certifiable_high_risk_rate"]
        + metrics["ambiguous_abstention_rate"]
        + metrics["empty_set_rate"],
        1.0,
    )


def test_clinical_counterfactual_immutability_and_validity() -> None:
    """Verify counterfactual generator strictly preserves immutable features and reduces risk."""
    rng = np.random.RandomState(7)
    feature_names = ["age", "sex", "systolic_bp", "cholesterol"]
    X_train = rng.normal(0.0, 1.0, size=(300, 4)).astype(np.float32)

    # Model where age, systolic_bp, and cholesterol drive positive risk
    model = _SimpleLinearClinicalNet(weights=[0.8, 0.3, 1.5, 1.2], bias=0.2)

    generator = ClinicalCounterfactualGenerator(
        model=model,
        feature_names=feature_names,
        immutable_features=("age", "sex"),
        target_probability=0.25,
        lr=0.08,
        max_iter=300,
    ).fit(X_train)

    # High-risk patient
    x_high_risk = np.array([1.2, 1.0, 1.8, 1.5], dtype=np.float32)
    result = generator.generate(x_high_risk)

    assert result["original_probability"] > 0.85
    assert result["validity_achieved"] is True
    assert result["counterfactual_probability"] <= 0.25
    # Strict immutability: age (idx 0) and sex (idx 1) must have zero modification
    assert result["immutable_violation_max"] == 0.0
    assert np.allclose(result["x_counterfactual"][:2], x_high_risk[:2], atol=1e-7)
    # Physiological box bounds respected
    assert np.all(result["x_counterfactual"] >= generator.lower_bounds_ - 1e-5)
    assert np.all(result["x_counterfactual"] <= generator.upper_bounds_ + 1e-5)
