"""Causal Inference Analysis in Clinical Prediction: Observational vs. Interventional Risk.

Reference:
Pearl, J., & Mackenzie, D. (2018).
The Book of Why: The New Science of Cause and Effect. Basic Books.

Implements Propensity Score Matching (PSM) and Inverse Probability of
Treatment Weighting (IPTW) to estimate causal Average Treatment Effects (ATE)
and contrasts them against naive observational risk associations.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def compute_propensity_scores(
    X: np.ndarray,
    treatment: np.ndarray,
) -> np.ndarray:
    """Fit logistic propensity score model e(X) = P(Treatment=1 | X)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, treatment)
    # Predicted probability of receiving treatment
    propensities = model.predict_proba(X_scaled)[:, 1]
    # Trim extreme propensities to ensure positivity assumption
    return np.clip(propensities, 0.01, 0.99)


def compute_iptw_ate(
    outcome: np.ndarray,
    treatment: np.ndarray,
    propensities: np.ndarray,
    stabilized: bool = True,
) -> dict[str, float]:
    """Compute Average Treatment Effect via Inverse Probability of Treatment Weighting.

    Formula from Pearl (2018):
    w_i = T_i / e(X_i) + (1 - T_i) / (1 - e(X_i))
    """
    treatment = treatment.astype(float)
    outcome = outcome.astype(float)

    p_t = np.mean(treatment)
    if stabilized:
        # Stabilized weights reduce variance from small propensities
        weights = (treatment * p_t / propensities) + ((1.0 - treatment) * (1.0 - p_t) / (1.0 - propensities))
    else:
        weights = (treatment / propensities) + ((1.0 - treatment) / (1.0 - propensities))

    treated_mask = treatment == 1.0
    control_mask = treatment == 0.0

    treated_mean = np.sum(outcome[treated_mask] * weights[treated_mask]) / np.sum(weights[treated_mask])
    control_mean = np.sum(outcome[control_mask] * weights[control_mask]) / np.sum(weights[control_mask])

    ate = treated_mean - control_mean
    risk_ratio = treated_mean / max(control_mean, 1e-6)

    return {
        "causal_ate_iptw": float(ate),
        "causal_risk_ratio": float(risk_ratio),
        "treated_counterfactual_mean": float(treated_mean),
        "control_counterfactual_mean": float(control_mean),
    }


def compute_nearest_neighbor_psm(
    outcome: np.ndarray,
    treatment: np.ndarray,
    propensities: np.ndarray,
    caliper: float = 0.1,
) -> dict[str, float]:
    """Propensity Score Matching (1:1 nearest neighbor with caliper)."""
    treated_idx = np.where(treatment == 1)[0]
    control_idx = np.where(treatment == 0)[0]

    matched_treated_outcomes = []
    matched_control_outcomes = []

    for t_idx in treated_idx:
        t_score = propensities[t_idx]
        diffs = np.abs(propensities[control_idx] - t_score)
        min_idx = np.argmin(diffs)
        if diffs[min_idx] <= caliper:
            matched_treated_outcomes.append(outcome[t_idx])
            matched_control_outcomes.append(outcome[control_idx[min_idx]])

    if not matched_treated_outcomes:
        return {"causal_ate_matching": float("nan"), "n_matched_pairs": 0}

    ate_match = np.mean(matched_treated_outcomes) - np.mean(matched_control_outcomes)
    return {
        "causal_ate_matching": float(ate_match),
        "n_matched_pairs": len(matched_treated_outcomes),
    }


def compare_observational_vs_causal(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    treatment_name: str = "Intervention",
) -> dict[str, float | str]:
    """Demonstrate the divergence between observational correlation and causal ATE."""
    # 1. Naive Observational Risk Difference: E[Y | T=1] - E[Y | T=0]
    obs_treated = float(np.mean(outcome[treatment == 1]))
    obs_control = float(np.mean(outcome[treatment == 0]))
    obs_diff = obs_treated - obs_control
    obs_rr = obs_treated / max(obs_control, 1e-6)

    # 2. Causal Propensity Estimation
    propensities = compute_propensity_scores(X, treatment)

    # 3. Causal ATE via IPTW
    iptw_res = compute_iptw_ate(outcome, treatment, propensities)

    # 4. Causal ATE via Matching
    psm_res = compute_nearest_neighbor_psm(outcome, treatment, propensities)

    # Confounder bias = Observational Difference - Causal ATE
    confounding_bias = obs_diff - iptw_res["causal_ate_iptw"]

    return {
        "treatment_name": treatment_name,
        "observational_treated_risk": obs_treated,
        "observational_control_risk": obs_control,
        "observational_risk_difference": obs_diff,
        "observational_risk_ratio": obs_rr,
        "causal_ate_iptw": iptw_res["causal_ate_iptw"],
        "causal_risk_ratio_iptw": iptw_res["causal_risk_ratio"],
        "causal_ate_matching": psm_res["causal_ate_matching"],
        "n_matched_pairs": psm_res["n_matched_pairs"],
        "confounding_bias": float(confounding_bias),
    }
