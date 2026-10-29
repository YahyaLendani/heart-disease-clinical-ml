"""Defense-ready methodological upgrades for clinical AI:
1. Cross-Fitted Doubly Robust AIPW Causal Inference + VanderWeele E-Values
2. Split & Mondrian Conformal Prediction for Distribution-Free Risk Triage
3. Actionable Clinical Counterfactual Recourse with Immutability & Manifold Constraints
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib.lines import Line2D
from sklearn.base import clone
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# =============================================================================
# UPGRADE 1: DOUBLY ROBUST AIPW CAUSAL INFERENCE & E-VALUE SENSITIVITY
# =============================================================================


def compute_e_value(risk_ratio: float, ci_bound_closest_to_null: float | None = None) -> dict[str, float]:
    """Compute VanderWeele & Ding (2017) E-values for unmeasured confounding.

    Parameters
    ----------
    risk_ratio : float
        Point estimate of the causal Risk Ratio (RR = E[Y(1)] / E[Y(0)]).
    ci_bound_closest_to_null : float | None
        The 95% confidence interval bound of RR closest to 1.0. If the CI
        crosses 1.0, pass 1.0 (which yields E-value = 1.0).

    Returns
    -------
    dict[str, float]
        Dictionary containing `e_value_point` and `e_value_ci`.
    """
    if risk_ratio <= 0 or not np.isfinite(risk_ratio):
        return {"e_value_point": 1.0, "e_value_ci": 1.0}

    rr_star = risk_ratio if risk_ratio >= 1.0 else (1.0 / risk_ratio)
    if np.isclose(rr_star, 1.0):
        e_val_point = 1.0
    else:
        e_val_point = float(rr_star + np.sqrt(rr_star * (rr_star - 1.0)))

    e_val_ci = 1.0
    if ci_bound_closest_to_null is not None and np.isfinite(ci_bound_closest_to_null) and ci_bound_closest_to_null > 0:
        if (risk_ratio >= 1.0 and ci_bound_closest_to_null <= 1.0) or (
            risk_ratio < 1.0 and ci_bound_closest_to_null >= 1.0
        ):
            e_val_ci = 1.0
        else:
            bound_star = (
                ci_bound_closest_to_null
                if ci_bound_closest_to_null >= 1.0
                else (1.0 / ci_bound_closest_to_null)
            )
            e_val_ci = float(bound_star + np.sqrt(bound_star * (bound_star - 1.0)))

    return {
        "e_value_point": e_val_point,
        "e_value_ci": e_val_ci,
    }


def compute_doubly_robust_aipw(
    X_confounders: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    treatment_name: str = "Clinical Intervention",
    n_splits: int = 5,
    propensity_clip: tuple[float, float] = (0.025, 0.975),
    seed: int = 42,
) -> dict[str, float | str]:
    """Estimate causal ATE and Risk Ratio via Cross-Fitted Doubly Robust AIPW + E-values.

    Uses K-fold cross-fitting (Chernozhukov et al., 2018) so nuisance models
    (propensity score e(X) and outcome regressions mu_0(X), mu_1(X)) are never
    evaluated on the same samples used to fit them.
    """
    X = np.asarray(X_confounders, dtype=float)
    T = np.asarray(treatment, dtype=int).ravel()
    Y = np.asarray(outcome, dtype=float).ravel()
    N = len(Y)

    if N < 20 or len(np.unique(T)) < 2:
        raise ValueError("Insufficient sample size or treatment variation for AIPW estimation.")

    # Observational baseline
    obs_r1 = float(np.mean(Y[T == 1]))
    obs_r0 = float(np.mean(Y[T == 0]))
    obs_diff = obs_r1 - obs_r0

    # Cross-fitted nuisance predictions
    e_hat = np.zeros(N, dtype=float)
    mu1_hat = np.zeros(N, dtype=float)
    mu0_hat = np.zeros(N, dtype=float)

    effective_splits = min(n_splits, int(np.min(np.bincount(T))))
    effective_splits = max(2, effective_splits)
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)

    base_propensity = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    is_binary_outcome = set(np.unique(Y)).issubset({0.0, 1.0})
    if is_binary_outcome:
        base_outcome = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=seed
        )
    else:
        base_outcome = RandomForestRegressor(
            n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=seed
        )

    for train_idx, eval_idx in skf.split(X, T):
        X_tr, T_tr, Y_tr = X[train_idx], T[train_idx], Y[train_idx]
        X_ev = X[eval_idx]

        # 1. Propensity model e(X) = P(T=1 | X)
        prop_model = clone(base_propensity)
        prop_model.fit(X_tr, T_tr)
        e_hat[eval_idx] = prop_model.predict_proba(X_ev)[:, 1]

        # 2. Outcome model under treatment mu_1(X) = E[Y | T=1, X]
        mask_1 = T_tr == 1
        if np.sum(mask_1) > 2 and len(np.unique(Y_tr[mask_1])) > 1:
            m1 = clone(base_outcome)
            m1.fit(X_tr[mask_1], Y_tr[mask_1])
            mu1_hat[eval_idx] = (
                m1.predict_proba(X_ev)[:, 1] if is_binary_outcome else m1.predict(X_ev)
            )
        else:
            mu1_hat[eval_idx] = float(np.mean(Y_tr[mask_1])) if np.any(mask_1) else obs_r1

        # 3. Outcome model under control mu_0(X) = E[Y | T=0, X]
        mask_0 = T_tr == 0
        if np.sum(mask_0) > 2 and len(np.unique(Y_tr[mask_0])) > 1:
            m0 = clone(base_outcome)
            m0.fit(X_tr[mask_0], Y_tr[mask_0])
            mu0_hat[eval_idx] = (
                m0.predict_proba(X_ev)[:, 1] if is_binary_outcome else m0.predict(X_ev)
            )
        else:
            mu0_hat[eval_idx] = float(np.mean(Y_tr[mask_0])) if np.any(mask_0) else obs_r0

    e_clipped = np.clip(e_hat, propensity_clip[0], propensity_clip[1])

    # IPTW estimator (for direct comparison)
    iptw_y1 = T * Y / e_clipped
    iptw_y0 = (1 - T) * Y / (1.0 - e_clipped)
    ate_iptw = float(np.mean(iptw_y1 - iptw_y0))

    # Doubly Robust AIPW pseudo-outcomes (influence functions)
    psi_1 = mu1_hat + (T * (Y - mu1_hat)) / e_clipped
    psi_0 = mu0_hat + ((1 - T) * (Y - mu0_hat)) / (1.0 - e_clipped)
    psi_diff = psi_1 - psi_0

    ate_aipw = float(np.mean(psi_diff))
    se_aipw = float(np.sqrt(np.var(psi_diff, ddof=1) / N))
    ci_lower = ate_aipw - 1.96 * se_aipw
    ci_upper = ate_aipw + 1.96 * se_aipw

    # Potential outcome risk estimates & Causal Risk Ratio (for E-value)
    r1_aipw = float(np.clip(np.mean(psi_1), 1e-4, 1.0 - 1e-4))
    r0_aipw = float(np.clip(np.mean(psi_0), 1e-4, 1.0 - 1e-4))
    causal_rr = r1_aipw / r0_aipw

    # Delta-method standard error on log(RR)
    se_psi1 = np.sqrt(np.var(psi_1, ddof=1) / N)
    se_psi0 = np.sqrt(np.var(psi_0, ddof=1) / N)
    cov_10 = np.cov(psi_1, psi_0, ddof=1)[0, 1] / N
    var_log_rr = max(
        1e-12,
        (se_psi1**2) / (r1_aipw**2)
        + (se_psi0**2) / (r0_aipw**2)
        - 2.0 * cov_10 / (r1_aipw * r0_aipw),
    )
    se_log_rr = np.sqrt(var_log_rr)
    rr_ci_lower = float(np.exp(np.log(causal_rr) - 1.96 * se_log_rr))
    rr_ci_upper = float(np.exp(np.log(causal_rr) + 1.96 * se_log_rr))

    if rr_ci_lower <= 1.0 <= rr_ci_upper:
        ci_bound_null = 1.0
    elif causal_rr >= 1.0:
        ci_bound_null = rr_ci_lower
    else:
        ci_bound_null = rr_ci_upper

    e_vals = compute_e_value(causal_rr, ci_bound_closest_to_null=ci_bound_null)

    return {
        "treatment_name": treatment_name,
        "observational_risk_difference": float(obs_diff),
        "causal_ate_iptw": float(ate_iptw),
        "causal_ate_aipw": float(ate_aipw),
        "aipw_se": float(se_aipw),
        "aipw_ci_lower": float(ci_lower),
        "aipw_ci_upper": float(ci_upper),
        "confounding_bias_aipw": float(obs_diff - ate_aipw),
        "causal_risk_ratio": float(causal_rr),
        "rr_ci_lower": float(rr_ci_lower),
        "rr_ci_upper": float(rr_ci_upper),
        "e_value_point": float(e_vals["e_value_point"]),
        "e_value_ci": float(e_vals["e_value_ci"]),
    }


# =============================================================================
# UPGRADE 2: SPLIT & MONDRIAN CONFORMAL PREDICTION FOR RISK TRIAGE
# =============================================================================


class SplitConformalClassifier:
    """Distribution-free finite-sample risk set prediction for binary clinical outcomes.

    Supports both standard Marginal Split Conformal Prediction and Mondrian
    (class-conditional) Conformal Prediction to guarantee coverage separately for
    cases (Y=1) and controls (Y=0) under class imbalance.
    """

    def __init__(self, alpha: float = 0.10, mondrian: bool = True) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError("Miscoverage rate alpha must be strictly in (0, 1).")
        self.alpha = float(alpha)
        self.mondrian = bool(mondrian)
        self.q_hat_marginal_: float | None = None
        self.q_hat_class_: dict[int, float] = {}

    @staticmethod
    def _finite_sample_quantile(scores: np.ndarray, alpha: float) -> float:
        n = len(scores)
        if n == 0:
            return 1.0
        level = np.ceil((n + 1) * (1.0 - alpha)) / n
        level = float(np.clip(level, 0.0, 1.0))
        return float(np.quantile(scores, level, method="higher"))

    def fit(self, y_cal: np.ndarray, prob_cal: np.ndarray) -> SplitConformalClassifier:
        """Calibrate non-conformity score thresholds strictly on validation data."""
        y_cal_arr = np.asarray(y_cal, dtype=int).ravel()
        prob_cal_arr = np.clip(np.asarray(prob_cal, dtype=float).ravel(), 0.0, 1.0)

        # Non-conformity score s_i = 1 - p(Y_i | X_i)
        scores = np.where(y_cal_arr == 1, 1.0 - prob_cal_arr, prob_cal_arr)
        self.q_hat_marginal_ = self._finite_sample_quantile(scores, self.alpha)

        if self.mondrian:
            for c in (0, 1):
                mask = y_cal_arr == c
                if np.any(mask):
                    self.q_hat_class_[c] = self._finite_sample_quantile(scores[mask], self.alpha)
                else:
                    self.q_hat_class_[c] = self.q_hat_marginal_
        else:
            self.q_hat_class_ = {0: self.q_hat_marginal_, 1: self.q_hat_marginal_}

        return self

    def predict_sets(self, prob_test: np.ndarray) -> np.ndarray:
        """Return boolean inclusion matrix of shape (N, 2) for classes {0, 1}."""
        if self.q_hat_marginal_ is None:
            raise RuntimeError("SplitConformalClassifier must be calibrated with .fit() first.")

        p1 = np.clip(np.asarray(prob_test, dtype=float).ravel(), 0.0, 1.0)
        p0 = 1.0 - p1

        score_0 = 1.0 - p0  # = p1
        score_1 = 1.0 - p1  # = p0

        include_0 = score_0 <= self.q_hat_class_[0]
        include_1 = score_1 <= self.q_hat_class_[1]
        return np.column_stack([include_0, include_1])

    def evaluate(self, y_test: np.ndarray, prob_test: np.ndarray) -> dict[str, float]:
        """Evaluate empirical coverage, efficiency (set size), and clinical abstention rates."""
        y_arr = np.asarray(y_test, dtype=int).ravel()
        pred_sets = self.predict_sets(prob_test)

        covered = pred_sets[np.arange(len(y_arr)), y_arr]
        set_sizes = np.sum(pred_sets, axis=1)

        pos_mask = y_arr == 1
        neg_mask = y_arr == 0

        cert_low_risk = float(np.mean(pred_sets[:, 0] & ~pred_sets[:, 1]))
        cert_high_risk = float(np.mean(~pred_sets[:, 0] & pred_sets[:, 1]))
        ambiguous_abstain = float(np.mean(pred_sets[:, 0] & pred_sets[:, 1]))
        empty_rate = float(np.mean(~pred_sets[:, 0] & ~pred_sets[:, 1]))

        return {
            "target_coverage": float(1.0 - self.alpha),
            "empirical_coverage": float(np.mean(covered)),
            "positive_class_coverage": float(np.mean(covered[pos_mask])) if np.any(pos_mask) else 1.0,
            "negative_class_coverage": float(np.mean(covered[neg_mask])) if np.any(neg_mask) else 1.0,
            "average_set_size": float(np.mean(set_sizes)),
            "singleton_rate": float(np.mean(set_sizes == 1)),
            "ambiguous_abstention_rate": ambiguous_abstain,
            "certifiable_low_risk_rate": cert_low_risk,
            "certifiable_high_risk_rate": cert_high_risk,
            "empty_set_rate": empty_rate,
            "q_hat_class_0": float(self.q_hat_class_[0]),
            "q_hat_class_1": float(self.q_hat_class_[1]),
        }


# =============================================================================
# UPGRADE 3: ACTIONABLE CLINICAL COUNTERFACTUAL RECOURSE
# =============================================================================


class ClinicalCounterfactualGenerator:
    """Generate actionable, physiologically plausible counterfactual explanations.

    Enforces:
    1. Hard immutability constraints on unmodifiable attributes (e.g., age, sex).
    2. Physiological box constraints [lower_j, upper_j] derived from training quantiles.
    3. Normalized L1/L2 proximity weighted by training Median Absolute Deviation (MAD).
    4. Mahalanobis manifold regularization via Ledoit-Wolf shrinkage precision matrix.
    """

    def __init__(
        self,
        model: nn.Module,
        feature_names: Sequence[str],
        immutable_features: Sequence[str] = ("age", "sex", "gender", "fbs", "had_icu", "has_ed_visit"),
        target_probability: float = 0.25,
        margin_kappa: float = 0.15,
        lambda_l1: float = 0.05,
        lambda_l2: float = 0.02,
        lambda_mahalanobis: float = 0.01,
        max_iter: int = 250,
        lr: float = 0.05,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.immutable_features = set(immutable_features)
        self.target_probability = float(target_probability)
        self.margin_kappa = float(margin_kappa)
        self.lambda_l1 = float(lambda_l1)
        self.lambda_l2 = float(lambda_l2)
        self.lambda_mahalanobis = float(lambda_mahalanobis)
        self.max_iter = int(max_iter)
        self.lr = float(lr)

        self.immutable_indices_: list[int] = [
            i for i, name in enumerate(self.feature_names) if name in self.immutable_features
        ]
        self.actionable_indices_: list[int] = [
            i for i, name in enumerate(self.feature_names) if name not in self.immutable_features
        ]
        self.lower_bounds_: np.ndarray | None = None
        self.upper_bounds_: np.ndarray | None = None
        self.mad_weights_: np.ndarray | None = None
        self.precision_matrix_: np.ndarray | None = None

    def fit(
        self,
        X_train: np.ndarray,
        custom_bounds: dict[str, tuple[float, float]] | None = None,
    ) -> ClinicalCounterfactualGenerator:
        """Fit physiological bounds, MAD scales, and precision matrix strictly on X_train."""
        X = np.asarray(X_train, dtype=np.float32)
        n_features = X.shape[1]
        if n_features != len(self.feature_names):
            raise ValueError("X_train column count does not match feature_names length.")

        # 1. Physiological bounds (1st to 99th training percentiles unless overridden)
        self.lower_bounds_ = np.percentile(X, 1.0, axis=0).astype(np.float32)
        self.upper_bounds_ = np.percentile(X, 99.0, axis=0).astype(np.float32)
        if custom_bounds:
            for feat, (lo, hi) in custom_bounds.items():
                if feat in self.feature_names:
                    idx = self.feature_names.index(feat)
                    self.lower_bounds_[idx] = float(lo)
                    self.upper_bounds_[idx] = float(hi)

        # 2. Inverse MAD feature weighting for unit-invariant sparsity
        medians = np.median(X, axis=0)
        mad = np.median(np.abs(X - medians), axis=0)
        std_fallback = np.std(X, axis=0)
        scale = np.where(mad > 1e-4, mad * 1.4826, np.where(std_fallback > 1e-4, std_fallback, 1.0))
        self.mad_weights_ = (1.0 / scale).astype(np.float32)

        # 3. Shrinkage-regularized precision matrix for Mahalanobis plausibility
        lw = LedoitWolf().fit(X)
        self.precision_matrix_ = lw.precision_.astype(np.float32)
        return self

    def generate(self, x_factual: np.ndarray) -> dict[str, np.ndarray | float | bool | dict[str, float]]:
        """Generate an actionable counterfactual for a single patient vector x_factual."""
        x0 = np.asarray(x_factual, dtype=np.float32).reshape(1, -1)
        return self.generate_batch(x0)[0]

    def generate_batch(self, X_factual: np.ndarray) -> list[dict[str, np.ndarray | float | bool | dict[str, float]]]:
        """Generate counterfactuals for many patients in one optimisation.

        Every patient's loss term depends only on that patient's row and Adam
        updates each coordinate independently, so optimising the rows together
        follows the same per-patient trajectories as one-at-a-time calls while
        running the model once per step for the whole batch.
        """
        if self.lower_bounds_ is None or self.mad_weights_ is None or self.precision_matrix_ is None:
            raise RuntimeError("ClinicalCounterfactualGenerator must be fitted on X_train first.")

        self.model.eval()
        device = next(self.model.parameters()).device

        x0_np = np.asarray(X_factual, dtype=np.float32)
        if x0_np.ndim == 1:
            x0_np = x0_np.reshape(1, -1)
        n_rows = x0_np.shape[0]
        if n_rows == 0:
            return []
        x0_t = torch.tensor(x0_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            orig_probs = torch.sigmoid(self.model(x0_t).view(-1)).cpu().numpy()

        target_logit = float(np.log(self.target_probability / (1.0 - self.target_probability)))
        x_cf = x0_t.clone().detach().requires_grad_(True)

        w_t = torch.tensor(self.mad_weights_, dtype=torch.float32, device=device)
        prec_t = torch.tensor(self.precision_matrix_, dtype=torch.float32, device=device)
        lo_t = torch.tensor(self.lower_bounds_, dtype=torch.float32, device=device).unsqueeze(0)
        hi_t = torch.tensor(self.upper_bounds_, dtype=torch.float32, device=device).unsqueeze(0)
        immutable = self.immutable_indices_

        optimizer = torch.optim.Adam([x_cf], lr=self.lr)

        best_cf = x0_t.clone().detach()
        best_cost = torch.full((n_rows,), float("inf"), device=device)
        found_valid = torch.zeros(n_rows, dtype=torch.bool, device=device)

        for _ in range(self.max_iter):
            optimizer.zero_grad()
            logit = self.model(x_cf).view(-1)

            # Hinge-logit validity loss with safety margin kappa
            validity_loss = torch.relu(logit - target_logit + self.margin_kappa)

            diff = x_cf - x0_t
            weighted_diff = w_t * diff
            l1_loss = torch.sum(torch.abs(weighted_diff), dim=1)
            l2_loss = torch.sum(weighted_diff**2, dim=1)
            mahal_loss = torch.sum((diff @ prec_t) * diff, dim=1)

            reg_cost = (
                self.lambda_l1 * l1_loss
                + self.lambda_l2 * l2_loss
                + self.lambda_mahalanobis * mahal_loss
            )
            total_loss = torch.sum(validity_loss + reg_cost)
            total_loss.backward()
            optimizer.step()

            # Hard projection step: enforce physiological bounds & strict immutability
            with torch.no_grad():
                x_cf.data = torch.max(torch.min(x_cf.data, hi_t), lo_t)
                if immutable:
                    x_cf.data[:, immutable] = x0_t[:, immutable]

                curr_prob = torch.sigmoid(self.model(x_cf).view(-1))
                curr_reg = reg_cost.detach()
                improved = (curr_prob <= self.target_probability) & (curr_reg < best_cost)
                best_cost = torch.where(improved, curr_reg, best_cost)
                best_cf = torch.where(improved.unsqueeze(1), x_cf.detach(), best_cf)
                found_valid |= improved

        with torch.no_grad():
            # Rows that never reached the target keep their final iterate.
            best_cf = torch.where(found_valid.unsqueeze(1), best_cf, x_cf.detach())
            cf_probs = torch.sigmoid(self.model(best_cf).view(-1)).cpu().numpy()

        cf_np = best_cf.cpu().numpy()
        results = []
        for row in range(n_rows):
            delta = cf_np[row] - x0_np[row]
            mahal_dist = float(np.sqrt(max(0.0, float(delta @ self.precision_matrix_ @ delta))))
            active_changes = {
                self.feature_names[i]: float(delta[i])
                for i in self.actionable_indices_
                if abs(delta[i]) > 1e-3
            }
            cf_prob = float(cf_probs[row])
            results.append({
                "x_factual": x0_np[row],
                "x_counterfactual": cf_np[row],
                "delta": delta,
                "original_probability": float(orig_probs[row]),
                "counterfactual_probability": cf_prob,
                "validity_achieved": bool(cf_prob <= self.target_probability),
                "sparsity_count": len(active_changes),
                "mahalanobis_distance": mahal_dist,
                "immutable_violation_max": float(
                    np.max(np.abs(delta[immutable])) if immutable else 0.0
                ),
                "recommended_interventions": active_changes,
            })
        return results

    def evaluate_cohort_recourse(
        self,
        X_test: np.ndarray,
        high_risk_threshold: float = 0.50,
        max_patients: int = 25,
    ) -> dict[str, float]:
        """Benchmark recourse validity, sparsity, and manifold plausibility on high-risk test patients."""
        self.model.eval()
        device = next(self.model.parameters()).device
        X_arr = np.asarray(X_test, dtype=np.float32)

        with torch.no_grad():
            probs = (
                torch.sigmoid(self.model(torch.tensor(X_arr, dtype=torch.float32, device=device)))
                .cpu()
                .numpy()
                .ravel()
            )

        high_risk_idx = np.where(probs >= high_risk_threshold)[0][:max_patients]
        if len(high_risk_idx) == 0:
            high_risk_idx = np.argsort(probs)[::-1][: min(10, len(X_arr))]

        batch = self.generate_batch(X_arr[high_risk_idx])
        validities = [float(res["validity_achieved"]) for res in batch]
        sparsities = [float(res["sparsity_count"]) for res in batch]
        mahal_dists = [float(res["mahalanobis_distance"]) for res in batch]
        risk_drops = [float(res["original_probability"] - res["counterfactual_probability"]) for res in batch]
        immut_violations = [float(res["immutable_violation_max"]) for res in batch]

        return {
            "n_evaluated": int(len(high_risk_idx)),
            "recourse_validity_rate": float(np.mean(validities)),
            "mean_features_modified": float(np.mean(sparsities)),
            "mean_mahalanobis_distance": float(np.mean(mahal_dists)),
            "mean_probability_reduction": float(np.mean(risk_drops)),
            "max_immutable_violation": float(np.max(immut_violations)),
        }


# =============================================================================
# PUBLICATION FIGURES: CONFORMAL TRIAGE, COUNTERFACTUALS & AIPW LOVE PLOT
# =============================================================================


def _apply_publication_style() -> None:
    """Apply consistent, journal-grade Matplotlib styling across all figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.labelweight": "bold",
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.titlesize": 13,
            "figure.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def plot_conformal_triage_curve(
    conformal_classifier: SplitConformalClassifier,
    y_val: np.ndarray,
    prob_val: np.ndarray,
    y_test: np.ndarray,
    prob_test: np.ndarray,
    output_path: str | Path = "artifacts/figures/conformal_triage_curve.png",
) -> dict[str, Any]:
    """Generate a 2-panel publication figure for Conformal Prediction Risk Triage.

    Panel A: Target (1 - alpha) vs. Empirical Marginal and Class-Conditional Coverage.
    Panel B: Clinical Triage Zone Proportions (Certifiable Low-Risk {0}, Certifiable
             High-Risk {1}, and Ambiguous Abstention {0, 1}) across miscoverage alpha.
    """
    _apply_publication_style()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    y_val_arr = np.asarray(y_val, dtype=int).ravel()
    prob_val_arr = np.clip(np.asarray(prob_val, dtype=float).ravel(), 0.0, 1.0)
    y_test_arr = np.asarray(y_test, dtype=int).ravel()
    prob_test_arr = np.clip(np.asarray(prob_test, dtype=float).ravel(), 0.0, 1.0)

    # Ensure primary classifier is calibrated and evaluate at operating alpha
    conformal_classifier.fit(y_val_arr, prob_val_arr)
    op_alpha = float(conformal_classifier.alpha)
    op_metrics = conformal_classifier.evaluate(y_test_arr, prob_test_arr)

    # Sweep miscoverage alpha from 0.01 to 0.35
    alphas = np.linspace(0.01, 0.35, 35)
    target_covs = 1.0 - alphas

    emp_cov_list = []
    pos_cov_list = []
    neg_cov_list = []
    low_risk_list = []
    high_risk_list = []
    abstain_list = []
    empty_list = []

    for a in alphas:
        clf_a = SplitConformalClassifier(alpha=float(a), mondrian=conformal_classifier.mondrian)
        clf_a.fit(y_val_arr, prob_val_arr)
        m = clf_a.evaluate(y_test_arr, prob_test_arr)
        emp_cov_list.append(m["empirical_coverage"])
        pos_cov_list.append(m["positive_class_coverage"])
        neg_cov_list.append(m["negative_class_coverage"])
        low_risk_list.append(m["certifiable_low_risk_rate"])
        high_risk_list.append(m["certifiable_high_risk_rate"])
        abstain_list.append(m["ambiguous_abstention_rate"])
        empty_list.append(m["empty_set_rate"])

    emp_cov = np.array(emp_cov_list)
    pos_cov = np.array(pos_cov_list)
    neg_cov = np.array(neg_cov_list)
    low_risk = np.array(low_risk_list)
    high_risk = np.array(high_risk_list)
    abstain = np.array(abstain_list)
    empty = np.array(empty_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=300)
    mode_label = "Mondrian Class-Conditional" if conformal_classifier.mondrian else "Marginal Split"

    # --- Panel A: Finite-Sample Coverage Calibration ---
    ax1.plot(
        target_covs * 100,
        target_covs * 100,
        linestyle="--",
        color="#555555",
        linewidth=1.5,
        label="Ideal Finite-Sample Guarantee ($y = x$)",
    )
    ax1.plot(
        target_covs * 100,
        emp_cov * 100,
        color="#1f4e78",
        linewidth=2.4,
        label="Empirical Marginal Coverage",
    )
    ax1.plot(
        target_covs * 100,
        pos_cov * 100,
        color="#d95f02",
        linewidth=2.0,
        linestyle="-.",
        label="Positive Class ($Y=1$) Coverage",
    )
    ax1.plot(
        target_covs * 100,
        neg_cov * 100,
        color="#1b9e77",
        linewidth=2.0,
        linestyle=":",
        label="Negative Class ($Y=0$) Coverage",
    )

    op_target_pct = (1.0 - op_alpha) * 100.0
    op_emp_pct = op_metrics["empirical_coverage"] * 100.0
    ax1.axvline(
        op_target_pct,
        color="#b22222",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
        label=f"Operating Point ($1-\\alpha = {op_target_pct:.0f}\\%$)",
    )
    ax1.scatter([op_target_pct], [op_emp_pct], color="#b22222", s=55, zorder=5)
    ax1.annotate(
        f"Emp: {op_emp_pct:.1f}%\n(Y=1: {op_metrics['positive_class_coverage']*100:.1f}%)",
        xy=(op_target_pct, op_emp_pct),
        xytext=(op_target_pct - 12.5, max(68.0, op_emp_pct - 14.0)),
        fontsize=8.5,
        fontweight="bold",
        color="#1f4e78",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f4f8fb", edgecolor="#1f4e78", alpha=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.15", color="#1f4e78"),
    )

    ax1.set_title(f"A. {mode_label} Coverage Guarantee")
    ax1.set_xlabel("Target Confidence Level $1 - \\alpha$ (%)")
    ax1.set_ylabel("Empirical Test Set Coverage (%)")
    ax1.set_xlim(65, 99.5)
    ax1.set_ylim(60, 101.5)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)

    # --- Panel B: Clinical Triage Zone Decomposition ---
    ax2.stackplot(
        target_covs * 100,
        low_risk * 100,
        high_risk * 100,
        abstain * 100,
        empty * 100,
        labels=[
            "Certifiable Low-Risk Singleton $\\hat{C}(x) = \\{0\\}$",
            "Certifiable High-Risk Singleton $\\hat{C}(x) = \\{1\\}$",
            "Ambiguous / Abstain for Clinical Review $\\hat{C}(x) = \\{0, 1\\}$",
            "Empty Prediction Set $\\hat{C}(x) = \\emptyset$",
        ],
        colors=["#2b8cbe", "#e34a33", "#fdbb84", "#cccccc"],
        alpha=0.85,
    )

    ax2.axvline(op_target_pct, color="#111111", linestyle="--", linewidth=1.5)
    ax2.annotate(
        f"At $1-\\alpha = {op_target_pct:.0f}\\%$:\n"
        f"• Low-Risk {{0}}: {op_metrics['certifiable_low_risk_rate']*100:.1f}%\n"
        f"• High-Risk {{1}}: {op_metrics['certifiable_high_risk_rate']*100:.1f}%\n"
        f"• Abstain {{0,1}}: {op_metrics['ambiguous_abstention_rate']*100:.1f}%",
        xy=(op_target_pct, 50.0),
        xytext=(67.0, 56.0),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#333333", alpha=0.92),
        arrowprops=dict(arrowstyle="->", color="#222222", lw=1.2),
    )

    ax2.set_title("B. Clinical Risk Triage & Abstention Trade-Off")
    ax2.set_xlabel("Target Confidence Level $1 - \\alpha$ (%)")
    ax2.set_ylabel("Proportion of Test Cohort (%)")
    ax2.set_xlim(65, 99.5)
    ax2.set_ylim(0, 100)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)

    fig.suptitle(
        "Distribution-Free Conformal Risk Triage: Calibration & Selective Abstention",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "output_path": str(out_file),
        "operating_alpha": op_alpha,
        "operating_metrics": op_metrics,
    }


def plot_counterfactual_case_study(
    generator: ClinicalCounterfactualGenerator,
    X_test: np.ndarray,
    feature_names: Sequence[str] | None = None,
    output_path: str | Path = "artifacts/figures/counterfactual_case_study.png",
) -> dict[str, Any]:
    """Generate a 2-panel publication plot of patient-level counterfactual recourse.

    Panel A: Standardized feature profile showing Factual vs. Counterfactual values,
             actionable physiological interventions, and locked immutable features.
    Panel B: Predicted probability reduction across the safety margin kappa and
             recourse quality metrics (Sparsity, Mahalanobis plausibility, Immutability).
    """
    _apply_publication_style()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    X_arr = np.asarray(X_test, dtype=np.float32)
    names = list(feature_names) if feature_names is not None else list(generator.feature_names)

    generator.model.eval()
    device = next(generator.model.parameters()).device
    with torch.no_grad():
        probs = (
            torch.sigmoid(generator.model(torch.tensor(X_arr, dtype=torch.float32, device=device)))
            .cpu()
            .numpy()
            .ravel()
        )

    # Select a representative high-risk patient that achieves valid recourse
    sorted_indices = np.argsort(probs)[::-1]
    candidates_idx = sorted_indices[: min(25, len(sorted_indices))]
    candidates = generator.generate_batch(X_arr[candidates_idx])
    selected_idx = int(candidates_idx[0])
    cf_result = candidates[0]

    for idx, candidate in zip(candidates_idx, candidates, strict=True):
        if candidate["validity_achieved"] and candidate["sparsity_count"] >= 1:
            selected_idx = int(idx)
            cf_result = candidate
            break

    x_fact = np.asarray(cf_result["x_factual"], dtype=float)
    x_cf = np.asarray(cf_result["x_counterfactual"], dtype=float)
    delta = np.asarray(cf_result["delta"], dtype=float)
    mad_weights = np.asarray(generator.mad_weights_, dtype=float)

    # Normalize feature coordinates into [0, 1] physiological range for visual alignment
    lo = np.asarray(generator.lower_bounds_, dtype=float)
    hi = np.asarray(generator.upper_bounds_, dtype=float)
    span = np.where((hi - lo) > 1e-6, hi - lo, 1.0)
    norm_fact = np.clip((x_fact - lo) / span, 0.0, 1.0)
    norm_cf = np.clip((x_cf - lo) / span, 0.0, 1.0)

    # Order features: Immutable features at the bottom, actionable sorted by |weighted delta| at top
    immutable_set = set(generator.immutable_indices_)
    weighted_abs_delta = np.abs(delta * mad_weights)

    actionable_order = sorted(
        [i for i in range(len(names)) if i not in immutable_set],
        key=lambda i: weighted_abs_delta[i],
    )
    immutable_order = sorted(list(immutable_set), reverse=True)
    plot_order = immutable_order + actionable_order

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14.0, max(5.6, 0.42 * len(names) + 1.5)), dpi=300, gridspec_kw={"width_ratios": [1.85, 1.0]}
    )

    y_positions = np.arange(len(plot_order))
    y_labels = []

    for pos, feat_idx in enumerate(plot_order):
        fname = names[feat_idx]
        is_immutable = feat_idx in immutable_set
        f_val = x_fact[feat_idx]
        c_val = x_cf[feat_idx]
        d_val = delta[feat_idx]
        nf = norm_fact[feat_idx]
        nc = norm_cf[feat_idx]

        # Physiological box-constraint background track
        ax1.plot([0.0, 1.0], [pos, pos], color="#e6e6e6", linewidth=5.0, solid_capstyle="round", zorder=1)

        if is_immutable:
            y_labels.append(f"{fname}  [LOCKED]")
            ax1.axhspan(pos - 0.38, pos + 0.38, color="#f2f2f2", alpha=0.8, zorder=0)
            ax1.scatter(
                [nf],
                [pos],
                marker="s",
                s=95,
                facecolor="#636363",
                edgecolor="#111111",
                linewidth=1.5,
                zorder=4,
            )
            ax1.text(
                min(0.98, nf + 0.03),
                pos,
                f"Immutable ({f_val:.2f}, $\\Delta=0$)",
                va="center",
                ha="left",
                fontsize=8.0,
                fontstyle="italic",
                color="#444444",
            )
        else:
            y_labels.append(fname)
            is_modified = abs(d_val) > 1e-3
            if is_modified:
                ax1.annotate(
                    "",
                    xy=(nc, pos),
                    xytext=(nf, pos),
                    arrowprops=dict(arrowstyle="-|>", color="#006d2c", lw=2.2, mutation_scale=13),
                    zorder=3,
                )
                ax1.scatter([nf], [pos], marker="o", s=70, color="#cb181d", zorder=4)
                ax1.scatter([nc], [pos], marker="D", s=70, color="#006d2c", zorder=5)
                sign_str = "+" if d_val > 0 else ""
                ax1.text(
                    max(nf, nc) + 0.025,
                    pos,
                    f"{f_val:.2f} $\\rightarrow$ {c_val:.2f} ({sign_str}{d_val:.2f})",
                    va="center",
                    ha="left",
                    fontsize=8.3,
                    fontweight="bold",
                    color="#00441b",
                )
            else:
                ax1.scatter([nf], [pos], marker="o", s=55, color="#888888", alpha=0.8, zorder=3)
                ax1.text(
                    min(0.98, nf + 0.025),
                    pos,
                    f"{f_val:.2f} (Unchanged)",
                    va="center",
                    ha="left",
                    fontsize=8.0,
                    color="#666666",
                )

    if immutable_order and actionable_order:
        sep_y = len(immutable_order) - 0.5
        ax1.axhline(sep_y, color="#444444", linestyle="--", linewidth=1.1, alpha=0.7)
        ax1.text(
            0.01,
            sep_y + 0.15,
            "▲ Actionable Physiological Features",
            fontsize=8.0,
            fontweight="bold",
            color="#00441b",
        )
        ax1.text(
            0.01,
            sep_y - 0.35,
            "▼ Locked Immutable Features (Hard Projection $\\Delta = 0$)",
            fontsize=8.0,
            fontweight="bold",
            color="#444444",
        )

    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(y_labels)
    ax1.set_xlim(-0.04, 1.38)
    ax1.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax1.set_xticklabels(["1st Pct\n(Min Bound)", "25%", "50%", "75%", "99th Pct\n(Max Bound)"])
    ax1.set_xlabel("Normalized Physiological Range $[L_j, U_j]$")
    ax1.set_title(f"A. Patient #{selected_idx} Counterfactual Recourse Profile")
    ax1.grid(axis="x", linestyle="--", alpha=0.3)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Factual High-Risk State $x$", markerfacecolor="#cb181d", markersize=8),
        Line2D([0], [0], marker="D", color="w", label="Actionable Counterfactual $x_{\\mathrm{cf}}$", markerfacecolor="#006d2c", markersize=8),
        Line2D([0], [0], marker="s", color="w", label="Locked Immutable Attribute ($\\Delta=0$)", markerfacecolor="#636363", markeredgecolor="#111111", markersize=8),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", frameon=True, facecolor="white", framealpha=0.95)

    # --- Panel B: Risk Reduction & Manifold Plausibility ---
    orig_p = float(cf_result["original_probability"]) * 100.0
    cf_p = float(cf_result["counterfactual_probability"]) * 100.0
    target_p = generator.target_probability * 100.0

    bars = ax2.bar(
        ["Factual State\n$P(Y=1 \\mid x)$", "Post-Recourse\n$P(Y=1 \\mid x_{\\mathrm{cf}})$"],
        [orig_p, cf_p],
        color=["#cb181d", "#006d2c"],
        width=0.48,
        edgecolor="#222222",
        linewidth=1.2,
        zorder=3,
    )

    for bar, val in zip(bars, [orig_p, cf_p], strict=True):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 2.2,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax2.axhline(
        target_p,
        color="#1f4e78",
        linestyle="--",
        linewidth=1.8,
        label=f"Target Safe Threshold $\\tau = {target_p:.1f}\\%$",
        zorder=4,
    )
    ax2.axhspan(0.0, target_p, color="#e5f5e0", alpha=0.45, zorder=1, label="Certified Low-Risk Zone")

    summary_text = (
        f"Recourse Diagnostic Summary\n"
        f"───────────────────────────\n"
        f"• Validity Achieved: {'YES' if cf_result['validity_achieved'] else 'NO'}\n"
        f"• Absolute Risk Drop: -{orig_p - cf_p:.1f}%\n"
        f"• Sparsity ($L_0$ Modified): {cf_result['sparsity_count']} / {len(names)} features\n"
        f"• Mahalanobis Dist $D_M$: {cf_result['mahalanobis_distance']:.2f}\n"
        f"• Max Immutable Violation: {cf_result['immutable_violation_max']:.4f}"
    )
    ax2.text(
        0.50,
        0.60,
        summary_text,
        transform=ax2.transAxes,
        fontsize=8.8,
        va="center",
        ha="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", edgecolor="#1f4e78", linewidth=1.2),
        zorder=5,
    )

    ax2.set_ylim(0, 108)
    ax2.set_ylabel("Predicted Clinical Risk Probability (%)")
    ax2.set_title("B. Risk Reduction & Manifold Diagnostics")
    ax2.grid(axis="y", linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", frameon=True, facecolor="white")

    fig.suptitle(
        "Actionable Clinical Counterfactual Recourse with Hard Immutability & Manifold Constraints",
        y=1.02,
    )
    plt.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "output_path": str(out_file),
        "patient_index": selected_idx,
        "counterfactual_result": cf_result,
    }


def plot_aipw_love_plot(
    X_confounders: np.ndarray,
    treatment: np.ndarray,
    propensity_scores: np.ndarray | None = None,
    feature_names: Sequence[str] | None = None,
    output_path: str | Path = "artifacts/figures/aipw_love_plot.png",
) -> dict[str, Any]:
    """Compute exact Standardized Mean Differences (SMD) and plot a Covariate Balance Love Plot.

    Exact Standardized Mean Difference (Austin, 2009) Formula:
        s_pool = sqrt((Var(X_1) + Var(X_0)) / 2)
        SMD_unadjusted = (Mean(X_1) - Mean(X_0)) / s_pool
        SMD_iptw       = (WeightedMean(X_1, w) - WeightedMean(X_0, w)) / s_pool
    where stabilized/clipped IPTW weights are w_i = T_i / e(X_i) + (1 - T_i) / (1 - e(X_i)).
    """
    _apply_publication_style()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    X = np.asarray(X_confounders, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    T = np.asarray(treatment, dtype=int).ravel()
    n_samples, n_features = X.shape

    if feature_names is None:
        names = [f"Confounder {j + 1}" for j in range(n_features)]
    else:
        names = list(feature_names)

    # Estimate cross-fitted propensity scores if not provided
    if propensity_scores is None:
        e_hat = np.zeros(n_samples, dtype=float)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        base_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        for train_idx, eval_idx in skf.split(X, T):
            model = clone(base_lr)
            model.fit(X[train_idx], T[train_idx])
            e_hat[eval_idx] = model.predict_proba(X[eval_idx])[:, 1]
    else:
        e_hat = np.asarray(propensity_scores, dtype=float).ravel()

    e_clipped = np.clip(e_hat, 0.025, 0.975)

    # Inverse Probability of Treatment Weights (IPTW)
    weights = np.where(T == 1, 1.0 / e_clipped, 1.0 / (1.0 - e_clipped))

    mask_1 = T == 1
    mask_0 = T == 0
    X1, X0 = X[mask_1], X[mask_0]
    w1, w0 = weights[mask_1], weights[mask_0]

    # Unadjusted means and sample variances (ddof=1)
    mean_1_unadj = np.mean(X1, axis=0)
    mean_0_unadj = np.mean(X0, axis=0)
    var_1_unadj = np.var(X1, axis=0, ddof=1)
    var_0_unadj = np.var(X0, axis=0, ddof=1)

    # Pooled baseline standard deviation: sqrt((s_1^2 + s_0^2) / 2)
    s_pool = np.sqrt((var_1_unadj + var_0_unadj) / 2.0)
    s_pool = np.where(s_pool > 1e-8, s_pool, 1.0)

    smd_unadjusted = (mean_1_unadj - mean_0_unadj) / s_pool

    # IPTW-weighted means
    mean_1_iptw = np.average(X1, weights=w1, axis=0)
    mean_0_iptw = np.average(X0, weights=w0, axis=0)
    smd_iptw = (mean_1_iptw - mean_0_iptw) / s_pool

    abs_smd_unadj = np.abs(smd_unadjusted)
    abs_smd_iptw = np.abs(smd_iptw)

    # Sort covariates by unadjusted |SMD| ascending (so largest imbalance is at the top)
    sort_idx = np.argsort(abs_smd_unadj)
    sorted_names = [names[i] for i in sort_idx]
    sorted_unadj = abs_smd_unadj[sort_idx]
    sorted_iptw = abs_smd_iptw[sort_idx]

    fig, ax = plt.subplots(figsize=(9.5, max(5.0, 0.38 * n_features + 1.6)), dpi=300)
    y_pos = np.arange(n_features)

    # Horizontal connecting lines showing bias reduction per confounder
    for y, u_val, w_val in zip(y_pos, sorted_unadj, sorted_iptw, strict=True):
        ax.plot([w_val, u_val], [y, y], color="#888888", linestyle="-", linewidth=1.4, zorder=2)

    ax.scatter(
        sorted_unadj,
        y_pos,
        color="#d95f02",
        marker="o",
        s=70,
        label="Unadjusted Observational Cohort",
        zorder=4,
    )
    ax.scatter(
        sorted_iptw,
        y_pos,
        color="#1f4e78",
        marker="D",
        s=65,
        label="IPTW Doubly-Robust Pseudo-Population",
        zorder=5,
    )

    # Standard clinical covariate balance threshold at |SMD| = 0.10
    ax.axvline(
        0.10,
        color="#b22222",
        linestyle="--",
        linewidth=1.6,
        label="Clinical Balance Threshold ($|\\mathrm{SMD}| = 0.10$)",
        zorder=3,
    )
    ax.axvspan(0.0, 0.10, color="#e5f5e0", alpha=0.40, zorder=1, label="Well-Balanced Region ($|\\mathrm{SMD}| < 0.10$)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    max_x = max(0.25, float(np.max(sorted_unadj)) * 1.15)
    ax.set_xlim(-0.01, max_x)
    ax.set_xlabel("Absolute Standardized Mean Difference ($|\\mathrm{SMD}|$)")
    ax.set_title("Covariate Balance Love Plot: Confounding Reduction via Propensity Weighting")
    ax.grid(axis="x", linestyle="--", alpha=0.3)

    n_balanced_before = int(np.sum(abs_smd_unadj < 0.10))
    n_balanced_after = int(np.sum(abs_smd_iptw < 0.10))
    summary_box = (
        f"Balance Diagnostics ($|\\mathrm{{SMD}}| < 0.10$)\n"
        f"• Unadjusted: {n_balanced_before}/{n_features} covariates balanced\n"
        f"• Mean $|\\mathrm{{SMD}}|$ (Unadj): { float(np.mean(abs_smd_unadj)):.3f}\n"
        f"• Post-IPTW: {n_balanced_after}/{n_features} covariates balanced\n"
        f"• Mean $|\\mathrm{{SMD}}|$ (IPTW): {float(np.mean(abs_smd_iptw)):.3f}"
    )
    ax.text(
        0.97,
        0.18,
        summary_box,
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#1f4e78", alpha=0.95),
        zorder=6,
    )

    ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95)
    plt.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "output_path": str(out_file),
        "feature_names": names,
        "smd_unadjusted": {names[i]: float(smd_unadjusted[i]) for i in range(n_features)},
        "smd_iptw": {names[i]: float(smd_iptw[i]) for i in range(n_features)},
        "mean_abs_smd_unadjusted": float(np.mean(abs_smd_unadj)),
        "mean_abs_smd_iptw": float(np.mean(abs_smd_iptw)),
        "balanced_fraction_iptw": float(n_balanced_after / max(1, n_features)),
    }


def evaluate_loco_conformal(
    df_combined: pd.DataFrame,
    feature_cols: Sequence[str] | None = None,
    site_col: str = "site",
    target_col: str = "target",
    alpha: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate Leave-One-Center-Out (LOCO) cross-hospital generalization of Conformal Prediction.

    For each international hospital center:
    1. Trains on all other centers.
    2. Calibrates Mondrian Conformal Prediction on an internal validation fold.
    3. Evaluates out-of-domain finite-sample coverage and triage efficiency on the unseen center.
    """
    if feature_cols is None:
        ignore_cols = {site_col, target_col, "id", "dataset"}
        feature_cols = [c for c in df_combined.columns if c not in ignore_cols]

    unique_sites = sorted(df_combined[site_col].dropna().unique().tolist())
    records = []

    for test_site in unique_sites:
        train_df = df_combined[df_combined[site_col] != test_site].copy()
        test_df = df_combined[df_combined[site_col] == test_site].copy()

        y_train_full = train_df[target_col].values.astype(int)
        y_test = test_df[target_col].values.astype(int)

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_raw = imputer.fit_transform(train_df[feature_cols].values)
        X_train_scaled = scaler.fit_transform(X_train_raw)

        X_test_raw = imputer.transform(test_df[feature_cols].values)
        X_test_scaled = scaler.transform(X_test_raw)

        # 80/20 train/calibration split within training sites
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fit_idx, cal_idx = next(skf.split(X_train_scaled, y_train_full))

        X_fit, y_fit = X_train_scaled[fit_idx], y_train_full[fit_idx]
        X_cal, y_cal = X_train_scaled[cal_idx], y_train_full[cal_idx]

        model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        model.fit(X_fit, y_fit)

        prob_cal = model.predict_proba(X_cal)[:, 1]
        prob_test = model.predict_proba(X_test_scaled)[:, 1]

        # Standard fixed-threshold accuracy & coverage
        pred_fixed = (prob_test >= 0.5).astype(int)
        fixed_acc = float(np.mean(pred_fixed == y_test))

        conformal = SplitConformalClassifier(alpha=alpha, mondrian=True).fit(y_cal, prob_cal)
        conf_eval = conformal.evaluate(y_test, prob_test)

        records.append({
            "test_site": test_site,
            "n_train_sites": len(unique_sites) - 1,
            "n_test_samples": len(test_df),
            "test_disease_prevalence": float(np.mean(y_test)),
            "fixed_threshold_accuracy": fixed_acc,
            "target_conformal_coverage": 1.0 - alpha,
            "loco_empirical_coverage": conf_eval["empirical_coverage"],
            "positive_class_coverage": conf_eval["positive_class_coverage"],
            "negative_class_coverage": conf_eval["negative_class_coverage"],
            "average_set_size": conf_eval["average_set_size"],
            "ambiguous_abstention_rate": conf_eval["ambiguous_abstention_rate"],
        })

    return pd.DataFrame(records)


