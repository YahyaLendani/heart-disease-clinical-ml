"""Defense-ready methodological upgrades for clinical AI:
1. Cross-Fitted Doubly Robust AIPW Causal Inference + VanderWeele E-Values
2. Split & Mondrian Conformal Prediction for Distribution-Free Risk Triage
3. Actionable Clinical Counterfactual Recourse with Immutability & Manifold Constraints
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.base import clone
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn


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
        if self.lower_bounds_ is None or self.mad_weights_ is None or self.precision_matrix_ is None:
            raise RuntimeError("ClinicalCounterfactualGenerator must be fitted on X_train first.")

        self.model.eval()
        device = next(self.model.parameters()).device

        x0_np = np.asarray(x_factual, dtype=np.float32).ravel()
        x0_t = torch.tensor(x0_np, dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            orig_logit = self.model(x0_t).view(-1)
            orig_prob = float(torch.sigmoid(orig_logit).item())

        target_logit = float(np.log(self.target_probability / (1.0 - self.target_probability)))
        x_cf = x0_t.clone().detach().requires_grad_(True)

        w_t = torch.tensor(self.mad_weights_, dtype=torch.float32, device=device)
        prec_t = torch.tensor(self.precision_matrix_, dtype=torch.float32, device=device)
        lo_t = torch.tensor(self.lower_bounds_, dtype=torch.float32, device=device)
        hi_t = torch.tensor(self.upper_bounds_, dtype=torch.float32, device=device)

        optimizer = torch.optim.Adam([x_cf], lr=self.lr)

        best_cf = x0_t.clone().detach()
        best_cost = float("inf")
        found_valid = False

        for _ in range(self.max_iter):
            optimizer.zero_grad()
            logit = self.model(x_cf).view(-1)
            prob = torch.sigmoid(logit)

            # Hinge-logit validity loss with safety margin kappa
            validity_loss = torch.relu(logit - target_logit + self.margin_kappa).squeeze()

            diff = (x_cf - x0_t).squeeze(0)
            weighted_diff = w_t * diff
            l1_loss = torch.sum(torch.abs(weighted_diff))
            l2_loss = torch.sum(weighted_diff**2)
            mahal_loss = torch.matmul(diff.unsqueeze(0), torch.matmul(prec_t, diff.unsqueeze(1))).squeeze()

            reg_cost = (
                self.lambda_l1 * l1_loss
                + self.lambda_l2 * l2_loss
                + self.lambda_mahalanobis * mahal_loss
            )
            total_loss = validity_loss + reg_cost
            total_loss.backward()
            optimizer.step()

            # Hard projection step: enforce physiological bounds & strict immutability
            with torch.no_grad():
                x_cf.data = torch.max(torch.min(x_cf.data, hi_t.unsqueeze(0)), lo_t.unsqueeze(0))
                if self.immutable_indices_:
                    x_cf.data[:, self.immutable_indices_] = x0_t[:, self.immutable_indices_]

                curr_prob = float(torch.sigmoid(self.model(x_cf).view(-1)).item())
                curr_reg = float(reg_cost.item())
                if curr_prob <= self.target_probability and curr_reg < best_cost:
                    best_cost = curr_reg
                    best_cf = x_cf.clone().detach()
                    found_valid = True

        if not found_valid:
            best_cf = x_cf.clone().detach()

        with torch.no_grad():
            cf_logit = self.model(best_cf).view(-1)
            cf_prob = float(torch.sigmoid(cf_logit).item())

        cf_np = best_cf.cpu().numpy().ravel()
        delta = cf_np - x0_np
        mahal_dist = float(np.sqrt(max(0.0, float(delta @ self.precision_matrix_ @ delta))))
        active_changes = {
            self.feature_names[i]: float(delta[i])
            for i in self.actionable_indices_
            if abs(delta[i]) > 1e-3
        }

        return {
            "x_factual": x0_np,
            "x_counterfactual": cf_np,
            "delta": delta,
            "original_probability": orig_prob,
            "counterfactual_probability": cf_prob,
            "validity_achieved": bool(cf_prob <= self.target_probability),
            "sparsity_count": len(active_changes),
            "mahalanobis_distance": mahal_dist,
            "immutable_violation_max": float(
                np.max(np.abs(delta[self.immutable_indices_])) if self.immutable_indices_ else 0.0
            ),
            "recommended_interventions": active_changes,
        }

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

        validities = []
        sparsities = []
        mahal_dists = []
        risk_drops = []
        immut_violations = []

        for idx in high_risk_idx:
            res = self.generate(X_arr[idx])
            validities.append(float(res["validity_achieved"]))
            sparsities.append(float(res["sparsity_count"]))
            mahal_dists.append(float(res["mahalanobis_distance"]))
            risk_drops.append(float(res["original_probability"] - res["counterfactual_probability"]))
            immut_violations.append(float(res["immutable_violation_max"]))

        return {
            "n_evaluated": int(len(high_risk_idx)),
            "recourse_validity_rate": float(np.mean(validities)),
            "mean_features_modified": float(np.mean(sparsities)),
            "mean_mahalanobis_distance": float(np.mean(mahal_dists)),
            "mean_probability_reduction": float(np.mean(risk_drops)),
            "max_immutable_violation": float(np.max(immut_violations)),
        }
