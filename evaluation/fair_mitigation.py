"""Algorithmic Fairness Disparity Mitigation: Equal Opportunity Threshold Tuning.

Reference:
Barocas, S., Hardt, M., & Narayanan, A. (2020).
Fairness and Machine Learning: Limitations and Opportunities. MIT Press.
fairmlbook.org.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def find_equal_opportunity_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive_group: np.ndarray,
    target_tpr_tol: float = 0.05,
    n_threshold_steps: int = 100,
) -> dict[str, float | dict]:
    """Optimize group-specific classification thresholds to enforce Equal Opportunity.

    Finds (t_0, t_1) such that:
        |TPR(group=0, t_0) - TPR(group=1, t_1)| <= target_tpr_tol
    while maximizing the overall combined F1 score.

    Parameters
    ----------
    y_true : binary labels (0/1)
    y_prob : predicted probability scores
    sensitive_group : binary sensitive group indicator (e.g., 0=Female, 1=Male)
    target_tpr_tol : maximum allowable TPR difference between groups
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    sensitive_group = np.asarray(sensitive_group, dtype=int)

    mask_0 = sensitive_group == 0
    mask_1 = sensitive_group == 1

    y_true_0, y_prob_0 = y_true[mask_0], y_prob[mask_0]
    y_true_1, y_prob_1 = y_true[mask_1], y_prob[mask_1]

    thresholds = np.linspace(0.05, 0.95, n_threshold_steps)

    # Precompute TPR and predictions for all candidate thresholds per group
    tpr_0_list, tpr_1_list = [], []
    for t in thresholds:
        # Group 0
        pred_0 = (y_prob_0 >= t).astype(int)
        tn0, fp0, fn0, tp0 = confusion_matrix(y_true_0, pred_0, labels=[0, 1]).ravel()
        tpr_0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0.0
        tpr_0_list.append(tpr_0)

        # Group 1
        pred_1 = (y_prob_1 >= t).astype(int)
        tn1, fp1, fn1, tp1 = confusion_matrix(y_true_1, pred_1, labels=[0, 1]).ravel()
        tpr_1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0.0
        tpr_1_list.append(tpr_1)

    best_combined_f1 = -1.0
    best_t0, best_t1 = 0.5, 0.5
    best_tpr_diff = 1.0

    # Grid search over threshold pairs
    for i, t0 in enumerate(thresholds):
        tpr_0 = tpr_0_list[i]
        for j, t1 in enumerate(thresholds):
            tpr_1 = tpr_1_list[j]
            tpr_diff = abs(tpr_0 - tpr_1)

            if tpr_diff <= target_tpr_tol:
                # Compute joint predictions
                pred_joint = np.zeros_like(y_true)
                pred_joint[mask_0] = (y_prob_0 >= t0).astype(int)
                pred_joint[mask_1] = (y_prob_1 >= t1).astype(int)

                joint_f1 = f1_score(y_true, pred_joint, zero_division=0)
                if joint_f1 > best_combined_f1:
                    best_combined_f1 = joint_f1
                    best_t0 = t0
                    best_t1 = t1
                    best_tpr_diff = tpr_diff

    # If strict tolerance wasn't met, fall back to minimum disparity pair
    if best_combined_f1 < 0:
        min_disp = 1.0
        for i, t0 in enumerate(thresholds):
            for j, t1 in enumerate(thresholds):
                diff = abs(tpr_0_list[i] - tpr_1_list[j])
                if diff < min_disp:
                    min_disp = diff
                    best_t0, best_t1 = t0, t1
                    best_tpr_diff = diff
        pred_joint = np.zeros_like(y_true)
        pred_joint[mask_0] = (y_prob_0 >= best_t0).astype(int)
        pred_joint[mask_1] = (y_prob_1 >= best_t1).astype(int)
        best_combined_f1 = f1_score(y_true, pred_joint, zero_division=0)

    # Compute unmitigated baseline (single threshold = 0.5)
    unmitigated_pred = (y_prob >= 0.5).astype(int)
    unmit_tn0, unmit_fp0, unmit_fn0, unmit_tp0 = confusion_matrix(
        y_true_0, unmitigated_pred[mask_0], labels=[0, 1]
    ).ravel()
    unmit_tpr_0 = unmit_tp0 / (unmit_tp0 + unmit_fn0) if (unmit_tp0 + unmit_fn0) > 0 else 0.0

    unmit_tn1, unmit_fp1, unmit_fn1, unmit_tp1 = confusion_matrix(
        y_true_1, unmitigated_pred[mask_1], labels=[0, 1]
    ).ravel()
    unmit_tpr_1 = unmit_tp1 / (unmit_tp1 + unmit_fn1) if (unmit_tp1 + unmit_fn1) > 0 else 0.0

    unmit_tpr_diff = abs(unmit_tpr_0 - unmit_tpr_1)
    unmit_f1 = f1_score(y_true, unmitigated_pred, zero_division=0)

    return {
        "optimal_threshold_group_0": float(best_t0),
        "optimal_threshold_group_1": float(best_t1),
        "mitigated_tpr_diff": float(best_tpr_diff),
        "mitigated_joint_f1": float(best_combined_f1),
        "unmitigated_tpr_diff": float(unmit_tpr_diff),
        "unmitigated_joint_f1": float(unmit_f1),
        "disparity_reduction_pct": float(max(0.0, (unmit_tpr_diff - best_tpr_diff) / max(unmit_tpr_diff, 1e-6) * 100.0)),
    }
