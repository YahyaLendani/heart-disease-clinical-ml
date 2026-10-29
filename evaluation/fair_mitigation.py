"""Algorithmic Fairness Disparity Mitigation: Equal Opportunity Threshold Tuning.

Reference:
Barocas, S., Hardt, M., & Narayanan, A. (2020).
Fairness and Machine Learning: Limitations and Opportunities. MIT Press.
fairmlbook.org.
"""

from __future__ import annotations

import numpy as np

OBJECTIVES = ("youden", "f1")


def _counts_by_threshold(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> dict[str, np.ndarray]:
    """TP/FP/FN/TN for every candidate threshold at once (vectorised)."""
    preds = y_prob[:, None] >= thresholds[None, :]
    pos = (y_true == 1)[:, None]
    neg = ~pos
    return {
        "tp": np.sum(preds & pos, axis=0).astype(float),
        "fp": np.sum(preds & neg, axis=0).astype(float),
        "fn": np.sum(~preds & pos, axis=0).astype(float),
        "tn": np.sum(~preds & neg, axis=0).astype(float),
    }


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return np.divide(num, den, out=np.zeros_like(num, dtype=float), where=den > 0)


def _scores(tp, fp, fn, tn) -> dict[str, np.ndarray]:
    tpr = _safe_div(tp, tp + fn)
    tnr = _safe_div(tn, tn + fp)
    return {
        "youden": tpr + tnr - 1.0,
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "tnr": tnr,
    }


def find_equal_opportunity_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    sensitive_group: np.ndarray,
    target_tpr_tol: float = 0.05,
    n_threshold_steps: int = 100,
    objective: str = "youden",
) -> dict[str, float | dict]:
    """Choose group-specific thresholds (t_0, t_1) under an Equal Opportunity constraint.

    Among pairs with ``|TPR_0(t_0) - TPR_1(t_1)| <= target_tpr_tol``, pick the
    pair that maximises the pooled ``objective``:

    * ``"youden"`` (default): sensitivity + specificity - 1. Flagging every
      patient scores 0, so parity cannot be bought by abandoning specificity.
    * ``"f1"``: the original objective. On small or high-prevalence samples it
      tends to pick thresholds near 0 (flag almost everyone).

    If no pair meets the tolerance, the pair with the smallest TPR gap is used.
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    sensitive_group = np.asarray(sensitive_group, dtype=int)

    mask_0 = sensitive_group == 0
    mask_1 = sensitive_group == 1
    thresholds = np.linspace(0.05, 0.95, n_threshold_steps)

    c0 = _counts_by_threshold(y_true[mask_0], y_prob[mask_0], thresholds)
    c1 = _counts_by_threshold(y_true[mask_1], y_prob[mask_1], thresholds)
    tpr_0 = _safe_div(c0["tp"], c0["tp"] + c0["fn"])
    tpr_1 = _safe_div(c1["tp"], c1["tp"] + c1["fn"])

    # Pooled confusion counts for every (t_0, t_1) pair: shape (T, T).
    joint = {k: c0[k][:, None] + c1[k][None, :] for k in c0}
    scores = _scores(joint["tp"], joint["fp"], joint["fn"], joint["tn"])
    tpr_gap = np.abs(tpr_0[:, None] - tpr_1[None, :])

    feasible = tpr_gap <= target_tpr_tol
    if feasible.any():
        masked = np.where(feasible, scores[objective], -np.inf)
        i, j = np.unravel_index(int(np.argmax(masked)), masked.shape)
    else:
        i, j = np.unravel_index(int(np.argmin(tpr_gap)), tpr_gap.shape)

    best_t0, best_t1 = float(thresholds[i]), float(thresholds[j])
    best_tpr_diff = float(tpr_gap[i, j])

    # Unmitigated baseline: one shared threshold of 0.5.
    base = _counts_by_threshold(y_true, y_prob, np.array([0.5]))
    b0 = _counts_by_threshold(y_true[mask_0], y_prob[mask_0], np.array([0.5]))
    b1 = _counts_by_threshold(y_true[mask_1], y_prob[mask_1], np.array([0.5]))
    unmit_tpr_diff = float(abs(
        _safe_div(b0["tp"], b0["tp"] + b0["fn"])[0] - _safe_div(b1["tp"], b1["tp"] + b1["fn"])[0]
    ))
    base_scores = _scores(base["tp"], base["fp"], base["fn"], base["tn"])

    return {
        "objective": objective,
        "optimal_threshold_group_0": best_t0,
        "optimal_threshold_group_1": best_t1,
        "mitigated_tpr_diff": best_tpr_diff,
        "mitigated_joint_f1": float(scores["f1"][i, j]),
        "mitigated_joint_youden": float(scores["youden"][i, j]),
        "mitigated_specificity": float(scores["tnr"][i, j]),
        "unmitigated_tpr_diff": unmit_tpr_diff,
        "unmitigated_joint_f1": float(base_scores["f1"][0]),
        "unmitigated_joint_youden": float(base_scores["youden"][0]),
        "unmitigated_specificity": float(base_scores["tnr"][0]),
        "constraint_met": bool(feasible.any()),
        "disparity_reduction_pct": float(
            max(0.0, (unmit_tpr_diff - best_tpr_diff) / max(unmit_tpr_diff, 1e-6) * 100.0)
        ),
    }
