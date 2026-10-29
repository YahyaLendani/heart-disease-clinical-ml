"""Shared helpers for cross-validation, scoring, parallelism and confidence intervals.

Every training script uses these so that metrics are computed the same way
everywhere and each model is asked for predictions only once per fold.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

BASE_METRICS = ("accuracy", "f1", "roc_auc", "pr_auc", "brier")


def resolve_n_jobs(n_jobs: int | None) -> int:
    """Turn a scikit-learn style ``n_jobs`` (-1 = all cores) into a positive worker count."""
    cores = os.cpu_count() or 1
    if n_jobs is None or n_jobs == 0:
        return 1
    if n_jobs < 0:
        return max(1, cores + 1 + n_jobs)
    return int(n_jobs)


def _positive_proba(estimator, X) -> np.ndarray:
    return np.asarray(estimator.predict_proba(X))[:, 1]


def binary_metrics(
    y_true,
    y_prob,
    threshold: float = 0.5,
    extended: bool = False,
) -> dict[str, float]:
    """Accuracy, F1, ROC-AUC, PR-AUC and Brier score from probabilities."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) > 1
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if two_classes else float("nan"),
        "brier": float(brier_score_loss(y_true, y_prob)),
    }
    if extended:
        metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    return metrics


def multi_metric_scorer(
    threshold: float = 0.5,
    extended: bool = False,
    before_predict: Callable | None = None,
) -> Callable:
    """Build one ``cross_validate`` scorer that predicts once and returns every metric.

    scikit-learn calls a scorer dict entry once per metric, so five custom
    scorers meant five ``predict_proba`` calls per fold. This returns a
    single callable that yields a dict, which ``cross_validate`` accepts.
    """

    def _score(estimator, X, y) -> dict[str, float]:
        if before_predict is not None:
            before_predict(estimator)
        return binary_metrics(y, _positive_proba(estimator, X), threshold=threshold, extended=extended)

    return _score


def summarize_cv_scores(scores: dict, metrics: tuple[str, ...] = BASE_METRICS) -> dict[str, float]:
    """Mean and standard deviation of ``cross_validate`` test scores."""
    summary: dict[str, float] = {}
    for name in metrics:
        values = np.asarray(scores[f"test_{name}"], dtype=float)
        summary[f"cv_{name}_mean"] = float(np.nanmean(values))
        summary[f"cv_{name}_std"] = float(np.nanstd(values))
    return summary


# ---------------------------------------------------------------------------
# Fast metric kernels for bootstrapping (match scikit-learn exactly)
# ---------------------------------------------------------------------------


def _fast_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from scipy.stats import rankdata

    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(y_prob)
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _fast_average_precision(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.sum() == 0:
        return float("nan")
    order = np.argsort(-y_prob, kind="mergesort")
    scores = y_prob[order]
    labels = y_true[order]
    distinct = np.where(np.diff(scores))[0]
    threshold_idx = np.r_[distinct, len(labels) - 1]
    tps = np.cumsum(labels)[threshold_idx]
    fps = 1 + threshold_idx - tps
    precision = tps / (tps + fps)
    recall = tps / tps[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _fast_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = y_prob >= threshold
    tp = float(np.sum(y_pred & (y_true == 1)))
    fp = float(np.sum(y_pred & (y_true == 0)))
    fn = float(np.sum(~y_pred & (y_true == 1)))
    denom = 2 * tp + fp + fn
    return {
        "accuracy": float(np.mean(y_pred == (y_true == 1))),
        "f1": float(2 * tp / denom) if denom > 0 else 0.0,
        "roc_auc": _fast_roc_auc(y_true, y_prob),
        "pr_auc": _fast_average_precision(y_true, y_prob),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
    }


def bootstrap_metric_cis(
    y_true,
    y_prob,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    threshold: float = 0.5,
) -> dict[str, tuple[float, float]]:
    """Stratified bootstrap percentile intervals for the five base metrics.

    Positives and negatives are resampled separately so every replicate keeps
    the observed class balance (and AUC is always defined).
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0 or n_bootstraps <= 0:
        point = binary_metrics(y_true, y_prob, threshold=threshold)
        return {name: (point[name], point[name]) for name in BASE_METRICS}

    rng = np.random.default_rng(seed)
    samples = {name: np.empty(n_bootstraps) for name in BASE_METRICS}
    for b in range(n_bootstraps):
        idx = np.concatenate([
            rng.choice(pos_idx, size=len(pos_idx), replace=True),
            rng.choice(neg_idx, size=len(neg_idx), replace=True),
        ])
        values = _fast_metrics(y_true[idx], y_prob[idx], threshold)
        for name in BASE_METRICS:
            samples[name][b] = values[name]

    alpha = (1.0 - ci) / 2.0
    return {
        name: (
            float(np.nanpercentile(values, 100.0 * alpha)),
            float(np.nanpercentile(values, 100.0 * (1.0 - alpha))),
        )
        for name, values in samples.items()
    }


def add_bootstrap_cis(
    row: dict,
    y_true,
    y_prob,
    prefix: str = "test_",
    seed: int = 42,
    n_bootstraps: int = 1000,
    name_map: dict[str, str] | None = None,
) -> dict:
    """Add ``<prefix><metric>_ci_low`` / ``_ci_high`` columns to a result row in place."""
    name_map = name_map or {}
    cis = bootstrap_metric_cis(y_true, y_prob, n_bootstraps=n_bootstraps, seed=seed)
    for metric, (low, high) in cis.items():
        key = name_map.get(metric, f"{prefix}{metric}")
        row[f"{key}_ci_low"] = low
        row[f"{key}_ci_high"] = high
    return row


def format_ci(value: float, low: float, high: float, digits: int = 3) -> str:
    """Render ``0.912 [0.851, 0.960]``."""
    return f"{value:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def to_markdown_table(frame, float_digits: int = 4) -> str:
    """Render a DataFrame as a GitHub Markdown table without optional dependencies."""
    headers = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        cells = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                cells.append(f"{value:.{float_digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
