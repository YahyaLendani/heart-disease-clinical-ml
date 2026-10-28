from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def _youden_j_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> float:
    """Compute Youden's J (Sensitivity + Specificity - 1) at a fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    return float(sensitivity + specificity - 1.0)


def compute_clinical_metrics(
    y_true,
    y_prob,
    y_val_true=None,
    y_val_prob=None,
) -> dict:
    """Compute PR-AUC, Brier Score, and optimal thresholds for a single model.

    If `y_val_true` and `y_val_prob` are provided, optimal decision thresholds
    are tuned strictly on the validation data and evaluated out-of-sample on
    (`y_true`, `y_prob`) to eliminate test-set threshold leakage.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

    if y_val_true is not None and y_val_prob is not None:
        y_val_true_arr = np.asarray(y_val_true, dtype=int)
        y_val_prob_arr = np.asarray(y_val_prob, dtype=float)
        threshold_f1, _ = find_optimal_threshold(y_val_true_arr, y_val_prob_arr, method="f1")
        threshold_youden, _ = find_optimal_threshold(y_val_true_arr, y_val_prob_arr, method="youden")
        f1_at_opt = float(f1_score(y_true, (y_prob >= threshold_f1).astype(int), zero_division=0))
        youden_j = _youden_j_at_threshold(y_true, y_prob, threshold_youden)
    else:
        threshold_f1, f1_at_opt = find_optimal_threshold(y_true, y_prob, method="f1")
        threshold_youden, youden_j = find_optimal_threshold(y_true, y_prob, method="youden")

    return {
        "pr_auc": float(pr_auc),
        "brier_score": float(brier),
        "optimal_threshold_f1": float(threshold_f1),
        "f1_at_optimal": float(f1_at_opt),
        "optimal_threshold_youden": float(threshold_youden),
        "youden_j": float(youden_j),
    }


def compute_bootstrap_ci(
    y_true,
    y_prob,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = roc_auc_score,
    n_bootstraps: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute stratified bootstrap point estimate and 95% confidence interval.

    Returns
    -------
    tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    point_est = float(metric_fn(y_true, y_prob))

    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return point_est, point_est, point_est

    rng = np.random.RandomState(seed)
    scores = []
    for _ in range(n_bootstraps):
        boot_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        boot_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        boot_idx = np.concatenate([boot_pos, boot_neg])
        try:
            val = float(metric_fn(y_true[boot_idx], y_prob[boot_idx]))
            scores.append(val)
        except Exception:
            continue

    if not scores:
        return point_est, point_est, point_est

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(scores, 100.0 * alpha))
    upper = float(np.percentile(scores, 100.0 * (1.0 - alpha)))
    return point_est, lower, upper


def compute_net_benefit(
    y_true,
    y_prob,
    thresholds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Compute Decision Curve Analysis (DCA) Net Benefit via 2D NumPy broadcasting.

    Net Benefit(pt) = TP/N - FP/N * (pt / (1 - pt))
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    N = len(y_true)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    else:
        thresholds = np.asarray(thresholds, dtype=float).ravel()

    if N == 0:
        zeros = np.zeros_like(thresholds)
        return {
            "thresholds": thresholds,
            "net_benefit_model": zeros,
            "net_benefit_all": zeros.copy(),
            "net_benefit_none": zeros.copy(),
        }

    # 2D broadcast: (N, 1) >= (1, T) -> (N, T) boolean matrix
    preds = y_prob[:, None] >= thresholds[None, :]
    y_pos = (y_true == 1)[:, None]

    tp = np.sum(preds & y_pos, axis=0, dtype=float)
    fp = np.sum(preds & (~y_pos), axis=0, dtype=float)

    w = thresholds / np.clip(1.0 - thresholds, 1e-12, None)
    prevalence = float(np.mean(y_true))

    net_benefit_model = (tp / N) - (fp / N) * w
    net_benefit_all = prevalence - (1.0 - prevalence) * w

    return {
        "thresholds": thresholds,
        "net_benefit_model": net_benefit_model,
        "net_benefit_all": net_benefit_all,
        "net_benefit_none": np.zeros_like(thresholds),
    }


def find_optimal_threshold(y_true, y_prob, method: str = "f1") -> tuple[float, float]:
    """Find the optimal decision threshold for binary classification (fully vectorized)."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_prob)

    if method == "f1":
        if len(thresholds_pr) == 0:
            return 0.5, 0.0
        f1_scores = np.zeros(len(thresholds_pr))
        denom = precisions[:-1] + recalls[:-1]
        np.divide(
            2 * precisions[:-1] * recalls[:-1],
            denom,
            out=f1_scores,
            where=denom > 0,
        )
        best_idx = int(np.argmax(f1_scores))
        return float(thresholds_pr[best_idx]), float(f1_scores[best_idx])

    if method == "youden":
        thresholds = np.unique(y_prob)
        if len(thresholds) == 0:
            return 0.5, 0.0
        if len(thresholds) > 500:
            thresholds = np.linspace(float(y_prob.min()), float(y_prob.max()), 500)

        # Vectorized Youden's J across all candidate thresholds: (N, T)
        preds = y_prob[:, None] >= thresholds[None, :]
        y_pos = (y_true == 1)[:, None]
        y_neg = ~y_pos

        tp = np.sum(preds & y_pos, axis=0, dtype=float)
        fn = np.sum((~preds) & y_pos, axis=0, dtype=float)
        tn = np.sum((~preds) & y_neg, axis=0, dtype=float)
        fp = np.sum(preds & y_neg, axis=0, dtype=float)

        sensitivity = tp / np.maximum(tp + fn, 1.0)
        specificity = tn / np.maximum(tn + fp, 1.0)
        j_scores = sensitivity + specificity - 1.0

        best_idx = int(np.argmax(j_scores))
        return float(thresholds[best_idx]), float(j_scores[best_idx])

    raise ValueError(f"Unsupported method: {method}. Use 'f1' or 'youden'.")


def plot_calibration_curves(
    y_true,
    model_probas_dict: dict[str, np.ndarray],
    dataset_name: str,
    output_dir: str | Path,
    n_bins: int = 10,
) -> Path:
    """Plot calibration curves (reliability diagrams) with adaptive quantile binning."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping calibration curve plot.")
        return Path(output_dir) / f"calibration_{dataset_name}.png"

    y_true = np.asarray(y_true, dtype=int)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n_pos = int(np.sum(y_true == 1))
    if len(y_true) < 100 or n_pos < 15:
        effective_bins = min(5, max(3, n_pos))
        strategy = "quantile"
    else:
        effective_bins = n_bins
        strategy = "uniform"

    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]
    ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfectly calibrated", alpha=0.7)

    for idx, (model_name, y_prob) in enumerate(model_probas_dict.items()):
        y_prob = np.asarray(y_prob, dtype=float)
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=effective_bins, strategy=strategy,
            )
        except Exception:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_prob, n_bins=effective_bins, strategy="uniform",
            )
        color = colors[idx % len(colors)]

        brier = brier_score_loss(y_true, y_prob)
        ax_cal.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            color=color,
            linewidth=1.8,
            markersize=6,
            label=f"{model_name} (Brier={brier:.4f})",
        )

        ax_hist.hist(y_prob, bins=25, alpha=0.4, color=color, label=model_name, range=(0, 1))

    ax_cal.set_ylabel("Fraction of Positives", fontsize=12)
    ax_cal.set_title(
        f"Calibration Curves ({strategy.capitalize()}, {effective_bins} bins) — {dataset_name.upper()}",
        fontsize=13,
    )
    ax_cal.legend(loc="lower right", fontsize=9)
    ax_cal.set_xlim([0, 1])
    ax_cal.set_ylim([0, 1])
    ax_cal.grid(alpha=0.3)

    ax_hist.set_xlabel("Predicted Probability", fontsize=12)
    ax_hist.set_ylabel("Count", fontsize=12)
    ax_hist.set_title("Distribution of Predicted Probabilities", fontsize=11)
    ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.grid(alpha=0.3)

    fig.tight_layout()
    save_path = output_path / f"calibration_{dataset_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved calibration curve: {save_path.as_posix()}")
    return save_path


def plot_precision_recall_curves(
    y_true,
    model_probas_dict: dict[str, np.ndarray],
    dataset_name: str,
    output_dir: str | Path,
) -> Path:
    """Plot precision-recall curves for multiple models."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PR curve plot.")
        return Path(output_dir) / f"pr_curve_{dataset_name}.png"

    y_true = np.asarray(y_true, dtype=int)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]

    prevalence = float(np.mean(y_true))
    ax.axhline(
        y=prevalence,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"Baseline (prevalence={prevalence:.3f})",
    )

    for idx, (model_name, y_prob) in enumerate(model_probas_dict.items()):
        y_prob = np.asarray(y_prob, dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        color = colors[idx % len(colors)]

        ax.plot(
            recall,
            precision,
            color=color,
            linewidth=1.8,
            label=f"{model_name} (PR-AUC={pr_auc:.4f})",
        )

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curves — {dataset_name.upper()} Dataset", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save_path = output_path / f"pr_curve_{dataset_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved precision-recall curve: {save_path.as_posix()}")
    return save_path


def plot_decision_curve(
    y_true,
    model_probas_dict: dict[str, np.ndarray],
    dataset_name: str,
    output_dir: str | Path,
    thresholds: np.ndarray | None = None,
) -> Path:
    """Plot Decision Curve Analysis (DCA) Net Benefit across threshold probabilities."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping DCA curve plot.")
        return Path(output_dir) / f"dca_curve_{dataset_name}.png"

    y_true = np.asarray(y_true, dtype=int)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]

    prevalence = float(np.mean(y_true))
    w = thresholds / (1.0 - thresholds)
    net_benefit_all = prevalence - (1.0 - prevalence) * w

    ax.plot(thresholds, net_benefit_all, "--", color="gray", label=f"Treat All (prev={prevalence:.2f})", linewidth=1.5)
    ax.axhline(0, color="black", linestyle=":", label="Treat None (NB=0)", linewidth=1.2)

    for idx, (model_name, y_prob) in enumerate(model_probas_dict.items()):
        res = compute_net_benefit(y_true, y_prob, thresholds=thresholds)
        color = colors[idx % len(colors)]
        ax.plot(thresholds, res["net_benefit_model"], color=color, label=model_name, linewidth=2.0)

    ax.set_xlim([0, 1])
    ax.set_ylim([-0.05, max(prevalence * 1.15, 0.25)])
    ax.set_xlabel("Decision Threshold Probability ($p_t$)", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title(f"Decision Curve Analysis (Net Benefit) — {dataset_name.upper()}", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    save_path = output_path / f"dca_curve_{dataset_name}.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    fig.savefig(output_path / "dca_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved DCA curve: {save_path.as_posix()}")
    return save_path


