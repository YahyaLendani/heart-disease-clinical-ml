"""Clinical evaluation metrics: PR-AUC, Brier Score, calibration curves, threshold tuning.

This module provides reusable functions for clinically relevant model evaluation,
designed for imbalanced binary classification tasks such as hospital mortality prediction.
"""

from pathlib import Path

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
)


def compute_clinical_metrics(y_true, y_prob) -> dict:
    """Compute PR-AUC, Brier Score, and optimal thresholds for a single model.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    y_prob : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict
        Keys: pr_auc, brier_score, optimal_threshold_f1, f1_at_optimal,
        optimal_threshold_youden, youden_j.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    pr_auc = average_precision_score(y_true, y_prob)
    brier = brier_score_loss(y_true, y_prob)

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


def find_optimal_threshold(y_true, y_prob, method: str = "f1") -> tuple[float, float]:
    """Find the optimal decision threshold for binary classification.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0/1).
    y_prob : array-like
        Predicted probabilities for the positive class.
    method : str
        'f1' — maximizes F1-score.
        'youden' — maximizes Youden's J statistic (sensitivity + specificity - 1).

    Returns
    -------
    tuple[float, float]
        (best_threshold, best_score)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    # Generate candidate thresholds from the precision-recall curve
    # (which already enumerates all unique predicted probability values)
    precisions, recalls, thresholds_pr = precision_recall_curve(y_true, y_prob)

    if method == "f1":
        # Compute F1 at each threshold from the PR curve
        # precisions and recalls have one extra element, thresholds has len-1
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
        # Youden's J = sensitivity + specificity - 1 = TPR - FPR
        # Scan a grid of thresholds derived from predicted probabilities
        thresholds = np.unique(y_prob)
        if len(thresholds) > 500:
            thresholds = np.linspace(y_prob.min(), y_prob.max(), 500)

        best_j = -1.0
        best_t = 0.5
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            tn = int(np.sum((y_pred == 0) & (y_true == 0)))
            fp = int(np.sum((y_pred == 1) & (y_true == 0)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            sensitivity = tp / max(tp + fn, 1)
            specificity = tn / max(tn + fp, 1)
            j = sensitivity + specificity - 1
            if j > best_j:
                best_j = j
                best_t = t
        return float(best_t), float(best_j)

    raise ValueError(f"Unsupported method: {method}. Use 'f1' or 'youden'.")


def plot_calibration_curves(
    y_true,
    model_probas_dict: dict[str, np.ndarray],
    dataset_name: str,
    output_dir: str | Path,
    n_bins: int = 10,
) -> Path:
    """Plot calibration curves (reliability diagrams) for multiple models.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    model_probas_dict : dict
        Mapping of model name → predicted probability array.
    dataset_name : str
        Name for the output file (e.g., 'mimic', 'heart').
    output_dir : str or Path
        Directory to save the figure.
    n_bins : int
        Number of bins for the calibration curve.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping calibration curve plot.")
        return Path(output_dir) / f"calibration_{dataset_name}.png"

    y_true = np.asarray(y_true, dtype=int)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, (ax_cal, ax_hist) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
    )

    # Color palette for models
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]

    # Diagonal — perfect calibration
    ax_cal.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Perfectly calibrated", alpha=0.7)

    for idx, (model_name, y_prob) in enumerate(model_probas_dict.items()):
        y_prob = np.asarray(y_prob, dtype=float)
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="uniform",
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

        ax_hist.hist(y_prob, bins=30, alpha=0.4, color=color, label=model_name, range=(0, 1))

    ax_cal.set_ylabel("Fraction of Positives", fontsize=12)
    ax_cal.set_title(f"Calibration Curves — {dataset_name.upper()} Dataset", fontsize=14)
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
    print(f"Saved calibration curve: {save_path}")
    return save_path


def plot_precision_recall_curves(
    y_true,
    model_probas_dict: dict[str, np.ndarray],
    dataset_name: str,
    output_dir: str | Path,
) -> Path:
    """Plot precision-recall curves for multiple models.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    model_probas_dict : dict
        Mapping of model name → predicted probability array.
    dataset_name : str
        Name for the output file (e.g., 'mimic', 'heart').
    output_dir : str or Path
        Directory to save the figure.

    Returns
    -------
    Path
        Path to the saved figure.
    """
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

    # Baseline: prevalence line
    prevalence = float(np.mean(y_true))
    ax.axhline(y=prevalence, color="gray", linestyle="--", linewidth=1.2, label=f"Baseline (prevalence={prevalence:.3f})")

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
    print(f"Saved precision-recall curve: {save_path}")
    return save_path
