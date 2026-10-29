from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

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


def _title(kind: str, dataset_name: str, cohort: str | None) -> str:
    return f"{kind} — {cohort}" if cohort else f"{kind} — {dataset_name}"


def plot_calibration_curves(
    y_true,
    model_probas_dict: dict[str, np.ndarray],
    dataset_name: str,
    output_dir: str | Path,
    n_bins: int = 10,
    cohort: str | None = None,
) -> Path:
    """Reliability diagram plus predicted-probability distributions on the test split."""
    try:
        import matplotlib.pyplot as plt

        from evaluation.plot_style import (
            apply_style,
            finish_axes,
            reference_color,
            series_color,
            subtitle,
        )
    except ImportError:
        print("matplotlib is not installed; skipping calibration curve plot.")
        return Path(output_dir) / f"calibration_{dataset_name}.png"

    apply_style()
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
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}, sharex=True,
    )
    ax_cal.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color=reference_color(), label="Perfect calibration")

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
        color = series_color(idx)
        brier = brier_score_loss(y_true, y_prob)
        ax_cal.plot(
            mean_predicted_value, fraction_of_positives, marker="o", color=color,
            markeredgecolor=fig.get_facecolor(), markeredgewidth=2, markersize=8,
            label=f"{model_name} (Brier {brier:.3f})",
        )
        # Outlined histograms stay readable where models overlap.
        ax_hist.hist(y_prob, bins=20, range=(0, 1), histtype="step", linewidth=1.8, color=color)

    ax_cal.set_ylabel("Observed fraction with disease")
    ax_cal.set_title(_title("Calibration", dataset_name, cohort))
    subtitle(ax_cal, f"Test split, {len(y_true)} patients · {effective_bins} {strategy} bins")
    ax_cal.legend(loc="upper left")
    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    finish_axes(ax_cal)

    ax_hist.set_xlabel("Predicted probability")
    ax_hist.set_ylabel("Patients")
    finish_axes(ax_hist)

    save_path = output_path / f"calibration_{dataset_name}.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved calibration curve: {save_path.as_posix()}")
    return save_path


def plot_precision_recall_curves(
    y_true,
    model_probas_dict: dict[str, np.ndarray],
    dataset_name: str,
    output_dir: str | Path,
    cohort: str | None = None,
) -> Path:
    """Precision-recall curves for several models on the test split."""
    try:
        import matplotlib.pyplot as plt

        from evaluation.plot_style import (
            apply_style,
            finish_axes,
            reference_color,
            series_color,
            subtitle,
        )
    except ImportError:
        print("matplotlib is not installed; skipping PR curve plot.")
        return Path(output_dir) / f"pr_curve_{dataset_name}.png"

    apply_style()
    y_true = np.asarray(y_true, dtype=int)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    prevalence = float(np.mean(y_true))
    ax.axhline(prevalence, color=reference_color(), linestyle="--", linewidth=1.2,
               label=f"No-skill (prevalence {prevalence:.2f})")

    for idx, (model_name, y_prob) in enumerate(model_probas_dict.items()):
        y_prob = np.asarray(y_prob, dtype=float)
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, color=series_color(idx), drawstyle="steps-post",
                label=f"{model_name} (PR-AUC {pr_auc:.3f})")

    ax.set_xlabel("Recall (sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(_title("Precision-recall", dataset_name, cohort))
    subtitle(ax, f"Test split, {len(y_true)} patients")
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    finish_axes(ax)

    save_path = output_path / f"pr_curve_{dataset_name}.png"
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved precision-recall curve: {save_path.as_posix()}")
    return save_path


def plot_decision_curve(
    y_true,
    model_probas_dict: dict[str, np.ndarray],
    dataset_name: str,
    output_dir: str | Path,
    thresholds: np.ndarray | None = None,
    cohort: str | None = None,
) -> Path:
    """Decision Curve Analysis: net benefit across threshold probabilities."""
    try:
        import matplotlib.pyplot as plt

        from evaluation.plot_style import (
            apply_style,
            finish_axes,
            reference_color,
            series_color,
            subtitle,
        )
    except ImportError:
        print("matplotlib is not installed; skipping DCA curve plot.")
        return Path(output_dir) / f"dca_curve_{dataset_name}.png"

    apply_style()
    y_true = np.asarray(y_true, dtype=int)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    fig, ax = plt.subplots(figsize=(8, 6))
    prevalence = float(np.mean(y_true))
    w = thresholds / (1.0 - thresholds)
    net_benefit_all = prevalence - (1.0 - prevalence) * w

    ax.plot(thresholds, net_benefit_all, linestyle="--", linewidth=1.2, color=reference_color(), label="Treat all")
    ax.plot(thresholds, np.zeros_like(thresholds), linestyle=":", linewidth=1.4, color=reference_color(), label="Treat none")
    for idx, (model_name, y_prob) in enumerate(model_probas_dict.items()):
        res = compute_net_benefit(y_true, y_prob, thresholds=thresholds)
        ax.plot(thresholds, res["net_benefit_model"], color=series_color(idx), label=model_name)

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, max(prevalence * 1.15, 0.25))
    ax.set_xlabel("Threshold probability $p_t$")
    ax.set_ylabel("Net benefit")
    ax.set_title(_title("Decision curve", dataset_name, cohort))
    subtitle(ax, f"Test split, {len(y_true)} patients · disease prevalence {prevalence:.2f}")
    ax.legend(loc="upper right")
    finish_axes(ax)

    save_path = output_path / f"dca_curve_{dataset_name}.png"
    fig.savefig(save_path)
    fig.savefig(output_path / "dca_curve.png")
    plt.close(fig)
    print(f"Saved DCA curve: {save_path.as_posix()}")
    return save_path
