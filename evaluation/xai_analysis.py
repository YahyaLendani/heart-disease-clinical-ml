"""Explainable AI (XAI) analysis using SHAP.

Provides functions to compute SHAP values and generate interpretability
plots for tree-based models (TreeExplainer) and PyTorch MLP models
(KernelExplainer).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None  # type: ignore[assignment]

try:
    import shap
except ImportError:
    shap = None  # type: ignore[assignment]


MODEL_DISPLAY_NAMES = {
    "mlp_adam": "HeartMLP (Adam)",
    "mlp_sgd": "HeartMLP (SGD)",
    "mlp": "MLP",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
}
DATASET_DISPLAY_NAMES = {"heart": "heart disease", "mimic": "MIMIC-IV demo"}


def _display_name(model_name: str, dataset_name: str) -> str:
    """Readable "Model (dataset)" label for figure titles."""
    model = MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title())
    dataset = DATASET_DISPLAY_NAMES.get(dataset_name, dataset_name)
    return f"{model} ({dataset})"


def _ensure_deps() -> None:
    if shap is None:
        raise ImportError("shap is required. Install with: pip install shap")
    if plt is None:
        raise ImportError("matplotlib is required. Install with: pip install matplotlib")


def _get_feature_names(preprocessor: Any) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        return [f"feature_{i}" for i in range(preprocessor.n_features_in_)]


def _top_features(shap_values: np.ndarray, feature_names: list[str], top_n: int = 5) -> list[dict]:
    """Return top N features ranked by mean |SHAP value|."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    indices = np.argsort(mean_abs)[::-1][:top_n]
    return [
        {
            "rank": rank + 1,
            "feature": feature_names[idx],
            "mean_abs_shap": float(mean_abs[idx]),
        }
        for rank, idx in enumerate(indices)
    ]


def _save_summary_plot(
    shap_values: np.ndarray,
    X_display: np.ndarray,
    feature_names: list[str],
    output_path: Path,
    title: str,
) -> None:
    """Generate and save a SHAP beeswarm summary plot."""
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_display,
        feature_names=feature_names,
        show=False,
        plot_size=None,
    )
    current_fig = plt.gcf()
    current_fig.suptitle(title, fontsize=14, y=1.02)
    current_fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved SHAP summary plot: {output_path}")


def _save_waterfall_plot(
    shap_explanation: Any,
    idx: int,
    output_path: Path,
    title: str,
) -> None:
    """Generate and save a SHAP waterfall plot for a single observation."""
    plt.figure(figsize=(10, 7))
    shap.plots.waterfall(shap_explanation[idx], show=False)
    current_fig = plt.gcf()
    current_fig.suptitle(title, fontsize=13, y=1.02)
    current_fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    current_fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"Saved SHAP waterfall plot: {output_path}")


def _find_risk_indices(shap_values: np.ndarray) -> tuple[int, int]:
    """Return indices of highest-risk and lowest-risk patients by sum of SHAP values."""
    sums = shap_values.sum(axis=1)
    high_risk_idx = int(np.argmax(sums))
    low_risk_idx = int(np.argmin(sums))
    return high_risk_idx, low_risk_idx


# ---------------------------------------------------------------------------
# Public API: Tree-based models
# ---------------------------------------------------------------------------


def generate_shap_tree(
    model_pipeline: Any,
    X_train: pd.DataFrame | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    feature_names: list[str] | None,
    dataset_name: str,
    model_name: str,
    output_dir: Path,
) -> dict:
    """Compute SHAP values for a tree-based sklearn Pipeline and save plots.

    Parameters
    ----------
    model_pipeline : sklearn Pipeline with ("preprocess", ...) and ("classifier", ...) steps.
    X_train, X_test : Raw (pre-transform) DataFrames or already-transformed arrays.
    feature_names : Optional list of feature names after preprocessing.
    dataset_name : Label used in filenames (e.g., "heart", "mimic").
    model_name : Label for the model (e.g., "xgboost", "random_forest").
    output_dir : Directory for output figures.

    Returns
    -------
    dict with keys: dataset, model, top_features (list[dict]).
    """
    _ensure_deps()

    from sklearn.pipeline import Pipeline

    # Extract preprocessor + classifier from pipeline
    if isinstance(model_pipeline, Pipeline):
        preprocessor = model_pipeline.named_steps.get("preprocess")
        classifier = model_pipeline.named_steps.get("classifier")
        # TreeExplainer only needs the test rows (X_train is part of the public signature).
        if preprocessor is not None:
            if isinstance(X_test, pd.DataFrame):
                X_test_transformed = preprocessor.transform(X_test)
            else:
                X_test_transformed = X_test
            if feature_names is None:
                feature_names = _get_feature_names(preprocessor)
        else:
            X_test_transformed = np.asarray(X_test)
            classifier = model_pipeline
    else:
        classifier = model_pipeline
        X_test_transformed = np.asarray(X_test)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_test_transformed.shape[1])]

    # Compute SHAP values
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_transformed)

    # Normalize SHAP values to 2D (n_samples, n_features) for the positive class.
    # TreeExplainer returns different formats depending on the model:
    #   - list of 2D arrays [class_0, class_1] (e.g., some XGBoost configs)
    #   - 3D array (n_samples, n_features, n_classes) (e.g., RandomForest)
    #   - 2D array (n_samples, n_features) (e.g., XGBoost binary)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # positive class slice

    # Resolve base value for the positive class
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        base_value = float(np.asarray(ev)[1])
    else:
        base_value = float(ev)

    # Build Explanation object for waterfall plots
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(shap_values.shape[0], base_value),
        data=X_test_transformed,
        feature_names=feature_names,
    )

    # Filenames
    prefix = f"{dataset_name}_{model_name}"
    figures_dir = output_dir / "figures"

    _save_summary_plot(
        shap_values,
        X_test_transformed,
        feature_names,
        figures_dir / f"{prefix}_shap_summary.png",
        title=f"SHAP Summary — {_display_name(model_name, dataset_name)}",
    )

    high_idx, low_idx = _find_risk_indices(shap_values)

    _save_waterfall_plot(
        explanation,
        high_idx,
        figures_dir / f"{prefix}_shap_waterfall_high_risk.png",
        title=f"Highest-Risk Patient — {_display_name(model_name, dataset_name)}",
    )
    _save_waterfall_plot(
        explanation,
        low_idx,
        figures_dir / f"{prefix}_shap_waterfall_low_risk.png",
        title=f"Lowest-Risk Patient — {_display_name(model_name, dataset_name)}",
    )

    top = _top_features(shap_values, feature_names, top_n=5)
    return {"dataset": dataset_name, "model": model_name, "top_features": top}


# ---------------------------------------------------------------------------
# Public API: PyTorch MLP (KernelExplainer)
# ---------------------------------------------------------------------------


def generate_shap_mlp(
    model: Any,
    X_train_np: np.ndarray,
    X_test_np: np.ndarray,
    feature_names: list[str] | None,
    dataset_name: str,
    model_name: str,
    output_dir: Path,
    n_background: int = 50,
) -> dict:
    """Compute SHAP values for a PyTorch MLP using KernelExplainer.

    Parameters
    ----------
    model : A PyTorch nn.Module in eval mode.
    X_train_np : Preprocessed training data (NumPy).
    X_test_np : Preprocessed test data (NumPy).
    feature_names : Feature names after preprocessing.
    dataset_name : Label for filenames.
    model_name : Label for the model (e.g., "mlp_adam").
    output_dir : Directory for output figures.
    n_background : Number of k-means background samples.

    Returns
    -------
    dict with keys: dataset, model, top_features (list[dict]).
    """
    _ensure_deps()
    import torch

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_test_np.shape[1])]

    # Wrap PyTorch model as a callable returning probabilities
    device = next(model.parameters()).device
    model.eval()

    def predict_fn(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32).to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs

    # Use k-means summary of training data as background
    background = shap.kmeans(X_train_np, min(n_background, len(X_train_np)))

    print(f"Computing SHAP values for {model_name} on {dataset_name} "
          f"({X_test_np.shape[0]} test samples, {n_background} background samples)...")
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(X_test_np, silent=True)
    shap_values = np.asarray(shap_values)

    # Build Explanation object for waterfall plots
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.full(shap_values.shape[0], explainer.expected_value),
        data=X_test_np,
        feature_names=feature_names,
    )

    # Filenames
    prefix = f"{dataset_name}_{model_name}"
    figures_dir = output_dir / "figures"

    _save_summary_plot(
        shap_values,
        X_test_np,
        feature_names,
        figures_dir / f"{prefix}_shap_summary.png",
        title=f"SHAP Summary — {_display_name(model_name, dataset_name)}",
    )

    high_idx, low_idx = _find_risk_indices(shap_values)

    _save_waterfall_plot(
        explanation,
        high_idx,
        figures_dir / f"{prefix}_shap_waterfall_high_risk.png",
        title=f"Highest-Risk Patient — {_display_name(model_name, dataset_name)}",
    )
    _save_waterfall_plot(
        explanation,
        low_idx,
        figures_dir / f"{prefix}_shap_waterfall_low_risk.png",
        title=f"Lowest-Risk Patient — {_display_name(model_name, dataset_name)}",
    )

    top = _top_features(shap_values, feature_names, top_n=5)
    return {"dataset": dataset_name, "model": model_name, "top_features": top}


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------


def save_xai_summary_csv(results: list[dict], output_dir: Path) -> Path:
    """Save top-features summary to CSV for thesis auto-update.

    Parameters
    ----------
    results : list of dicts returned by generate_shap_tree / generate_shap_mlp.
    output_dir : Directory to write the CSV.

    Returns
    -------
    Path to the saved CSV file.
    """
    rows: list[dict] = []
    for result in results:
        for feat in result["top_features"]:
            rows.append(
                {
                    "dataset": result["dataset"],
                    "model": result["model"],
                    "rank": feat["rank"],
                    "feature": feat["feature"],
                    "mean_abs_shap": feat["mean_abs_shap"],
                }
            )
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    csv_path = figures_dir / "shap_top_features.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved SHAP top features summary: {csv_path}")
    return csv_path
