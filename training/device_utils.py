"""CPU/GPU selection shared by the PyTorch and XGBoost training scripts.

``auto`` (the default everywhere) uses a GPU when one is usable and falls
back to the CPU otherwise. Asking for ``cuda`` explicitly still fails loudly
when no GPU is available, so a misconfigured GPU run is never silently slow.
"""

from __future__ import annotations

import shutil

import numpy as np

DEVICE_CHOICES = ("auto", "cpu", "cuda")


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return shutil.which("nvidia-smi") is not None
    return bool(torch.cuda.is_available())


def resolve_torch_device(requested_device: str = "auto"):
    """Return a ``torch.device`` for ``auto``, ``cpu`` or ``cuda``."""
    import torch

    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but no CUDA-compatible GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_xgb_device(requested_device: str = "auto") -> str:
    """Return ``"cuda"`` or ``"cpu"`` for XGBoost, probing the GPU build when needed."""
    from xgboost import XGBClassifier
    from xgboost.core import XGBoostError

    if requested_device == "cpu":
        return "cpu"
    if requested_device == "auto" and not _cuda_available():
        return "cpu"

    try:
        probe = XGBClassifier(
            n_estimators=1,
            max_depth=1,
            objective="binary:logistic",
            tree_method="hist",
            device="cuda",
            n_jobs=1,
        )
        probe.fit(
            np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32),
            np.array([0, 1, 0, 1], dtype=np.int32),
            verbose=False,
        )
        return "cuda"
    except (XGBoostError, ValueError) as error:
        if requested_device == "cuda":
            raise RuntimeError(
                "CUDA was requested for XGBoost, but GPU execution is unavailable. "
                "Ensure your XGBoost build supports CUDA."
            ) from error
        print("XGBoost GPU not available; falling back to CPU.")
        return "cpu"


def align_xgb_inference_device(model) -> None:
    """Move a fitted CUDA XGBoost booster to CPU for prediction on NumPy/pandas input."""
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    from xgboost.core import XGBoostError

    classifier = model.named_steps.get("classifier") if isinstance(model, Pipeline) else model
    if not isinstance(classifier, XGBClassifier):
        return
    if classifier.get_xgb_params().get("device") != "cuda":
        return
    try:
        classifier.get_booster().set_param({"device": "cpu"})
    except (XGBoostError, ValueError):
        return


def cv_n_jobs_for(model, requested_n_jobs: int) -> int:
    """Choose how many folds to run in parallel for ``model``.

    Small tabular models are fastest with one thread each and folds spread
    across cores. Two cases stay serial: GPU XGBoost (one GPU, one process)
    and estimators that already use several threads internally.
    """
    from sklearn.pipeline import Pipeline

    from evaluation.cv_utils import resolve_n_jobs

    classifier = model.named_steps.get("classifier") if isinstance(model, Pipeline) else model
    if classifier is None and isinstance(model, Pipeline):
        classifier = model.steps[-1][1]
    get_params = getattr(classifier, "get_xgb_params", None)
    if get_params is not None and get_params().get("device") == "cuda":
        return 1
    inner_jobs = getattr(classifier, "n_jobs", 1)
    if inner_jobs not in (None, 1):
        return 1
    return resolve_n_jobs(requested_n_jobs)
