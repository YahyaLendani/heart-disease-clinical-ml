"""Evaluation, reporting, and thesis update tools.

Names are imported lazily (PEP 562) so that importing one light module, such
as ``evaluation.clinical_metrics``, does not pull in PyTorch, SHAP or XGBoost.
This keeps script start-up and parallel worker start-up fast.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "GradCAM": ".gradcam_analysis",
    "ClinicalCounterfactualGenerator": ".advanced_methodology",
    "SplitConformalClassifier": ".advanced_methodology",
    "compute_doubly_robust_aipw": ".advanced_methodology",
    "compute_e_value": ".advanced_methodology",
    "evaluate_loco_conformal": ".advanced_methodology",
    "plot_aipw_love_plot": ".advanced_methodology",
    "plot_conformal_triage_curve": ".advanced_methodology",
    "plot_counterfactual_case_study": ".advanced_methodology",
    "find_equal_opportunity_thresholds": ".fair_mitigation",
    "compare_observational_vs_causal": ".causal_analysis",
    "compute_iptw_ate": ".causal_analysis",
    "compute_nearest_neighbor_psm": ".causal_analysis",
    "compute_propensity_scores": ".causal_analysis",
    "compute_clinical_metrics": ".clinical_metrics",
    "find_optimal_threshold": ".clinical_metrics",
    "compute_subgroup_metrics": ".fairness_analysis",
    "bootstrap_metric_cis": ".cv_utils",
    "multi_metric_scorer": ".cv_utils",
    "resolve_n_jobs": ".cv_utils",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
