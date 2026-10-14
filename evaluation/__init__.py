"""Evaluation, reporting, and thesis update tools."""

from .causal_analysis import (
    compare_observational_vs_causal,
    compute_iptw_ate,
    compute_nearest_neighbor_psm,
    compute_propensity_scores,
)
from .advanced_methodology import (
    ClinicalCounterfactualGenerator,
    SplitConformalClassifier,
    compute_doubly_robust_aipw,
    compute_e_value,
    evaluate_loco_conformal,
    plot_aipw_love_plot,
    plot_conformal_triage_curve,
    plot_counterfactual_case_study,
)
from .clinical_metrics import compute_clinical_metrics, find_optimal_threshold
from .fair_mitigation import find_equal_opportunity_thresholds
from .fairness_analysis import compute_subgroup_metrics
from .gradcam_analysis import GradCAM

__all__ = [
    "GradCAM",
    "ClinicalCounterfactualGenerator",
    "SplitConformalClassifier",
    "compute_doubly_robust_aipw",
    "compute_e_value",
    "evaluate_loco_conformal",
    "plot_aipw_love_plot",
    "plot_conformal_triage_curve",
    "plot_counterfactual_case_study",
    "find_equal_opportunity_thresholds",
    "compare_observational_vs_causal",
    "compute_iptw_ate",
    "compute_nearest_neighbor_psm",
    "compute_propensity_scores",
    "compute_clinical_metrics",
    "find_optimal_threshold",
    "compute_subgroup_metrics",
]

