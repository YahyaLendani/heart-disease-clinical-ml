"""Regression tests for the evaluation-protocol fixes and speed work.

Covers: parallelism settings, the single-pass CV scorer, bootstrap metric
kernels, lazy package imports, batched counterfactual search, tensor
mini-batching, data-path resolution, the MIMIC leakage check, the CLI, and
short end-to-end runs of the training scripts.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from evaluation.cv_utils import (  # noqa: E402
    _fast_average_precision,
    _fast_roc_auc,
    binary_metrics,
    bootstrap_metric_cis,
    multi_metric_scorer,
    resolve_n_jobs,
    summarize_cv_scores,
)

# ---------------------------------------------------------------------------
# Parallelism settings
# ---------------------------------------------------------------------------


def test_resolve_n_jobs_honours_explicit_values():
    cores = os.cpu_count() or 1
    assert resolve_n_jobs(-1) == cores
    assert resolve_n_jobs(1) == 1
    assert resolve_n_jobs(3) == 3
    assert resolve_n_jobs(None) == 1
    assert resolve_n_jobs(-2) == max(1, cores - 1)


def test_cross_center_models_use_requested_threads():
    """Old code parsed `cpu_count() or 1 if ... else n` as `cpu_count() or (...)`."""
    from sklearn.compose import ColumnTransformer

    from experiments.cross_center_generalization import get_models

    rf = get_models(seed=0, n_jobs=2)["Random Forest"](ColumnTransformer([]))
    assert rf.named_steps["clf"].n_jobs == 2


def test_cv_n_jobs_runs_folds_in_parallel_for_single_thread_models():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline

    from training.device_utils import cv_n_jobs_for

    single = Pipeline([("classifier", RandomForestClassifier(n_jobs=1))])
    multi = Pipeline([("classifier", RandomForestClassifier(n_jobs=-1))])
    assert cv_n_jobs_for(single, 2) == 2
    assert cv_n_jobs_for(multi, 2) == 1  # avoid nested oversubscription


# ---------------------------------------------------------------------------
# Scoring and confidence intervals
# ---------------------------------------------------------------------------


class _CountingLogReg(LogisticRegression):
    calls = 0

    def predict_proba(self, X):
        type(self).calls += 1
        return super().predict_proba(X)

    def predict(self, X):  # pragma: no cover - must never be needed
        raise AssertionError("scorer should derive labels from predict_proba")


def test_multi_metric_scorer_predicts_once_per_fold():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] + rng.normal(scale=0.5, size=120) > 0).astype(int)
    _CountingLogReg.calls = 0
    scores = cross_validate(
        _CountingLogReg(), X, y, cv=StratifiedKFold(4, shuffle=True, random_state=0),
        scoring=multi_metric_scorer(), n_jobs=1,
    )
    assert _CountingLogReg.calls == 4
    summary = summarize_cv_scores(scores)
    for metric in ("accuracy", "f1", "roc_auc", "pr_auc", "brier"):
        assert f"cv_{metric}_mean" in summary and f"cv_{metric}_std" in summary


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_metric_kernels_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=80)
    # Rounded scores create ties, the tricky case for both kernels.
    p = np.round(rng.random(80), 1)
    assert _fast_roc_auc(y, p) == pytest.approx(roc_auc_score(y, p), abs=1e-12)
    assert _fast_average_precision(y, p) == pytest.approx(average_precision_score(y, p), abs=1e-12)


def test_bootstrap_cis_bracket_point_estimates():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, size=100)
    p = np.clip(0.6 * y + rng.normal(0.2, 0.2, size=100), 0, 1)
    point = binary_metrics(y, p)
    cis = bootstrap_metric_cis(y, p, n_bootstraps=300, seed=0)
    for metric, (low, high) in cis.items():
        assert low <= high
        assert low - 1e-9 <= point[metric] <= high + 1e-9, metric
    assert cis == bootstrap_metric_cis(y, p, n_bootstraps=300, seed=0)  # deterministic


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "statement",
    [
        "import evaluation.clinical_metrics",
        "import evaluation.cv_utils",
        "from models import add_cardiovascular_features",
        "import training.device_utils",
    ],
)
def test_light_modules_do_not_import_torch(statement):
    code = f"import sys; {statement}; print('torch' in sys.modules)"
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "False"


def test_lazy_package_exports_resolve():
    import evaluation
    import models
    import training

    assert callable(evaluation.compute_clinical_metrics)
    assert callable(models.HeartMLP)
    assert callable(training.DeepEnsemble)
    with pytest.raises(AttributeError):
        evaluation.not_a_real_name  # noqa: B018


# ---------------------------------------------------------------------------
# Batched counterfactuals
# ---------------------------------------------------------------------------


def test_batched_counterfactuals_match_single_patient_runs():
    from evaluation.advanced_methodology import ClinicalCounterfactualGenerator
    from models import HeartMLP

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(150, 6)).astype(np.float32)
    model = HeartMLP(input_dim=6, hidden_dims=(16, 8), dropout=0.0).eval()
    gen = ClinicalCounterfactualGenerator(
        model, [f"f{i}" for i in range(5)] + ["age"], max_iter=60, target_probability=0.3
    ).fit(X_train)

    X = X_train[:6]
    batch = gen.generate_batch(X)
    singles = [gen.generate(x) for x in X]
    assert len(batch) == len(singles) == 6
    for b, s in zip(batch, singles, strict=True):
        np.testing.assert_allclose(b["x_counterfactual"], s["x_counterfactual"], atol=1e-4)
        assert b["validity_achieved"] == s["validity_achieved"]
        assert b["immutable_violation_max"] == 0.0  # "age" never changes


# ---------------------------------------------------------------------------
# Tensor mini-batching
# ---------------------------------------------------------------------------


def test_iterate_minibatches_covers_rows_and_skips_singletons():
    from training.torch_utils import iterate_minibatches

    X = torch.arange(65, dtype=torch.float32).view(-1, 1)
    y = torch.zeros(65)
    g = torch.Generator().manual_seed(0)
    seen = torch.cat([xb for xb, _ in iterate_minibatches(X, y, 32, shuffle=True, generator=g)])
    # 65 = 32 + 32 + 1: the lone trailing row is skipped (BatchNorm needs >1 sample).
    assert len(seen) == 64
    assert len(set(seen.view(-1).tolist())) == 64

    full = torch.cat([xb for xb, _ in iterate_minibatches(X[:64], y[:64], 32, shuffle=False)])
    assert torch.equal(full, X[:64])


# ---------------------------------------------------------------------------
# Data paths and MIMIC leakage check
# ---------------------------------------------------------------------------


def test_resolve_data_path_finds_bare_names():
    from project_paths import HEART_CSV, RAW_DATA_DIR, resolve_data_path

    assert resolve_data_path("cleaned_heart_disease.csv") == HEART_CSV
    assert resolve_data_path("heart.csv") == RAW_DATA_DIR / "heart.csv"
    assert HEART_CSV.exists()


def test_mimic_split_check_compares_subject_ids():
    """The old check mixed index spaces and failed on every grouped split."""
    from training.mimic_train_mlp import split_data

    rng = np.random.default_rng(0)
    subjects = pd.Series(np.repeat(np.arange(60), 3))
    y = pd.Series((rng.random(len(subjects)) < 0.3).astype(int))
    X = pd.DataFrame({"a": rng.normal(size=len(subjects))})
    X_train, X_val, X_test, *_ = split_data(X, y, subjects, seed=0)
    train_s = set(subjects.loc[X_train.index])
    val_s = set(subjects.loc[X_val.index])
    test_s = set(subjects.loc[X_test.index])
    assert train_s.isdisjoint(test_s) and val_s.isdisjoint(test_s) and train_s.isdisjoint(val_s)


# ---------------------------------------------------------------------------
# CLI and end-to-end runs
# ---------------------------------------------------------------------------


def test_cli_steps_point_at_importable_mains():
    import importlib

    from run_pipeline import STEPS

    for step, (module_name, _) in STEPS.items():
        if step == "test":
            continue
        assert callable(importlib.import_module(module_name).main), step


def test_cli_step_help_exits_cleanly():
    from run_pipeline import main

    with pytest.raises(SystemExit) as exc:
        main(["baselines", "--help"])
    assert exc.value.code == 0


def test_baselines_end_to_end_reports_cis_and_val_thresholds(tmp_path):
    from training.baseline_models import main

    summary = main([
        "--model", "logreg", "--cv-folds", "3", "--cv-repeats", "1",
        "--n-jobs", "1", "--output-dir", str(tmp_path),
    ])
    row = summary.iloc[0]
    assert row["test_roc_auc_ci_low"] <= row["test_roc_auc"] <= row["test_roc_auc_ci_high"]
    assert (tmp_path / "baseline_comparison.csv").exists()
    assert row["cv_n_folds_total"] == 3


def test_mlp_end_to_end_selects_by_validation(tmp_path):
    from training.train_model import main

    results = main([
        "--optimizer", "both", "--epochs", "3", "--patience", "2", "--cv-folds", "2",
        "--cv-repeats", "1", "--n-jobs", "1", "--device", "cpu", "--output-dir", str(tmp_path),
    ])
    assert {r["optimizer"] for r in results} == {"sgd", "adam"}
    for r in results:
        assert "test_roc_auc_ci_low" in r and "cv_roc_auc_mean" in r
    assert (tmp_path / "mlp_holdout_summary.csv").exists()
    assert len(pd.read_csv(tmp_path / "mlp_cv_fold_metrics_adam.csv")) == 2


def test_parallel_and_serial_folds_give_identical_results():
    """Folds always train single-threaded, so n_jobs changes speed, not numbers."""
    from experiments.run_reading_list_suite import run_architecture_cv
    from project_paths import HEART_CSV

    df = pd.read_csv(HEART_CSV)
    serial, _ = run_architecture_cv(df, folds=2, repeats=1, seed=0, device="cpu", n_jobs=1)
    parallel, _ = run_architecture_cv(df, folds=2, repeats=1, seed=0, device="cpu", n_jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_ensemble_oof_identical_for_any_n_jobs():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from training.ensemble_models import PyTorchMLPClassifier, generate_oof_and_fit_base_models

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 5)))
    y = pd.Series((X[0] + rng.normal(scale=0.5, size=120) > 0).astype(int))
    specs = [
        ("lr", Pipeline([("s", StandardScaler()), ("c", LogisticRegression())])),
        ("mlp", Pipeline([("s", StandardScaler()), ("c", PyTorchMLPClassifier(epochs=3, device="cpu"))])),
    ]
    oof_1, _ = generate_oof_and_fit_base_models(specs, X, y, cv_folds=3, n_jobs=1)
    oof_2, _ = generate_oof_and_fit_base_models(specs, X, y, cv_folds=3, n_jobs=2)
    np.testing.assert_array_equal(oof_1, oof_2)


# ---------------------------------------------------------------------------
# Paired model comparison (DeLong + Holm)
# ---------------------------------------------------------------------------


def _brute_force_delong(y, pa, pb):
    """Textbook DeLong via explicit pairwise placement values (O(m*n))."""
    pos, neg = y == 1, y == 0
    out = []
    for p in (pa, pb):
        x, z = p[pos][:, None], p[neg][None, :]
        psi = (x > z).astype(float) + 0.5 * (x == z)
        out.append((psi.mean(), psi.mean(axis=1), psi.mean(axis=0)))
    (a1, v10a, v01a), (a2, v10b, v01b) = out
    m, n = pos.sum(), neg.sum()
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    cov = s10 / m + s01 / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    return a1, a2, np.sqrt(var)


@pytest.mark.parametrize("seed", [0, 1])
def test_delong_matches_brute_force_and_sklearn(seed):
    from evaluation.model_comparison import delong_test

    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=90)
    pa = np.round(np.clip(0.4 * y + rng.normal(0.3, 0.25, 90), 0, 1), 2)  # ties included
    pb = np.round(np.clip(0.2 * y + rng.normal(0.4, 0.25, 90), 0, 1), 2)
    res = delong_test(y, pa, pb)
    a1, a2, se = _brute_force_delong(y, pa, pb)
    assert res["auc_a"] == pytest.approx(roc_auc_score(y, pa), abs=1e-12)
    assert res["auc_a"] == pytest.approx(a1, abs=1e-12) and res["auc_b"] == pytest.approx(a2, abs=1e-12)
    assert res["se"] == pytest.approx(se, rel=1e-9)
    assert 0.0 <= res["p_value"] <= 1.0


def test_delong_identical_models_have_p_one():
    from evaluation.model_comparison import delong_test

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50)
    p = rng.random(50)
    res = delong_test(y, p, p)
    assert res["auc_diff"] == 0.0 and res["p_value"] == 1.0


def test_holm_adjustment_known_values():
    from evaluation.model_comparison import holm_adjust

    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_prediction_files_merge_and_skip_mismatches(tmp_path):
    from evaluation.model_comparison import load_test_predictions, save_test_predictions

    y = np.array([0, 1, 0, 1])
    save_test_predictions(tmp_path, "a", "d.csv", [3, 1, 2, 0], y, {"m1": [0.1, 0.9, 0.2, 0.8]})
    save_test_predictions(tmp_path, "b", "d.csv", [3, 1, 2, 0], y, {"m2": [0.3, 0.7, 0.4, 0.6]})
    save_test_predictions(tmp_path, "c", "d.csv", [9, 1, 2, 0], y, {"m3": [0.5] * 4})  # other rows
    save_test_predictions(tmp_path, "d", "other.csv", [3, 1, 2, 0], y, {"m4": [0.5] * 4})
    merged = load_test_predictions(tmp_path / "test_predictions", dataset="d.csv")
    assert list(merged["row_id"]) == [0, 1, 2, 3]
    assert {"m1", "m2"} <= set(merged.columns) and not {"m3", "m4"} & set(merged.columns)


# ---------------------------------------------------------------------------
# Fairness objective, search grid, data codes
# ---------------------------------------------------------------------------


def test_youden_objective_never_scores_flag_everyone_above_zero():
    from evaluation.fair_mitigation import find_equal_opportunity_thresholds

    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.7).astype(int)  # high prevalence pushes F1 toward "flag all"
    p = np.clip(0.25 * y + rng.normal(0.4, 0.2, 200), 0, 1)
    g = rng.integers(0, 2, 200)
    f1 = find_equal_opportunity_thresholds(y, p, g, objective="f1")
    yj = find_equal_opportunity_thresholds(y, p, g, objective="youden")
    assert yj["mitigated_joint_youden"] >= f1["mitigated_joint_youden"] - 1e-12
    assert yj["mitigated_specificity"] > 0.2
    with pytest.raises(ValueError):
        find_equal_opportunity_thresholds(y, p, g, objective="accuracy")


def test_hyperparameter_grids_have_eight_configs_each():
    from training.baseline_models import (
        RF_BASE_PARAMS,
        RF_PARAM_GRID,
        XGB_BASE_PARAMS,
        XGB_PARAM_GRID,
        expand_grid,
    )

    rf = expand_grid(RF_BASE_PARAMS, RF_PARAM_GRID)
    xgb = expand_grid(XGB_BASE_PARAMS, XGB_PARAM_GRID)
    assert len(rf) == 8 and len(xgb) == 8
    assert all(c["n_jobs"] == 1 for c in rf + xgb)


def test_invalid_codes_become_missing():
    from clean_data import recode_invalid_codes

    df = pd.DataFrame({"ca": [0, 4, 2], "thal": [0, 1, 3], "chol": [200, 0, 180], "trestbps": [0, 120, 130]})
    out = recode_invalid_codes(df)
    assert out.isna().sum().to_dict() == {"ca": 1, "thal": 1, "chol": 1, "trestbps": 1}
    # UCI coding of thal (3/6/7) has no placeholder zero and must be left alone.
    uci = recode_invalid_codes(pd.DataFrame({"thal": [3.0, 6.0, 7.0]}))
    assert uci["thal"].notna().all()


def test_combined_cohort_has_no_zero_coded_measurements():
    from project_paths import HEART_COMBINED_CSV

    df = pd.read_csv(HEART_COMBINED_CSV)
    assert len(df) == 918
    assert (df["chol"] == 0).sum() == 0 and (df["trestbps"] == 0).sum() == 0
