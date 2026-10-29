"""Paired statistical comparison of models evaluated on the same test patients.

Every training script uses the same seed-controlled 80/20 first split, so all
models are scored on identical test patients. Each script saves its per-patient
test probabilities to ``<output-dir>/test_predictions/<source>.csv``; this
module merges them and compares every pair of models with DeLong's test for
two correlated ROC curves (DeLong et al., 1988; fast algorithm of Sun & Xu,
2014), adjusting the p-values for multiple comparisons with Holm's method.

A confidence interval per model (bootstrap) answers "how precise is this
AUC?"; this answers "is model A's AUC different from model B's on the same
patients?", which is the question a model ranking needs.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PREDICTIONS_SUBDIR = "test_predictions"
META_COLUMNS = ("row_id", "y_true", "dataset")


# ---------------------------------------------------------------------------
# Saving predictions
# ---------------------------------------------------------------------------


def save_test_predictions(
    output_dir: str | Path,
    source: str,
    dataset: str,
    row_ids,
    y_true,
    model_probas: dict[str, np.ndarray],
) -> Path:
    """Write per-patient test probabilities for one script's models."""
    out_dir = Path(output_dir) / PREDICTIONS_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "row_id": np.asarray(row_ids, dtype=int),
        "y_true": np.asarray(y_true, dtype=int),
        "dataset": Path(str(dataset)).name,
    })
    for name, probs in model_probas.items():
        frame[name] = np.asarray(probs, dtype=float).ravel()
    path = out_dir / f"{source}.csv"
    frame.to_csv(path, index=False)
    return path


def load_test_predictions(predictions_dir: str | Path, dataset: str | None = None) -> pd.DataFrame:
    """Merge all prediction files that cover the same dataset and test patients."""
    files = sorted(Path(predictions_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files in {predictions_dir}.")
    frames = [pd.read_csv(f) for f in files]
    if dataset is None:
        dataset = frames[0]["dataset"].iloc[0]
    dataset = Path(str(dataset)).name

    merged: pd.DataFrame | None = None
    for path, frame in zip(files, frames, strict=True):
        if frame["dataset"].iloc[0] != dataset:
            print(f"Skipping {path.name}: predictions are for {frame['dataset'].iloc[0]}, not {dataset}.")
            continue
        frame = frame.drop(columns=["dataset"]).sort_values("row_id").reset_index(drop=True)
        if merged is None:
            merged = frame
            continue
        if not np.array_equal(merged["row_id"].values, frame["row_id"].values):
            print(f"Skipping {path.name}: it was scored on different test patients.")
            continue
        if not np.array_equal(merged["y_true"].values, frame["y_true"].values):
            raise ValueError(f"{path.name}: labels disagree with other prediction files for the same rows.")
        merged = merged.merge(frame.drop(columns=["y_true"]), on="row_id")
    if merged is None:
        raise ValueError(f"No prediction files for dataset {dataset}.")
    merged.insert(2, "dataset", dataset)
    return merged


# ---------------------------------------------------------------------------
# DeLong test
# ---------------------------------------------------------------------------


def _midrank(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties replaced by their average rank."""
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    n = len(x)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def delong_test(y_true, prob_a, prob_b) -> dict[str, float]:
    """DeLong's test for the difference between two correlated ROC AUCs.

    Returns both AUCs, their difference (A - B), its standard error, a 95% CI
    for the difference, the z statistic and the two-sided p-value.
    """
    y = np.asarray(y_true, dtype=int).ravel()
    preds = np.vstack([np.asarray(prob_a, dtype=float).ravel(), np.asarray(prob_b, dtype=float).ravel()])
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m < 2 or n < 2:
        raise ValueError("DeLong's test needs at least two positive and two negative cases.")

    tx = np.vstack([_midrank(p[pos]) for p in preds])
    ty = np.vstack([_midrank(p[neg]) for p in preds])
    tz = np.vstack([_midrank(np.concatenate([p[pos], p[neg]])) for p in preds])

    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n

    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    se = float(np.sqrt(max(var, 0.0)))
    if se == 0.0:
        z, p_value = 0.0, 1.0
    else:
        z = diff / se
        p_value = float(2.0 * norm.sf(abs(z)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": diff,
        "se": se,
        "diff_ci_low": diff - 1.959964 * se,
        "diff_ci_high": diff + 1.959964 * se,
        "z": float(z),
        "p_value": p_value,
    }


def holm_adjust(p_values) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    k = len(p)
    order = np.argsort(p)
    adjusted = np.empty(k)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (k - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def compare_all_models(predictions: pd.DataFrame, exclude: tuple[str, ...] = ()) -> pd.DataFrame:
    """Pairwise DeLong tests for every model column, sorted by adjusted p-value."""
    models = [c for c in predictions.columns if c not in META_COLUMNS and c not in exclude]
    y = predictions["y_true"].values
    rows = []
    for a, b in combinations(models, 2):
        result = delong_test(y, predictions[a].values, predictions[b].values)
        rows.append({"model_a": a, "model_b": b, **result})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = holm_adjust(table["p_value"].values)
    table["significant_holm_0.05"] = table["p_holm"] < 0.05
    return table.sort_values(["p_holm", "p_value"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pairwise DeLong comparison of models on the shared test split.")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--dataset", type=str, default=None, help="Only compare predictions made on this dataset.")
    parser.add_argument(
        "--exclude", nargs="*", default=["Multimodal Cross-Attention (synthetic-signal demo)"],
        help="Model columns to leave out (the synthetic-signal demo by default).",
    )
    return parser


def main(argv: list[str] | None = None) -> pd.DataFrame:
    from evaluation.cv_utils import to_markdown_table
    from project_paths import resolve_output_dir

    args = build_arg_parser().parse_args(argv)
    output_dir = resolve_output_dir(args.output_dir)
    predictions = load_test_predictions(output_dir / PREDICTIONS_SUBDIR, dataset=args.dataset)
    table = compare_all_models(predictions, exclude=tuple(args.exclude or ()))

    n_models = len([c for c in predictions.columns if c not in META_COLUMNS and c not in (args.exclude or ())])
    n_pos = int(predictions["y_true"].sum())
    print(
        f"Compared {n_models} models pairwise ({len(table)} tests) on {len(predictions)} test patients "
        f"({n_pos} with disease) from {predictions['dataset'].iloc[0]}."
    )
    n_sig = int(table["significant_holm_0.05"].sum()) if not table.empty else 0
    print(f"Pairs with a significant ROC-AUC difference after Holm correction: {n_sig} of {len(table)}")

    csv_path = output_dir / "model_comparison.csv"
    md_path = output_dir / "model_comparison.md"
    table.to_csv(csv_path, index=False)
    header = (
        "# Paired Model Comparison (DeLong test, Holm-adjusted)\n\n"
        f"{len(predictions)} shared test patients ({n_pos} with disease) from "
        f"`{predictions['dataset'].iloc[0]}`. `auc_diff` = AUC(model_a) - AUC(model_b); "
        f"{n_sig} of {len(table)} pairs differ at Holm-adjusted p < 0.05.\n\n"
    )
    md_path.write_text(header + to_markdown_table(table), encoding="utf-8")
    print(f"Saved model comparison: {csv_path}")
    return table


if __name__ == "__main__":
    main()
