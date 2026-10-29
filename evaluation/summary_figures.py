"""Headline figures for the README, built from the result tables.

* ``model_auc_forest.png`` — every model's holdout ROC-AUC with its bootstrap
  95% CI, on the 4-center cohort and (if present) the Cleveland-only run.
* ``loco_conformal_coverage.png`` — conformal coverage and fixed-threshold
  accuracy at each unseen hospital against the 90% coverage target.

Each figure is written in a light version and a ``_dark`` version so the
README can serve the one matching the reader's theme.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.plot_style import (
    apply_style,
    finish_axes,
    legend_above,
    palette,
    series_color,
    subtitle,
)

OPTIMIZER_NAMES = {"adam": "Adam", "sgd": "SGD"}

CENTER_NAMES = {
    "cleveland": "Cleveland",
    "hungarian": "Budapest",
    "switzerland": "Zurich",
    "va_long_beach": "Long Beach VA",
}


def _auc_rows(run_dir: Path) -> pd.DataFrame:
    """Model name, ROC-AUC and CI from every result table in ``run_dir``."""
    rows = []

    def add(name, row, auc="test_roc_auc"):
        rows.append({
            "model": name,
            "auc": float(row[auc]),
            "low": float(row[f"{auc}_ci_low"]),
            "high": float(row[f"{auc}_ci_high"]),
        })

    path = run_dir / "baseline_comparison.csv"
    if path.exists():
        for _, r in pd.read_csv(path).iterrows():
            add(r["model"], r)
    path = run_dir / "mlp_holdout_summary.csv"
    if path.exists():
        for _, r in pd.read_csv(path).iterrows():
            add(f"HeartMLP ({OPTIMIZER_NAMES.get(r['optimizer'], r['optimizer'])})", r)
    path = run_dir / "reading_list_advancements_summary.csv"
    if path.exists():
        frame = pd.read_csv(path)
        if "is_synthetic_demo" in frame.columns:
            frame = frame[~frame["is_synthetic_demo"].astype(bool)]
        for _, r in frame.iterrows():
            add(r["model"].replace(" (Kipf & Welling)", "").replace(" (Velickovic et al.)", ""), r)
    path = run_dir / "ensemble_comparison.csv"
    if path.exists():
        for _, r in pd.read_csv(path).iterrows():
            if "Ensemble" in r["model"] and "test_roc_auc_ci_low" in r.index:
                add(r["model"].replace("Hybrid ", "").replace(" (Ours)", ""), r)
    return pd.DataFrame(rows)


def draw_auc_forest(primary: pd.DataFrame, secondary: pd.DataFrame | None, n_sig: int | None,
                    n_pairs: int | None, mode: str = "light"):
    import matplotlib.pyplot as plt

    p = apply_style(mode)
    order = primary.sort_values("auc").reset_index(drop=True)
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9, 0.48 * len(order) + 1.9))
    series = [("UCI 4-center cohort (184 test patients)", primary, 0, 0.17)]
    if secondary is not None and not secondary.empty:
        series.append(("Cleveland only (61 test patients)", secondary, 1, -0.17))
    else:
        series[0] = (series[0][0], primary, 0, 0.0)

    for label, frame, slot, offset in series:
        lookup = frame.set_index("model")
        names = [m for m in order["model"] if m in lookup.index]
        ys = np.array([y[order.index[order["model"] == m][0]] for m in names]) + offset
        auc = lookup.loc[names, "auc"].to_numpy()
        color = series_color(slot)
        ax.hlines(ys, lookup.loc[names, "low"], lookup.loc[names, "high"], color=color, linewidth=2)
        ax.scatter(auc, ys, s=64, color=color, edgecolors=p["surface"], linewidths=2, zorder=3, label=label)

    # Primary-cohort values as a right-hand column (the table view lives in the README).
    for yi, value in zip(y, order["auc"], strict=True):
        ax.annotate(f"{value:.3f}", xy=(1.0, yi + (0.17 if len(series) > 1 else 0)), xycoords=("axes fraction", "data"),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=p["ink_secondary"])

    ax.set_yticks(y)
    ax.set_yticklabels(order["model"])
    ax.set_xlim(0.70, 1.0)
    ax.set_xlabel("Holdout ROC-AUC with bootstrap 95% confidence interval")
    ax.grid(axis="y", visible=False)
    ax.set_title("Every model's interval overlaps the others")
    note = "Paired DeLong tests on the shared test patients"
    if n_sig is not None and n_pairs is not None:
        note += f": {n_sig} of {n_pairs} model pairs differ after Holm correction"
    subtitle(ax, note, legend_row=True)
    legend_above(ax)
    finish_axes(ax)
    fig.tight_layout()
    return fig


def draw_loco_coverage(loco: pd.DataFrame, mode: str = "light"):
    import matplotlib.pyplot as plt

    p = apply_style(mode)
    order = loco.sort_values("loco_empirical_coverage").reset_index(drop=True)
    y = np.arange(len(order))
    labels = [
        f"{CENTER_NAMES.get(c, c)}  (n={int(n)}, {prev * 100:.0f}% disease)"
        for c, n, prev in zip(order["test_site"], order["n_test_samples"], order["test_disease_prevalence"], strict=True)
    ]
    fig, ax = plt.subplots(figsize=(9, 3.9))
    ax.axvline(90, color=p["muted"], linestyle="--", linewidth=1.2)
    ax.annotate("90% target", xy=(90, -0.5), xytext=(4, 0), textcoords="offset points",
                fontsize=9, color=p["ink_secondary"], va="center")

    cov = order["loco_empirical_coverage"].to_numpy() * 100
    acc = order["fixed_threshold_accuracy"].to_numpy() * 100
    ax.hlines(y, np.minimum(cov, acc), np.maximum(cov, acc), color=p["grid"], linewidth=2, zorder=1)
    ax.scatter(acc, y, s=64, color=series_color(1), edgecolors=p["surface"], linewidths=2, zorder=3,
               label="Fixed-threshold accuracy")
    ax.scatter(cov, y, s=64, color=series_color(0), edgecolors=p["surface"], linewidths=2, zorder=3,
               label="Conformal coverage (alpha = 0.10)")
    for yi, value in zip(y, cov, strict=True):
        ax.annotate(f"{value:.1f}%", xy=(value, yi), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=p["ink_secondary"])

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(50, 100)
    ax.set_xlabel("Percent of patients at the held-out hospital")
    ax.grid(axis="y", visible=False)
    ax.set_title("Conformal coverage falls short at unseen hospitals")
    subtitle(ax, "Logistic regression trained and calibrated on three hospitals, evaluated on the fourth",
             legend_row=True)
    legend_above(ax)
    finish_axes(ax)
    fig.tight_layout()
    return fig


def _save_both(draw, path: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    written = []
    for mode, out in (("light", path), ("dark", path.with_name(f"{path.stem}_dark{path.suffix}"))):
        fig = draw(mode)
        fig.savefig(out, dpi=170)
        plt.close(fig)
        written.append(out)
    apply_style("light")
    return written


def main(argv: list[str] | None = None) -> list[Path]:
    from project_paths import resolve_output_dir

    parser = argparse.ArgumentParser(description="Build the README summary figures from the result tables.")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--secondary-dir", type=str, default="artifacts/cleveland")
    args = parser.parse_args(argv)

    output_dir = resolve_output_dir(args.output_dir)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    secondary_dir = Path(args.secondary_dir)
    if not secondary_dir.is_absolute():
        secondary_dir = PROJECT_ROOT / secondary_dir

    written: list[Path] = []
    primary = _auc_rows(output_dir)
    if not primary.empty:
        secondary = _auc_rows(secondary_dir) if secondary_dir.exists() and secondary_dir != output_dir else None
        n_sig = n_pairs = None
        comparison = output_dir / "model_comparison.csv"
        if comparison.exists():
            table = pd.read_csv(comparison)
            n_sig, n_pairs = int(table["significant_holm_0.05"].sum()), len(table)
        written += _save_both(
            lambda mode: draw_auc_forest(primary, secondary, n_sig, n_pairs, mode),
            figures_dir / "model_auc_forest.png",
        )

    loco_path = output_dir / "loco_conformal_generalization.csv"
    if loco_path.exists():
        loco = pd.read_csv(loco_path)
        written += _save_both(lambda mode: draw_loco_coverage(loco, mode), figures_dir / "loco_conformal_coverage.png")

    for path in written:
        print(f"Saved summary figure: {path}")
    return written


if __name__ == "__main__":
    palette()  # keep the style import used when run as a script
    main()
