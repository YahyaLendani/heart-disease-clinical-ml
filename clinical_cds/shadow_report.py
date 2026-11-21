"""Prospective Silent-Mode Shadow Trial Evaluation Utility (``heart-pipeline shadow-report``).

Joins a de-identified ``cds_audit.jsonl`` log written by ``POST /v1/shadow/stage1`` or
``POST /v1/shadow/stage2`` against a delayed clinical outcomes CSV keyed by ``request_id``
(zero PHI required) and generates a TRIPOD+AI / DECIDE-AI aligned evaluation report:

- Discrimination: ROC-AUC (with 95% bootstrap CI), PR-AUC
- Calibration: Brier Score, Expected Calibration Error (ECE)
- Conformal Triage Safety: Empirical coverage, deferral rate, OOD override rate
- Operational Telemetry: Mean & P95 latency (ms)
- Clinical Utility: Decision Curve Net Benefit at 30% and 50% thresholds
- Subgroup Fairness Audit: Per-group sensitivity, FNR, specificity, ROC-AUC (if ``sex`` or
  ``subgroup`` columns are supplied in the outcomes CSV)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from models import compute_ece
from project_paths import ARTIFACTS_DIR


def _bootstrap_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(float(roc_auc_score(y_true[idx], probs[idx])))
    if not vals:
        return 0.5, 0.5
    return round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)


def _net_benefit(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    n = max(len(y_true), 1)
    pred = probs >= threshold
    tp = float(np.sum(pred & (y_true == 1)))
    fp = float(np.sum(pred & (y_true == 0)))
    return (tp / n) - (fp / n) * (threshold / (1.0 - threshold))


def _subgroup_metrics(df: pd.DataFrame, group_col: str) -> dict[str, dict[str, float | int | None]]:
    out: dict[str, dict[str, float | int | None]] = {}
    for gval, sub in df.groupby(group_col, dropna=True):
        y = sub["target"].values.astype(int)
        p = sub["calibrated_risk_probability"].values.astype(np.float64)
        pred = (p >= 0.5).astype(int)
        pos = y == 1
        neg = y == 0
        sens = float(np.mean(pred[pos] == 1)) if pos.any() else None
        spec = float(np.mean(pred[neg] == 0)) if neg.any() else None
        fnr = float(1.0 - sens) if sens is not None else None
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
        cov = float(np.mean(sub["is_covered"].values))
        defer = float(np.mean(sub["triage_decision"].values == "DEFER_TO_CLINICIAN"))
        out[str(gval)] = {
            "n": int(len(sub)),
            "prevalence": round(float(np.mean(y)), 4),
            "roc_auc": round(auc, 4) if auc is not None else None,
            "sensitivity": round(sens, 4) if sens is not None else None,
            "specificity": round(spec, 4) if spec is not None else None,
            "fnr": round(fnr, 4) if fnr is not None else None,
            "conformal_coverage": round(cov, 4),
            "deferral_rate": round(defer, 4),
        }
    return out


def generate_shadow_trial_report(
    audit_log_path: str | Path,
    outcomes_csv: str | Path,
    stage: str = "all",
    output_dir: str | Path = ARTIFACTS_DIR,
    seed: int = 42,
) -> dict[str, Any]:
    """Join de-identified shadow audit JSONL with delayed clinical outcomes and write reports."""
    audit_path = Path(audit_log_path)
    if not audit_path.exists():
        raise FileNotFoundError(f"Audit log not found: {audit_path}")

    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if stage in ("stage1", "stage2"):
        records = [r for r in records if r.get("stage", "stage1") == stage]
    if not records:
        raise ValueError(f"No audit records found in {audit_path} for stage='{stage}'.")

    audit_df = pd.DataFrame(records)
    outcomes_df = pd.read_csv(outcomes_csv)
    if "request_id" not in outcomes_df.columns or "target" not in outcomes_df.columns:
        raise ValueError("Outcomes CSV must include 'request_id' and binary 'target' columns.")

    merged = audit_df.merge(outcomes_df, on="request_id", how="inner")
    merged = merged.dropna(subset=["target", "calibrated_risk_probability"]).reset_index(drop=True)
    if len(merged) < 5:
        raise ValueError(
            f"Only {len(merged)} matched shadow records found (need >= 5). "
            "Check that request_id values match between audit JSONL and outcomes CSV."
        )

    y_true = merged["target"].astype(int).values
    probs = merged["calibrated_risk_probability"].astype(np.float64).values
    latencies = merged["latency_ms"].astype(np.float64).values

    # Determine conformal coverage from logged prediction set or triage decision
    covered_flags: list[int] = []
    for _, row in merged.iterrows():
        y_i = int(row["target"])
        cset = row.get("conformal_prediction_set")
        if isinstance(cset, list):
            covered_flags.append(int(y_i in cset))
        else:
            td = str(row.get("triage_decision", "DEFER_TO_CLINICIAN"))
            if td == "DEFER_TO_CLINICIAN":
                covered_flags.append(1)
            elif td == "HIGH_RISK":
                covered_flags.append(int(y_i == 1))
            else:
                covered_flags.append(int(y_i == 0))
    merged["is_covered"] = covered_flags

    has_both_classes = len(np.unique(y_true)) == 2
    roc_auc = round(float(roc_auc_score(y_true, probs)), 4) if has_both_classes else None
    roc_ci = list(_bootstrap_ci(y_true, probs, seed=seed)) if has_both_classes else None
    pr_auc = round(float(average_precision_score(y_true, probs)), 4) if has_both_classes else None

    deferral_rate = float(np.mean(merged["triage_decision"].values == "DEFER_TO_CLINICIAN"))
    ood_rate = float(np.mean(merged.get("is_ood", pd.Series(False, index=merged.index)).astype(bool)))

    subgroups: dict[str, dict[str, Any]] = {}
    for candidate_col in ("sex", "age_group", "subgroup", "site"):
        if candidate_col in merged.columns:
            subgroups[candidate_col] = _subgroup_metrics(merged, candidate_col)

    report: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "reporting_framework": "TRIPOD+AI / DECIDE-AI Silent-Mode Shadow Evaluation",
        "stage_filter": stage,
        "n_audit_records": int(len(audit_df)),
        "n_matched_outcomes": int(len(merged)),
        "outcome_prevalence": round(float(np.mean(y_true)), 4),
        "discrimination": {
            "roc_auc": roc_auc,
            "roc_auc_ci_95": roc_ci,
            "pr_auc": pr_auc,
        },
        "calibration": {
            "brier": round(float(brier_score_loss(y_true, probs)), 4),
            "ece": round(float(compute_ece(probs, y_true)), 4),
        },
        "conformal_safety": {
            "empirical_coverage": round(float(np.mean(covered_flags)), 4),
            "deferral_rate": round(deferral_rate, 4),
            "ood_override_rate": round(ood_rate, 4),
            "triage_breakdown": {
                str(k): int(v) for k, v in merged["triage_decision"].value_counts().items()
            },
        },
        "clinical_utility": {
            "net_benefit_30pct": round(_net_benefit(y_true, probs, 0.30), 4),
            "net_benefit_50pct": round(_net_benefit(y_true, probs, 0.50), 4),
        },
        "operations": {
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        },
        "subgroups": subgroups,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "shadow_trial_report.json"
    md_path = out_dir / "shadow_trial_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    auc_str = (
        f"{roc_auc:.4f} [{roc_ci[0]:.4f}, {roc_ci[1]:.4f}]"
        if roc_auc is not None and roc_ci is not None
        else "N/A (single class)"
    )
    pr_str = f"{pr_auc:.4f}" if pr_auc is not None else "N/A"
    md_lines = [
        "# Silent-Mode Clinical Shadow Trial Report (TRIPOD+AI / DECIDE-AI)",
        "",
        f"- **Generated (UTC):** `{report['generated_utc']}`",
        f"- **Stage Filter:** `{stage}` | **Matched Patients:** `{len(merged):,}` / `{len(audit_df):,}` logged",
        f"- **Observed Event Prevalence:** `{report['outcome_prevalence'] * 100:.1f}%`",
        "",
        "## Primary Performance & Safety Summary",
        "",
        "| Metric | Observed Shadow Value |",
        "| :--- | :---: |",
        f"| **ROC-AUC [95% Bootstrap CI]** | **{auc_str}** |",
        f"| **PR-AUC** | {pr_str} |",
        f"| **Brier Score ↓** | {report['calibration']['brier']:.4f} |",
        f"| **Expected Calibration Error (ECE) ↓** | {report['calibration']['ece']:.4f} |",
        f"| **Empirical Conformal Coverage** | **{report['conformal_safety']['empirical_coverage'] * 100:.1f}%** |",
        f"| **Clinician Deferral Rate** | {report['conformal_safety']['deferral_rate'] * 100:.1f}% |",
        f"| **OOD Safety Override Rate** | {report['conformal_safety']['ood_override_rate'] * 100:.1f}% |",
        f"| **Decision Curve Net Benefit @ 30% / 50%** | {report['clinical_utility']['net_benefit_30pct']:.4f} / {report['clinical_utility']['net_benefit_50pct']:.4f} |",
        f"| **Mean / P95 Inference Latency** | {report['operations']['mean_latency_ms']:.2f} ms / {report['operations']['p95_latency_ms']:.2f} ms |",
    ]
    if subgroups:
        md_lines.extend(["", "## Subgroup Fairness Audit", ""])
        for col, groups in subgroups.items():
            md_lines.append(f"### Stratified by `{col}`")
            md_lines.append("")
            md_lines.append("| Group | N | Prevalence | ROC-AUC | Sensitivity | FNR | Coverage | Deferral |")
            md_lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for gname, g in groups.items():
                g_auc = f"{g['roc_auc']:.4f}" if g["roc_auc"] is not None else "—"
                g_sens = f"{g['sensitivity'] * 100:.1f}%" if g["sensitivity"] is not None else "—"
                g_fnr = f"{g['fnr'] * 100:.1f}%" if g["fnr"] is not None else "—"
                md_lines.append(
                    f"| `{gname}` | {g['n']} | {g['prevalence'] * 100:.1f}% | {g_auc} | "
                    f"{g_sens} | {g_fnr} | {g['conformal_coverage'] * 100:.1f}% | {g['deferral_rate'] * 100:.1f}% |"
                )
            md_lines.append("")

    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Shadow trial report written -> {md_path} ({len(merged)} matched patients)")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate a TRIPOD+AI / DECIDE-AI validation report from shadow-mode audit logs."
    )
    parser.add_argument("--audit-log", type=str, required=True, help="Path to cds_audit.jsonl")
    parser.add_argument(
        "--outcomes",
        type=str,
        required=True,
        help="Path to delayed outcomes CSV containing request_id and target columns.",
    )
    parser.add_argument("--stage", choices=["all", "stage1", "stage2"], default="all")
    parser.add_argument("--output-dir", type=str, default=str(ARTIFACTS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    generate_shadow_trial_report(
        audit_log_path=args.audit_log,
        outcomes_csv=args.outcomes,
        stage=args.stage,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
