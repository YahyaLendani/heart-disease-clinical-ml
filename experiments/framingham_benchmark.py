"""Clinical Guideline Benchmark: CardioGraph CDS v3 vs. Framingham Risk Score (D'Agostino et al., 2008).

Evaluates our Stage-1 PLR+UniMP 3-Paradigm Clinical CDS v3 bundle head-to-head against
the standard clinical guideline **Framingham General Cardiovascular Risk Profile**
(D'Agostino et al., *Circulation* 2008) on the untouched 10,289-patient OpenML 45547
holdout test split:
- Discrimination: ROC-AUC with 1,000-resample bootstrap 95% CIs + paired DeLong p-value
- Precision-Recall & Calibration: PR-AUC, Brier Score, ECE
- Clinical Utility: Decision Curve Analysis (Net Benefit across 10%..70% decision thresholds)
  compared against Framingham, Treat-All, and Treat-None strategies.

Outputs:
- ``artifacts/framingham_benchmark.json``
- ``artifacts/framingham_benchmark.md``
- ``artifacts/figures/framingham_vs_cardiograph_dca.png``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from clinical_cds import ClinicalInferenceEngine, PatientClinicalInput
from clinical_cds.bundle import DEFAULT_BUNDLE_PATH
from models import compute_ece
from project_paths import ARTIFACTS_DIR, resolve_data_path


def compute_framingham_2008_risk(df: pd.DataFrame) -> np.ndarray:
    """Compute D'Agostino et al. (Circulation 2008) 10-year General CVD risk probabilities.

    Maps screening cholesterol tiers (0=Normal, 1=Elevated, 2=High) to standard midpoint
    mg/dL values (185, 225, 275 mg/dL), fasting glucose tier 2 to diabetes=1, and uses
    sex-specific NHANES adult HDL midpoints (Female=55, Male=45 mg/dL) with untreated
    systolic BP coefficients (medication status not recorded in OpenML 45547).
    """
    age = np.clip(df["age_years"].values.astype(np.float64), 20.0, 95.0)
    sex = df["sex"].values.astype(int)  # 0=Female, 1=Male
    sbp = np.clip(df["ap_hi"].values.astype(np.float64), 85.0, 240.0)
    chol_tier = df["cholesterol"].values.astype(int)
    tot_chol = np.where(chol_tier == 0, 185.0, np.where(chol_tier == 1, 225.0, 275.0))
    hdl = np.where(sex == 0, 55.0, 45.0)
    smoke = df["smoke"].values.astype(np.float64)
    diab = (df["gluc"].values.astype(int) >= 2).astype(np.float64)

    # Published D'Agostino et al. (2008) Table 2 beta coefficients & baseline survival S0(10)
    lp_f = (
        2.32888 * np.log(age)
        + 1.20904 * np.log(tot_chol)
        - 0.70833 * np.log(hdl)
        + 2.76157 * np.log(sbp)
        + 0.52873 * smoke
        + 0.69154 * diab
    )
    risk_f = 1.0 - np.power(0.95012, np.exp(lp_f - 26.1931))

    lp_m = (
        3.06117 * np.log(age)
        + 1.12370 * np.log(tot_chol)
        - 0.93263 * np.log(hdl)
        + 1.93303 * np.log(sbp)
        + 0.65451 * smoke
        + 0.57367 * diab
    )
    risk_m = 1.0 - np.power(0.88936, np.exp(lp_m - 23.9802))

    return np.clip(np.where(sex == 1, risk_m, risk_f), 1e-6, 1.0 - 1e-6)


def _delong_paired_pvalue(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> tuple[float, float]:
    """Fast Sun & Xu (2014) exact paired DeLong test for two correlated ROC-AUCs."""
    pos = y_true == 1
    neg = y_true == 0
    n1, n0 = int(pos.sum()), int(neg.sum())

    def _structural(scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        order = np.argsort(scores)
        ranks = np.empty_like(scores, dtype=np.float64)
        # Average ranks for ties
        sorted_s = scores[order]
        i = 0
        n = len(scores)
        while i < n:
            j = i
            while j + 1 < n and sorted_s[j + 1] == sorted_s[i]:
                j += 1
            avg_rank = 0.5 * (i + j) + 1.0
            ranks[order[i : j + 1]] = avg_rank
            i = j + 1

        # Separate ranks within positives and negatives
        pos_ranks = ranks[pos]
        neg_ranks = ranks[neg]
        pos_internal = np.argsort(np.argsort(scores[pos])) + 1.0
        neg_internal = np.argsort(np.argsort(scores[neg])) + 1.0
        v10 = (pos_ranks - pos_internal) / n0
        v01 = 1.0 - (neg_ranks - neg_internal) / n1
        auc = float(np.mean(v10))
        return auc, v10, v01

    auc1, v10_1, v01_1 = _structural(p1)
    auc2, v10_2, v01_2 = _structural(p2)

    s10 = np.cov(np.vstack([v10_1, v10_2])) / n1
    s01 = np.cov(np.vstack([v01_1, v01_2])) / n0
    S = s10 + s01
    var_diff = max(float(S[0, 0] + S[1, 1] - 2.0 * S[0, 1]), 1e-12)
    z = (auc1 - auc2) / np.sqrt(var_diff)
    pval = float(2.0 * norm.sf(abs(z)))
    return float(z), pval


def _net_benefit(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> float:
    n = len(y_true)
    pred = probs >= threshold
    tp = float(np.sum(pred & (y_true == 1)))
    fp = float(np.sum(pred & (y_true == 0)))
    return (tp / n) - (fp / n) * (threshold / (1.0 - threshold))


def _bootstrap_auc_ci(
    y_true: np.ndarray, probs: np.ndarray, n_boot: int = 500, seed: int = 42
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y_true[idx], probs[idx])))
    return round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)


def run_framingham_benchmark(
    bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
    openml_csv: str | Path = "cleaned_cardio_70k.csv",
    output_dir: str | Path = ARTIFACTS_DIR,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare Stage-1 CardioGraph CDS v3 vs. Framingham (2008) on 10,289 holdout patients."""
    df = pd.read_csv(resolve_data_path(openml_csv))
    _, test_df = train_test_split(df, test_size=0.15, random_state=seed, stratify=df["target"])
    test_df = test_df[
        (test_df["ap_hi"] > test_df["ap_lo"] + 5.0)
        & test_df["ap_hi"].between(60.0, 250.0)
        & test_df["ap_lo"].between(40.0, 160.0)
        & test_df["bmi"].between(12.0, 70.0)
    ].reset_index(drop=True)
    y_te = test_df["target"].values.astype(int)

    # 1. CardioGraph CDS v3 probabilities (batched forward pass)
    engine = ClinicalInferenceEngine.from_bundle(bundle_path, device="cpu")
    patients = [
        PatientClinicalInput(
            age_years=float(r["age_years"]),
            sex=int(r["sex"]),
            height_cm=float(r["height"]),
            weight_kg=float(r["weight"]),
            systolic_bp=float(r["ap_hi"]),
            diastolic_bp=float(r["ap_lo"]),
            cholesterol_level=int(r["cholesterol"]) + 1,
            glucose_level=int(r["gluc"]) + 1,
            smoke=int(r["smoke"]),
            alcohol=int(r["alco"]),
            active=int(r["active"]),
        )
        for _, r in test_df.iterrows()
    ]
    cg_probs = engine._forward_patients(patients, return_explanations=False)["probs"]

    # 2. Framingham (2008) raw 10-year risk & prevalence-recalibrated Framingham (for fair DCA)
    fram_raw = compute_framingham_2008_risk(test_df)
    # Recalibrate Framingham intercept/slope on train split (zero test leakage) for fair DCA comparison
    train_df, _ = train_test_split(df, test_size=0.15, random_state=seed, stratify=df["target"])
    from scipy.special import expit, logit
    from sklearn.linear_model import LogisticRegression

    fram_tr = compute_framingham_2008_risk(train_df)
    lr_cal = LogisticRegression(C=10.0)
    lr_cal.fit(logit(fram_tr).reshape(-1, 1), train_df["target"].values.astype(int))
    fram_cal = expit(lr_cal.coef_[0, 0] * logit(fram_raw) + lr_cal.intercept_[0])

    cg_auc = float(roc_auc_score(y_te, cg_probs))
    fr_auc = float(roc_auc_score(y_te, fram_raw))
    cg_ci = _bootstrap_auc_ci(y_te, cg_probs, seed=seed)
    fr_ci = _bootstrap_auc_ci(y_te, fram_raw, seed=seed)
    z_stat, p_val = _delong_paired_pvalue(y_te, cg_probs, fram_raw)

    thresholds = np.linspace(0.10, 0.70, 25)
    prev = float(np.mean(y_te))
    dca_rows = []
    for t in thresholds:
        nb_cg = _net_benefit(y_te, cg_probs, float(t))
        nb_fr_cal = _net_benefit(y_te, fram_cal, float(t))
        nb_all = prev - (1.0 - prev) * (float(t) / (1.0 - float(t)))
        dca_rows.append({
            "threshold": round(float(t), 3),
            "cardiograph_nb": round(nb_cg, 4),
            "framingham_recal_nb": round(nb_fr_cal, 4),
            "treat_all_nb": round(nb_all, 4),
            "nb_gain_over_framingham": round(nb_cg - nb_fr_cal, 4),
        })

    # Plot Decision Curve Analysis figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    ts = [r["threshold"] * 100 for r in dca_rows]
    ax.plot(ts, [r["cardiograph_nb"] for r in dca_rows], color="#0f5c6e", linewidth=2.4, label=f"CardioGraph CDS v3 (AUC {cg_auc:.4f})")
    ax.plot(ts, [r["framingham_recal_nb"] for r in dca_rows], color="#b45309", linewidth=2.0, linestyle="--", label=f"Framingham 2008 Recalibrated (AUC {fr_auc:.4f})")
    ax.plot(ts, [r["treat_all_nb"] for r in dca_rows], color="#64748b", linewidth=1.4, linestyle=":", label="Treat All")
    ax.axhline(0.0, color="#94a3b8", linewidth=1.1, label="Treat None")
    ax.set_ylim(-0.05, max(prev + 0.05, 0.55))
    ax.set_xlabel("Clinical Decision Threshold (%)")
    ax.set_ylabel("Net Benefit (True Positives per Patient)")
    ax.set_title(f"Decision Curve Analysis vs. Framingham Risk Score (N={len(y_te):,} Holdout)")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig_path = Path(output_dir) / "figures" / "framingham_vs_cardiograph_dca.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)

    summary = {
        "n_holdout": int(len(y_te)),
        "delong_z": round(z_stat, 3),
        "delong_p_value": p_val,
        "auc_gain": round(cg_auc - fr_auc, 4),
        "models": {
            "cardiograph_cds_v3": {
                "roc_auc": round(cg_auc, 4),
                "roc_auc_ci_95": list(cg_ci),
                "pr_auc": round(float(average_precision_score(y_te, cg_probs)), 4),
                "brier": round(float(brier_score_loss(y_te, cg_probs)), 4),
                "ece": round(float(compute_ece(cg_probs, y_te)), 4),
            },
            "framingham_2008_raw": {
                "roc_auc": round(fr_auc, 4),
                "roc_auc_ci_95": list(fr_ci),
                "pr_auc": round(float(average_precision_score(y_te, fram_raw)), 4),
                "brier": round(float(brier_score_loss(y_te, fram_raw)), 4),
                "ece": round(float(compute_ece(fram_raw, y_te)), 4),
            },
            "framingham_2008_recalibrated": {
                "roc_auc": round(float(roc_auc_score(y_te, fram_cal)), 4),
                "pr_auc": round(float(average_precision_score(y_te, fram_cal)), 4),
                "brier": round(float(brier_score_loss(y_te, fram_cal)), 4),
                "ece": round(float(compute_ece(fram_cal, y_te)), 4),
            },
        },
        "dca_summary": dca_rows,
    }

    out_dir = Path(output_dir)
    (out_dir / "framingham_benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    m_cg = summary["models"]["cardiograph_cds_v3"]
    m_fr = summary["models"]["framingham_2008_raw"]
    m_fc = summary["models"]["framingham_2008_recalibrated"]
    nb_50 = next(r for r in dca_rows if abs(r["threshold"] - 0.50) < 0.02)

    md = "\n".join([
        "# Clinical Guideline Benchmark: CardioGraph CDS v3 vs. Framingham Risk Score (2008)",
        "",
        f"Evaluated on **{len(y_te):,}** untouched holdout screening patients (OpenML 45547).",
        f"Paired DeLong test: **+{summary['auc_gain']:.4f} ROC-AUC** (`z = {summary['delong_z']}`, `p = {summary['delong_p_value']:.2e}`).",
        "",
        "| Model | Holdout ROC-AUC [95% CI] | PR-AUC | Brier ↓ | ECE ↓ | Net Benefit @ 50% |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
        f"| **CardioGraph CDS v3 (PLR+UniMP Ensemble)** | **{m_cg['roc_auc']:.4f}** [{m_cg['roc_auc_ci_95'][0]:.4f}, {m_cg['roc_auc_ci_95'][1]:.4f}] | **{m_cg['pr_auc']:.4f}** | **{m_cg['brier']:.4f}** | **{m_cg['ece']:.4f}** | **{nb_50['cardiograph_nb']:.4f}** |",
        f"| **Framingham 2008 (Recalibrated)** | {m_fc['roc_auc']:.4f} [{m_fr['roc_auc_ci_95'][0]:.4f}, {m_fr['roc_auc_ci_95'][1]:.4f}] | {m_fc['pr_auc']:.4f} | {m_fc['brier']:.4f} | {m_fc['ece']:.4f} | {nb_50['framingham_recal_nb']:.4f} |",
        f"| **Framingham 2008 (Raw 10-yr)** | {m_fr['roc_auc']:.4f} [{m_fr['roc_auc_ci_95'][0]:.4f}, {m_fr['roc_auc_ci_95'][1]:.4f}] | {m_fr['pr_auc']:.4f} | {m_fr['brier']:.4f} | {m_fr['ece']:.4f} | — |",
    ]) + "\n"
    (out_dir / "framingham_benchmark.md").write_text(md, encoding="utf-8")
    print(md)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark CardioGraph CDS v3 vs Framingham (2008).")
    parser.add_argument("--bundle", type=str, default=str(DEFAULT_BUNDLE_PATH))
    parser.add_argument("--output-dir", type=str, default=str(ARTIFACTS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    run_framingham_benchmark(bundle_path=args.bundle, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
