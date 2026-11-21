"""Hospital Local Site Recalibration Utility (``heart-pipeline calibrate-site``).

Refits Temperature Scaling ``T*``, multi-alpha sex-stratified Mondrian conformal
quantiles (85% / 90% / 95%), OOD threshold, and PSI baseline histogram on a local
hospital cohort without retraining any neural or tree weights (< 2 seconds on CPU).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from models import ModelWithTemperature, compute_ece
from project_paths import ARTIFACTS_DIR, resolve_data_path

from .bundle import (
    DEFAULT_BUNDLE_PATH,
    SUPPORTED_ALPHAS,
    ClinicalModelBundle,
    _sha256_file,
    compute_multialpha_mondrian_quantiles,
    engineer_screening_interactions,
)


def recalibrate_bundle_for_site(
    local_csv: str | Path,
    site_name: str,
    base_bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
    output_path: str | Path | None = None,
    device: str = "cpu",
) -> tuple[Path, dict[str, Any]]:
    """Recalibrate Temperature T*, Mondrian quantiles, OOD & PSI baseline on local hospital data."""
    bundle = ClinicalModelBundle.load(base_bundle_path, device=device)
    if bundle.raw_payload is None:
        raise RuntimeError("Loaded bundle is missing raw payload required for re-export.")

    df = pd.read_csv(resolve_data_path(local_csv)).dropna(subset=["target"]).reset_index(drop=True)
    if len(df) < 30:
        raise ValueError(f"Local calibration cohort too small ({len(df)} rows); need at least 30.")
    if "aha_bp_stage" in bundle.cat_cols and "aha_bp_stage" not in df.columns:
        df = engineer_screening_interactions(df)

    dev = next(bundle.model.parameters()).device
    X_cat = df[bundle.cat_cols].values.astype(np.int64)
    X_cont_raw = df[bundle.cont_cols].values.astype(np.float32)
    X_cont = (X_cont_raw - bundle.cont_mean) / np.where(bundle.cont_scale == 0, 1.0, bundle.cont_scale)
    X_tab = np.hstack([X_cat.astype(np.float32), X_cont]).astype(np.float32)
    y_int = df["target"].values.astype(int)
    sex_int = df["sex"].values.astype(int)

    xc_t = torch.from_numpy(X_cat).to(dev)
    xct_t = torch.from_numpy(X_cont).to(dev)

    bundle.model.eval()
    logits_parts, dist_parts = [], []
    k_eff = min(bundle.k_neighbors, bundle.H_bank_norm.size(0))
    with torch.inference_mode():
        for s in range(0, len(df), 1024):
            h_q, _ = bundle.model.encode_tabular(xc_t[s : s + 1024], xct_t[s : s + 1024])
            q_norm = F.normalize(h_q, p=2, dim=1, eps=1e-8)
            sims = torch.mm(q_norm, bundle.H_bank_norm.t())
            topk_sims, topk_idx = torch.topk(sims, k=k_eff, dim=1)
            lg = bundle.model.forward_minibatch(
                x_cat=xc_t[s : s + 1024],
                x_cont=xct_t[s : s + 1024],
                nbr_h_tab=bundle.H_tab_bank[topk_idx],
                nbr_labels=bundle.bank_outcomes_t[topk_idx],
            )
            logits_parts.append(lg)
            dist_parts.append((1.0 - topk_sims[:, 0]).cpu().numpy())

    local_logits = torch.cat(logits_parts, dim=0)
    local_min_dists = np.concatenate(dist_parts)

    temp_scaler = ModelWithTemperature(bundle.model).to(dev)
    cal_report = temp_scaler.calibrate(
        local_logits.detach().clone(),
        torch.from_numpy(y_int.astype(np.float32)).to(dev).view(-1, 1),
    )
    new_temp = float(temp_scaler.temperature.item())

    with torch.inference_mode():
        local_gat_probs = torch.sigmoid(temp_scaler.temperature_scale(local_logits)).cpu().numpy().ravel()

    w_gat, w_xgb, w_hgb = bundle.ensemble_weights
    local_probs = w_gat * local_gat_probs
    total_w = w_gat
    if bundle.xgb_model is not None:
        local_probs = local_probs + w_xgb * bundle.xgb_model.predict_proba(X_tab)[:, 1]
        total_w += w_xgb
    if bundle.hgb_model is not None:
        local_probs = local_probs + w_hgb * bundle.hgb_model.predict_proba(X_tab)[:, 1]
        total_w += w_hgb
    local_probs_raw = np.clip(local_probs / max(total_w, 1e-8), 1e-5, 1.0 - 1e-5)

    # Fit post-blend local Platt recalibration (fixes both slope & hospital prevalence shift)
    if len(np.unique(y_int)) == 2:
        platt_lr = LogisticRegression(C=5.0, max_iter=500)
        platt_lr.fit(logit(local_probs_raw).reshape(-1, 1), y_int)
        platt_a = float(platt_lr.coef_[0, 0])
        platt_b = float(platt_lr.intercept_[0])
        local_probs = expit(platt_a * logit(local_probs_raw) + platt_b)
    else:
        platt_a, platt_b = 1.0, 0.0
        local_probs = local_probs_raw

    new_quantiles = compute_multialpha_mondrian_quantiles(
        local_probs, y_int, sex_int, alphas=SUPPORTED_ALPHAS
    )
    new_ood_thresh = float(np.quantile(local_min_dists, 0.995))

    hist_edges = np.linspace(0.0, 1.0, 11)
    hist_counts, _ = np.histogram(local_probs, bins=hist_edges)
    hist_props = np.clip(hist_counts / max(len(local_probs), 1), 1e-6, 1.0)
    hist_props = hist_props / hist_props.sum()

    payload = dict(bundle.raw_payload)
    payload["model_version"] = f"{bundle.model_version}+{site_name.lower().replace(' ', '_')}"
    payload["temperature"] = new_temp
    payload["platt_a"] = platt_a
    payload["platt_b"] = platt_b
    payload["conformal_quantiles"] = new_quantiles
    payload["ood_distance_threshold"] = new_ood_thresh
    payload["val_risk_hist_edges"] = torch.from_numpy(hist_edges.astype(np.float64))
    payload["val_risk_hist_props"] = torch.from_numpy(hist_props.astype(np.float64))

    updated_card = dict(bundle.model_card)
    updated_card["model_version"] = payload["model_version"]
    updated_card["site_recalibration"] = {
        "site_name": site_name,
        "recalibrated_utc": datetime.now(timezone.utc).isoformat(),
        "n_local_patients": int(len(df)),
        "local_prevalence": round(float(np.mean(y_int)), 4),
        "temperature_before": round(bundle.temperature, 4),
        "temperature_after": round(new_temp, 4),
        "platt_a": round(platt_a, 4),
        "platt_b": round(platt_b, 4),
        "local_roc_auc": round(float(roc_auc_score(y_int, local_probs)), 4),
        "local_brier": round(float(brier_score_loss(y_int, local_probs)), 4),
        "local_ece_before": round(float(compute_ece(local_probs_raw, y_int)), 4),
        "local_ece_after": round(float(compute_ece(local_probs, y_int)), 4),
        "gat_ece_before": round(float(cal_report["ece_before"]), 4),
        "gat_ece_after": round(float(cal_report["ece_after"]), 4),
    }
    payload["model_card"] = updated_card

    out = Path(output_path) if output_path else ARTIFACTS_DIR / f"clinical_cds_bundle_{site_name.lower().replace(' ', '_')}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    sha = _sha256_file(out)
    updated_card["bundle_sha256"] = sha
    out.with_suffix(".model_card.json").write_text(json.dumps(updated_card, indent=2), encoding="utf-8")

    print(
        f"Recalibrated bundle for site '{site_name}' ({len(df)} patients) -> {out} "
        f"(T*: {bundle.temperature:.3f} -> {new_temp:.3f} | Local AUC={updated_card['site_recalibration']['local_roc_auc']})"
    )
    return out, updated_card


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Locally recalibrate Clinical CDS Bundle for a hospital site.")
    parser.add_argument("--local-csv", type=str, required=True, help="Local labeled hospital calibration CSV.")
    parser.add_argument("--site-name", type=str, required=True, help="Hospital site identifier.")
    parser.add_argument("--base-bundle", type=str, default=str(DEFAULT_BUNDLE_PATH))
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args(argv)
    recalibrate_bundle_for_site(
        local_csv=args.local_csv,
        site_name=args.site_name,
        base_bundle_path=args.base_bundle,
        output_path=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
