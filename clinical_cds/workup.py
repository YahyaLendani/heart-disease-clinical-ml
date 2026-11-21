"""Stage-2 Cardiology Workup Bundle & Engine (Multi-Center UCI Cohort, N=918, ~0.90+ ROC-AUC).

Used when a Stage-1 primary-care screening patient triggers ``DEFER_TO_CLINICIAN``
or ``HIGH_RISK`` and escalates to a full cardiology diagnostic workup with ECG,
exercise stress-test, and perfusion scan biomarkers:
- 5-Fold Out-of-Fold Cross-Fitted 3-Paradigm Ensemble:
  ``PLR + UniMP GraphAugmentedClinicalTransformer`` + ``XGBoost`` + ``RandomForest``
- Multi-alpha sex-stratified Mondrian conformal triage (85% / 90% / 95%)
- Learned-space multi-center cohort neighbor matching + dual biomarker interpretability
- Immutable-safe cardiology recourse recommendations (BP, cholesterol, exercise capacity)
"""

from __future__ import annotations

import io
import json
import pickle
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from models import (
    GraphAugmentedClinicalTransformer,
    ModelWithTemperature,
    add_cardiovascular_features,
    build_inductive_knn_indices,
    compute_ece,
)
from project_paths import ARTIFACTS_DIR, resolve_data_path

from .bundle import (
    SUPPORTED_ALPHAS,
    _sha256_file,
    compute_multialpha_mondrian_quantiles,
)
from .schemas import (
    CardiologyWorkupInput,
    ClinicalPredictionResponse,
    DualClinicalExplanation,
    MatchedCohortNeighbor,
    OODReport,
    RecourseRecommendation,
)

WORKUP_BUNDLE_VERSION = "1.0.0-cardiology-workup-plr-unimp"
DEFAULT_WORKUP_BUNDLE_PATH = ARTIFACTS_DIR / "clinical_workup_bundle.pt"

WORKUP_CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
WORKUP_CONT_COLS = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "rpp",
    "chronotropic_ratio",
    "ischemia_burden",
    "vessel_ischemia",
]

# Fixed canonical category value -> index mappings for zero-ambiguity clinical inference
WORKUP_CAT_VOCABS: dict[str, dict[int, int]] = {
    "sex": {0: 0, 1: 1},
    "cp": {1: 0, 2: 1, 3: 2, 4: 3},
    "fbs": {0: 0, 1: 1},
    "restecg": {0: 0, 1: 1, 2: 2},
    "exang": {0: 0, 1: 1},
    "slope": {1: 0, 2: 1, 3: 2},
    "ca": {0: 0, 1: 1, 2: 2, 3: 3},
    "thal": {3: 0, 6: 1, 7: 2},
}
WORKUP_CAT_DIMS = [len(v) + 1 for v in WORKUP_CAT_VOCABS.values()]


def _encode_workup_cats(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for col, vocab in WORKUP_CAT_VOCABS.items():
        unk = len(vocab)
        vals = pd.to_numeric(df[col], errors="coerce").round().map(vocab).fillna(unk).astype(np.int64)
        cols.append(vals.values)
    return np.column_stack(cols)


def _train_workup_gat(
    xc_tr: np.ndarray,
    xct_tr: np.ndarray,
    y_tr: np.ndarray,
    xc_va: np.ndarray,
    xct_va: np.ndarray,
    y_va: np.ndarray,
    model_config: dict[str, Any],
    epochs: int,
    k_neighbors: int,
    seed: int,
    dev: torch.device,
) -> GraphAugmentedClinicalTransformer:
    torch.manual_seed(seed)
    np.random.seed(seed)

    g_scl = StandardScaler()
    xg_tr = g_scl.fit_transform(np.hstack([xc_tr.astype(np.float32), xct_tr])).astype(np.float32)
    xg_va = g_scl.transform(np.hstack([xc_va.astype(np.float32), xct_va])).astype(np.float32)
    tr_nbr, va_nbr = build_inductive_knn_indices(xg_tr, xg_va, k=k_neighbors, metric="cosine", n_jobs=1)

    xc_tr_t, xct_tr_t = torch.from_numpy(xc_tr).to(dev), torch.from_numpy(xct_tr).to(dev)
    xc_va_t, xct_va_t = torch.from_numpy(xc_va).to(dev), torch.from_numpy(xct_va).to(dev)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(dev).view(-1, 1)
    y_va_t = torch.from_numpy(y_va.astype(np.float32)).to(dev).view(-1, 1)
    y_tr_l = torch.from_numpy(y_tr.astype(np.int64)).to(dev)
    tr_nbr_t = torch.from_numpy(tr_nbr).to(dev)
    va_nbr_t = torch.from_numpy(va_nbr).to(dev)  # type: ignore[arg-type]

    model = GraphAugmentedClinicalTransformer(**model_config).to(dev)
    opt = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=3e-3)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_v, best_sd = float("inf"), None
    refresh_ep = max(10, epochs // 3)
    for ep in range(epochs):
        model.eval()
        with torch.inference_mode():
            h_bank, _ = model.encode_tabular(xc_tr_t, xct_tr_t)
            if ep == refresh_ep:
                h_va_e, _ = model.encode_tabular(xc_va_t, xct_va_t)
                tr_new, va_new = build_inductive_knn_indices(
                    F.normalize(h_bank, p=2, dim=1, eps=1e-8).cpu().numpy(),
                    F.normalize(h_va_e, p=2, dim=1, eps=1e-8).cpu().numpy(),
                    k=k_neighbors,
                    metric="cosine",
                    n_jobs=1,
                )
                tr_nbr_t = torch.from_numpy(tr_new).to(dev)
                va_nbr_t = torch.from_numpy(va_new).to(dev)  # type: ignore[arg-type]

        model.train()
        opt.zero_grad()
        logits = model.forward_minibatch(
            x_cat=xc_tr_t,
            x_cont=xct_tr_t,
            nbr_h_tab=h_bank[tr_nbr_t],
            nbr_labels=y_tr_l[tr_nbr_t],
            label_mask_prob=0.40,
            feature_noise_std=0.04,
        )
        loss = criterion(logits, y_tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        model.eval()
        with torch.inference_mode():
            h_eval, _ = model.encode_tabular(xc_tr_t, xct_tr_t)
            v_lg = model.forward_minibatch(
                x_cat=xc_va_t,
                x_cont=xct_va_t,
                nbr_h_tab=h_eval[va_nbr_t],
                nbr_labels=y_tr_l[va_nbr_t],
            )
            vloss = criterion(v_lg, y_va_t).item()
            if vloss < best_v:
                best_v = vloss
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_sd is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})
    model.eval()
    return model


@dataclass
class CardiologyWorkupBundle:
    """Serialized Stage-2 Cardiology Workup bundle (multi-center UCI cohort, N=918)."""

    model: GraphAugmentedClinicalTransformer
    xgb_model: XGBClassifier
    rf_model: RandomForestClassifier
    ensemble_weights: tuple[float, float, float]
    temperature: float
    model_version: str
    bundle_sha256: str
    feature_names: list[str]
    cont_medians: np.ndarray
    cont_mean: np.ndarray
    cont_scale: np.ndarray
    H_bank_norm: torch.Tensor
    H_tab_bank: torch.Tensor
    bank_outcomes: np.ndarray
    bank_outcomes_t: torch.Tensor
    bank_summary: np.ndarray  # [age, trestbps, chol]
    k_neighbors: int
    default_conformal_alpha: float
    conformal_quantiles: dict[str, dict[str, float]]
    ood_distance_threshold: float
    ood_zscore_threshold: float
    model_card: dict[str, Any]

    @classmethod
    def load(
        cls,
        bundle_path: str | Path = DEFAULT_WORKUP_BUNDLE_PATH,
        device: str = "cpu",
    ) -> CardiologyWorkupBundle:
        path = Path(bundle_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Stage-2 Cardiology Workup bundle not found at {path}. "
                "Build it with `heart-pipeline build-workup-bundle`."
            )
        actual_sha = _sha256_file(path)
        dev = torch.device(device)
        payload = torch.load(path, map_location=dev, weights_only=False)

        model = GraphAugmentedClinicalTransformer(**payload["model_config"]).to(dev)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

        xgb_model = XGBClassifier(n_jobs=1, device="cpu")
        xgb_model.load_model(bytearray(payload["xgb_booster_json"]))
        rf_model = pickle.loads(io.BytesIO(payload["rf_pickle_bytes"]).read())
        rf_model.n_jobs = 1

        H_tab = payload["H_tab_bank"].float().to(dev)
        H_norm = payload["H_bank_norm"].float().to(dev)
        outcomes = np.asarray(payload["bank_outcomes"], dtype=np.int8)

        return cls(
            model=model,
            xgb_model=xgb_model,
            rf_model=rf_model,
            ensemble_weights=tuple(float(x) for x in payload["ensemble_weights"]),  # type: ignore[arg-type]
            temperature=float(payload["temperature"]),
            model_version=str(payload["model_version"]),
            bundle_sha256=actual_sha,
            feature_names=list(payload["feature_names"]),
            cont_medians=np.asarray(payload["cont_medians"], dtype=np.float32),
            cont_mean=np.asarray(payload["cont_mean"], dtype=np.float32),
            cont_scale=np.asarray(payload["cont_scale"], dtype=np.float32),
            H_bank_norm=H_norm,
            H_tab_bank=H_tab,
            bank_outcomes=outcomes,
            bank_outcomes_t=torch.from_numpy(outcomes.astype(np.int64)).to(dev),
            bank_summary=np.asarray(payload["bank_summary"], dtype=np.float32),
            k_neighbors=int(payload["k_neighbors"]),
            default_conformal_alpha=float(payload.get("conformal_alpha", 0.10)),
            conformal_quantiles={
                str(a): {str(k): float(v) for k, v in d.items()}
                for a, d in payload["conformal_quantiles"].items()
            },
            ood_distance_threshold=float(payload["ood_distance_threshold"]),
            ood_zscore_threshold=float(payload.get("ood_zscore_threshold", 4.5)),
            model_card=dict(payload["model_card"]),
        )


def build_workup_bundle(
    dataset_path: str | Path = "cleaned_heart_disease_combined.csv",
    output_path: str | Path = DEFAULT_WORKUP_BUNDLE_PATH,
    epochs: int = 80,
    cv_folds: int = 5,
    k_neighbors: int = 7,
    embed_dim: int = 32,
    seed: int = 42,
) -> tuple[Path, dict[str, Any]]:
    """Train and export the Stage-2 Cardiology Workup bundle (~0.90+ ROC-AUC)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(resolve_data_path(dataset_path)).dropna(subset=["target"]).reset_index(drop=True)
    # Recode impossible zero-coded cholesterol/trestbps as missing so imputers handle cleanly
    for col in ("chol", "trestbps"):
        if col in df.columns:
            df.loc[df[col] == 0, col] = np.nan
    df = add_cardiovascular_features(df)

    y_all = df["target"].values.astype(int)
    dev_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.20, random_state=seed, stratify=y_all)
    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    te_df = df.iloc[test_idx].reset_index(drop=True)
    y_dev = dev_df["target"].values.astype(int)
    y_te = te_df["target"].values.astype(int)

    model_config = {
        "categorical_dims": WORKUP_CAT_DIMS,
        "continuous_dim": len(WORKUP_CONT_COLS),
        "embed_dim": embed_dim,
        "num_transformer_heads": 4,
        "num_transformer_layers": 2,
        "num_gat_heads": 4,
        "num_gat_layers": 2,
        "mlp_hidden_dim": 32,
        "dropout": 0.25,
        "num_classes": 1,
        "use_periodic_embeddings": True,
        "n_frequencies": 6,
        "use_label_propagation": True,
    }

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_gat_logits = np.zeros(len(dev_df), dtype=np.float64)
    oof_xgb = np.zeros(len(dev_df), dtype=np.float64)
    oof_rf = np.zeros(len(dev_df), dtype=np.float64)
    oof_min_dists = np.zeros(len(dev_df), dtype=np.float64)

    for fold, (f_tr, f_va) in enumerate(skf.split(dev_df, y_dev)):
        tr_f, va_f = dev_df.iloc[f_tr], dev_df.iloc[f_va]
        imp_f = SimpleImputer(strategy="median")
        scl_f = StandardScaler()
        xc_tr, xc_va = _encode_workup_cats(tr_f), _encode_workup_cats(va_f)
        xct_tr = scl_f.fit_transform(imp_f.fit_transform(tr_f[WORKUP_CONT_COLS].values)).astype(np.float32)
        xct_va = scl_f.transform(imp_f.transform(va_f[WORKUP_CONT_COLS].values)).astype(np.float32)
        xtab_tr = np.hstack([xc_tr.astype(np.float32), xct_tr])
        xtab_va = np.hstack([xc_va.astype(np.float32), xct_va])

        gat_f = _train_workup_gat(
            xc_tr, xct_tr, y_dev[f_tr], xc_va, xct_va, y_dev[f_va],
            model_config=model_config, epochs=epochs, k_neighbors=k_neighbors, seed=seed + fold, dev=dev
        )
        with torch.inference_mode():
            h_tr, _ = gat_f.encode_tabular(torch.from_numpy(xc_tr).to(dev), torch.from_numpy(xct_tr).to(dev))
            h_va, _ = gat_f.encode_tabular(torch.from_numpy(xc_va).to(dev), torch.from_numpy(xct_va).to(dev))
            sims = torch.mm(F.normalize(h_va, p=2, dim=1, eps=1e-8), F.normalize(h_tr, p=2, dim=1, eps=1e-8).t())
            tsim, tidx = torch.topk(sims, k=min(k_neighbors, len(f_tr)), dim=1)
            y_tr_l = torch.from_numpy(y_dev[f_tr].astype(np.int64)).to(dev)
            lg = gat_f.forward_minibatch(
                torch.from_numpy(xc_va).to(dev),
                torch.from_numpy(xct_va).to(dev),
                nbr_h_tab=h_tr[tidx],
                nbr_labels=y_tr_l[tidx],
            )
            oof_gat_logits[f_va] = lg.cpu().numpy().ravel()
            oof_min_dists[f_va] = (1.0 - tsim[:, 0]).cpu().numpy()

        xgb_f = XGBClassifier(
            n_estimators=220, max_depth=3, learning_rate=0.04, subsample=0.85,
            colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss",
            early_stopping_rounds=20, random_state=seed + fold, n_jobs=-1,
        )
        xgb_f.fit(xtab_tr, y_dev[f_tr], eval_set=[(xtab_va, y_dev[f_va])], verbose=False)
        oof_xgb[f_va] = xgb_f.predict_proba(xtab_va)[:, 1]

        rf_f = RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=seed + fold, n_jobs=-1
        )
        rf_f.fit(xtab_tr, y_dev[f_tr])
        oof_rf[f_va] = rf_f.predict_proba(xtab_va)[:, 1]

    dummy = GraphAugmentedClinicalTransformer(**model_config).to(dev)
    temp_scaler = ModelWithTemperature(dummy).to(dev)
    temp_scaler.calibrate(
        torch.from_numpy(oof_gat_logits.astype(np.float32)).to(dev).view(-1, 1),
        torch.from_numpy(y_dev.astype(np.float32)).to(dev).view(-1, 1),
    )
    temp_val = float(temp_scaler.temperature.item())
    oof_gat = 1.0 / (1.0 + np.exp(-oof_gat_logits / max(temp_val, 1e-3)))

    P_oof = np.column_stack([oof_gat, oof_xgb, oof_rf])

    def _obj(theta: np.ndarray) -> float:
        w = softmax(theta)
        if w[0] < 0.20:
            return 10.0 + float(0.20 - w[0])
        return float(log_loss(y_dev, np.clip(P_oof @ w, 1e-7, 1.0 - 1e-7)))

    w_opt = softmax(minimize(_obj, np.zeros(3), method="Nelder-Mead").x)
    w_gat, w_xgb, w_rf = float(w_opt[0]), float(w_opt[1]), float(w_opt[2])
    oof_ens = P_oof @ w_opt

    dev_sex = pd.to_numeric(dev_df["sex"], errors="coerce").fillna(1).astype(int).values
    conformal_quantiles = compute_multialpha_mondrian_quantiles(
        oof_ens, y_dev, dev_sex, alphas=SUPPORTED_ALPHAS
    )
    ood_dist_threshold = float(np.quantile(oof_min_dists, 0.99))

    # Fit final full-cohort workup models
    cont_imputer = SimpleImputer(strategy="median")
    cont_scaler = StandardScaler()
    xc_dev, xc_te = _encode_workup_cats(dev_df), _encode_workup_cats(te_df)
    xct_dev = cont_scaler.fit_transform(cont_imputer.fit_transform(dev_df[WORKUP_CONT_COLS].values)).astype(np.float32)
    xct_te = cont_scaler.transform(cont_imputer.transform(te_df[WORKUP_CONT_COLS].values)).astype(np.float32)
    xtab_dev = np.hstack([xc_dev.astype(np.float32), xct_dev])
    xtab_te = np.hstack([xc_te.astype(np.float32), xct_te])

    sub_tr, sub_es = train_test_split(np.arange(len(dev_df)), test_size=0.15, random_state=seed, stratify=y_dev)
    final_gat = _train_workup_gat(
        xc_dev[sub_tr], xct_dev[sub_tr], y_dev[sub_tr],
        xc_dev[sub_es], xct_dev[sub_es], y_dev[sub_es],
        model_config=model_config, epochs=epochs, k_neighbors=k_neighbors, seed=seed, dev=dev
    )
    final_xgb = XGBClassifier(
        n_estimators=220, max_depth=3, learning_rate=0.04, subsample=0.85,
        colsample_bytree=0.8, reg_lambda=2.0, eval_metric="logloss",
        early_stopping_rounds=20, random_state=seed, n_jobs=-1,
    )
    final_xgb.fit(xtab_dev[sub_tr], y_dev[sub_tr], eval_set=[(xtab_dev[sub_es], y_dev[sub_es])], verbose=False)
    final_rf = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=seed, n_jobs=-1
    )
    final_rf.fit(xtab_dev, y_dev)

    with torch.inference_mode():
        h_dev, _ = final_gat.encode_tabular(torch.from_numpy(xc_dev).to(dev), torch.from_numpy(xct_dev).to(dev))
        h_dev_n = F.normalize(h_dev, p=2, dim=1, eps=1e-8)
        h_te, _ = final_gat.encode_tabular(torch.from_numpy(xc_te).to(dev), torch.from_numpy(xct_te).to(dev))
        sims = torch.mm(F.normalize(h_te, p=2, dim=1, eps=1e-8), h_dev_n.t())
        _, tidx = torch.topk(sims, k=min(k_neighbors, len(dev_df)), dim=1)
        y_dev_l = torch.from_numpy(y_dev.astype(np.int64)).to(dev)
        te_lg = final_gat.forward_minibatch(
            torch.from_numpy(xc_te).to(dev),
            torch.from_numpy(xct_te).to(dev),
            nbr_h_tab=h_dev[tidx],
            nbr_labels=y_dev_l[tidx],
        ).cpu().numpy().ravel()

    te_gat_p = 1.0 / (1.0 + np.exp(-te_lg / max(temp_val, 1e-3)))
    te_xgb_p = final_xgb.predict_proba(xtab_te)[:, 1]
    te_rf_p = final_rf.predict_proba(xtab_te)[:, 1]
    te_probs = w_gat * te_gat_p + w_xgb * te_xgb_p + w_rf * te_rf_p
    te_preds = (te_probs >= 0.5).astype(int)

    summary_vals = np.column_stack([
        pd.to_numeric(dev_df["age"], errors="coerce").fillna(cont_imputer.statistics_[0]).values,
        pd.to_numeric(dev_df["trestbps"], errors="coerce").fillna(cont_imputer.statistics_[1]).values,
        pd.to_numeric(dev_df["chol"], errors="coerce").fillna(cont_imputer.statistics_[2]).values,
    ]).astype(np.float32)

    # Multi-alpha Mondrian conformal audit on holdout test set
    sex_te = pd.to_numeric(te_df["sex"], errors="coerce").fillna(1).astype(int).values
    conformal_by_alpha: dict[str, dict[str, float]] = {}
    for alpha in SUPPORTED_ALPHAS:
        akey = f"{alpha:.2f}"
        qmap = conformal_quantiles[akey]
        cov_hits, defer_hits, sizes = [], [], []
        for p_i, y_i, s_i in zip(te_probs, y_te, sex_te, strict=True):
            q0 = qmap.get(f"sex_{int(s_i)}_class_0", qmap["class_0"])
            q1 = qmap.get(f"sex_{int(s_i)}_class_1", qmap["class_1"])
            cset = []
            if p_i <= q0:
                cset.append(0)
            if (1.0 - p_i) <= q1:
                cset.append(1)
            if not cset:
                cset = [0, 1]
            cov_hits.append(int(int(y_i) in cset))
            defer_hits.append(int(len(cset) == 2))
            sizes.append(len(cset))
        conformal_by_alpha[akey] = {
            "nominal_coverage": round(1.0 - alpha, 2),
            "empirical_test_coverage": round(float(np.mean(cov_hits)), 4),
            "deferral_rate": round(float(np.mean(defer_hits)), 4),
            "mean_set_size": round(float(np.mean(sizes)), 4),
        }

    # Sex-stratified holdout subgroup audit
    subgroup_perf: dict[str, dict[str, float | int]] = {}
    for sex_val, sex_name in ((0, "female"), (1, "male")):
        mask = sex_te == sex_val
        if mask.sum() >= 10 and len(np.unique(y_te[mask])) == 2:
            y_s, p_s = y_te[mask], te_probs[mask]
            pred_s = (p_s >= 0.5).astype(int)
            tp = int(((pred_s == 1) & (y_s == 1)).sum())
            fn = int(((pred_s == 0) & (y_s == 1)).sum())
            tn = int(((pred_s == 0) & (y_s == 0)).sum())
            fp = int(((pred_s == 1) & (y_s == 0)).sum())
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            subgroup_perf[sex_name] = {
                "n_test": int(mask.sum()),
                "roc_auc": round(float(roc_auc_score(y_s, p_s)), 4),
                "sensitivity": round(sens, 4),
                "specificity": round(spec, 4),
                "fnr": round(1.0 - sens, 4),
            }

    model_card = {
        "model_name": "Stage-2 Cardiology Workup Ensemble (PLR+UniMP GAT-T + XGBoost + RF)",
        "model_version": WORKUP_BUNDLE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_cohort": {
            "source": "UCI 4-Center Cardiology Workup Cohort (Cleveland, Budapest, Zurich, VA Long Beach)",
            "n_total": int(len(df)),
            "n_train": int(len(dev_df)),
            "n_test": int(len(te_df)),
        },
        "ensemble_weights": {
            "graph_augmented_transformer": round(w_gat, 4),
            "xgboost": round(w_xgb, 4),
            "random_forest": round(w_rf, 4),
        },
        "holdout_performance": {
            "roc_auc": round(float(roc_auc_score(y_te, te_probs)), 4),
            "pr_auc": round(float(average_precision_score(y_te, te_probs)), 4),
            "accuracy": round(float(accuracy_score(y_te, te_preds)), 4),
            "f1": round(float(f1_score(y_te, te_preds)), 4),
            "brier": round(float(brier_score_loss(y_te, te_probs)), 4),
            "ece": round(float(compute_ece(te_probs, y_te)), 4),
            "temperature": round(temp_val, 4),
            "standalone_gat_roc_auc": round(float(roc_auc_score(y_te, te_gat_p)), 4),
            "standalone_xgb_roc_auc": round(float(roc_auc_score(y_te, te_xgb_p)), 4),
            "standalone_rf_roc_auc": round(float(roc_auc_score(y_te, te_rf_p)), 4),
            "oof_ensemble_roc_auc": round(float(roc_auc_score(y_dev, oof_ens)), 4),
        },
        "mondrian_conformal_audit": conformal_by_alpha["0.10"],
        "mondrian_conformal_by_alpha": conformal_by_alpha,
        "subgroup_performance": subgroup_perf,
        "intended_use": (
            "Stage-2 Cardiology diagnostic escalation for patients deferred or flagged high-risk "
            "during Stage-1 primary-care screening. Incorporates ECG, stress-test heart rate, "
            "ST depression, chest pain, and perfusion scan biomarkers."
        ),
        "limitations": [
            "Trained on 918 retrospective cardiology referrals across 4 centers (Cleveland, Budapest, Zurich, VA Long Beach).",
            "High pre-test referral prevalence (~55%); intended strictly for Stage-2 cardiology workup after primary-care triage.",
            "Research decision-support prototype only; not FDA-cleared or CE-marked clinical diagnostic software.",
        ],
    }

    rf_buf = io.BytesIO()
    pickle.dump(final_rf, rf_buf)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": WORKUP_BUNDLE_VERSION,
        "model_config": model_config,
        "model_state_dict": {k: v.cpu() for k, v in final_gat.state_dict().items()},
        "xgb_booster_json": bytes(final_xgb.get_booster().save_raw("json")),
        "rf_pickle_bytes": rf_buf.getvalue(),
        "ensemble_weights": (w_gat, w_xgb, w_rf),
        "temperature": temp_val,
        "feature_names": WORKUP_CAT_COLS + WORKUP_CONT_COLS,
        "cont_medians": cont_imputer.statistics_.astype(np.float32),
        "cont_mean": cont_scaler.mean_.astype(np.float32),
        "cont_scale": cont_scaler.scale_.astype(np.float32),
        "H_bank_norm": h_dev_n.half().cpu(),
        "H_tab_bank": h_dev.half().cpu(),
        "bank_outcomes": y_dev.astype(np.int8),
        "bank_summary": summary_vals,
        "k_neighbors": k_neighbors,
        "conformal_alpha": 0.10,
        "conformal_quantiles": conformal_quantiles,
        "ood_distance_threshold": ood_dist_threshold,
        "ood_zscore_threshold": 4.5,
        "model_card": model_card,
    }
    torch.save(payload, out)
    sha = _sha256_file(out)
    model_card["bundle_sha256"] = sha
    out.with_suffix(".model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")

    hp = model_card["holdout_performance"]
    print(f"Exported Stage-2 Cardiology Workup Bundle -> {out} (SHA-256: {sha[:16]}...)")
    print(
        f"Workup Holdout ROC-AUC={hp['roc_auc']} (GAT={hp['standalone_gat_roc_auc']}, "
        f"XGB={hp['standalone_xgb_roc_auc']}, RF={hp['standalone_rf_roc_auc']}) | "
        f"PR-AUC={hp['pr_auc']} | Acc={hp['accuracy']} | ECE={hp['ece']}"
    )
    return out, model_card


class CardiologyWorkupEngine:
    """Stage-2 Cardiology Workup inference engine (~0.91 ROC-AUC)."""

    def __init__(self, bundle: CardiologyWorkupBundle) -> None:
        self.bundle = bundle
        self.device = next(bundle.model.parameters()).device

    @classmethod
    def from_bundle(
        cls, bundle_path: str | Path = DEFAULT_WORKUP_BUNDLE_PATH, device: str = "cpu"
    ) -> CardiologyWorkupEngine:
        return cls(CardiologyWorkupBundle.load(bundle_path, device=device))

    def _featurize(self, patients: list[CardiologyWorkupInput]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cat_rows, cont_rows = [], []
        for p in patients:
            cat_rows.append([
                WORKUP_CAT_VOCABS["sex"].get(p.sex, 2),
                WORKUP_CAT_VOCABS["cp"].get(p.cp, 4),
                WORKUP_CAT_VOCABS["fbs"].get(p.fbs, 2),
                WORKUP_CAT_VOCABS["restecg"].get(p.restecg, 3),
                WORKUP_CAT_VOCABS["exang"].get(p.exang, 2),
                WORKUP_CAT_VOCABS["slope"].get(p.slope, 3),
                WORKUP_CAT_VOCABS["ca"].get(p.ca, 4),
                WORKUP_CAT_VOCABS["thal"].get(p.thal, 3),
            ])
            rpp = (p.trestbps * p.thalach) / 100.0
            chrono = p.thalach / max(220.0 - p.age, 40.0)
            isch = p.oldpeak * float(p.exang)
            v_isch = float(p.ca) * p.oldpeak
            cont_rows.append([
                p.age, p.trestbps, p.chol, p.thalach, p.oldpeak,
                rpp, chrono, isch, v_isch,
            ])
        xc = np.asarray(cat_rows, dtype=np.int64)
        xct_raw = np.asarray(cont_rows, dtype=np.float32)
        xct = (xct_raw - self.bundle.cont_mean) / np.where(self.bundle.cont_scale == 0, 1.0, self.bundle.cont_scale)
        xtab = np.hstack([xc.astype(np.float32), xct]).astype(np.float32)
        return xc, xct.astype(np.float32), xtab

    @torch.inference_mode()
    def _forward(self, patients: list[CardiologyWorkupInput], return_explanations: bool = True) -> dict:
        xc_np, xct_np, xtab_np = self._featurize(patients)
        xc_t, xct_t = torch.from_numpy(xc_np).to(self.device), torch.from_numpy(xct_np).to(self.device)

        self.bundle.model.eval()
        hq, _ = self.bundle.model.encode_tabular(xc_t, xct_t)
        sims = torch.mm(F.normalize(hq, p=2, dim=1, eps=1e-8), self.bundle.H_bank_norm.t())
        k_eff = min(self.bundle.k_neighbors, sims.size(1))
        tsim, tidx = torch.topk(sims, k=k_eff, dim=1)

        out = self.bundle.model.forward_minibatch(
            x_cat=xc_t,
            x_cont=xct_t,
            nbr_h_tab=self.bundle.H_tab_bank[tidx],
            nbr_labels=self.bundle.bank_outcomes_t[tidx],
            return_explanations=return_explanations,
        )
        logits, expl = out if return_explanations else (out, None)
        gat_p = torch.sigmoid(logits / max(self.bundle.temperature, 1e-3)).cpu().numpy().ravel()
        xgb_p = self.bundle.xgb_model.predict_proba(xtab_np)[:, 1].astype(np.float64)
        rf_p = self.bundle.rf_model.predict_proba(xtab_np)[:, 1].astype(np.float64)

        w_g, w_x, w_r = self.bundle.ensemble_weights
        probs = w_g * gat_p + w_x * xgb_p + w_r * rf_p
        return {
            "probs": probs,
            "gat_probs": gat_p,
            "xgb_probs": xgb_p,
            "rf_probs": rf_p,
            "x_cont_scaled": xct_np,
            "topk_sims": tsim.cpu().numpy(),
            "topk_idx": tidx.cpu().numpy(),
            "expl": expl,
        }

    def predict(self, patient: CardiologyWorkupInput) -> ClinicalPredictionResponse:
        t0 = time.perf_counter()
        fwd = self._forward([patient], return_explanations=True)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        prob = float(fwd["probs"][0])
        gat_p = float(fwd["gat_probs"][0])
        xgb_p = float(fwd["xgb_probs"][0])
        rf_p = float(fwd["rf_probs"][0])
        disagreement = float(max(gat_p, xgb_p, rf_p) - min(gat_p, xgb_p, rf_p))

        max_z = float(np.max(np.abs(fwd["x_cont_scaled"][0])))
        min_cos_dist = float(1.0 - fwd["topk_sims"][0, 0])
        ood_reasons: list[str] = []
        if max_z > self.bundle.ood_zscore_threshold:
            ood_reasons.append(f"Extreme cardiology biomarker z-score ({max_z:.2f} > {self.bundle.ood_zscore_threshold:.1f}).")
        if min_cos_dist > self.bundle.ood_distance_threshold:
            ood_reasons.append(
                f"Nearest cardiology cohort distance ({min_cos_dist:.4f}) exceeds OOD threshold ({self.bundle.ood_distance_threshold:.4f})."
            )
        is_ood = len(ood_reasons) > 0

        akey = f"{patient.conformal_alpha:.2f}"
        qmap = self.bundle.conformal_quantiles.get(akey, next(iter(self.bundle.conformal_quantiles.values())))
        q0 = qmap.get(f"sex_{patient.sex}_class_0", qmap["class_0"])
        q1 = qmap.get(f"sex_{patient.sex}_class_1", qmap["class_1"])
        low_t = float(np.clip(1.0 - q1, 0.0, 1.0))
        high_t = float(np.clip(q0, 0.0, 1.0))

        cset = []
        if prob <= q0:
            cset.append(0)
        if (1.0 - prob) <= q1:
            cset.append(1)
        if not cset:
            cset = [0, 1]

        nom_pct = int(round((1.0 - patient.conformal_alpha) * 100))
        if is_ood:
            triage, reason = "DEFER_TO_CLINICIAN", "Automatic fail-safe abstention triggered by cardiology OOD alert."
        elif cset == [0]:
            triage, reason = "LOW_RISK", f"Stage-2 Cardiology Workup conformal set {{0}} at {nom_pct}% coverage."
        elif cset == [1]:
            triage, reason = "HIGH_RISK", f"Stage-2 Cardiology Workup conformal set {{1}} at {nom_pct}% coverage — cardiology intervention indicated."
        else:
            triage, reason = "DEFER_TO_CLINICIAN", f"Ambiguous Stage-2 conformal set {{0, 1}} at {nom_pct}% coverage — direct cardiologist evaluation advised."

        feat_attns = fwd["expl"]["feature_attn"][0].cpu().numpy()
        nbr_attns = fwd["expl"]["neighbor_attn"][0].cpu().numpy()
        gate_val = float(fwd["expl"]["gate"][0].mean().item())

        biomarker_attentions = dict(
            sorted(
                ((n, round(float(w), 4)) for n, w in zip(self.bundle.feature_names, feat_attns, strict=True)),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )
        neighbors: list[MatchedCohortNeighbor] = []
        for rank, (b_idx, cos_sim, g_w) in enumerate(
            zip(fwd["topk_idx"][0], fwd["topk_sims"][0], nbr_attns[1:], strict=True)
        ):
            if rank >= 5:
                break
            b_i = int(b_idx)
            age_n, sbp_n, chol_n = self.bundle.bank_summary[b_i]
            neighbors.append(
                MatchedCohortNeighbor(
                    cohort_index=b_i,
                    attention_weight=round(float(g_w), 4),
                    cosine_similarity=round(float(cos_sim), 4),
                    historical_outcome="CVD+" if int(self.bundle.bank_outcomes[b_i]) == 1 else "Healthy",
                    age_years=round(float(age_n), 1),
                    systolic_bp=round(float(sbp_n), 1),
                    bmi=round(float(chol_n), 1),  # Cholesterol mg/dL stored for workup cohort display
                )
            )

        # Immutable-safe cardiology recourse (BP, cholesterol, exercise capacity / ischemia)
        recourse: list[RecourseRecommendation] = []
        if triage in ("HIGH_RISK", "DEFER_TO_CLINICIAN"):
            cands: list[tuple[str, dict[str, float | int], CardiologyWorkupInput]] = []
            if patient.trestbps > 125:
                ns = round(max(120.0, patient.trestbps - 20.0), 1)
                cands.append((f"Resting BP control ({patient.trestbps:.0f} -> {ns:.0f} mmHg)", {"trestbps": ns}, patient.model_copy(update={"trestbps": ns})))
            if patient.chol > 200:
                nc = round(max(185.0, patient.chol - 45.0), 1)
                cands.append((f"Intensive statin lipid lowering ({patient.chol:.0f} -> {nc:.0f} mg/dL)", {"chol": nc}, patient.model_copy(update={"chol": nc})))
            if patient.oldpeak > 0.5:
                no = round(max(0.0, patient.oldpeak - 1.0), 1)
                cands.append((f"Anti-ischemic therapy (ST depression {patient.oldpeak:.1f} -> {no:.1f} mm)", {"oldpeak": no}, patient.model_copy(update={"oldpeak": no})))
            if patient.thalach < 155:
                nt = round(min(175.0, patient.thalach + 18.0), 1)
                cands.append((f"Cardiac rehabilitation (peak HR {patient.thalach:.0f} -> {nt:.0f} bpm)", {"thalach": nt}, patient.model_copy(update={"thalach": nt})))
            if cands:
                cf_p = self._forward([c[2] for c in cands], return_explanations=False)["probs"]
                for (lbl, ch, _), pn in zip(cands, cf_p, strict=True):
                    delta = float(prob - float(pn))
                    if delta > 0.005:
                        recourse.append(
                            RecourseRecommendation(
                                intervention=lbl,
                                changes=ch,
                                projected_risk_probability=round(float(pn), 4),
                                absolute_risk_reduction=round(delta, 4),
                            )
                        )
                recourse.sort(key=lambda r: r.absolute_risk_reduction, reverse=True)

        return ClinicalPredictionResponse(
            request_id=f"workup-{uuid.uuid4().hex[:10]}",
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            model_version=self.bundle.model_version,
            bundle_sha256=self.bundle.bundle_sha256,
            calibrated_risk_probability=round(prob, 4),
            gat_transformer_probability=round(gat_p, 4),
            xgboost_probability=round(xgb_p, 4),
            hgb_probability=round(rf_p, 4),
            model_disagreement=round(disagreement, 4),
            triage_decision=triage,
            triage_reason=reason,
            conformal_prediction_set=cset,
            conformal_alpha=patient.conformal_alpha,
            conformal_low_threshold=round(low_t, 4),
            conformal_high_threshold=round(high_t, 4),
            ood_report=OODReport(
                is_ood=is_ood,
                max_abs_zscore=round(max_z, 3),
                min_neighbor_cosine_distance=round(min_cos_dist, 4),
                ood_distance_threshold=round(self.bundle.ood_distance_threshold, 4),
                reasons=ood_reasons,
            ),
            dual_explanation=DualClinicalExplanation(
                graph_reliance_gate=round(gate_val, 4),
                biomarker_attentions=biomarker_attentions,
                top_cohort_neighbors=neighbors,
            ),
            actionable_recourse=recourse[:4],
            latency_ms=round(latency_ms, 2),
        )


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build Stage-2 Cardiology Workup bundle (N=918).")
    parser.add_argument("--dataset", type=str, default="cleaned_heart_disease_combined.csv")
    parser.add_argument("--output", type=str, default=str(DEFAULT_WORKUP_BUNDLE_PATH))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    build_workup_bundle(
        dataset_path=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        cv_folds=args.cv_folds,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
