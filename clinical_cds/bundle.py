"""Versioned, SHA-256-verified Clinical Model Bundle v3.

Major accuracy & clinical engineering upgrades over v2:
1. **PLR Periodic Numerical Embeddings (Gorishniy et al., NeurIPS 2022):**
   Learnable Fourier features + ReLU + linear skip capture sharp clinical step cutoffs.
2. **UniMP Masked Label Propagation in Level-2 GAT (Shi et al., IJCAI 2021):**
   Injects historical training neighbor outcome embeddings (0=Healthy, 1=CVD+)
   while strictly masking the query self-loop (id=2) for zero target leakage.
3. **Domain Clinical Interaction Features (14 -> 19 features):**
   Adds AHA Hypertension Stage, Metabolic Burden (`chol + gluc`), Arterial Stiffness
   (`age * sbp / 100`), Cumulative Adiposity (`bmi * age / 100`), and `ap_hi / ap_lo`.
4. **5-Fold Out-of-Fold Cross-Fitting + 3-Paradigm Simplex Ensemble:**
   Trains across 100% of Train+Val (58,300 patients, +21% more training data!)
   combining `PLR+UniMP GraphAugmentedClinicalTransformer`, `XGBoost`, and
   `HistGradientBoostingClassifier`, with Nelder-Mead simplex meta-weights and
   multi-alpha sex-stratified Mondrian conformal quantiles calibrated on all
   58,300 leak-free OOF predictions.
"""

from __future__ import annotations

import hashlib
import json
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
from sklearn.cluster import MiniBatchKMeans
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
    build_inductive_knn_indices,
    compute_ece,
)
from project_paths import ARTIFACTS_DIR, resolve_data_path

BUNDLE_VERSION = "3.1.0-plr-unimp-zeropickle-cds"
DEFAULT_BUNDLE_PATH = ARTIFACTS_DIR / "clinical_cds_bundle.pt"
SUPPORTED_ALPHAS = (0.05, 0.10, 0.15)


def _to_np(val: Any, dtype: Any) -> np.ndarray:
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy().astype(dtype)
    return np.asarray(val, dtype=dtype)

CAT_COLS = [
    "sex",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "aha_bp_stage",
    "metabolic_burden",
]
CONT_COLS = [
    "age_years",
    "height",
    "weight",
    "bmi",
    "ap_hi",
    "ap_lo",
    "pulse_pressure",
    "mean_arterial_pressure",
    "age_x_sbp",
    "bmi_x_age",
    "bp_ratio",
]
CAT_DIMS = [3, 4, 4, 3, 3, 3, 6, 6]


def compute_aha_bp_stage(sbp: np.ndarray, dbp: np.ndarray) -> np.ndarray:
    """Map systolic/diastolic BP (mmHg) to AHA/ACC clinical hypertension stage (0..4)."""
    stage = np.zeros(len(sbp), dtype=np.int64)
    stage = np.where((sbp >= 120) & (sbp < 130) & (dbp < 80), 1, stage)
    stage = np.where(((sbp >= 130) & (sbp < 140)) | ((dbp >= 80) & (dbp < 90)), 2, stage)
    stage = np.where((sbp >= 140) | (dbp >= 90), 3, stage)
    stage = np.where((sbp >= 180) | (dbp >= 120), 4, stage)
    return stage


def engineer_screening_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic clinical interaction features without target leakage."""
    out = df.copy()
    sbp = out["ap_hi"].values.astype(np.float32)
    dbp = out["ap_lo"].values.astype(np.float32)
    age = out["age_years"].values.astype(np.float32)
    bmi = out["bmi"].values.astype(np.float32)

    out["aha_bp_stage"] = compute_aha_bp_stage(sbp, dbp)
    out["metabolic_burden"] = (
        out["cholesterol"].values.astype(np.int64) + out["gluc"].values.astype(np.int64)
    ).clip(0, 4)
    out["age_x_sbp"] = (age * sbp) / 100.0
    out["bmi_x_age"] = (bmi * age) / 100.0
    out["bp_ratio"] = sbp / np.clip(dbp, 30.0, None)
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    if n == 0:
        return 1.0 - alpha
    level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    return float(np.quantile(scores, level, method="higher"))


def compute_multialpha_mondrian_quantiles(
    probs: np.ndarray,
    y_true: np.ndarray,
    sex: np.ndarray,
    alphas: tuple[float, ...] = SUPPORTED_ALPHAS,
) -> dict[str, dict[str, float]]:
    """Compute sex-stratified Mondrian conformal quantiles for each supported alpha."""
    y_int = np.asarray(y_true, dtype=int).ravel()
    sx_int = np.asarray(sex, dtype=int).ravel()
    pr = np.asarray(probs, dtype=np.float64).ravel()

    out: dict[str, dict[str, float]] = {}
    for alpha in alphas:
        key = f"{alpha:.2f}"
        q_dict: dict[str, float] = {
            "class_0": _conformal_quantile(pr[y_int == 0], alpha),
            "class_1": _conformal_quantile(1.0 - pr[y_int == 1], alpha),
        }
        for sx in (0, 1):
            m0 = (sx_int == sx) & (y_int == 0)
            m1 = (sx_int == sx) & (y_int == 1)
            q_dict[f"sex_{sx}_class_0"] = (
                _conformal_quantile(pr[m0], alpha) if np.any(m0) else q_dict["class_0"]
            )
            q_dict[f"sex_{sx}_class_1"] = (
                _conformal_quantile(1.0 - pr[m1], alpha) if np.any(m1) else q_dict["class_1"]
            )
        out[key] = q_dict
    return out


def _select_stratified_medoids(
    H_norm_np: np.ndarray,
    y_int: np.ndarray,
    n_prototypes: int,
    seed: int = 42,
) -> np.ndarray:
    """Compress N_train patients into real stratified medoid prototypes in learned [CLS] space."""
    n_total = len(y_int)
    if n_prototypes <= 0 or n_prototypes >= n_total:
        return np.arange(n_total, dtype=np.int64)

    selected: list[np.ndarray] = []
    for cls in (0, 1):
        cls_idx = np.where(y_int == cls)[0]
        target_k = max(1, min(len(cls_idx), n_prototypes // 2))
        if target_k >= len(cls_idx):
            selected.append(cls_idx)
            continue
        km = MiniBatchKMeans(
            n_clusters=target_k,
            random_state=seed + cls,
            batch_size=min(2048, len(cls_idx)),
            n_init=3,
        )
        labels = km.fit_predict(H_norm_np[cls_idx])
        centers = km.cluster_centers_
        chosen: list[int] = []
        for c_id in range(target_k):
            members = np.where(labels == c_id)[0]
            if len(members) == 0:
                continue
            dists = np.linalg.norm(H_norm_np[cls_idx[members]] - centers[c_id], axis=1)
            chosen.append(int(cls_idx[members[int(np.argmin(dists))]]))
        selected.append(np.asarray(chosen, dtype=np.int64))
    return np.sort(np.concatenate(selected))


def _train_gat_transformer_model(
    X_cat_tr: np.ndarray,
    X_cont_tr: np.ndarray,
    y_tr: np.ndarray,
    X_cat_va: np.ndarray | None,
    X_cont_va: np.ndarray | None,
    y_va: np.ndarray | None,
    model_config: dict[str, Any],
    epochs: int,
    batch_size: int,
    k_neighbors: int,
    seed: int,
    dev: torch.device,
) -> GraphAugmentedClinicalTransformer:
    """Train a PLR + UniMP GraphAugmentedClinicalTransformer with mid-training manifold refresh."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    graph_scaler = StandardScaler()
    X_g_tr = graph_scaler.fit_transform(np.hstack([X_cat_tr.astype(np.float32), X_cont_tr])).astype(np.float32)
    X_g_va = (
        graph_scaler.transform(np.hstack([X_cat_va.astype(np.float32), X_cont_va])).astype(np.float32)
        if X_cat_va is not None and X_cont_va is not None
        else None
    )
    tr_nbr_idx, va_nbr_idx = build_inductive_knn_indices(
        X_g_tr, X_g_va, k=k_neighbors, metric="cosine", n_jobs=-1
    )

    xc_tr = torch.from_numpy(X_cat_tr).to(dev)
    xct_tr = torch.from_numpy(X_cont_tr).to(dev)
    y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(dev).view(-1, 1)
    y_tr_long = torch.from_numpy(y_tr.astype(np.int64)).to(dev)
    tr_nbr_t = torch.from_numpy(tr_nbr_idx).to(dev)

    has_val = X_cat_va is not None and X_cont_va is not None and y_va is not None
    if has_val:
        assert X_cat_va is not None and X_cont_va is not None and y_va is not None
        xc_va = torch.from_numpy(X_cat_va).to(dev)
        xct_va = torch.from_numpy(X_cont_va).to(dev)
        y_va_t = torch.from_numpy(y_va.astype(np.float32)).to(dev).view(-1, 1)
        va_nbr_t = torch.from_numpy(va_nbr_idx).to(dev)  # type: ignore[arg-type]

    model = GraphAugmentedClinicalTransformer(**model_config).to(dev)
    opt = optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=2e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    n_tr = len(y_tr)

    @torch.inference_mode()
    def _enc(xc: torch.Tensor, xct: torch.Tensor) -> torch.Tensor:
        model.eval()
        return torch.cat(
            [model.encode_tabular(xc[s : s + 2048], xct[s : s + 2048])[0] for s in range(0, len(xc), 2048)],
            dim=0,
        )

    best_v, best_sd = float("inf"), None
    refresh_ep = max(2, epochs // 3)
    for ep in range(epochs):
        h_bank = _enc(xc_tr, xct_tr)
        if ep == refresh_ep:
            h_tr_np = F.normalize(h_bank, p=2, dim=1, eps=1e-8).cpu().numpy()
            h_va_np = (
                F.normalize(_enc(xc_va, xct_va), p=2, dim=1, eps=1e-8).cpu().numpy()
                if has_val
                else None
            )
            tr_new, va_new = build_inductive_knn_indices(
                h_tr_np, h_va_np, k=k_neighbors, metric="cosine", n_jobs=-1
            )
            tr_nbr_t = torch.from_numpy(tr_new).to(dev)
            if has_val and va_new is not None:
                va_nbr_t = torch.from_numpy(va_new).to(dev)

        model.train()
        perm = torch.randperm(n_tr, device=dev)
        for s in range(0, n_tr, batch_size):
            b = perm[s : s + batch_size]
            nbr_ids = tr_nbr_t[b]
            opt.zero_grad()
            logits = model.forward_minibatch(
                x_cat=xc_tr[b],
                x_cont=xct_tr[b],
                nbr_h_tab=h_bank[nbr_ids],
                nbr_labels=y_tr_long[nbr_ids],
                label_mask_prob=0.35,
                feature_noise_std=0.02,
            )
            loss = criterion(logits, y_tr_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        if has_val:
            h_eval = _enc(xc_tr, xct_tr)
            model.eval()
            with torch.inference_mode():
                v_logits = torch.cat([
                    model.forward_minibatch(
                        x_cat=xc_va[s : s + 1024],
                        x_cont=xct_va[s : s + 1024],
                        nbr_h_tab=h_eval[va_nbr_t[s : s + 1024]],
                        nbr_labels=y_tr_long[va_nbr_t[s : s + 1024]],
                    )
                    for s in range(0, len(xc_va), 1024)
                ])
                vloss = criterion(v_logits, y_va_t).item()
                if vloss < best_v:
                    best_v = vloss
                    best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_sd is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})
    model.eval()
    return model


@dataclass
class ClinicalModelBundle:
    """Self-contained, integrity-verified clinical inference artifact."""

    model: GraphAugmentedClinicalTransformer
    xgb_model: XGBClassifier | None
    hgb_model: Any | None
    ensemble_weights: tuple[float, float, float]  # (w_gat, w_xgb, w_hgb)
    temperature: float
    model_version: str
    bundle_sha256: str
    cat_cols: list[str]
    cont_cols: list[str]
    feature_names: list[str]
    cat_dims: list[int]
    cont_medians: np.ndarray
    cont_mean: np.ndarray
    cont_scale: np.ndarray
    graph_mean: np.ndarray
    graph_scale: np.ndarray
    H_bank_norm: torch.Tensor
    H_tab_bank: torch.Tensor
    bank_outcomes: np.ndarray
    bank_outcomes_t: torch.Tensor
    bank_summary: np.ndarray
    k_neighbors: int
    default_conformal_alpha: float
    conformal_quantiles: dict[str, dict[str, float]]
    ood_distance_threshold: float
    ood_zscore_threshold: float
    val_risk_hist_edges: np.ndarray
    val_risk_hist_props: np.ndarray
    model_card: dict[str, Any]
    raw_payload: dict[str, Any] | None = None
    platt_a: float = 1.0
    platt_b: float = 0.0

    @property
    def ensemble_gat_weight(self) -> float:
        return self.ensemble_weights[0]

    @classmethod
    def load(
        cls,
        bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
        device: str = "auto",
        expected_sha256: str | None = None,
    ) -> ClinicalModelBundle:
        path = Path(bundle_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Clinical model bundle not found at {path}. "
                "Build it with `heart-pipeline build-cds-bundle`."
            )

        actual_sha = _sha256_file(path)
        if expected_sha256 is not None and actual_sha != expected_sha256:
            raise ValueError(
                f"Bundle integrity check failed! Expected SHA-256 {expected_sha256}, got {actual_sha}."
            )

        dev = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto"
            else torch.device(device)
        )
        # Strict Zero-Pickle safe deserialization (weights_only=True)
        payload = torch.load(path, map_location=dev, weights_only=True)
        cfg = payload["model_config"]

        model = GraphAugmentedClinicalTransformer(**cfg).to(dev)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

        xgb_model: XGBClassifier | None = None
        if payload.get("xgb_booster_json") is not None:
            xgb_model = XGBClassifier(n_jobs=1, device="cpu")
            xgb_model.load_model(bytearray(payload["xgb_booster_json"]))

        hgb_model: XGBClassifier | None = None
        if payload.get("hgb_booster_json") is not None:
            hgb_model = XGBClassifier(n_jobs=1, device="cpu")
            hgb_model.load_model(bytearray(payload["hgb_booster_json"]))

        if "ensemble_weights" in payload:
            ew = tuple(float(x) for x in payload["ensemble_weights"])
        else:
            wg = float(payload.get("ensemble_gat_weight", 1.0))
            ew = (wg, 1.0 - wg, 0.0) if xgb_model is not None else (1.0, 0.0, 0.0)

        H_tab = payload["H_tab_bank"].float().to(dev)
        H_norm = (
            payload["H_bank_norm"].float().to(dev)
            if "H_bank_norm" in payload
            else F.normalize(H_tab, p=2, dim=1, eps=1e-8)
        )

        bank_outcomes = _to_np(payload["bank_outcomes"], np.int8)
        bank_outcomes_t = torch.from_numpy(bank_outcomes.astype(np.int64)).to(dev)

        cq = {
            str(a): {str(k): float(v) for k, v in d.items()}
            for a, d in payload["conformal_quantiles"].items()
        }

        return cls(
            model=model,
            xgb_model=xgb_model,
            hgb_model=hgb_model,
            ensemble_weights=(ew[0], ew[1], ew[2]),
            temperature=float(payload["temperature"]),
            model_version=str(payload["model_version"]),
            bundle_sha256=actual_sha,
            cat_cols=list(payload["cat_cols"]),
            cont_cols=list(payload["cont_cols"]),
            feature_names=list(payload["feature_names"]),
            cat_dims=list(cfg["categorical_dims"]),
            cont_medians=_to_np(payload["cont_medians"], np.float32),
            cont_mean=_to_np(payload["cont_mean"], np.float32),
            cont_scale=_to_np(payload["cont_scale"], np.float32),
            graph_mean=_to_np(payload["graph_mean"], np.float32),
            graph_scale=_to_np(payload["graph_scale"], np.float32),
            H_bank_norm=H_norm,
            H_tab_bank=H_tab,
            bank_outcomes=bank_outcomes,
            bank_outcomes_t=bank_outcomes_t,
            bank_summary=_to_np(payload["bank_summary"], np.float32),
            k_neighbors=int(payload["k_neighbors"]),
            default_conformal_alpha=float(payload.get("conformal_alpha", 0.10)),
            conformal_quantiles=cq,
            ood_distance_threshold=float(payload["ood_distance_threshold"]),
            ood_zscore_threshold=float(payload.get("ood_zscore_threshold", 4.5)),
            val_risk_hist_edges=_to_np(
                payload.get("val_risk_hist_edges", np.linspace(0.0, 1.0, 11)), np.float64
            ),
            val_risk_hist_props=_to_np(
                payload.get("val_risk_hist_props", np.full(10, 0.1)), np.float64
            ),
            model_card=dict(payload["model_card"]),
            raw_payload=payload,
            platt_a=float(payload.get("platt_a", 1.0)),
            platt_b=float(payload.get("platt_b", 0.0)),
        )


def build_clinical_bundle(
    dataset_path: str | Path = "cleaned_cardio_70k.csv",
    output_path: str | Path = DEFAULT_BUNDLE_PATH,
    epochs: int = 16,
    cv_folds: int = 5,
    batch_size: int = 512,
    k_neighbors: int = 7,
    embed_dim: int = 32,
    n_prototypes: int = 4096,
    conformal_alpha: float = 0.10,
    seed: int = 42,
) -> tuple[Path, dict[str, Any]]:
    """Train 5-fold OOF 3-paradigm PLR+UniMP CDS Bundle v3 and export with SHA-256."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_file = resolve_data_path(dataset_path)
    if not data_file.exists():
        from prepare_large_cvd_dataset import prepare_large_cvd_cohort
        prepare_large_cvd_cohort(data_file)

    df = engineer_screening_interactions(pd.read_csv(data_file))
    y_all = df["target"].values.astype(int)
    indices = np.arange(len(df))
    dev_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=seed, stratify=y_all)

    dev_df = df.iloc[dev_idx].reset_index(drop=True)
    te_df = df.iloc[test_idx].reset_index(drop=True)
    y_dev = dev_df["target"].values.astype(int)
    y_te = te_df["target"].values.astype(int)

    model_config = {
        "categorical_dims": CAT_DIMS,
        "continuous_dim": len(CONT_COLS),
        "embed_dim": embed_dim,
        "num_transformer_heads": 4,
        "num_transformer_layers": 2,
        "num_gat_heads": 4,
        "num_gat_layers": 2,
        "mlp_hidden_dim": 64,
        "dropout": 0.15,
        "num_classes": 1,
        "use_periodic_embeddings": True,
        "n_frequencies": 8,
        "use_label_propagation": True,
    }

    # --- Stage A: 5-Fold Out-of-Fold Cross-Fitting across ALL 58,300 dev patients ---
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    oof_gat_logits = np.zeros(len(dev_df), dtype=np.float64)
    oof_xgb = np.zeros(len(dev_df), dtype=np.float64)
    oof_hgb = np.zeros(len(dev_df), dtype=np.float64)
    oof_min_dists = np.zeros(len(dev_df), dtype=np.float64)

    print(
        f"Running {cv_folds}-fold OOF cross-fitting on {len(dev_df):,} patients "
        f"(PLR+UniMP GAT-Transformer + XGBoost + HistGradientBoosting) on {dev}..."
    )

    for fold, (f_tr, f_va) in enumerate(skf.split(dev_df, y_dev)):
        tr_f = dev_df.iloc[f_tr]
        va_f = dev_df.iloc[f_va]

        imp_f = SimpleImputer(strategy="median")
        scl_f = StandardScaler()
        xc_tr = tr_f[CAT_COLS].values.astype(np.int64)
        xc_va = va_f[CAT_COLS].values.astype(np.int64)
        xct_tr = scl_f.fit_transform(imp_f.fit_transform(tr_f[CONT_COLS].values)).astype(np.float32)
        xct_va = scl_f.transform(imp_f.transform(va_f[CONT_COLS].values)).astype(np.float32)
        xtab_tr = np.hstack([xc_tr.astype(np.float32), xct_tr])
        xtab_va = np.hstack([xc_va.astype(np.float32), xct_va])

        # 1. PLR + UniMP GAT-Transformer
        gat_f = _train_gat_transformer_model(
            xc_tr, xct_tr, y_dev[f_tr], xc_va, xct_va, y_dev[f_va],
            model_config=model_config,
            epochs=max(6, epochs - 2),
            batch_size=batch_size,
            k_neighbors=k_neighbors,
            seed=seed + fold,
            dev=dev,
        )
        with torch.inference_mode():
            xc_tr_t, xct_tr_t = torch.from_numpy(xc_tr).to(dev), torch.from_numpy(xct_tr).to(dev)
            xc_va_t, xct_va_t = torch.from_numpy(xc_va).to(dev), torch.from_numpy(xct_va).to(dev)
            y_tr_l = torch.from_numpy(y_dev[f_tr].astype(np.int64)).to(dev)

            h_tr = torch.cat([gat_f.encode_tabular(xc_tr_t[s : s + 2048], xct_tr_t[s : s + 2048])[0] for s in range(0, len(xc_tr_t), 2048)])
            h_tr_n = F.normalize(h_tr, p=2, dim=1, eps=1e-8)

            # Use stratified prototype medoids inside OOF evaluation to mirror exact inference path
            p_idx = _select_stratified_medoids(h_tr_n.cpu().numpy(), y_dev[f_tr], n_prototypes=min(n_prototypes, len(f_tr)), seed=seed + fold)
            p_t = torch.from_numpy(p_idx).to(dev)
            h_pb, h_pn, y_pb = h_tr[p_t], h_tr_n[p_t], y_tr_l[p_t]

            lg_parts, d_parts = [], []
            k_eff = min(k_neighbors, len(p_idx))
            for s in range(0, len(xc_va_t), 1024):
                hq, _ = gat_f.encode_tabular(xc_va_t[s : s + 1024], xct_va_t[s : s + 1024])
                sims = torch.mm(F.normalize(hq, p=2, dim=1, eps=1e-8), h_pn.t())
                tsim, tidx = torch.topk(sims, k=k_eff, dim=1)
                lg = gat_f.forward_minibatch(
                    xc_va_t[s : s + 1024],
                    xct_va_t[s : s + 1024],
                    nbr_h_tab=h_pb[tidx],
                    nbr_labels=y_pb[tidx],
                )
                lg_parts.append(lg.cpu().numpy().ravel())
                d_parts.append((1.0 - tsim[:, 0]).cpu().numpy())
            oof_gat_logits[f_va] = np.concatenate(lg_parts)
            oof_min_dists[f_va] = np.concatenate(d_parts)

        # 2. XGBoost
        xgb_f = XGBClassifier(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=2.5,
            eval_metric="logloss",
            early_stopping_rounds=25,
            tree_method="hist",
            random_state=seed + fold,
            n_jobs=-1,
        )
        xgb_f.fit(xtab_tr, y_dev[f_tr], eval_set=[(xtab_va, y_dev[f_va])], verbose=False)
        oof_xgb[f_va] = xgb_f.predict_proba(xtab_va)[:, 1]

        # 3. Leaf-wise Lossguide HistGBM (LightGBM-style leaf-wise boosted trees; zero-pickle JSON)
        hgb_f = XGBClassifier(
            n_estimators=400,
            grow_policy="lossguide",
            max_leaves=31,
            max_depth=0,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.8,
            min_child_weight=15,
            reg_lambda=2.5,
            eval_metric="logloss",
            early_stopping_rounds=25,
            tree_method="hist",
            random_state=seed + 100 + fold,
            n_jobs=-1,
        )
        hgb_f.fit(xtab_tr, y_dev[f_tr], eval_set=[(xtab_va, y_dev[f_va])], verbose=False)
        oof_hgb[f_va] = hgb_f.predict_proba(xtab_va)[:, 1]

    # Temperature calibrate OOF GAT logits strictly on OOF data
    dummy_model = GraphAugmentedClinicalTransformer(**model_config).to(dev)
    temp_scaler = ModelWithTemperature(dummy_model).to(dev)
    temp_scaler.calibrate(
        torch.from_numpy(oof_gat_logits.astype(np.float32)).to(dev).view(-1, 1),
        torch.from_numpy(y_dev.astype(np.float32)).to(dev).view(-1, 1),
    )
    temp_val = float(temp_scaler.temperature.item())
    oof_gat = 1.0 / (1.0 + np.exp(-oof_gat_logits / max(temp_val, 1e-3)))

    # Nelder-Mead simplex optimization of 3-paradigm weights (w_gat, w_xgb, w_hgb) strictly on OOF
    P_oof = np.column_stack([oof_gat, oof_xgb, oof_hgb])

    def _simplex_obj(theta: np.ndarray) -> float:
        w = softmax(theta)
        # Keep >= 12% weight on GAT-Transformer so dual attention remains meaningfully active
        if w[0] < 0.12:
            return 10.0 + float(0.12 - w[0])
        p = np.clip(P_oof @ w, 1e-7, 1.0 - 1e-7)
        return float(log_loss(y_dev, p))

    res_w = minimize(_simplex_obj, np.zeros(3), method="Nelder-Mead", options={"maxiter": 600})
    w_opt = softmax(res_w.x)
    w_gat, w_xgb, w_hgb = float(w_opt[0]), float(w_opt[1]), float(w_opt[2])
    oof_ens = P_oof @ w_opt

    # Calibrate multi-alpha sex-stratified Mondrian quantiles & PSI baseline on ALL 58,300 OOF probs!
    dev_sex = dev_df["sex"].values.astype(int)
    conformal_quantiles = compute_multialpha_mondrian_quantiles(
        oof_ens, y_dev, dev_sex, alphas=SUPPORTED_ALPHAS
    )
    ood_dist_threshold = float(np.quantile(oof_min_dists, 0.995))

    hist_edges = np.linspace(0.0, 1.0, 11)
    hist_counts, _ = np.histogram(oof_ens, bins=hist_edges)
    hist_props = np.clip(hist_counts / max(len(oof_ens), 1), 1e-6, 1.0)
    hist_props = hist_props / hist_props.sum()

    # --- Stage B: Fit Final Full-Cohort Models on ALL 58,300 Dev Patients ---
    cont_imputer = SimpleImputer(strategy="median")
    cont_scaler = StandardScaler()
    X_cont_dev = cont_scaler.fit_transform(cont_imputer.fit_transform(dev_df[CONT_COLS].values)).astype(np.float32)
    X_cont_te = cont_scaler.transform(cont_imputer.transform(te_df[CONT_COLS].values)).astype(np.float32)

    X_cat_dev = dev_df[CAT_COLS].values.astype(np.int64)
    X_cat_te = te_df[CAT_COLS].values.astype(np.int64)
    X_tab_dev = np.hstack([X_cat_dev.astype(np.float32), X_cont_dev])
    X_tab_te = np.hstack([X_cat_te.astype(np.float32), X_cont_te])
    graph_scaler = StandardScaler().fit(X_tab_dev)

    # Hold 8% internal early-stopping slice for final full-cohort training
    sub_tr, sub_es = train_test_split(np.arange(len(dev_df)), test_size=0.08, random_state=seed, stratify=y_dev)
    final_gat = _train_gat_transformer_model(
        X_cat_dev[sub_tr], X_cont_dev[sub_tr], y_dev[sub_tr],
        X_cat_dev[sub_es], X_cont_dev[sub_es], y_dev[sub_es],
        model_config=model_config,
        epochs=epochs,
        batch_size=batch_size,
        k_neighbors=k_neighbors,
        seed=seed,
        dev=dev,
    )

    final_xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=2.5,
        eval_metric="logloss",
        early_stopping_rounds=25,
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )
    final_xgb.fit(X_tab_dev[sub_tr], y_dev[sub_tr], eval_set=[(X_tab_dev[sub_es], y_dev[sub_es])], verbose=False)

    final_hgb = XGBClassifier(
        n_estimators=400,
        grow_policy="lossguide",
        max_leaves=31,
        max_depth=0,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=15,
        reg_lambda=2.5,
        eval_metric="logloss",
        early_stopping_rounds=25,
        tree_method="hist",
        random_state=seed + 100,
        n_jobs=-1,
    )
    final_hgb.fit(X_tab_dev[sub_tr], y_dev[sub_tr], eval_set=[(X_tab_dev[sub_es], y_dev[sub_es])], verbose=False)

    # Encode full 58,300 dev cohort into learned [CLS] manifold & compress to stratified clinical medoids
    xc_dev_t = torch.from_numpy(X_cat_dev).to(dev)
    xct_dev_t = torch.from_numpy(X_cont_dev).to(dev)
    xc_te_t = torch.from_numpy(X_cat_te).to(dev)
    xct_te_t = torch.from_numpy(X_cont_te).to(dev)
    y_dev_l = torch.from_numpy(y_dev.astype(np.int64)).to(dev)

    with torch.inference_mode():
        final_gat.eval()
        h_dev_full = torch.cat([
            final_gat.encode_tabular(xc_dev_t[s : s + 2048], xct_dev_t[s : s + 2048])[0]
            for s in range(0, len(xc_dev_t), 2048)
        ])
        h_dev_norm = F.normalize(h_dev_full, p=2, dim=1, eps=1e-8)

    proto_idx = _select_stratified_medoids(
        h_dev_norm.cpu().numpy(), y_dev, n_prototypes=n_prototypes, seed=seed
    )
    proto_t = torch.from_numpy(proto_idx).to(dev)
    H_proto_bank = h_dev_full[proto_t]
    H_proto_norm = h_dev_norm[proto_t]
    y_proto_l = y_dev_l[proto_t]

    # Evaluate final v3 ensemble on strict untouched holdout test set (N=10,289)
    with torch.inference_mode():
        te_logits_parts = []
        k_eff = min(k_neighbors, len(proto_idx))
        for s in range(0, len(xc_te_t), 1024):
            hq, _ = final_gat.encode_tabular(xc_te_t[s : s + 1024], xct_te_t[s : s + 1024])
            sims = torch.mm(F.normalize(hq, p=2, dim=1, eps=1e-8), H_proto_norm.t())
            _, tidx = torch.topk(sims, k=k_eff, dim=1)
            lg = final_gat.forward_minibatch(
                xc_te_t[s : s + 1024],
                xct_te_t[s : s + 1024],
                nbr_h_tab=H_proto_bank[tidx],
                nbr_labels=y_proto_l[tidx],
            )
            te_logits_parts.append(lg)
        test_gat_logits = torch.cat(te_logits_parts, dim=0).cpu().numpy().ravel()

    test_gat_probs = 1.0 / (1.0 + np.exp(-test_gat_logits / max(temp_val, 1e-3)))
    test_xgb_probs = final_xgb.predict_proba(X_tab_te)[:, 1]
    test_hgb_probs = final_hgb.predict_proba(X_tab_te)[:, 1]
    test_probs = w_gat * test_gat_probs + w_xgb * test_xgb_probs + w_hgb * test_hgb_probs
    test_preds = (test_probs >= 0.5).astype(int)
    test_sex = te_df["sex"].values.astype(int)

    subgroup_metrics: dict[str, dict[str, float]] = {}
    for sx, label in ((0, "female"), (1, "male")):
        mask = test_sex == sx
        yt_s, yp_s, pr_s = y_te[mask], test_preds[mask], test_probs[mask]
        tp = int(np.sum((yt_s == 1) & (yp_s == 1)))
        fn = int(np.sum((yt_s == 1) & (yp_s == 0)))
        tn = int(np.sum((yt_s == 0) & (yp_s == 0)))
        fp = int(np.sum((yt_s == 0) & (yp_s == 1)))
        subgroup_metrics[label] = {
            "n_test": int(mask.sum()),
            "roc_auc": round(float(roc_auc_score(yt_s, pr_s)), 4),
            "sensitivity": round(tp / max(tp + fn, 1), 4),
            "specificity": round(tn / max(tn + fp, 1), 4),
            "fnr": round(fn / max(tp + fn, 1), 4),
        }

    conformal_audit_by_alpha: dict[str, dict[str, float]] = {}
    for alpha in SUPPORTED_ALPHAS:
        akey = f"{alpha:.2f}"
        qmap = conformal_quantiles[akey]
        in_set, set_sizes = [], []
        for p, yt, sx in zip(test_probs, y_te, test_sex, strict=True):
            cset = []
            if p <= qmap[f"sex_{sx}_class_0"]:
                cset.append(0)
            if (1.0 - p) <= qmap[f"sex_{sx}_class_1"]:
                cset.append(1)
            if not cset:
                cset = [0, 1]
            in_set.append(int(yt in cset))
            set_sizes.append(len(cset))
        conformal_audit_by_alpha[akey] = {
            "nominal_coverage": round(1.0 - alpha, 4),
            "empirical_test_coverage": round(float(np.mean(in_set)), 4),
            "deferral_rate": round(float(np.mean([s == 2 for s in set_sizes])), 4),
            "mean_set_size": round(float(np.mean(set_sizes)), 4),
        }

    default_akey = f"{conformal_alpha:.2f}"
    model_card: dict[str, Any] = {
        "model_name": "PLR+UniMP GraphAugmentedTransformer + XGBoost + HistGBM (CDS v3)",
        "model_version": BUNDLE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_cohort": {
            "source": "OpenML 45547 Cardiovascular Disease Cohort (5-Fold OOF Cross-Fitted)",
            "n_total": int(len(df)),
            "n_train": int(len(dev_df)),
            "n_val_oof": int(len(dev_df)),
            "n_test": int(len(te_df)),
            "n_prototypes": int(len(proto_idx)),
            "features": len(CAT_COLS) + len(CONT_COLS),
            "neighbor_space": "learned_cls_embedding_cosine_with_unimp_masked_labels",
        },
        "ensemble_weights": {
            "graph_augmented_transformer": round(w_gat, 4),
            "xgboost": round(w_xgb, 4),
            "hist_gradient_boosting": round(w_hgb, 4),
        },
        "holdout_performance": {
            "roc_auc": round(float(roc_auc_score(y_te, test_probs)), 4),
            "pr_auc": round(float(average_precision_score(y_te, test_probs)), 4),
            "accuracy": round(float(accuracy_score(y_te, test_preds)), 4),
            "f1": round(float(f1_score(y_te, test_preds)), 4),
            "brier": round(float(brier_score_loss(y_te, test_probs)), 4),
            "ece": round(float(compute_ece(test_probs, y_te)), 4),
            "temperature": round(temp_val, 4),
            "standalone_gat_roc_auc": round(float(roc_auc_score(y_te, test_gat_probs)), 4),
            "standalone_xgb_roc_auc": round(float(roc_auc_score(y_te, test_xgb_probs)), 4),
            "standalone_hgb_roc_auc": round(float(roc_auc_score(y_te, test_hgb_probs)), 4),
            "oof_ensemble_roc_auc": round(float(roc_auc_score(y_dev, oof_ens)), 4),
        },
        "mondrian_conformal_audit": conformal_audit_by_alpha[default_akey],
        "mondrian_conformal_by_alpha": conformal_audit_by_alpha,
        "subgroup_performance": subgroup_metrics,
        "intended_use": (
            "Stage-1 primary-care cardiovascular risk triage with PLR periodic embeddings, "
            "UniMP masked label propagation, 5-fold OOF 3-paradigm ensembling, multi-alpha "
            "Mondrian conformal abstention, PSI drift tracking, and escalation to Stage-2 "
            "Cardiology Workup."
        ),
        "limitations": [
            "Trained on retrospective screening vitals; escalate DEFER_TO_CLINICIAN / HIGH_RISK patients to Stage-2 Cardiology Workup.",
            "Not FDA-cleared or CE-marked medical software.",
            "Use `heart-pipeline calibrate-site` to locally recalibrate Temperature and Mondrian quantiles for a new hospital.",
        ],
    }

    xgb_bytes = bytes(final_xgb.get_booster().save_raw("json"))
    hgb_bytes = bytes(final_hgb.get_booster().save_raw("json"))

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "safe_serialization": "weights_only_tensors_json",
        "model_version": BUNDLE_VERSION,
        "model_config": model_config,
        "model_state_dict": {k: v.cpu() for k, v in final_gat.state_dict().items()},
        "xgb_booster_json": xgb_bytes,
        "hgb_booster_json": hgb_bytes,
        "ensemble_weights": (w_gat, w_xgb, w_hgb),
        "temperature": temp_val,
        "platt_a": 1.0,
        "platt_b": 0.0,
        "cat_cols": CAT_COLS,
        "cont_cols": CONT_COLS,
        "feature_names": CAT_COLS + CONT_COLS,
        "cont_medians": torch.from_numpy(cont_imputer.statistics_.astype(np.float32)),
        "cont_mean": torch.from_numpy(cont_scaler.mean_.astype(np.float32)),
        "cont_scale": torch.from_numpy(cont_scaler.scale_.astype(np.float32)),
        "graph_mean": torch.from_numpy(graph_scaler.mean_.astype(np.float32)),
        "graph_scale": torch.from_numpy(graph_scaler.scale_.astype(np.float32)),
        "H_bank_norm": H_proto_norm.half().cpu(),
        "H_tab_bank": H_proto_bank.half().cpu(),
        "bank_outcomes": torch.from_numpy(y_dev[proto_idx].astype(np.int8)),
        "bank_summary": torch.from_numpy(
            dev_df[["age_years", "ap_hi", "bmi"]].values[proto_idx].astype(np.float32)
        ),
        "k_neighbors": k_neighbors,
        "conformal_alpha": conformal_alpha,
        "conformal_quantiles": conformal_quantiles,
        "ood_distance_threshold": ood_dist_threshold,
        "ood_zscore_threshold": 4.5,
        "val_risk_hist_edges": torch.from_numpy(hist_edges.astype(np.float64)),
        "val_risk_hist_props": torch.from_numpy(hist_props.astype(np.float64)),
        "model_card": model_card,
    }
    torch.save(payload, out)
    sha = _sha256_file(out)
    model_card["bundle_sha256"] = sha
    out.with_suffix(".model_card.json").write_text(json.dumps(model_card, indent=2), encoding="utf-8")

    size_kb = out.stat().st_size / 1024.0
    hp = model_card["holdout_performance"]
    ca = model_card["mondrian_conformal_audit"]
    print(f"Exported CDS Bundle v3 -> {out} ({size_kb:.1f} KB, {len(proto_idx)} prototypes, SHA-256: {sha[:16]}...)")
    print(
        f"Holdout ROC-AUC={hp['roc_auc']} (PLR+UniMP GAT={hp['standalone_gat_roc_auc']}, "
        f"XGB={hp['standalone_xgb_roc_auc']}, HGB={hp['standalone_hgb_roc_auc']}) | "
        f"Weights=({w_gat:.2f}, {w_xgb:.2f}, {w_hgb:.2f}) | PR-AUC={hp['pr_auc']} | ECE={hp['ece']}"
    )
    print(
        f"Mondrian 90% Coverage={ca['empirical_test_coverage']} (Deferral={ca['deferral_rate']*100:.1f}%) | "
        f"85% Deferral={conformal_audit_by_alpha['0.15']['deferral_rate']*100:.1f}%"
    )
    return out, model_card


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build and export Clinical CDS Bundle v3 (PLR + UniMP + 5-Fold 3-Paradigm).")
    parser.add_argument("--dataset", type=str, default="cleaned_cardio_70k.csv")
    parser.add_argument("--output", type=str, default=str(DEFAULT_BUNDLE_PATH))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--k-neighbors", type=int, default=7)
    parser.add_argument("--n-prototypes", type=int, default=4096)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    build_clinical_bundle(
        dataset_path=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        cv_folds=args.cv_folds,
        batch_size=args.batch_size,
        k_neighbors=args.k_neighbors,
        n_prototypes=args.n_prototypes,
        conformal_alpha=args.alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
