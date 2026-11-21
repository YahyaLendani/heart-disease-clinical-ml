"""Versioned, SHA-256-verified Clinical Model Bundle with pre-encoded neighbor bank.

Performance optimization for clinical production latency:
Instead of re-running the Level-1 Feature Transformer on k historical neighbors
for every incoming patient, ``build_clinical_bundle`` pre-encodes the entire
training cohort's ``[CLS]`` embeddings (``H_bank``) and L2-normalized graph
similarity vectors (``X_graph_bank_norm``) once at export time. At inference
time, exact cosine k-NN retrieval against 48,000+ patients reduces to a single
BLAS matrix-vector product (~1 ms on CPU, <0.2 ms on GPU).
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models import (
    GraphAugmentedClinicalTransformer,
    ModelWithTemperature,
    build_inductive_knn_indices,
    compute_ece,
)
from project_paths import ARTIFACTS_DIR, resolve_data_path

BUNDLE_VERSION = "1.0.0-gat-transformer-cds"
DEFAULT_BUNDLE_PATH = ARTIFACTS_DIR / "clinical_cds_bundle.pt"

CAT_COLS = ["sex", "cholesterol", "gluc", "smoke", "alco", "active"]
CONT_COLS = [
    "age_years",
    "height",
    "weight",
    "bmi",
    "ap_hi",
    "ap_lo",
    "pulse_pressure",
    "mean_arterial_pressure",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split conformal quantile ceil((n + 1)(1 - alpha)) / n."""
    n = len(scores)
    if n == 0:
        return 1.0 - alpha
    level = min(1.0, float(np.ceil((n + 1) * (1.0 - alpha)) / n))
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class ClinicalModelBundle:
    """Self-contained, integrity-verified clinical inference artifact."""

    model: GraphAugmentedClinicalTransformer
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
    # Pre-encoded training neighbor bank for sub-millisecond exact cosine k-NN
    X_graph_bank_norm: torch.Tensor  # (N_bank, F) L2-normalized float32
    H_tab_bank: torch.Tensor  # (N_bank, D) float32
    bank_outcomes: np.ndarray  # (N_bank,) int8
    bank_summary: np.ndarray  # (N_bank, 3) float32: [age_years, ap_hi, bmi]
    k_neighbors: int
    # Mondrian conformal thresholds (per class, plus per sex x class)
    conformal_alpha: float
    q_class_0: float
    q_class_1: float
    q_by_sex_class: dict[str, float]
    # OOD threshold (99th percentile of validation nearest-neighbor cosine distance)
    ood_distance_threshold: float
    ood_zscore_threshold: float
    model_card: dict[str, Any]

    @classmethod
    def load(
        cls,
        bundle_path: str | Path = DEFAULT_BUNDLE_PATH,
        device: str = "auto",
        expected_sha256: str | None = None,
    ) -> ClinicalModelBundle:
        """Load and SHA-256-verify a serialized ClinicalModelBundle."""
        path = Path(bundle_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Clinical model bundle not found at {path}. "
                "Build it with `python -m clinical_cds.bundle`."
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
        payload = torch.load(path, map_location=dev, weights_only=False)
        cfg = payload["model_config"]

        model = GraphAugmentedClinicalTransformer(**cfg).to(dev)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()

        return cls(
            model=model,
            temperature=float(payload["temperature"]),
            model_version=str(payload["model_version"]),
            bundle_sha256=actual_sha,
            cat_cols=list(payload["cat_cols"]),
            cont_cols=list(payload["cont_cols"]),
            feature_names=list(payload["feature_names"]),
            cat_dims=list(cfg["categorical_dims"]),
            cont_medians=np.asarray(payload["cont_medians"], dtype=np.float32),
            cont_mean=np.asarray(payload["cont_mean"], dtype=np.float32),
            cont_scale=np.asarray(payload["cont_scale"], dtype=np.float32),
            graph_mean=np.asarray(payload["graph_mean"], dtype=np.float32),
            graph_scale=np.asarray(payload["graph_scale"], dtype=np.float32),
            X_graph_bank_norm=payload["X_graph_bank_norm"].float().to(dev),
            H_tab_bank=payload["H_tab_bank"].float().to(dev),
            bank_outcomes=np.asarray(payload["bank_outcomes"], dtype=np.int8),
            bank_summary=np.asarray(payload["bank_summary"], dtype=np.float32),
            k_neighbors=int(payload["k_neighbors"]),
            conformal_alpha=float(payload["conformal_alpha"]),
            q_class_0=float(payload["q_class_0"]),
            q_class_1=float(payload["q_class_1"]),
            q_by_sex_class={k: float(v) for k, v in payload["q_by_sex_class"].items()},
            ood_distance_threshold=float(payload["ood_distance_threshold"]),
            ood_zscore_threshold=float(payload.get("ood_zscore_threshold", 4.0)),
            model_card=dict(payload["model_card"]),
        )


def build_clinical_bundle(
    dataset_path: str | Path = "cleaned_cardio_70k.csv",
    output_path: str | Path = DEFAULT_BUNDLE_PATH,
    epochs: int = 18,
    batch_size: int = 512,
    k_neighbors: int = 7,
    embed_dim: int = 32,
    conformal_alpha: float = 0.10,
    seed: int = 42,
    max_bank_size: int = 48011,
) -> tuple[Path, dict[str, Any]]:
    """Train, calibrate, conformalize, and export a production ClinicalModelBundle."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_file = resolve_data_path(dataset_path)
    if not data_file.exists():
        from prepare_large_cvd_dataset import prepare_large_cvd_cohort
        prepare_large_cvd_cohort(data_file)

    df = pd.read_csv(data_file)
    y = df["target"].values.astype(int)
    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=seed, stratify=y)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.15 / 0.85, random_state=seed, stratify=y[train_val_idx]
    )

    tr_df = df.iloc[train_idx].reset_index(drop=True)
    va_df = df.iloc[val_idx].reset_index(drop=True)
    te_df = df.iloc[test_idx].reset_index(drop=True)

    # Fit preprocessing strictly on training split
    cont_imputer = SimpleImputer(strategy="median")
    cont_scaler = StandardScaler()
    X_cont_tr = cont_scaler.fit_transform(cont_imputer.fit_transform(tr_df[CONT_COLS].values)).astype(np.float32)
    X_cont_va = cont_scaler.transform(cont_imputer.transform(va_df[CONT_COLS].values)).astype(np.float32)
    X_cont_te = cont_scaler.transform(cont_imputer.transform(te_df[CONT_COLS].values)).astype(np.float32)

    X_cat_tr = tr_df[CAT_COLS].values.astype(np.int64)
    X_cat_va = va_df[CAT_COLS].values.astype(np.int64)
    X_cat_te = te_df[CAT_COLS].values.astype(np.int64)
    # Fixed cardinalities: sex(2), chol(3), gluc(3), smoke(2), alco(2), active(2) + <UNK>
    cat_dims = [3, 4, 4, 3, 3, 3]

    graph_scaler = StandardScaler()
    X_graph_tr = graph_scaler.fit_transform(np.hstack([X_cat_tr, X_cont_tr])).astype(np.float32)
    X_graph_va = graph_scaler.transform(np.hstack([X_cat_va, X_cont_va])).astype(np.float32)
    X_graph_te = graph_scaler.transform(np.hstack([X_cat_te, X_cont_te])).astype(np.float32)

    y_tr = tr_df["target"].values.astype(np.float32)
    y_va = va_df["target"].values.astype(np.float32)
    y_te = te_df["target"].values.astype(np.float32)

    tr_nbr_idx, va_nbr_idx = build_inductive_knn_indices(
        X_graph_tr, X_graph_va, k=k_neighbors, metric="cosine", n_jobs=-1
    )
    _, te_nbr_idx = build_inductive_knn_indices(
        X_graph_tr, X_graph_te, k=k_neighbors, metric="cosine", n_jobs=-1
    )

    x_cat_tr_t = torch.from_numpy(X_cat_tr).to(dev)
    x_cont_tr_t = torch.from_numpy(X_cont_tr).to(dev)
    y_tr_t = torch.from_numpy(y_tr).to(dev).view(-1, 1)

    x_cat_va_t = torch.from_numpy(X_cat_va).to(dev)
    x_cont_va_t = torch.from_numpy(X_cont_va).to(dev)
    y_va_t = torch.from_numpy(y_va).to(dev).view(-1, 1)

    x_cat_te_t = torch.from_numpy(X_cat_te).to(dev)
    x_cont_te_t = torch.from_numpy(X_cont_te).to(dev)

    tr_nbr_t = torch.from_numpy(tr_nbr_idx).to(dev)
    va_nbr_t = torch.from_numpy(va_nbr_idx).to(dev)  # type: ignore[arg-type]
    te_nbr_t = torch.from_numpy(te_nbr_idx).to(dev)  # type: ignore[arg-type]

    model_config = {
        "categorical_dims": cat_dims,
        "continuous_dim": len(CONT_COLS),
        "embed_dim": embed_dim,
        "num_transformer_heads": 4,
        "num_transformer_layers": 2,
        "num_gat_heads": 4,
        "num_gat_layers": 2,
        "mlp_hidden_dim": 64,
        "dropout": 0.15,
        "num_classes": 1,
    }
    model = GraphAugmentedClinicalTransformer(**model_config).to(dev)
    opt = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=2e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    n_tr = len(y_tr)

    @torch.inference_mode()
    def _encode_bank() -> torch.Tensor:
        model.eval()
        return torch.cat(
            [
                model.encode_tabular(x_cat_tr_t[s : s + 2048], x_cont_tr_t[s : s + 2048])[0]
                for s in range(0, n_tr, 2048)
            ],
            dim=0,
        )

    best_v, best_sd = float("inf"), None
    for _ in range(epochs):
        h_bank = _encode_bank()
        model.train()
        perm = torch.randperm(n_tr, device=dev)
        for s in range(0, n_tr, batch_size):
            b = perm[s : s + batch_size]
            opt.zero_grad()
            logits = model.forward_minibatch(
                x_cat=x_cat_tr_t[b],
                x_cont=x_cont_tr_t[b],
                nbr_h_tab=h_bank[tr_nbr_t[b]],
                feature_noise_std=0.02,
            )
            loss = criterion(logits, y_tr_t[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        h_eval = _encode_bank()
        model.eval()
        with torch.inference_mode():
            v_logits = torch.cat([
                model.forward_minibatch(
                    x_cat=x_cat_va_t[s : s + 1024],
                    x_cont=x_cont_va_t[s : s + 1024],
                    nbr_h_tab=h_eval[va_nbr_t[s : s + 1024]],
                )
                for s in range(0, len(y_va), 1024)
            ])
            vloss = criterion(v_logits, y_va_t).item()
            if vloss < best_v:
                best_v = vloss
                best_sd = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_sd is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})

    # Final pre-encoded bank & temperature calibration strictly on validation set
    h_bank_final = _encode_bank()
    model.eval()
    with torch.inference_mode():
        val_logits = torch.cat([
            model.forward_minibatch(
                x_cat=x_cat_va_t[s : s + 1024],
                x_cont=x_cont_va_t[s : s + 1024],
                nbr_h_tab=h_bank_final[va_nbr_t[s : s + 1024]],
            )
            for s in range(0, len(y_va), 1024)
        ])
        test_logits = torch.cat([
            model.forward_minibatch(
                x_cat=x_cat_te_t[s : s + 1024],
                x_cont=x_cont_te_t[s : s + 1024],
                nbr_h_tab=h_bank_final[te_nbr_t[s : s + 1024]],
            )
            for s in range(0, len(y_te), 1024)
        ])

    temp_scaler = ModelWithTemperature(model).to(dev)
    temp_scaler.calibrate(val_logits.detach().clone(), y_va_t.detach().clone())
    temp_val = float(temp_scaler.temperature.item())

    with torch.inference_mode():
        val_probs = torch.sigmoid(temp_scaler.temperature_scale(val_logits)).cpu().numpy().ravel()
        test_probs = torch.sigmoid(temp_scaler.temperature_scale(test_logits)).cpu().numpy().ravel()

    # Mondrian Conformal Quantiles strictly on validation set
    y_va_int = y_va.astype(int)
    nonconf_0 = val_probs[y_va_int == 0]  # 1 - P(y=0) = P(y=1)
    nonconf_1 = 1.0 - val_probs[y_va_int == 1]
    q_class_0 = _conformal_quantile(nonconf_0, conformal_alpha)
    q_class_1 = _conformal_quantile(nonconf_1, conformal_alpha)

    q_by_sex_class: dict[str, float] = {}
    val_sex = va_df["sex"].values.astype(int)
    for sx in (0, 1):
        m0 = (val_sex == sx) & (y_va_int == 0)
        m1 = (val_sex == sx) & (y_va_int == 1)
        q_by_sex_class[f"sex_{sx}_class_0"] = _conformal_quantile(val_probs[m0], conformal_alpha)
        q_by_sex_class[f"sex_{sx}_class_1"] = _conformal_quantile(1.0 - val_probs[m1], conformal_alpha)

    # Calibrate OOD nearest-neighbor cosine distance threshold on validation set (99.5th pct)
    X_bank_t = F.normalize(torch.from_numpy(X_graph_tr).to(dev), p=2, dim=1, eps=1e-8)
    X_va_norm = F.normalize(torch.from_numpy(X_graph_va).to(dev), p=2, dim=1, eps=1e-8)
    min_dists = []
    with torch.inference_mode():
        for s in range(0, len(X_va_norm), 1024):
            sims = torch.mm(X_va_norm[s : s + 1024], X_bank_t.t())
            max_sim, _ = sims.max(dim=1)
            min_dists.append((1.0 - max_sim).cpu().numpy())
    val_min_dists = np.concatenate(min_dists)
    ood_dist_threshold = float(np.quantile(val_min_dists, 0.995))

    # Subgroup fairness & conformal audit on holdout test set for Model Card
    test_preds = (test_probs >= 0.5).astype(int)
    y_te_int = y_te.astype(int)
    test_sex = te_df["sex"].values.astype(int)
    subgroup_metrics: dict[str, dict[str, float]] = {}
    for sx, label in ((0, "female"), (1, "male")):
        mask = test_sex == sx
        yt_s, yp_s, pr_s = y_te_int[mask], test_preds[mask], test_probs[mask]
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

    # Verify Mondrian conformal coverage on holdout test set
    in_set = []
    set_sizes = []
    for p, yt, sx in zip(test_probs, y_te_int, test_sex, strict=True):
        q0 = q_by_sex_class[f"sex_{sx}_class_0"]
        q1 = q_by_sex_class[f"sex_{sx}_class_1"]
        cset = []
        if p <= q0:
            cset.append(0)
        if (1.0 - p) <= q1:
            cset.append(1)
        if not cset:
            cset = [0, 1]
        in_set.append(int(yt in cset))
        set_sizes.append(len(cset))

    model_card = {
        "model_name": "GraphAugmentedClinicalTransformer (CDS Bundle)",
        "model_version": BUNDLE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_cohort": {
            "source": "OpenML 45547 Cardiovascular Disease Cohort (physiologically cleaned & deduplicated)",
            "n_total": int(len(df)),
            "n_train": int(len(tr_df)),
            "n_val": int(len(va_df)),
            "n_test": int(len(te_df)),
        },
        "holdout_performance": {
            "roc_auc": round(float(roc_auc_score(y_te_int, test_probs)), 4),
            "pr_auc": round(float(average_precision_score(y_te_int, test_probs)), 4),
            "accuracy": round(float(accuracy_score(y_te_int, test_preds)), 4),
            "f1": round(float(f1_score(y_te_int, test_preds)), 4),
            "brier": round(float(brier_score_loss(y_te_int, test_probs)), 4),
            "ece": round(float(compute_ece(test_probs, y_te_int)), 4),
            "temperature": round(temp_val, 4),
        },
        "mondrian_conformal_audit": {
            "nominal_coverage": round(1.0 - conformal_alpha, 4),
            "empirical_test_coverage": round(float(np.mean(in_set)), 4),
            "deferral_rate": round(float(np.mean([s == 2 for s in set_sizes])), 4),
            "mean_set_size": round(float(np.mean(set_sizes)), 4),
        },
        "subgroup_performance": subgroup_metrics,
        "intended_use": (
            "Clinician-in-the-loop cardiovascular risk triage and dual-level interpretability "
            "research prototype. Outputs calibrated risk, Mondrian conformal abstention, OOD "
            "safety alerts, biomarker attentions, matched cohort neighbors, and mutable-only recourse."
        ),
        "limitations": [
            "Trained on retrospective screening data; requires prospective hospital EHR validation before clinical use.",
            "Not FDA-cleared or CE-marked medical software.",
            "Automatically defers uncertain or out-of-distribution patients to clinician review.",
        ],
    }

    # Store compact float16 pre-encoded bank for fast load & minimal footprint (~6 MB vs >40 MB)
    keep_n = min(n_tr, max_bank_size)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_version": BUNDLE_VERSION,
        "model_config": model_config,
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "temperature": temp_val,
        "cat_cols": CAT_COLS,
        "cont_cols": CONT_COLS,
        "feature_names": CAT_COLS + CONT_COLS,
        "cont_medians": cont_imputer.statistics_.astype(np.float32),
        "cont_mean": cont_scaler.mean_.astype(np.float32),
        "cont_scale": cont_scaler.scale_.astype(np.float32),
        "graph_mean": graph_scaler.mean_.astype(np.float32),
        "graph_scale": graph_scaler.scale_.astype(np.float32),
        "X_graph_bank_norm": X_bank_t[:keep_n].half().cpu(),
        "H_tab_bank": h_bank_final[:keep_n].half().cpu(),
        "bank_outcomes": tr_df["target"].values[:keep_n].astype(np.int8),
        "bank_summary": tr_df[["age_years", "ap_hi", "bmi"]].values[:keep_n].astype(np.float32),
        "k_neighbors": k_neighbors,
        "conformal_alpha": conformal_alpha,
        "q_class_0": q_class_0,
        "q_class_1": q_class_1,
        "q_by_sex_class": q_by_sex_class,
        "ood_distance_threshold": ood_dist_threshold,
        "ood_zscore_threshold": 4.0,
        "model_card": model_card,
    }
    torch.save(payload, out)
    sha = _sha256_file(out)
    card_path = out.with_suffix(".model_card.json")
    model_card["bundle_sha256"] = sha
    card_path.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
    print(f"Exported Clinical Model Bundle -> {out} (SHA-256: {sha[:16]}...)")
    print(f"Holdout AUC={model_card['holdout_performance']['roc_auc']} | "
          f"Mondrian Coverage={model_card['mondrian_conformal_audit']['empirical_test_coverage']} "
          f"(Deferral={model_card['mondrian_conformal_audit']['deferral_rate']*100:.1f}%)")
    return out, model_card


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build and export the Clinical CDS Model Bundle.")
    parser.add_argument("--dataset", type=str, default="cleaned_cardio_70k.csv")
    parser.add_argument("--output", type=str, default=str(DEFAULT_BUNDLE_PATH))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--k-neighbors", type=int, default=7)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    build_clinical_bundle(
        dataset_path=args.dataset,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        k_neighbors=args.k_neighbors,
        conformal_alpha=args.alpha,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
