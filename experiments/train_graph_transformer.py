"""Train, Cross-Validate, Scale, and Interpret the Graph-Augmented Clinical Transformer.

Supports three rigorous clinical evaluation regimes with strict zero-leakage guarantees:

1. ``uci_cv`` — UCI Multi-Center Cardiovascular Cohort (N=918, 4 hospitals):
   5-Fold Stratified Cross-Validation comparing TabTransformer (No Graph),
   Inductive GAT (No Transformer), and Graph-Augmented Transformer (Ours),
   plus dual-level biomarker + cohort-neighbor interpretability visualization.

2. ``mimic_cv`` — MIMIC-IV ICU In-Hospital Mortality (N=275, 100 unique patients):
   5-Fold Patient-Grouped Cross-Validation (``StratifiedGroupKFold`` by ``subject_id``)
   verifying cross-domain generalization from Cardiology to Critical Care with zero
   longitudinal patient leakage.

3. ``large_scale`` — OpenML 45547 Cardiovascular Cohort (N=68,589 unique patients):
   Scalable Mini-Batch Inductive Neighbor Attention (``forward_minibatch`` +
   ``build_inductive_knn_indices``) running in O(B * k) memory on GPU/CPU with
   validation-calibrated Temperature Scaling.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (
    GATClassifier,
    GraphAugmentedClinicalTransformer,
    ModelWithTemperature,
    TabTransformer,
    add_cardiovascular_features,
    build_inductive_knn_indices,
    build_inductive_patient_graph,
    compute_ece,
)
from project_paths import resolve_data_path, resolve_output_dir

UCI_CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
MIMIC_CAT_COLS = [
    "gender",
    "admission_type",
    "admission_location",
    "insurance",
    "marital_status",
    "race",
    "has_ed_visit",
    "prior_circulatory",
    "prior_diabetes",
    "prior_ckd",
]
LARGE_CVD_CAT_COLS = ["sex", "cholesterol", "gluc", "smoke", "alco", "active"]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _compute_metrics(name: str, y_true: np.ndarray, probs: np.ndarray) -> dict[str, float | str]:
    preds = (probs >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "pr_auc": float(average_precision_score(y_true, probs)),
        "brier": float(brier_score_loss(y_true, probs)),
        "ece": float(compute_ece(probs, y_true)),
    }


def _preprocess_fold(
    tr_df: pd.DataFrame,
    va_df: pd.DataFrame,
    te_df: pd.DataFrame,
    cat_cols: list[str],
    cont_cols: list[str],
    target_col: str,
) -> dict:
    """Fit all imputers, scalers, and categorical vocabularies strictly on tr_df."""
    tr = tr_df.copy()
    va = va_df.copy()
    te = te_df.copy()

    cat_dims: list[int] = []
    for col in cat_cols:
        tr_col = tr[col].astype(str).replace("nan", np.nan)
        mode_val = tr_col.mode().iloc[0] if not tr_col.mode().empty else "MISSING"
        tr_clean = tr_col.fillna(mode_val)
        va_clean = va[col].astype(str).replace("nan", np.nan).fillna(mode_val)
        te_clean = te[col].astype(str).replace("nan", np.nan).fillna(mode_val)

        vocab = {v: idx for idx, v in enumerate(sorted(tr_clean.unique()))}
        unk_idx = len(vocab)
        tr[col] = tr_clean.map(vocab).fillna(unk_idx).astype(int)
        va[col] = va_clean.map(vocab).fillna(unk_idx).astype(int)
        te[col] = te_clean.map(vocab).fillna(unk_idx).astype(int)
        cat_dims.append(len(vocab) + 1)

    cont_imputer = SimpleImputer(strategy="median")
    cont_scaler = StandardScaler()
    X_cont_tr = cont_scaler.fit_transform(cont_imputer.fit_transform(tr[cont_cols].values)).astype(np.float32)
    X_cont_va = cont_scaler.transform(cont_imputer.transform(va[cont_cols].values)).astype(np.float32)
    X_cont_te = cont_scaler.transform(cont_imputer.transform(te[cont_cols].values)).astype(np.float32)

    X_cat_tr = tr[cat_cols].values.astype(np.int64)
    X_cat_va = va[cat_cols].values.astype(np.int64)
    X_cat_te = te[cat_cols].values.astype(np.int64)

    graph_scaler = StandardScaler()
    X_graph_tr = graph_scaler.fit_transform(np.hstack([X_cat_tr, X_cont_tr])).astype(np.float32)
    X_graph_va = graph_scaler.transform(np.hstack([X_cat_va, X_cont_va])).astype(np.float32)
    X_graph_te = graph_scaler.transform(np.hstack([X_cat_te, X_cont_te])).astype(np.float32)

    return {
        "cat_dims": cat_dims,
        "feature_names": cat_cols + cont_cols,
        "X_cat_tr": X_cat_tr,
        "X_cat_va": X_cat_va,
        "X_cat_te": X_cat_te,
        "X_cont_tr": X_cont_tr,
        "X_cont_va": X_cont_va,
        "X_cont_te": X_cont_te,
        "X_graph_tr": X_graph_tr,
        "X_graph_va": X_graph_va,
        "X_graph_te": X_graph_te,
        "y_tr": tr[target_col].values.astype(np.float32),
        "y_va": va[target_col].values.astype(np.float32),
        "y_te": te[target_col].values.astype(np.float32),
    }


def _train_eval_full_graph_fold(
    splits: dict,
    epochs: int,
    k_neighbors: int,
    embed_dim: int,
    seed: int,
    dev: torch.device,
    feature_noise_std: float = 0.03,
) -> tuple[list[dict[str, float | str]], dict]:
    """Train and evaluate TabTransformer, Inductive GAT, and Graph-Augmented Transformer on one fold."""
    n_tr = len(splits["y_tr"])
    x_cat_tr = torch.from_numpy(splits["X_cat_tr"]).to(dev)
    x_cat_va = torch.from_numpy(splits["X_cat_va"]).to(dev)
    x_cat_te = torch.from_numpy(splits["X_cat_te"]).to(dev)

    x_cont_tr = torch.from_numpy(splits["X_cont_tr"]).to(dev)
    x_cont_va = torch.from_numpy(splits["X_cont_va"]).to(dev)
    x_cont_te = torch.from_numpy(splits["X_cont_te"]).to(dev)

    y_tr_t = torch.from_numpy(splits["y_tr"]).to(dev).view(-1, 1)
    y_va_t = torch.from_numpy(splits["y_va"]).to(dev).view(-1, 1)

    adj_tr = build_inductive_patient_graph(splits["X_graph_tr"], None, k=k_neighbors, metric="cosine").to(dev)
    adj_tr_va = build_inductive_patient_graph(splits["X_graph_tr"], splits["X_graph_va"], k=k_neighbors, metric="cosine").to(dev)
    adj_tr_te = build_inductive_patient_graph(splits["X_graph_tr"], splits["X_graph_te"], k=k_neighbors, metric="cosine").to(dev)

    x_cat_tr_va = torch.cat([x_cat_tr, x_cat_va], dim=0)
    x_cont_tr_va = torch.cat([x_cont_tr, x_cont_va], dim=0)
    x_cat_tr_te = torch.cat([x_cat_tr, x_cat_te], dim=0)
    x_cont_tr_te = torch.cat([x_cont_tr, x_cont_te], dim=0)

    pos_count = max(float(splits["y_tr"].sum()), 1.0)
    neg_count = max(float(len(splits["y_tr"]) - splits["y_tr"].sum()), 1.0)
    # Mild capped pos_weight prevents logit explosion on heavily imbalanced ICU mortality
    pw = min(neg_count / pos_count, 5.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], dtype=torch.float32, device=dev))

    fold_rows: list[dict[str, float | str]] = []

    # 1. TabTransformer (No Graph)
    set_seed(seed)
    tab_model = TabTransformer(
        categorical_dims=splits["cat_dims"],
        continuous_dim=splits["X_cont_tr"].shape[1],
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        mlp_hidden_dims=(32, 16),
        dropout=0.2,
    ).to(dev)
    opt_tab = optim.AdamW(tab_model.parameters(), lr=3e-3, weight_decay=1e-3)
    best_v, best_sd = float("inf"), None
    for _ in range(epochs):
        tab_model.train()
        opt_tab.zero_grad()
        loss = criterion(tab_model(x_cat_tr, x_cont_tr), y_tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tab_model.parameters(), 1.0)
        opt_tab.step()
        tab_model.eval()
        with torch.inference_mode():
            vloss = criterion(tab_model(x_cat_va, x_cont_va), y_va_t).item()
            if vloss < best_v:
                best_v = vloss
                best_sd = {k: v.cpu().clone() for k, v in tab_model.state_dict().items()}
    if best_sd is not None:
        tab_model.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})
    tab_model.eval()
    with torch.inference_mode():
        tab_probs = torch.sigmoid(tab_model(x_cat_te, x_cont_te)).cpu().numpy().ravel()
    fold_rows.append(_compute_metrics("TabTransformer (No Graph)", splits["y_te"], tab_probs))

    # 2. Inductive GAT (No Transformer)
    set_seed(seed)
    x_g_tr = torch.from_numpy(splits["X_graph_tr"]).to(dev)
    x_g_tr_va = torch.from_numpy(np.vstack([splits["X_graph_tr"], splits["X_graph_va"]])).to(dev)
    x_g_tr_te = torch.from_numpy(np.vstack([splits["X_graph_tr"], splits["X_graph_te"]])).to(dev)

    gat_model = GATClassifier(
        in_features=x_g_tr.size(1),
        hidden_dim=16,
        num_heads=4,
        out_features=1,
        dropout=0.2,
    ).to(dev)
    opt_gat = optim.AdamW(gat_model.parameters(), lr=5e-3, weight_decay=1e-3)
    best_v, best_sd = float("inf"), None
    for _ in range(epochs):
        gat_model.train()
        opt_gat.zero_grad()
        loss = criterion(gat_model(x_g_tr, adj_tr), y_tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gat_model.parameters(), 1.0)
        opt_gat.step()
        gat_model.eval()
        with torch.inference_mode():
            vloss = criterion(gat_model(x_g_tr_va, adj_tr_va)[n_tr:], y_va_t).item()
            if vloss < best_v:
                best_v = vloss
                best_sd = {k: v.cpu().clone() for k, v in gat_model.state_dict().items()}
    if best_sd is not None:
        gat_model.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})
    gat_model.eval()
    with torch.inference_mode():
        gat_probs = torch.sigmoid(gat_model(x_g_tr_te, adj_tr_te)[n_tr:]).cpu().numpy().ravel()
    fold_rows.append(_compute_metrics("Inductive GAT (No Transformer)", splits["y_te"], gat_probs))

    # 3. Graph-Augmented Clinical Transformer (Ours)
    set_seed(seed)
    gt_model = GraphAugmentedClinicalTransformer(
        categorical_dims=splits["cat_dims"],
        continuous_dim=splits["X_cont_tr"].shape[1],
        embed_dim=embed_dim,
        num_transformer_heads=4,
        num_transformer_layers=2,
        num_gat_heads=4,
        num_gat_layers=2,
        mlp_hidden_dim=32,
        dropout=0.2,
    ).to(dev)
    opt_gt = optim.AdamW(gt_model.parameters(), lr=3e-3, weight_decay=2e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt_gt, T_max=epochs, eta_min=1e-4)

    best_v, best_sd = float("inf"), None
    for _ in range(epochs):
        gt_model.train()
        opt_gt.zero_grad()
        logits_tr = gt_model(
            x_cat_tr, x_cont_tr, adj_tr, feature_noise_std=feature_noise_std
        )
        loss = criterion(logits_tr, y_tr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gt_model.parameters(), 1.0)
        opt_gt.step()
        scheduler.step()

        gt_model.eval()
        with torch.inference_mode():
            vloss = criterion(gt_model(x_cat_tr_va, x_cont_tr_va, adj_tr_va)[n_tr:], y_va_t).item()
            if vloss < best_v:
                best_v = vloss
                best_sd = {k: v.cpu().clone() for k, v in gt_model.state_dict().items()}

    if best_sd is not None:
        gt_model.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})

    gt_model.eval()
    with torch.inference_mode():
        val_logits = gt_model(x_cat_tr_va, x_cont_tr_va, adj_tr_va)[n_tr:]
        test_logits_all, test_expl = gt_model(
            x_cat_tr_te, x_cont_tr_te, adj_tr_te, return_explanations=True
        )
        test_logits = test_logits_all[n_tr:]

    temp_scaler = ModelWithTemperature(gt_model).to(dev)
    temp_scaler.calibrate(val_logits.detach().clone(), y_va_t.detach().clone())
    with torch.inference_mode():
        gt_probs = torch.sigmoid(temp_scaler.temperature_scale(test_logits)).cpu().numpy().ravel()

    fold_rows.append(_compute_metrics("Graph-Augmented Transformer (Ours)", splits["y_te"], gt_probs))

    extras = {
        "gt_model": gt_model,
        "temperature": float(temp_scaler.temperature.item()),
        "gt_probs": gt_probs,
        "test_expl": test_expl,
        "x_cat_tr_te": x_cat_tr_te,
        "x_cont_tr_te": x_cont_tr_te,
        "adj_tr_te": adj_tr_te,
        "n_tr": n_tr,
    }
    return fold_rows, extras


def _aggregate_cv_rows(all_rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(all_rows)
    metric_cols = ["accuracy", "f1", "roc_auc", "pr_auc", "brier", "ece"]
    summary = []
    for model_name, grp in df.groupby("model", sort=False):
        row: dict[str, float | str] = {"model": str(model_name)}
        for m in metric_cols:
            row[f"{m}_mean"] = float(grp[m].mean())
            row[f"{m}_std"] = float(grp[m].std(ddof=0))
        summary.append(row)
    return pd.DataFrame(summary)


def run_uci_5fold_cv(
    dataset_path: str = "cleaned_heart_disease_combined.csv",
    cv_folds: int = 5,
    epochs: int = 80,
    k_neighbors: int = 7,
    embed_dim: int = 32,
    seed: int = 42,
    output_dir: str = "artifacts",
) -> tuple[pd.DataFrame, dict]:
    """5-Fold Stratified CV on UCI Multi-Center CVD Cohort (N=918)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = resolve_data_path(dataset_path)
    df = pd.read_csv(path).dropna(subset=["target"]).reset_index(drop=True)
    df = add_cardiovascular_features(df)

    drop_cols = {"target", "diagnosis", "disease_code", "site"}
    cat_cols = [c for c in UCI_CAT_COLS if c in df.columns]
    cont_cols = [
        c
        for c in df.columns
        if c not in drop_cols and c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    y = df["target"].values.astype(int)

    print("\n" + "=" * 78)
    print(f"1. UCI MULTI-CENTER CVD ({cv_folds}-FOLD STRATIFIED CV, N={len(df)}) | Device: {dev}")
    print("=" * 78)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    all_rows: list[dict] = []
    feat_attn_accum: list[np.ndarray] = []
    gate_vals: list[float] = []
    fold0_artifacts: dict = {}

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(df, y)):
        tr_idx, va_idx = train_test_split(
            train_val_idx,
            test_size=0.15,
            random_state=seed + fold_idx,
            stratify=y[train_val_idx],
        )
        splits = _preprocess_fold(
            df.iloc[tr_idx], df.iloc[va_idx], df.iloc[test_idx], cat_cols, cont_cols, "target"
        )
        fold_rows, extras = _train_eval_full_graph_fold(
            splits=splits,
            epochs=epochs,
            k_neighbors=k_neighbors,
            embed_dim=embed_dim,
            seed=seed + fold_idx,
            dev=dev,
            feature_noise_std=0.03,
        )
        all_rows.extend(fold_rows)

        n_tr = extras["n_tr"]
        feat_attn_accum.append(extras["test_expl"]["feature_attn"][n_tr:].mean(dim=0).cpu().numpy())
        gate_vals.append(float(extras["test_expl"]["gate"][n_tr:].mean().item()))

        if fold_idx == 0:
            fold0_artifacts = {"splits": splits, "extras": extras}

    summary_df = _aggregate_cv_rows(all_rows)
    for _, r in summary_df.iterrows():
        print(
            f"  {r['model']:<36} | ROC-AUC: {r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f} | "
            f"PR-AUC: {r['pr_auc_mean']:.4f}±{r['pr_auc_std']:.4f} | "
            f"Acc: {r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f} | "
            f"Brier: {r['brier_mean']:.4f}"
        )

    mean_feat_attn = np.mean(feat_attn_accum, axis=0)
    feature_names = cat_cols + cont_cols
    global_feat_attn = dict(
        sorted(
            ((f, float(w)) for f, w in zip(feature_names, mean_feat_attn, strict=True)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    mean_gate = float(np.mean(gate_vals))

    # Build dual interpretability case study from fold 0
    s0 = fold0_artifacts["splits"]
    e0 = fold0_artifacts["extras"]
    pos_idx = np.where(s0["y_te"] == 1)[0]
    case_local_idx = int(pos_idx[int(np.argmax(e0["gt_probs"][pos_idx]))])
    case_expl = e0["gt_model"].explain_prediction(
        e0["x_cat_tr_te"],
        e0["x_cont_tr_te"],
        e0["adj_tr_te"],
        patient_idx=e0["n_tr"] + case_local_idx,
        feature_names=feature_names,
        top_k_neighbors=5,
    )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_p / "graph_transformer_uci_cv.csv", index=False)

    # Save updated 3-panel dual-interpretability figure using 5-fold CV averages
    _plot_interpretability_panel(
        global_feat_attn=global_feat_attn,
        case_expl=case_expl,
        y_train=s0["y_tr"],
        case_true_label=int(s0["y_te"][case_local_idx]),
        cv_summary_df=summary_df,
        output_path=out_p / "figures" / "graph_transformer_interpretability.png",
    )

    return summary_df, {"global_feat_attn": global_feat_attn, "mean_gate": mean_gate}


def run_mimic_grouped_cv(
    dataset_path: str = "mimic_hospital_mortality.csv",
    cv_folds: int = 5,
    epochs: int = 70,
    k_neighbors: int = 5,
    embed_dim: int = 32,
    seed: int = 42,
    output_dir: str = "artifacts",
) -> pd.DataFrame:
    """5-Fold Patient-Grouped CV on MIMIC-IV ICU Mortality (zero subject_id overlap)."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = resolve_data_path(dataset_path)
    df = pd.read_csv(path).dropna(subset=["hospital_expire_flag"]).reset_index(drop=True)

    target_col = "hospital_expire_flag"
    drop_cols = {target_col, "subject_id", "hadm_id"}
    cat_cols = [c for c in MIMIC_CAT_COLS if c in df.columns]
    cont_cols = [
        c
        for c in df.columns
        if c not in drop_cols and c not in cat_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    y = df[target_col].values.astype(int)
    groups = df["subject_id"].values

    print("\n" + "=" * 78)
    print(
        f"2. MIMIC-IV ICU MORTALITY ({cv_folds}-FOLD PATIENT-GROUPED CV, "
        f"N={len(df)}, Patients={df['subject_id'].nunique()}) | Device: {dev}"
    )
    print("=" * 78)

    sgkf = StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    all_rows: list[dict] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(sgkf.split(df, y, groups=groups)):
        # Ensure strict zero patient overlap between train+val and test
        assert set(groups[train_val_idx]).isdisjoint(set(groups[test_idx]))

        # Inner patient-grouped val split
        inner_sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed + fold_idx)
        tr_rel, va_rel = next(
            inner_sgkf.split(df.iloc[train_val_idx], y[train_val_idx], groups=groups[train_val_idx])
        )
        tr_idx = train_val_idx[tr_rel]
        va_idx = train_val_idx[va_rel]
        assert set(groups[tr_idx]).isdisjoint(set(groups[va_idx]))

        splits = _preprocess_fold(
            df.iloc[tr_idx], df.iloc[va_idx], df.iloc[test_idx], cat_cols, cont_cols, target_col
        )
        fold_rows, _ = _train_eval_full_graph_fold(
            splits=splits,
            epochs=epochs,
            k_neighbors=k_neighbors,
            embed_dim=embed_dim,
            seed=seed + fold_idx,
            dev=dev,
            feature_noise_std=0.05,
        )
        all_rows.extend(fold_rows)

    summary_df = _aggregate_cv_rows(all_rows)
    for _, r in summary_df.iterrows():
        print(
            f"  {r['model']:<36} | ROC-AUC: {r['roc_auc_mean']:.4f}±{r['roc_auc_std']:.4f} | "
            f"PR-AUC: {r['pr_auc_mean']:.4f}±{r['pr_auc_std']:.4f} | "
            f"Acc: {r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f} | "
            f"Brier: {r['brier_mean']:.4f}"
        )

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_p / "graph_transformer_mimic_cv.csv", index=False)
    return summary_df


def run_large_scale_minibatch_study(
    dataset_path: str = "cleaned_cardio_70k.csv",
    epochs: int = 20,
    batch_size: int = 512,
    k_neighbors: int = 7,
    embed_dim: int = 32,
    seed: int = 42,
    output_dir: str = "artifacts",
) -> tuple[pd.DataFrame, dict]:
    """Scalable O(B * k) Mini-Batch Inductive Graph-Augmented Transformer on N=68,589 cohort."""
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = resolve_data_path(dataset_path)
    if not path.exists():
        from prepare_large_cvd_dataset import prepare_large_cvd_cohort
        prepare_large_cvd_cohort(path)

    df = pd.read_csv(path)
    target_col = "target"
    cat_cols = [c for c in LARGE_CVD_CAT_COLS if c in df.columns]
    cont_cols = [c for c in df.columns if c != target_col and c not in cat_cols]
    y = df[target_col].values.astype(int)

    indices = np.arange(len(df))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.15, random_state=seed, stratify=y)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.15 / 0.85, random_state=seed, stratify=y[train_val_idx]
    )

    print("\n" + "=" * 78)
    print(
        f"3. LARGE-SCALE CARDIOVASCULAR COHORT (N={len(df):,} | "
        f"Train={len(train_idx):,}, Val={len(val_idx):,}, Test={len(test_idx):,}) | Device: {dev}"
    )
    print("=" * 78)

    splits = _preprocess_fold(
        df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx], cat_cols, cont_cols, target_col
    )

    t0 = time.perf_counter()
    tr_nbr_idx, val_nbr_idx = build_inductive_knn_indices(
        splits["X_graph_tr"], splits["X_graph_va"], k=k_neighbors, metric="cosine", n_jobs=-1
    )
    _, te_nbr_idx = build_inductive_knn_indices(
        splits["X_graph_tr"], splits["X_graph_te"], k=k_neighbors, metric="cosine", n_jobs=-1
    )
    knn_sec = time.perf_counter() - t0
    print(f"Built strictly inductive O(N*k) k-NN neighbor index (k={k_neighbors}) in {knn_sec:.2f}s")

    # Keep feature banks on GPU for blazing-fast mini-batch neighbor gathering
    x_cat_tr_bank = torch.from_numpy(splits["X_cat_tr"]).to(dev)
    x_cont_tr_bank = torch.from_numpy(splits["X_cont_tr"]).to(dev)
    y_tr_bank = torch.from_numpy(splits["y_tr"]).to(dev).view(-1, 1)

    x_cat_va = torch.from_numpy(splits["X_cat_va"]).to(dev)
    x_cont_va = torch.from_numpy(splits["X_cont_va"]).to(dev)
    y_va_t = torch.from_numpy(splits["y_va"]).to(dev).view(-1, 1)

    x_cat_te = torch.from_numpy(splits["X_cat_te"]).to(dev)
    x_cont_te = torch.from_numpy(splits["X_cont_te"]).to(dev)

    tr_nbr_t = torch.from_numpy(tr_nbr_idx).to(dev)
    va_nbr_t = torch.from_numpy(val_nbr_idx).to(dev)  # type: ignore[arg-type]
    te_nbr_t = torch.from_numpy(te_nbr_idx).to(dev)  # type: ignore[arg-type]

    criterion = nn.BCEWithLogitsLoss()
    n_tr = len(splits["y_tr"])
    results: list[dict[str, float | str]] = []

    # --- Baseline at Scale: TabTransformer (No Cohort Graph) ---
    set_seed(seed)
    t_start = time.perf_counter()
    tab_model = TabTransformer(
        categorical_dims=splits["cat_dims"],
        continuous_dim=len(cont_cols),
        embed_dim=embed_dim,
        num_heads=4,
        num_layers=2,
        mlp_hidden_dims=(64, 32),
        dropout=0.15,
    ).to(dev)
    opt_tab = optim.AdamW(tab_model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched_tab = optim.lr_scheduler.CosineAnnealingLR(opt_tab, T_max=epochs, eta_min=1e-4)
    best_v, best_sd = float("inf"), None

    for _ in range(epochs):
        tab_model.train()
        perm = torch.randperm(n_tr, device=dev)
        for start in range(0, n_tr, batch_size):
            b = perm[start : start + batch_size]
            opt_tab.zero_grad()
            loss = criterion(tab_model(x_cat_tr_bank[b], x_cont_tr_bank[b]), y_tr_bank[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(tab_model.parameters(), 1.0)
            opt_tab.step()
        sched_tab.step()

        tab_model.eval()
        with torch.inference_mode():
            v_logits = torch.cat([
                tab_model(x_cat_va[s : s + 1024], x_cont_va[s : s + 1024])
                for s in range(0, len(x_cat_va), 1024)
            ])
            vloss = criterion(v_logits, y_va_t).item()
            if vloss < best_v:
                best_v = vloss
                best_sd = {k: v.cpu().clone() for k, v in tab_model.state_dict().items()}

    if best_sd is not None:
        tab_model.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})
    tab_model.eval()
    with torch.inference_mode():
        tab_test_logits = torch.cat([
            tab_model(x_cat_te[s : s + 1024], x_cont_te[s : s + 1024])
            for s in range(0, len(x_cat_te), 1024)
        ])
        tab_probs = torch.sigmoid(tab_test_logits).cpu().numpy().ravel()
    tab_sec = time.perf_counter() - t_start
    tab_row = _compute_metrics("TabTransformer (No Graph)", splits["y_te"], tab_probs)
    tab_row["train_seconds"] = round(tab_sec, 2)
    results.append(tab_row)

    # --- Flagship at Scale: Mini-Batch Inductive Graph-Augmented Transformer ---
    set_seed(seed)
    t_start = time.perf_counter()
    gt_model = GraphAugmentedClinicalTransformer(
        categorical_dims=splits["cat_dims"],
        continuous_dim=len(cont_cols),
        embed_dim=embed_dim,
        num_transformer_heads=4,
        num_transformer_layers=2,
        num_gat_heads=4,
        num_gat_layers=2,
        mlp_hidden_dim=64,
        dropout=0.15,
    ).to(dev)
    opt_gt = optim.AdamW(gt_model.parameters(), lr=2e-3, weight_decay=2e-4)
    sched_gt = optim.lr_scheduler.CosineAnnealingLR(opt_gt, T_max=epochs, eta_min=1e-4)
    best_v, best_sd = float("inf"), None

    @torch.inference_mode()
    def _encode_train_bank() -> torch.Tensor:
        gt_model.eval()
        chunks = []
        for s in range(0, n_tr, 2048):
            h_chunk, _ = gt_model.encode_tabular(
                x_cat_tr_bank[s : s + 2048], x_cont_tr_bank[s : s + 2048]
            )
            chunks.append(h_chunk)
        return torch.cat(chunks, dim=0)

    for _ in range(epochs):
        # Refresh frozen neighbor embedding bank at epoch start -> O(N) instead of O(N*k) Transformer passes
        h_bank = _encode_train_bank()
        gt_model.train()
        perm = torch.randperm(n_tr, device=dev)
        for start in range(0, n_tr, batch_size):
            b = perm[start : start + batch_size]
            nbr_h = h_bank[tr_nbr_t[b]]  # (B, k, D)
            opt_gt.zero_grad()
            logits = gt_model.forward_minibatch(
                x_cat=x_cat_tr_bank[b],
                x_cont=x_cont_tr_bank[b],
                nbr_h_tab=nbr_h,
                feature_noise_std=0.02,
            )
            loss = criterion(logits, y_tr_bank[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gt_model.parameters(), 1.0)
            opt_gt.step()
        sched_gt.step()

        # Inductive validation using frozen training bank
        h_bank_eval = _encode_train_bank()
        gt_model.eval()
        with torch.inference_mode():
            val_chunks = []
            for s in range(0, len(x_cat_va), 1024):
                nbr_h_va = h_bank_eval[va_nbr_t[s : s + 1024]]
                val_chunks.append(
                    gt_model.forward_minibatch(
                        x_cat=x_cat_va[s : s + 1024],
                        x_cont=x_cont_va[s : s + 1024],
                        nbr_h_tab=nbr_h_va,
                    )
                )
            v_logits = torch.cat(val_chunks, dim=0)
            vloss = criterion(v_logits, y_va_t).item()
            if vloss < best_v:
                best_v = vloss
                best_sd = {k: v.cpu().clone() for k, v in gt_model.state_dict().items()}

    if best_sd is not None:
        gt_model.load_state_dict({k: v.to(dev) for k, v in best_sd.items()})

    # Final calibrated inductive test evaluation
    h_bank_final = _encode_train_bank()
    gt_model.eval()
    with torch.inference_mode():
        val_logits = torch.cat([
            gt_model.forward_minibatch(
                x_cat=x_cat_va[s : s + 1024],
                x_cont=x_cont_va[s : s + 1024],
                nbr_h_tab=h_bank_final[va_nbr_t[s : s + 1024]],
            )
            for s in range(0, len(x_cat_va), 1024)
        ])

        test_logits_list, feat_attn_list, gate_list = [], [], []
        for s in range(0, len(x_cat_te), 1024):
            lg, expl = gt_model.forward_minibatch(
                x_cat=x_cat_te[s : s + 1024],
                x_cont=x_cont_te[s : s + 1024],
                nbr_h_tab=h_bank_final[te_nbr_t[s : s + 1024]],
                return_explanations=True,
            )
            test_logits_list.append(lg)
            feat_attn_list.append(expl["feature_attn"].cpu())
            gate_list.append(expl["gate"].cpu())

        test_logits = torch.cat(test_logits_list, dim=0)
        mean_feat_attn = torch.cat(feat_attn_list, dim=0).mean(dim=0).numpy()
        mean_gate = float(torch.cat(gate_list, dim=0).mean().item())

    temp_scaler = ModelWithTemperature(gt_model).to(dev)
    temp_scaler.calibrate(val_logits.detach().clone(), y_va_t.detach().clone())
    with torch.inference_mode():
        gt_probs = torch.sigmoid(temp_scaler.temperature_scale(test_logits)).cpu().numpy().ravel()

    gt_sec = time.perf_counter() - t_start
    gt_row = _compute_metrics("Graph-Augmented Transformer (Ours)", splits["y_te"], gt_probs)
    gt_row["train_seconds"] = round(gt_sec, 2)
    results.append(gt_row)

    res_df = pd.DataFrame(results)
    for _, r in res_df.iterrows():
        print(
            f"  {r['model']:<36} | ROC-AUC: {r['roc_auc']:.4f} | "
            f"PR-AUC: {r['pr_auc']:.4f} | Acc: {r['accuracy']:.4f} | "
            f"Brier: {r['brier']:.4f} | ECE: {r['ece']:.4f} | Time: {r['train_seconds']:.1f}s"
        )

    global_feat_attn = dict(
        sorted(
            ((f, float(w)) for f, w in zip(splits["feature_names"], mean_feat_attn, strict=True)),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    print(f"Large-Scale Mean Graph Reliance Gate: {mean_gate:.3f} | Calibrated T*: {temp_scaler.temperature.item():.3f}")
    print("Top 5 Global Biomarker Attentions (N=68,589):")
    for feat, w in list(global_feat_attn.items())[:5]:
        print(f"  - {feat:<24}: {w*100:.2f}%")

    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_p / "graph_transformer_large_70k.csv", index=False)
    torch.save(
        {
            "model_state_dict": gt_model.state_dict(),
            "temperature": float(temp_scaler.temperature.item()),
            "feature_names": splits["feature_names"],
            "cat_dims": splits["cat_dims"],
            "continuous_dim": len(cont_cols),
        },
        out_p / "graph_transformer_70k_checkpoint.pt",
    )

    return res_df, {
        "global_feat_attn": global_feat_attn,
        "mean_gate": mean_gate,
        "temperature": float(temp_scaler.temperature.item()),
        "n_total": len(df),
    }


def _plot_interpretability_panel(
    global_feat_attn: dict[str, float],
    case_expl: dict,
    y_train: np.ndarray,
    case_true_label: int,
    cv_summary_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Create a 3-panel publication figure illustrating dual clinical interpretability."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), dpi=180)
    fig.patch.set_facecolor("white")

    # Panel A: Global Intra-Patient Biomarker Attention ([CLS] -> Feature)
    ax_a = axes[0]
    top_items = list(global_feat_attn.items())[:10][::-1]
    names_a = [k for k, _ in top_items]
    vals_a = [v * 100.0 for _, v in top_items]
    bars_a = ax_a.barh(names_a, vals_a, color="#1f5a8c", edgecolor="none", height=0.68)
    ax_a.set_xlabel("Mean [CLS] Attention Weight (%)", fontsize=10, fontweight="bold")
    ax_a.set_title("A. Level-1 Feature Transformer Attention\n(5-Fold Mean Biomarker Importance)", fontsize=11, fontweight="bold")
    ax_a.grid(axis="x", linestyle="--", alpha=0.35)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)
    for bar, v in zip(bars_a, vals_a, strict=True):
        ax_a.text(v + 0.25, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.5)

    # Panel B: Patient Case Study — Inter-Patient Cohort Neighbor Graph Attention
    ax_b = axes[1]
    neighbors = case_expl["top_cohort_neighbors"][:5][::-1]
    n_labels = [
        f"Train Pt #{n['neighbor_idx']} ({'CVD+' if y_train[n['neighbor_idx']] == 1 else 'Healthy'})"
        for n in neighbors
    ]
    n_weights = [n["attention_weight"] * 100.0 for n in neighbors]
    n_colors = [
        "#c0392b" if y_train[n["neighbor_idx"]] == 1 else "#2e8b57"
        for n in neighbors
    ]
    bars_b = ax_b.barh(n_labels, n_weights, color=n_colors, height=0.58)
    ax_b.set_xlabel("Inductive GAT Neighbor Attention (%)", fontsize=10, fontweight="bold")
    outcome_str = "CVD+" if case_true_label == 1 else "Healthy"
    ax_b.set_title(
        f"B. Level-2 Cohort Graph Attention (Case Study)\n"
        f"Pred Risk: {case_expl['risk_probability']*100:.1f}% | True: {outcome_str} | "
        f"Graph Gate: {case_expl['graph_reliance_gate']:.2f}",
        fontsize=10.5,
        fontweight="bold",
    )
    ax_b.grid(axis="x", linestyle="--", alpha=0.35)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    for bar, v in zip(bars_b, n_weights, strict=True):
        ax_b.text(v + 0.4, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.5)

    # Panel C: 5-Fold Stratified CV Ablation Comparison (with std error bars)
    ax_c = axes[2]
    metrics = ["roc_auc", "pr_auc", "f1"]
    metric_labels = ["ROC-AUC", "PR-AUC", "F1-Score"]
    x = np.arange(len(metrics))
    width = 0.25
    palette = ["#7f8c8d", "#2980b9", "#d35400"]

    for idx, (_, row) in enumerate(cv_summary_df.iterrows()):
        means = [row[f"{m}_mean"] for m in metrics]
        stds = [row[f"{m}_std"] for m in metrics]
        offset = (idx - (len(cv_summary_df) - 1) / 2) * width
        bars = ax_c.bar(
            x + offset,
            means,
            yerr=stds,
            capsize=3,
            width=width,
            label=row["model"],
            color=palette[idx % len(palette)],
            error_kw={"elinewidth": 1.1, "ecolor": "#2c3e50"},
        )
        for b, v, s in zip(bars, means, stds, strict=True):
            ax_c.text(b.get_x() + b.get_width() / 2, v + s + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax_c.set_xticks(x)
    ax_c.set_xticklabels(metric_labels, fontsize=10, fontweight="bold")
    ax_c.set_ylim(0.5, 1.05)
    ax_c.set_ylabel("5-Fold Stratified CV Score", fontsize=10, fontweight="bold")
    ax_c.set_title("C. Dual-Attention 5-Fold CV Ablation\n(UCI Multi-Center Cohort, N=918)", fontsize=11, fontweight="bold")
    ax_c.legend(loc="lower right", fontsize=8, frameon=True)
    ax_c.grid(axis="y", linestyle="--", alpha=0.35)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_multicohort_scale_panel(
    uci_df: pd.DataFrame,
    mimic_df: pd.DataFrame,
    large_df: pd.DataFrame,
    large_feat_attn: dict[str, float],
    output_path: Path,
) -> None:
    """Create a 3-panel multi-cohort & large-scale (N=68,589) verification figure."""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), dpi=180)
    fig.patch.set_facecolor("white")

    # Panel A: Cross-Cohort ROC-AUC (UCI 5-Fold CV, MIMIC-IV 5-Fold Grouped CV, 68.6k Large CVD)
    ax_a = axes[0]
    cohorts = ["UCI Multi-Center\n(N=918, 5-Fold CV)", "MIMIC-IV ICU\n(N=275, Grouped CV)", "OpenML Large CVD\n(N=68,589, Mini-Batch)"]
    tab_aucs = [
        float(uci_df.loc[uci_df["model"].str.contains("TabTransformer"), "roc_auc_mean"].iloc[0]),
        float(mimic_df.loc[mimic_df["model"].str.contains("TabTransformer"), "roc_auc_mean"].iloc[0]),
        float(large_df.loc[large_df["model"].str.contains("TabTransformer"), "roc_auc"].iloc[0]),
    ]
    gt_aucs = [
        float(uci_df.loc[uci_df["model"].str.contains("Graph-Augmented"), "roc_auc_mean"].iloc[0]),
        float(mimic_df.loc[mimic_df["model"].str.contains("Graph-Augmented"), "roc_auc_mean"].iloc[0]),
        float(large_df.loc[large_df["model"].str.contains("Graph-Augmented"), "roc_auc"].iloc[0]),
    ]
    x = np.arange(len(cohorts))
    w = 0.34
    b1 = ax_a.bar(x - w / 2, tab_aucs, width=w, label="TabTransformer (No Graph)", color="#7f8c8d")
    b2 = ax_a.bar(x + w / 2, gt_aucs, width=w, label="Graph-Augmented Transformer (Ours)", color="#d35400")
    for bars in (b1, b2):
        for b in bars:
            v = b.get_height()
            ax_a.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cohorts, fontsize=9, fontweight="bold")
    ax_a.set_ylim(0.5, 1.05)
    ax_a.set_ylabel("ROC-AUC", fontsize=10, fontweight="bold")
    ax_a.set_title("A. Cross-Cohort & Large-Scale Generalization\n(Cardiology + ICU Critical Care)", fontsize=11, fontweight="bold")
    ax_a.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax_a.grid(axis="y", linestyle="--", alpha=0.35)
    ax_a.spines["top"].set_visible(False)
    ax_a.spines["right"].set_visible(False)

    # Panel B: Large-Scale N=68,589 Biomarker Attention
    ax_b = axes[1]
    top_large = list(large_feat_attn.items())[:10][::-1]
    l_names = [k for k, _ in top_large]
    l_vals = [v * 100.0 for _, v in top_large]
    bars_l = ax_b.barh(l_names, l_vals, color="#1b6b59", height=0.66)
    ax_b.set_xlabel("Mean [CLS] Attention Weight (%)", fontsize=10, fontweight="bold")
    ax_b.set_title("B. Large-Scale (N=68,589) Biomarker Attention\n(O(B·k) Mini-Batch Inductive Mode)", fontsize=11, fontweight="bold")
    ax_b.grid(axis="x", linestyle="--", alpha=0.35)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    for bar, v in zip(bars_l, l_vals, strict=True):
        ax_b.text(v + 0.25, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=8.5)

    # Panel C: Calibration & Probability Quality (Brier Score across all 3 cohorts, lower is better)
    ax_c = axes[2]
    tab_briers = [
        float(uci_df.loc[uci_df["model"].str.contains("TabTransformer"), "brier_mean"].iloc[0]),
        float(mimic_df.loc[mimic_df["model"].str.contains("TabTransformer"), "brier_mean"].iloc[0]),
        float(large_df.loc[large_df["model"].str.contains("TabTransformer"), "brier"].iloc[0]),
    ]
    gt_briers = [
        float(uci_df.loc[uci_df["model"].str.contains("Graph-Augmented"), "brier_mean"].iloc[0]),
        float(mimic_df.loc[mimic_df["model"].str.contains("Graph-Augmented"), "brier_mean"].iloc[0]),
        float(large_df.loc[large_df["model"].str.contains("Graph-Augmented"), "brier"].iloc[0]),
    ]
    c1 = ax_c.bar(x - w / 2, tab_briers, width=w, label="TabTransformer (No Graph)", color="#7f8c8d")
    c2 = ax_c.bar(x + w / 2, gt_briers, width=w, label="Graph-Augmented Transformer (Ours)", color="#2471a3")
    for bars in (c1, c2):
        for b in bars:
            v = b.get_height()
            ax_c.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(cohorts, fontsize=9, fontweight="bold")
    ax_c.set_ylim(0.0, max(max(tab_briers), max(gt_briers)) + 0.06)
    ax_c.set_ylabel("Brier Score (Lower is Better)", fontsize=10, fontweight="bold")
    ax_c.set_title("C. Probabilistic Calibration Quality\n(Validation Temperature-Scaled Brier Score)", fontsize=11, fontweight="bold")
    ax_c.legend(loc="upper right", fontsize=8.5, frameon=True)
    ax_c.grid(axis="y", linestyle="--", alpha=0.35)
    ax_c.spines["top"].set_visible(False)
    ax_c.spines["right"].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Multi-Cohort & Large-Scale Graph-Augmented Clinical Transformer Suite.")
    parser.add_argument(
        "--mode",
        choices=["all", "uci_cv", "mimic_cv", "large_scale"],
        default="all",
        help="Which evaluation regime(s) to run.",
    )
    parser.add_argument("--dataset", type=str, default="cleaned_heart_disease_combined.csv")
    parser.add_argument("--mimic-dataset", type=str, default="mimic_hospital_mortality.csv")
    parser.add_argument("--large-dataset", type=str, default="cleaned_cardio_70k.csv")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--large-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--k-neighbors", type=int, default=7)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="artifacts")
    args = parser.parse_args(argv)
    out_dir = str(resolve_output_dir(args.output_dir))

    uci_df = mimic_df = large_df = None
    large_meta: dict = {}

    if args.mode in {"all", "uci_cv"}:
        uci_df, _ = run_uci_5fold_cv(
            dataset_path=args.dataset,
            cv_folds=args.cv_folds,
            epochs=args.epochs,
            k_neighbors=args.k_neighbors,
            embed_dim=args.embed_dim,
            seed=args.seed,
            output_dir=out_dir,
        )

    if args.mode in {"all", "mimic_cv"}:
        mimic_df = run_mimic_grouped_cv(
            dataset_path=args.mimic_dataset,
            cv_folds=args.cv_folds,
            epochs=min(args.epochs, 70),
            k_neighbors=min(args.k_neighbors, 5),
            embed_dim=args.embed_dim,
            seed=args.seed,
            output_dir=out_dir,
        )

    if args.mode in {"all", "large_scale"}:
        large_df, large_meta = run_large_scale_minibatch_study(
            dataset_path=args.large_dataset,
            epochs=args.large_epochs,
            batch_size=args.batch_size,
            k_neighbors=args.k_neighbors,
            embed_dim=args.embed_dim,
            seed=args.seed,
            output_dir=out_dir,
        )

    if uci_df is not None and mimic_df is not None and large_df is not None:
        out_p = Path(out_dir)
        fig_path = out_p / "figures" / "graph_transformer_multicohort_scale.png"
        _plot_multicohort_scale_panel(
            uci_df=uci_df,
            mimic_df=mimic_df,
            large_df=large_df,
            large_feat_attn=large_meta["global_feat_attn"],
            output_path=fig_path,
        )

        # Write consolidated Markdown summary report
        md_lines = [
            "# Graph-Augmented Clinical Transformer — Multi-Cohort & Large-Scale Benchmark\n",
            "## 1. UCI Multi-Center Heart Disease (N=918, 5-Fold Stratified CV)\n",
            "| Model | ROC-AUC | PR-AUC | Accuracy | F1 | Brier |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
        for _, r in uci_df.iterrows():
            md_lines.append(
                f"| {r['model']} | {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} | "
                f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f} | "
                f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} | "
                f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f} | "
                f"{r['brier_mean']:.4f} ± {r['brier_std']:.4f} |"
            )
        md_lines.extend([
            "\n## 2. MIMIC-IV ICU Mortality (N=275, 5-Fold Patient-Grouped CV)\n",
            "| Model | ROC-AUC | PR-AUC | Accuracy | F1 | Brier |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for _, r in mimic_df.iterrows():
            md_lines.append(
                f"| {r['model']} | {r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} | "
                f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f} | "
                f"{r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} | "
                f"{r['f1_mean']:.4f} ± {r['f1_std']:.4f} | "
                f"{r['brier_mean']:.4f} ± {r['brier_std']:.4f} |"
            )
        md_lines.extend([
            f"\n## 3. Large-Scale Cardiovascular Cohort (OpenML 45547, N={large_meta['n_total']:,}, Mini-Batch Inductive)\n",
            "| Model | ROC-AUC | PR-AUC | Accuracy | F1 | Brier | ECE | Train Time (s) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])
        for _, r in large_df.iterrows():
            md_lines.append(
                f"| {r['model']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | "
                f"{r['accuracy']:.4f} | {r['f1']:.4f} | {r['brier']:.4f} | "
                f"{r['ece']:.4f} | {r['train_seconds']:.1f}s |"
            )
        (out_p / "graph_transformer_benchmark.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"\nSaved multi-cohort summary & figure:\n  - {out_p / 'graph_transformer_benchmark.md'}\n  - {fig_path}")


if __name__ == "__main__":
    main()
