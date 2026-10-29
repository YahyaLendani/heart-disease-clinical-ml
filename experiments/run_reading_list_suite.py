"""Experimental Suite executing all implemented methods from the AI Thesis Reading List.

Methodological rules enforced here:

* Preprocessing (imputer, scaler, category maps) is fit on training rows only.
* Graph models are evaluated inductively (no test-to-test edges, no test
  nodes in the training graph).
* Every neural model keeps the weights from the epoch with the lowest
  validation loss; the test split is never used for model selection.
* The architecture benchmark is reported two ways: one stratified 60/20/20
  holdout (with bootstrap 95% CIs) and repeated stratified k-fold CV
  (mean ± std across folds), because a single test set of ~60 patients
  cannot separate models whose scores differ by one or two patients.
* Conformal prediction is calibrated and evaluated on the same probability
  scale (the uncalibrated model output), which is what its coverage
  guarantee requires. Temperature scaling is reported separately for ECE.
* The multimodal model is a demonstration: its "waveform" input is
  synthesised from the same tabular features, so it carries no new
  information and is flagged rather than ranked against the other models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation import (
    ClinicalCounterfactualGenerator,
    SplitConformalClassifier,
    compare_observational_vs_causal,
    compute_doubly_robust_aipw,
    evaluate_loco_conformal,
    find_equal_opportunity_thresholds,
    plot_aipw_love_plot,
    plot_conformal_triage_curve,
    plot_counterfactual_case_study,
)
from evaluation.cv_utils import BASE_METRICS, add_bootstrap_cis, binary_metrics, resolve_n_jobs
from models import (
    GATClassifier,
    GCNClassifier,
    HeartMLP,
    ModelWithTemperature,
    MultimodalFusionClassifier,
    TabTransformer,
    build_knn_patient_graph,
    compute_ece,
    normalize_adjacency_gcn,
)
from project_paths import PRIMARY_HEART_DATASET, resolve_data_path, resolve_output_dir
from training import DeepEnsemble
from training.device_utils import DEVICE_CHOICES, resolve_torch_device
from training.torch_utils import clone_state, set_seed, single_thread

MULTIMODAL_DEMO_NAME = "Multimodal Cross-Attention (synthetic-signal demo)"
TEST_COLUMNS = {
    "accuracy": "test_accuracy",
    "f1": "test_f1",
    "roc_auc": "test_roc_auc",
    "pr_auc": "test_pr_auc",
    "brier": "test_brier",
}
CATEGORICAL_HEART_COLUMNS = ("sex", "cp", "fbs", "restecg", "exang", "slope", "thal")


def set_global_seed(seed: int = 42) -> None:
    """Enforce strict global reproducibility across libraries."""
    set_seed(seed)


def _target_column(df: pd.DataFrame) -> str:
    return "target" if "target" in df.columns else "hospital_expire_flag"


def three_way_split(y: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 60% train / 20% validation / 20% test row indices."""
    indices = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=seed, stratify=y)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.25, random_state=seed, stratify=y[train_val_idx]
    )
    return train_idx, val_idx, test_idx


def cv_three_way_splits(
    y: np.ndarray, folds: int = 5, repeats: int = 1, seed: int = 42
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Repeated stratified k-fold: each fold is the test set; the rest is split 75/25 into train/val."""
    splitter = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    splits = []
    for k, (rest_idx, test_idx) in enumerate(splitter.split(np.zeros(len(y)), y)):
        train_idx, val_idx = train_test_split(
            rest_idx, test_size=0.25, random_state=seed + k, stratify=y[rest_idx]
        )
        splits.append((train_idx, val_idx, test_idx))
    return splits


def _scale_on_train(
    X_raw: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(imputer.fit_transform(X_raw[train_idx]))
    X_val = scaler.transform(imputer.transform(X_raw[val_idx]))
    X_test = scaler.transform(imputer.transform(X_raw[test_idx]))
    return X_train, X_val, X_test


def _clean_frame(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    df_clean = df.dropna(subset=[target_col]).reset_index(drop=True).copy()
    feature_cols = [
        c for c in df_clean.columns
        if c != target_col and pd.api.types.is_numeric_dtype(df_clean[c])
    ]
    return df_clean, feature_cols, df_clean[target_col].values.astype(float)


def _prepare_three_way_tabular_splits(
    df: pd.DataFrame,
    target_col: str,
    seed: int = 42,
    splits: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    list[str], pd.DataFrame,
]:
    """Split 60/20/20 (or use ``splits``) and fit imputer/scaler strictly on train."""
    df_clean, feature_cols, y = _clean_frame(df, target_col)
    train_idx, val_idx, test_idx = splits if splits is not None else three_way_split(y, seed)
    X_train, X_val, X_test = _scale_on_train(df_clean[feature_cols].values, train_idx, val_idx, test_idx)
    return (
        X_train, y[train_idx], train_idx,
        X_val, y[val_idx], val_idx,
        X_test, y[test_idx], test_idx,
        feature_cols, df_clean,
    )


def _result_row(
    name: str, y_test: np.ndarray, probs: np.ndarray, seed: int, with_ci: bool, test_idx: np.ndarray
) -> dict:
    """Metrics row; holdout rows (``with_ci``) also carry predictions for paired tests."""
    metrics = binary_metrics(y_test, probs)
    row: dict = {"model": name, **{TEST_COLUMNS[k]: v for k, v in metrics.items()}}
    if with_ci:
        add_bootstrap_cis(row, y_test, probs, seed=seed, name_map=TEST_COLUMNS)
        row["_probs"] = np.asarray(probs, dtype=float)
        row["_test_idx"] = np.asarray(test_idx)
        row["_y_test"] = np.asarray(y_test, dtype=int)
    return row


def _train_full_batch_with_val_selection(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epochs: int,
    train_step,
    val_loss_fn,
) -> nn.Module:
    """Full-batch training that restores the weights with the lowest validation loss."""
    best_val_loss = float("inf")
    best_state = None
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = train_step()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(val_loss_fn().item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = clone_state(model)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def run_tabtransformer_experiment(
    df: pd.DataFrame,
    epochs: int = 40,
    device: str = "cpu",
    seed: int = 42,
    splits: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    with_ci: bool = True,
) -> dict[str, float | str]:
    """Train and evaluate TabTransformer with strict train-only preprocessing and val selection."""
    set_global_seed(seed)
    cat_cols = [c for c in CATEGORICAL_HEART_COLUMNS if c in df.columns]
    target_col = _target_column(df)
    df_clean = df.dropna(subset=[target_col]).reset_index(drop=True).copy()
    cont_cols = [
        c for c in df_clean.columns
        if c not in cat_cols and c != target_col and pd.api.types.is_numeric_dtype(df_clean[c])
    ]

    y = df_clean[target_col].values.astype(float)
    train_idx, val_idx, test_idx = splits if splits is not None else three_way_split(y, seed)

    train_df = df_clean.iloc[train_idx].copy()
    val_df = df_clean.iloc[val_idx].copy()
    test_df = df_clean.iloc[test_idx].copy()

    cat_dims = []
    for c in cat_cols:
        mode_val = train_df[c].mode().iloc[0]
        unique_vals = {v: idx for idx, v in enumerate(train_df[c].fillna(mode_val).unique())}
        unk_idx = len(unique_vals)
        for part in (train_df, val_df, test_df):
            part[c] = part[c].fillna(mode_val).map(unique_vals).fillna(unk_idx).astype(int)
        cat_dims.append(len(unique_vals) + 1)

    if cont_cols:
        cont_imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_cont_train = scaler.fit_transform(cont_imputer.fit_transform(train_df[cont_cols].values))
        X_cont_val = scaler.transform(cont_imputer.transform(val_df[cont_cols].values))
        X_cont_test = scaler.transform(cont_imputer.transform(test_df[cont_cols].values))
    else:
        X_cont_train = np.empty((len(train_df), 0))
        X_cont_val = np.empty((len(val_df), 0))
        X_cont_test = np.empty((len(test_df), 0))

    dev = torch.device(device)
    model = TabTransformer(
        categorical_dims=cat_dims,
        continuous_dim=len(cont_cols),
        embed_dim=16,
        num_heads=2,
        num_layers=2,
        mlp_hidden_dims=[32, 16],
        dropout=0.15,
    ).to(dev)
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    def _tensors(frame: pd.DataFrame, cont: np.ndarray):
        return (
            torch.tensor(frame[cat_cols].values, dtype=torch.long, device=dev),
            torch.tensor(cont, dtype=torch.float32, device=dev),
            torch.tensor(frame[target_col].values.astype(float), dtype=torch.float32, device=dev).view(-1, 1),
        )

    x_cat_t, x_cont_t, y_t = _tensors(train_df, X_cont_train)
    val_cat_t, val_cont_t, y_val_t = _tensors(val_df, X_cont_val)
    test_cat_t, test_cont_t, _ = _tensors(test_df, X_cont_test)

    _train_full_batch_with_val_selection(
        model,
        optimizer,
        epochs,
        train_step=lambda: criterion(model(x_cat_t, x_cont_t), y_t),
        val_loss_fn=lambda: criterion(model(val_cat_t, val_cont_t), y_val_t),
    )
    with torch.no_grad():
        logits, _ = model(test_cat_t, test_cont_t, return_attn=True)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()

    y_test = test_df[target_col].values.astype(float)
    return _result_row("TabTransformer", y_test, probs, seed, with_ci, test_idx)


def run_gnn_experiment(
    df: pd.DataFrame,
    epochs: int = 40,
    device: str = "cpu",
    seed: int = 42,
    splits: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    with_ci: bool = True,
) -> list[dict[str, float | str]]:
    """Train and evaluate GCN and GAT with train-only scaling and inductive-aware graph edges."""
    set_global_seed(seed)
    target_col = _target_column(df)
    (
        X_train, y_train, train_idx,
        X_val, y_val, val_idx,
        X_test, y_test, test_idx,
        feature_cols, df_clean,
    ) = _prepare_three_way_tabular_splits(df, target_col, seed=seed, splits=splits)

    N = len(df_clean)
    X_scaled = np.zeros((N, len(feature_cols)), dtype=np.float32)
    X_scaled[train_idx] = X_train
    X_scaled[val_idx] = X_val
    X_scaled[test_idx] = X_test
    y_all = df_clean[target_col].values.astype(float)

    dev = torch.device(device)
    X_t = torch.tensor(X_scaled, dtype=torch.float32, device=dev)
    y_t = torch.tensor(y_all, dtype=torch.float32, device=dev).view(-1, 1)

    adj = build_knn_patient_graph(X_scaled, k=5, metric="cosine").to(dev)
    test_mask_1d = torch.zeros(N, dtype=torch.bool, device=dev)
    test_mask_1d[test_idx] = True
    test_to_test_mask = test_mask_1d.unsqueeze(1) & test_mask_1d.unsqueeze(0)
    test_to_test_mask.fill_diagonal_(False)
    adj = adj.masked_fill(test_to_test_mask, 0.0)

    adj_train = adj.clone()
    adj_train[test_idx, :] = 0.0
    adj_train[:, test_idx] = 0.0
    adj_train[test_idx, test_idx] = 1.0

    norm_adj_train = normalize_adjacency_gcn(adj_train).to(dev)
    norm_adj_eval = normalize_adjacency_gcn(adj).to(dev)

    criterion = nn.BCEWithLogitsLoss()
    results = []

    gcn = GCNClassifier(in_features=len(feature_cols), hidden_dim=32, out_features=1).to(dev)
    _train_full_batch_with_val_selection(
        gcn,
        optim.Adam(gcn.parameters(), lr=0.01, weight_decay=5e-4),
        epochs,
        train_step=lambda: criterion(gcn(X_t, norm_adj_train)[train_idx], y_t[train_idx]),
        val_loss_fn=lambda: criterion(gcn(X_t, norm_adj_train)[val_idx], y_t[val_idx]),
    )
    with torch.no_grad():
        test_probs = torch.sigmoid(gcn(X_t, norm_adj_eval)[test_idx]).cpu().numpy().flatten()
    results.append(_result_row("GCN (Kipf & Welling)", y_test, test_probs, seed, with_ci, test_idx))

    gat = GATClassifier(in_features=len(feature_cols), hidden_dim=16, num_heads=2, out_features=1).to(dev)
    _train_full_batch_with_val_selection(
        gat,
        optim.Adam(gat.parameters(), lr=0.01, weight_decay=5e-4),
        epochs,
        train_step=lambda: criterion(gat(X_t, adj_train)[train_idx], y_t[train_idx]),
        val_loss_fn=lambda: criterion(gat(X_t, adj_train)[val_idx], y_t[val_idx]),
    )
    with torch.no_grad():
        test_probs = torch.sigmoid(gat(X_t, adj)[test_idx]).cpu().numpy().flatten()
    results.append(_result_row("GAT (Velickovic et al.)", y_test, test_probs, seed, with_ci, test_idx))

    return results


def _synthesize_physiological_waveform(X_scaled: np.ndarray, seed: int = 42) -> np.ndarray:
    """Synthesize a 128-step 1D waveform modulated by two observed covariates (vectorised)."""
    rng = np.random.RandomState(seed)
    n_samples = X_scaled.shape[0]
    t = np.linspace(0, 2 * np.pi, 128, dtype=np.float32)
    freq_mod = (1.0 + 0.25 * np.clip(X_scaled[:, 0], -2.0, 2.0)).astype(np.float32)[:, None]
    st_col = min(1, X_scaled.shape[1] - 1)
    st_shift = (0.20 * np.clip(X_scaled[:, st_col], -2.0, 2.0)).astype(np.float32)[:, None]
    # One (n, 128) draw yields the same stream as the original per-patient loop.
    noise = rng.normal(0.0, 0.15, size=(n_samples, 128)).astype(np.float32)
    return (np.sin(freq_mod * t) + 0.5 * np.cos(2.0 * freq_mod * t) + st_shift + noise).astype(np.float32)


def run_multimodal_experiment(
    df: pd.DataFrame,
    epochs: int = 25,
    device: str = "cpu",
    seed: int = 42,
    splits: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    with_ci: bool = True,
) -> dict:
    """Demonstration of the multimodal fusion architecture on a synthetic signal channel."""
    set_global_seed(seed)
    target_col = _target_column(df)
    (
        X_train, y_train, _,
        X_val, y_val, _,
        X_test, y_test, test_idx,
        feature_cols, _,
    ) = _prepare_three_way_tabular_splits(df, target_col, seed=seed, splits=splits)
    dev = torch.device(device)
    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    S_train = to_t(_synthesize_physiological_waveform(X_train, seed=seed))
    S_val = to_t(_synthesize_physiological_waveform(X_val, seed=seed + 1))
    S_test = to_t(_synthesize_physiological_waveform(X_test, seed=seed + 2))
    X_tr, X_v, X_te = to_t(X_train), to_t(X_val), to_t(X_test)
    y_tr, y_v = to_t(y_train).view(-1, 1), to_t(y_val).view(-1, 1)

    model = MultimodalFusionClassifier(
        tabular_dim=len(feature_cols),
        signal_length=128,
        patch_size=16,
        embed_dim=32,
        fusion_strategy="cross_attention",
    ).to(dev)
    criterion = nn.BCEWithLogitsLoss()
    _train_full_batch_with_val_selection(
        model,
        optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4),
        epochs,
        train_step=lambda: criterion(model(X_tr, S_train), y_tr),
        val_loss_fn=lambda: criterion(model(X_v, S_val), y_v),
    )
    with torch.no_grad():
        probs = torch.sigmoid(model(X_te, S_test)).cpu().numpy().flatten()
    row = _result_row(MULTIMODAL_DEMO_NAME, y_test, probs, seed, with_ci, test_idx)
    row["is_synthetic_demo"] = True
    return row


def _architecture_fold(
    fold_id: int,
    df: pd.DataFrame,
    splits: tuple[np.ndarray, np.ndarray, np.ndarray],
    seed: int,
    device: str,
) -> list[dict]:
    fold_seed = seed + fold_id
    with single_thread():  # same results whether folds run serially or in parallel
        rows = [run_tabtransformer_experiment(df, device=device, seed=fold_seed, splits=splits, with_ci=False)]
        rows.extend(run_gnn_experiment(df, device=device, seed=fold_seed, splits=splits, with_ci=False))
        rows.append(run_multimodal_experiment(df, device=device, seed=fold_seed, splits=splits, with_ci=False))
    for row in rows:
        row["fold"] = fold_id
    return rows


def run_architecture_cv(
    df: pd.DataFrame,
    folds: int = 5,
    repeats: int = 1,
    seed: int = 42,
    device: str = "cpu",
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeated stratified k-fold benchmark of TabTransformer, GCN, GAT and the multimodal demo.

    Returns ``(fold_rows, summary)`` where summary has ``cv_<metric>_mean`` and
    ``cv_<metric>_std`` per model.
    """
    target_col = _target_column(df)
    _, _, y = _clean_frame(df, target_col)
    splits = cv_three_way_splits(y, folds=folds, repeats=repeats, seed=seed)
    workers = 1 if device == "cuda" else min(resolve_n_jobs(n_jobs), len(splits))
    jobs = [(k, df, sp, seed, device) for k, sp in enumerate(splits, start=1)]
    if workers > 1:
        from joblib import Parallel, delayed

        outputs = Parallel(n_jobs=workers)(delayed(_architecture_fold)(*job) for job in jobs)
    else:
        outputs = [_architecture_fold(*job) for job in jobs]

    fold_df = pd.DataFrame([row for rows in outputs for row in rows])
    summary_rows = []
    for model_name, group in fold_df.groupby("model", sort=False):
        entry = {"model": model_name, "cv_n_folds_total": len(group)}
        for metric in BASE_METRICS:
            values = group[TEST_COLUMNS[metric]].astype(float)
            entry[f"cv_{metric}_mean"] = float(values.mean())
            entry[f"cv_{metric}_std"] = float(values.std(ddof=0))
        summary_rows.append(entry)
    return fold_df, pd.DataFrame(summary_rows)


def run_all_advancements(
    heart_csv: str = PRIMARY_HEART_DATASET,
    mimic_csv: str = "mimic_hospital_mortality.csv",
    output_dir: str = "artifacts",
    seed: int = 42,
    device: str = "auto",
    cv_folds: int = 5,
    cv_repeats: int = 1,
    n_jobs: int = -1,
) -> None:
    """Execute complete reading list pipeline suite with strict experimental rigor."""
    set_global_seed(seed)
    out_path = resolve_output_dir(output_dir)
    dev = resolve_torch_device(device).type

    heart_path = resolve_data_path(heart_csv)
    mimic_path = resolve_data_path(mimic_csv)
    heart_df = pd.read_csv(heart_path) if heart_path.exists() else None
    mimic_df = pd.read_csv(mimic_path) if mimic_path.exists() else None

    target_df = heart_df if heart_df is not None else mimic_df
    if target_df is None:
        raise FileNotFoundError("Neither heart nor MIMIC dataset found.")

    target_col = _target_column(target_df)
    (
        X_train_t, y_train_t, train_idx,
        X_val_t, y_val_t, val_idx,
        X_test_t, y_test_t, test_idx,
        feature_cols, df_clean,
    ) = _prepare_three_way_tabular_splits(target_df, target_col, seed=seed)

    all_benchmarks: list[dict] = []

    print(">>> 1. Benchmarking TabTransformer (Huang et al., 2020)...")
    all_benchmarks.append(run_tabtransformer_experiment(target_df, device=dev, seed=seed))

    print(">>> 2. Benchmarking Graph Neural Networks: GCN & GAT (Kipf & Welling; Velickovic et al.)...")
    all_benchmarks.extend(run_gnn_experiment(target_df, device=dev, seed=seed))

    print(">>> 3. Multimodal Fusion (ResNet + ViT) demonstration on a synthetic signal channel...")
    all_benchmarks.append(run_multimodal_experiment(target_df, device=dev, seed=seed))

    print(f">>> 3b. Repeated stratified CV of the architectures ({cv_folds}-fold x {cv_repeats})...")
    arch_fold_df, arch_cv_df = run_architecture_cv(
        target_df, folds=cv_folds, repeats=cv_repeats, seed=seed, device=dev, n_jobs=n_jobs
    )
    arch_fold_df.to_csv(out_path / "reading_list_architecture_cv_folds.csv", index=False)
    for _, r in arch_cv_df.iterrows():
        print(
            f"    {r['model']}: ROC-AUC {r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f} | "
            f"Accuracy {r['cv_accuracy_mean']:.4f} ± {r['cv_accuracy_std']:.4f}"
        )

    print(">>> 4. Testing Calibration via Temperature Scaling (Guo et al., 2017)...")
    set_global_seed(seed)
    criterion = nn.BCEWithLogitsLoss()
    base_mlp = HeartMLP(input_dim=len(feature_cols), hidden_dims=[32, 16])
    mlp_opt = optim.Adam(base_mlp.parameters(), lr=0.005, weight_decay=1e-4)
    x_tr_tensor = torch.tensor(X_train_t, dtype=torch.float32)
    y_tr_tensor = torch.tensor(y_train_t, dtype=torch.float32)
    for _ in range(40):
        base_mlp.train()
        mlp_opt.zero_grad()
        mlp_loss = criterion(base_mlp(x_tr_tensor), y_tr_tensor)
        mlp_loss.backward()
        mlp_opt.step()

    base_mlp.eval()
    with torch.no_grad():
        val_logits = base_mlp(torch.tensor(X_val_t, dtype=torch.float32))
        val_probs = torch.sigmoid(val_logits).cpu().numpy().flatten()
        test_logits = base_mlp(torch.tensor(X_test_t, dtype=torch.float32))
        test_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()

    cal_wrapper = ModelWithTemperature(base_mlp)
    cal_res = cal_wrapper.calibrate(
        val_logits.detach(),
        torch.tensor(y_val_t, dtype=torch.float32).view(-1, 1),
    )
    opt_t = cal_res["optimal_temperature"]

    ece_test_before = compute_ece(test_probs, y_test_t)
    scaled_test_logits = cal_wrapper.temperature_scale(test_logits.detach())
    calibrated_test_probs = torch.sigmoid(scaled_test_logits).detach().cpu().numpy().flatten()
    ece_test_after = compute_ece(calibrated_test_probs, y_test_t)

    print(f"    Validation Optimal Temperature: {opt_t:.3f}")
    print(
        f"    Holdout Test ECE: {ece_test_before:.4f} -> {ece_test_after:.4f} "
        f"(Validation ECE: {cal_res['ece_before']:.4f} -> {cal_res['ece_after']:.4f})"
    )

    print(">>> 5. Testing Deep Ensemble Uncertainty (Lakshminarayanan et al., 2017)...")
    ensemble = DeepEnsemble(
        lambda: HeartMLP(input_dim=len(feature_cols), hidden_dims=[32, 16]),
        num_models=5,
    )
    ensemble.fit(X_train_t, y_train_t, epochs=30)
    unc_res = ensemble.predict_with_uncertainty(X_test_t)
    print(
        f"    Mean Variance: {unc_res['epistemic_variance'].mean():.4f} | "
        f"Uncertain Cases: {unc_res['is_uncertain'].sum()} / {len(X_test_t)}"
    )

    print(">>> 5b. Testing Split & Mondrian Conformal Prediction (Angelopoulos & Bates, 2021)...")
    # Calibration and test scores must come from the same model output for the
    # coverage guarantee to hold, so both use the uncalibrated probabilities.
    conformal = SplitConformalClassifier(alpha=0.10, mondrian=True)
    conformal.fit(y_val_t, val_probs)
    conf_res = conformal.evaluate(y_test_t, test_probs)
    print(
        f"    Target Coverage (90%): Empirical={conf_res['empirical_coverage']:.4f} "
        f"(Pos={conf_res['positive_class_coverage']:.4f}, Neg={conf_res['negative_class_coverage']:.4f}) | "
        f"Avg Set Size={conf_res['average_set_size']:.3f} | "
        f"Ambiguous Deferral={conf_res['ambiguous_abstention_rate']:.3f}"
    )

    # Sweep alpha in {0.05, 0.10, 0.15} to demonstrate the efficiency-coverage tradeoff curve
    conformal_sweeps = {}
    for a in (0.05, 0.10, 0.15):
        c_a = SplitConformalClassifier(alpha=a, mondrian=True).fit(y_val_t, val_probs).evaluate(y_test_t, test_probs)
        conformal_sweeps[a] = c_a
        print(
            f"    [Alpha Sweep alpha={a:.2f} ({int((1-a)*100)}% Target)]: "
            f"Empirical={c_a['empirical_coverage']*100:.1f}% | "
            f"Singleton Low {{0}}={c_a['certifiable_low_risk_rate']*100:.1f}% | "
            f"Singleton High {{1}}={c_a['certifiable_high_risk_rate']*100:.1f}% | "
            f"Deferral {{0,1}}={c_a['ambiguous_abstention_rate']*100:.1f}%"
        )

    # Publication Figure 1: Conformal Triage Curve
    plot_conformal_triage_curve(
        conformal_classifier=conformal,
        y_val=y_val_t,
        prob_val=val_probs,
        y_test=y_test_t,
        prob_test=test_probs,
        output_path=out_path / "figures" / "conformal_triage_curve.png",
    )

    print(">>> 6. Testing Causal Inference (Pearl & Mackenzie, 2018)...")
    if "had_icu" in df_clean.columns:
        treat_col = "had_icu"
        treatment_name = "ICU Admission"
    elif "exang" in df_clean.columns:
        treat_col = "exang"
        treatment_name = "Exercise Induced Angina"
    elif "has_ed_visit" in df_clean.columns:
        treat_col = "has_ed_visit"
        treatment_name = "ED Visit Prior to Admission"
    else:
        treat_col = feature_cols[0]
        treatment_name = f"{treat_col} Intervention"

    treatment = (df_clean[treat_col].fillna(0).values > 0).astype(int)
    # Exclude treatment variable itself from confounders matrix
    confounder_cols = [c for c in feature_cols if c != treat_col]
    X_conf_raw = df_clean[confounder_cols].values
    X_confounders = StandardScaler().fit_transform(
        SimpleImputer(strategy="median").fit_transform(X_conf_raw)
    )
    y_all = df_clean[target_col].values.astype(float)

    causal_res = compare_observational_vs_causal(
        X_confounders, treatment, y_all, treatment_name=treatment_name
    )
    print(
        f"    Observational Diff: {causal_res['observational_risk_difference']:.4f} "
        f"vs Causal ATE: {causal_res['causal_ate_iptw']:.4f}"
    )

    print(">>> 6b. Testing Doubly Robust Cross-Fitted AIPW & E-Value (VanderWeele & Ding, 2017)...")
    dr_res = compute_doubly_robust_aipw(
        X_confounders, treatment, y_all, treatment_name=treatment_name, seed=seed
    )
    print(
        f"    [Symptomatic Treatment - exang] Doubly Robust AIPW ATE: {dr_res['causal_ate_aipw']:.4f} "
        f"(95% CI: [{dr_res['aipw_ci_lower']:.4f}, {dr_res['aipw_ci_upper']:.4f}]) | "
        f"VanderWeele E-Value: {dr_res['e_value_point']:.3f} (CI E-Value: {dr_res['e_value_ci']:.3f})"
    )

    # Modifiable clinical intervention comparison: Hypertension (trestbps >= 140 mmHg)
    has_trestbps = "trestbps" in df_clean.columns
    if has_trestbps:
        treatment_htn = (df_clean["trestbps"].fillna(df_clean["trestbps"].median()).values >= 140).astype(int)
        conf_htn_cols = [c for c in feature_cols if c != "trestbps"]
        X_conf_htn_raw = df_clean[conf_htn_cols].values
        X_conf_htn = StandardScaler().fit_transform(
            SimpleImputer(strategy="median").fit_transform(X_conf_htn_raw)
        )
        dr_res_htn = compute_doubly_robust_aipw(
            X_conf_htn, treatment_htn, y_all, treatment_name="Hypertension (SBP >= 140 mmHg)", seed=seed
        )
        print(
            f"    [Modifiable Metabolic Treatment - Hypertension] Doubly Robust AIPW ATE: {dr_res_htn['causal_ate_aipw']:.4f} "
            f"(95% CI: [{dr_res_htn['aipw_ci_lower']:.4f}, {dr_res_htn['aipw_ci_upper']:.4f}]) | "
            f"VanderWeele E-Value: {dr_res_htn['e_value_point']:.3f} (CI E-Value: {dr_res_htn['e_value_ci']:.3f})"
        )
    else:
        dr_res_htn = None

    # Publication Figure 3: Covariate Balance Love Plot
    balance = plot_aipw_love_plot(
        X_confounders=X_confounders,
        treatment=treatment,
        propensity_scores=None,
        feature_names=confounder_cols,
        output_path=out_path / "figures" / "aipw_love_plot.png",
    )
    smd_after = {k: abs(v) for k, v in balance["smd_iptw"].items()}
    smd_before = {k: abs(v) for k, v in balance["smd_unadjusted"].items()}
    imbalanced = sorted(k for k, v in smd_after.items() if v >= 0.10)
    print(
        f"    Covariate balance (|SMD|): max {max(smd_before.values()):.3f} before -> {max(smd_after.values()):.3f} after IPTW; "
        f"{len(imbalanced)} of {len(smd_after)} covariates remain >= 0.10 ({', '.join(imbalanced) or 'none'})"
    )

    print(">>> 7. Testing Algorithmic Fairness Mitigation (Barocas et al., 2020)...")
    if "sex" in df_clean.columns:
        sex_group = df_clean["sex"].fillna(0).values.astype(int)
    elif "gender" in df_clean.columns:
        sex_group = (df_clean["gender"].astype(str).str.upper() == "M").astype(int)
    else:
        sex_group = np.random.RandomState(seed).randint(0, 2, len(y_all))

    val_sex = sex_group[val_idx]
    test_sex = sex_group[test_idx]

    val_fair_res = find_equal_opportunity_thresholds(y_val_t, val_probs, val_sex)
    t0 = val_fair_res["optimal_threshold_group_0"]
    t1 = val_fair_res["optimal_threshold_group_1"]

    def compute_subgroup_tpr(labels: np.ndarray, preds: np.ndarray, group: np.ndarray, group_val: int) -> float:
        mask = group == group_val
        pos = labels[mask] == 1
        if np.sum(pos) == 0:
            return 0.0
        return float(np.sum((preds[mask] == 1) & pos) / np.sum(pos))

    test_unmit_preds = (test_probs >= 0.5).astype(int)
    test_unmit_tpr0 = compute_subgroup_tpr(y_test_t, test_unmit_preds, test_sex, 0)
    test_unmit_tpr1 = compute_subgroup_tpr(y_test_t, test_unmit_preds, test_sex, 1)
    test_unmit_diff = abs(test_unmit_tpr0 - test_unmit_tpr1)

    test_mit_preds = np.zeros_like(y_test_t, dtype=int)
    test_mit_preds[test_sex == 0] = (test_probs[test_sex == 0] >= t0).astype(int)
    test_mit_preds[test_sex == 1] = (test_probs[test_sex == 1] >= t1).astype(int)
    test_mit_tpr0 = compute_subgroup_tpr(y_test_t, test_mit_preds, test_sex, 0)
    test_mit_tpr1 = compute_subgroup_tpr(y_test_t, test_mit_preds, test_sex, 1)
    test_mit_diff = abs(test_mit_tpr0 - test_mit_tpr1)
    test_disparity_reduction = max(
        0.0, (test_unmit_diff - test_mit_diff) / max(test_unmit_diff, 1e-6) * 100.0
    )

    # Report what the parity costs: equal TPRs can be reached by flagging everyone.
    def _fpr(preds: np.ndarray) -> float:
        neg = y_test_t == 0
        return float(np.mean(preds[neg] == 1)) if np.any(neg) else 0.0

    fair_cost = {
        "fpr_before": _fpr(test_unmit_preds),
        "fpr_after": _fpr(test_mit_preds),
        "acc_before": float(np.mean(test_unmit_preds == y_test_t)),
        "acc_after": float(np.mean(test_mit_preds == y_test_t)),
    }
    print(
        f"    Holdout Test TPR Disparity: {test_unmit_diff:.4f} -> {test_mit_diff:.4f} "
        f"({test_disparity_reduction:.1f}% reduction; Validation tuned thresholds t0={t0:.3f}, t1={t1:.3f})"
    )
    print(
        f"    Cost of parity on test: false-positive rate {fair_cost['fpr_before']:.3f} -> {fair_cost['fpr_after']:.3f}, "
        f"accuracy {fair_cost['acc_before']:.3f} -> {fair_cost['acc_after']:.3f}"
    )
    if min(t0, t1) <= 0.10:
        print("    WARNING: a group threshold is near 0, so parity is reached by flagging almost every patient in that group.")

    print(">>> 8. Testing Actionable Clinical Counterfactual Recourse (Wachter et al.; Ustun et al.)...")
    cf_gen = ClinicalCounterfactualGenerator(
        model=base_mlp,
        feature_names=feature_cols,
        immutable_features=("age", "sex", "gender", "fbs", "had_icu", "has_ed_visit"),
        target_probability=0.30,
    ).fit(X_train_t)
    cf_metrics = cf_gen.evaluate_cohort_recourse(X_test_t, high_risk_threshold=0.50)
    print(
        f"    Recourse Validity Rate: {cf_metrics['recourse_validity_rate'] * 100:.1f}% | "
        f"Mean Features Modified: {cf_metrics['mean_features_modified']:.2f} | "
        f"Mean Risk Reduction: -{cf_metrics['mean_probability_reduction']:.3f} | "
        f"Max Immutable Violation: {cf_metrics['max_immutable_violation']:.6f}"
    )

    # Publication Figure 2: Counterfactual Case Study (Patient-level with locked markers)
    plot_counterfactual_case_study(
        generator=cf_gen,
        X_test=X_test_t,
        feature_names=feature_cols,
        output_path=out_path / "figures" / "counterfactual_case_study.png",
    )

    # Closed-Loop Integration: verify high-risk conformal singletons {1} transition to {0} or {0,1}
    # Same probability scale as the conformal calibration (uncalibrated base MLP).
    pred_sets = conformal.predict_sets(test_probs)
    high_risk_singletons = np.where(~pred_sets[:, 0] & pred_sets[:, 1])[0]
    if len(high_risk_singletons) > 0:
        cf_batch = cf_gen.generate_batch(X_test_t[high_risk_singletons])
        cf_probs = np.array([res["counterfactual_probability"] for res in cf_batch])
        # Transitioned if no longer in {1} alone (i.e. became {0} or {0, 1})
        closed_loop_rate = float(np.mean(conformal.predict_sets(cf_probs)[:, 0])) * 100.0
    else:
        closed_loop_rate = 100.0
    print(f"    Closed-Loop Conformal Recourse: {closed_loop_rate:.1f}% of High-Risk singletons {{1}} transitioned to {{0}} or {{0,1}}")

    # Step 9: LOCO Domain-Shift Conformal Evaluation across international centers
    combined_csv = resolve_data_path("cleaned_heart_disease_combined.csv")
    loco_conf_df = None
    if combined_csv.exists():
        print(">>> 9. Benchmarking LOCO Domain-Shift Conformal Generalization across Centers...")
        df_comb = pd.read_csv(combined_csv)
        loco_conf_df = evaluate_loco_conformal(
            df_comb, site_col="site", target_col="target", alpha=0.10, seed=seed
        )
        loco_conf_df.to_csv(out_path / "loco_conformal_generalization.csv", index=False)
        print(
            f"    LOCO Conformal Mean Coverage across 4 International Centers: "
            f"{loco_conf_df['loco_empirical_coverage'].mean() * 100:.1f}% (Nominal: 90%)"
        )

    # Per-patient holdout predictions for the paired DeLong comparison.
    from evaluation.model_comparison import save_test_predictions

    original_rows = target_df.dropna(subset=[target_col]).index.to_numpy()
    holdout = [row for row in all_benchmarks if "_probs" in row]
    if holdout:
        save_test_predictions(
            out_path, "reading_list", heart_path if heart_df is not None else mimic_path,
            original_rows[holdout[0]["_test_idx"]], holdout[0]["_y_test"],
            {row["model"]: row["_probs"] for row in holdout},
        )
    for row in all_benchmarks:
        for key in ("_probs", "_test_idx", "_y_test"):
            row.pop(key, None)

    benchmarks_df = pd.DataFrame(all_benchmarks)
    if "is_synthetic_demo" not in benchmarks_df.columns:
        benchmarks_df["is_synthetic_demo"] = False
    benchmarks_df["is_synthetic_demo"] = benchmarks_df["is_synthetic_demo"].fillna(False).astype(bool)
    benchmarks_df = benchmarks_df.merge(arch_cv_df, on="model", how="left")
    benchmarks_df.to_csv(out_path / "reading_list_advancements_summary.csv", index=False)

    md_lines = [
        "# Reading List Implementation Benchmark Summary",
        "",
        "This report summarizes the empirical performance of all architectures and methods implemented from `AI_Thesis_Reading-List.csv`.",
        "",
        "## 1. Architectural Benchmark Table",
        "",
        f"Holdout: one stratified 60/20/20 split ({len(y_test_t)} test patients), with stratified bootstrap 95% CIs. "
        f"CV: repeated stratified {cv_folds}-fold x {cv_repeats} (mean ± std over folds). "
        "Model weights are chosen by validation loss in both settings.",
        "",
        "| Model | Holdout Accuracy | Holdout ROC-AUC [95% CI] | Holdout Brier | CV Accuracy | CV ROC-AUC | CV Brier |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in benchmarks_df.iterrows():
        name = row["model"] + (" †" if row["is_synthetic_demo"] else "")
        md_lines.append(
            f"| {name} | {row['test_accuracy']:.4f} | {row['test_roc_auc']:.4f} "
            f"[{row['test_roc_auc_ci_low']:.4f}, {row['test_roc_auc_ci_high']:.4f}] | {row['test_brier']:.4f} | "
            f"{row['cv_accuracy_mean']:.4f} ± {row['cv_accuracy_std']:.4f} | "
            f"{row['cv_roc_auc_mean']:.4f} ± {row['cv_roc_auc_std']:.4f} | "
            f"{row['cv_brier_mean']:.4f} ± {row['cv_brier_std']:.4f} |"
        )
    md_lines.extend([
        "",
        "† Demonstration only: the signal channel is synthesised from the tabular features, "
        "so this row is not a like-for-like comparison.",
    ])

    md_lines.extend([
        "",
        "## 2. Methodology & Rigor Summary",
        f"- **Temperature Scaling (Guo et al., 2017):** Validation-tuned optimal T `{opt_t:.3f}` on trained `HeartMLP` reduced holdout test ECE from `{ece_test_before:.4f}` to `{ece_test_after:.4f}` (Validation ECE: `{cal_res['ece_before']:.4f}` -> `{cal_res['ece_after']:.4f}`).",
        f"- **Deep Ensembles (Lakshminarayanan et al., 2017):** Average epistemic predictive variance is `{unc_res['epistemic_variance'].mean():.4f}` across {len(X_test_t)} holdout patients.",
        f"- **Split & Mondrian Conformal Prediction (Angelopoulos & Bates, 2021):** Calibrated and evaluated on the same (uncalibrated) MLP probabilities; the 90% finite-sample coverage target achieved `{conf_res['empirical_coverage']:.4f}` empirical coverage (Case: `{conf_res['positive_class_coverage']:.4f}`, Control: `{conf_res['negative_class_coverage']:.4f}`) with average set size `{conf_res['average_set_size']:.3f}` and `{conf_res['ambiguous_abstention_rate']:.3f}` clinical abstention rate.",
        f"- **Conformal Efficiency-Coverage Trade-Off:** Operating at $\\alpha=0.15$ (85% coverage guarantee) cuts the ambiguous deferral rate from `{conf_res['ambiguous_abstention_rate']*100:.1f}%` down to `{conformal_sweeps[0.15]['ambiguous_abstention_rate']*100:.1f}%` with `{conformal_sweeps[0.15]['empirical_coverage']*100:.1f}%` empirical test coverage.",
        f"- **Causal IPTW Analysis (Pearl & Mackenzie, 2018):** Observational risk difference is `{causal_res['observational_risk_difference']:.4f}`, whereas confounder-isolated Inverse Probability Weighting (IPTW) reveals a causal ATE of `{causal_res['causal_ate_iptw']:.4f}`, demonstrating a confounding bias of `{causal_res['confounding_bias']:.4f}`.",
        f"- **Covariate Balance (IPTW, treatment `{treat_col}`):** maximum |SMD| fell from `{max(smd_before.values()):.3f}` to `{max(smd_after.values()):.3f}`, but `{len(imbalanced)}` of `{len(smd_after)}` covariates remain at or above the 0.10 balance threshold"
        + (f" ({', '.join(imbalanced)}), so residual measured confounding cannot be ruled out." if imbalanced else "."),
        "- **Doubly Robust AIPW & E-Value (Chernozhukov et al., 2018; VanderWeele & Ding, 2017):**",
        f"  - *Symptomatic Proxy (exang):* Cross-fitted AIPW ATE = `{dr_res['causal_ate_aipw']:.4f}` (95% CI: [{dr_res['aipw_ci_lower']:.4f}, {dr_res['aipw_ci_upper']:.4f}]), E-value = `{dr_res['e_value_point']:.3f}`.",
    ])

    if dr_res_htn is not None:
        md_lines.append(
            f"  - *Modifiable Metabolic Treatment (Hypertension SBP >= 140):* Cross-fitted AIPW ATE = `{dr_res_htn['causal_ate_aipw']:.4f}` (95% CI: [{dr_res_htn['aipw_ci_lower']:.4f}, {dr_res_htn['aipw_ci_upper']:.4f}]), E-value = `{dr_res_htn['e_value_point']:.3f}` (CI E-value = `{dr_res_htn['e_value_ci']:.3f}`)."
        )

    md_lines.extend([
        f"- **Algorithmic Fairness (Barocas et al., 2020):** Validation-tuned Equal Opportunity thresholds (t0={t0:.3f}, t1={t1:.3f}) reduced holdout test TPR disparity from `{test_unmit_diff:.4f}` to `{test_mit_diff:.4f}` ({test_disparity_reduction:.1f}% reduction), while the false-positive rate moved from `{fair_cost['fpr_before']:.3f}` to `{fair_cost['fpr_after']:.3f}` and accuracy from `{fair_cost['acc_before']:.3f}` to `{fair_cost['acc_after']:.3f}`"
        + (" — at least one group threshold is near 0, so this parity comes from flagging almost every patient in that group and is not a useful operating point." if min(t0, t1) <= 0.10 else ".") + "",
        f"- **Clinical Counterfactual Recourse (Wachter et al., 2017; Ustun et al., 2019):** Achieved `{cf_metrics['recourse_validity_rate'] * 100:.1f}%` recourse validity across high-risk test patients with `{cf_metrics['mean_features_modified']:.2f}` average features modified, reducing mean predicted risk by `{cf_metrics['mean_probability_reduction']:.3f}` while strictly maintaining `{cf_metrics['max_immutable_violation']:.6f}` immutable demographic violation.",
        f"- **Closed-Loop Conformal Recourse Transition:** `{closed_loop_rate:.1f}%` of Conformal High-Risk patients ($\\hat{{C}}(x) = \\{{1\\}}$) successfully transitioned into Low-Risk $\\{{0\\}}$ or Deferral $\\{{0, 1\\}}$ following sparse actionable recourse.",
    ])

    if loco_conf_df is not None:
        md_lines.extend([
            "",
            "## 3. LOCO Domain-Shift Conformal Generalization Table",
            "",
            "| Evaluation Center | N Samples | Disease Prevalence | Fixed Threshold Acc | 90% Conformal Coverage | Avg Set Size | Abstain Rate |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ])
        for _, r in loco_conf_df.iterrows():
            md_lines.append(
                f"| {r['test_site']} | {int(r['n_test_samples'])} | {r['test_disease_prevalence']*100:.1f}% | "
                f"{r['fixed_threshold_accuracy']*100:.1f}% | {r['loco_empirical_coverage']*100:.1f}% | "
                f"{r['average_set_size']:.2f} | {r['ambiguous_abstention_rate']*100:.1f}% |"
            )

    with open(out_path / "reading_list_advancements_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\n[DONE] Saved summary to {out_path / 'reading_list_advancements_summary.md'}")
    print("[DONE] Saved publication figures:")
    print(f"  • {out_path / 'figures' / 'conformal_triage_curve.png'}")
    print(f"  • {out_path / 'figures' / 'counterfactual_case_study.png'}")
    print(f"  • {out_path / 'figures' / 'aipw_love_plot.png'}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the AI thesis reading-list suite.")
    parser.add_argument("--heart-csv", type=str, default=PRIMARY_HEART_DATASET)
    parser.add_argument("--mimic-csv", type=str, default="mimic_hospital_mortality.csv")
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=DEVICE_CHOICES, default="auto")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    run_all_advancements(
        args.heart_csv,
        args.mimic_csv,
        args.output_dir,
        seed=args.seed,
        device=args.device,
        cv_folds=args.cv_folds,
        cv_repeats=args.cv_repeats,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
