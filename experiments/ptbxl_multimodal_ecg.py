"""Real 12-Lead ECG Multimodal Cross-Attention Benchmark on PhysioNet PTB-XL (v1.0.3).

Replaces synthetic waveform surrogates with real clinical 10-second 12-lead ECG recordings
(100 Hz, shape ``[12, 1000]``) from the open-access **PTB-XL** database (Wagner et al.,
*Scientific Data* 2020) using the official patient-grouped splits:
- **Train:** ``strat_fold`` 1..8
- **Validation:** ``strat_fold`` 9
- **Untouched Holdout Test:** ``strat_fold`` 10 (strict zero ``patient_id`` overlap)

Task: Detect **Ischemic ECG Pathology** (``MI`` Myocardial Infarction or ``STTC`` ST/T ischemic
changes, ``target = 1``) vs. **Normal Sinus ECG** (``NORM``, ``target = 0``), comparing:
1. **Tabular Demographics Only** (``age, sex, height, weight, bmi``)
2. **12-Lead ECG 1D-ResNet** (raw ``[12, 1000]`` waveform only)
3. **Multimodal ECG + Tabular Cross-Attention** (gated multi-head cross-attention fusion)

Generates:
- ``artifacts/ptbxl_multimodal_ecg.json``
- ``artifacts/ptbxl_multimodal_ecg.md``
- ``artifacts/figures/ptbxl_multimodal_ecg_benchmark.png`` (ablation + real 12-lead Grad-CAM)
"""

from __future__ import annotations

import argparse
import ast
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from experiments.framingham_benchmark import _bootstrap_auc_ci, _delong_paired_pvalue
from models import compute_ece
from project_paths import ARTIFACTS_DIR, PROCESSED_DATA_DIR

PTBXL_BASE_URL = "https://physionet.org/files/ptb-xl/1.0.3"
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
TABULAR_COLS = ["age", "sex", "height", "weight", "bmi"]


class ResBlock1D(nn.Module):
    """Residual 1D convolutional block with Squeeze-and-Excitation lead recalibration."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 7, stride: int = 2, dropout: float = 0.15):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, stride=1, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_ch, max(out_ch // 4, 8), kernel_size=1),
            nn.GELU(),
            nn.Conv1d(max(out_ch // 4, 8), out_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.skip = (
            nn.Sequential(nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_ch))
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = self.bn2(self.conv2(h))
        h = h * self.se(h)
        return F.gelu(h + self.skip(x))


class ECGResNet1D(nn.Module):
    """Lightweight multi-scale 1D-ResNet for 12-lead 100 Hz ECG classification & Grad-CAM."""

    def __init__(self, in_leads: int = 12, d_model: int = 64, dropout: float = 0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_leads, 32, kernel_size=11, stride=2, padding=5, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )
        self.stage1 = ResBlock1D(32, 48, kernel_size=7, stride=2, dropout=dropout)
        self.stage2 = ResBlock1D(48, 64, kernel_size=5, stride=2, dropout=dropout)
        self.stage3 = ResBlock1D(64, d_model, kernel_size=5, stride=2, dropout=dropout)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 1),
        )

    def extract_feature_map(self, ecg: torch.Tensor) -> torch.Tensor:
        """Return temporal ECG feature tokens of shape ``(B, d_model, T')``."""
        h = self.stem(ecg)
        h = self.stage1(h)
        h = self.stage2(h)
        return self.stage3(h)

    def forward(self, ecg: torch.Tensor) -> torch.Tensor:
        fmap = self.extract_feature_map(ecg)
        pooled = torch.cat([fmap.mean(dim=-1), fmap.amax(dim=-1)], dim=1)
        return self.head(pooled)


class MultimodalECGTransformer(nn.Module):
    """Real 12-Lead ECG + Tabular Clinical Cross-Attention Architecture with Gated Fusion."""

    def __init__(self, n_tab_features: int = 5, d_model: int = 64, n_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.ecg_encoder = ECGResNet1D(in_leads=12, d_model=d_model, dropout=dropout)
        # Per-feature scalar tabular tokenizer (mirrors Level-1 clinical tokenizer)
        self.tab_w = nn.Parameter(torch.randn(n_tab_features, d_model) * 0.05)
        self.tab_b = nn.Parameter(torch.zeros(n_tab_features, d_model))
        self.tab_norm = nn.LayerNorm(d_model)
        self.ecg_norm = nn.LayerNorm(d_model)

        # Cross-attention: tabular clinical context queries temporal 12-lead ECG tokens
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.post_norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, ecg: torch.Tensor, tab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fmap = self.ecg_encoder.extract_feature_map(ecg)  # (B, D, T')
        ecg_tokens = self.ecg_norm(fmap.transpose(1, 2))  # (B, T', D)
        ecg_summary = fmap.mean(dim=-1)  # (B, D)

        # Tokenize tabular features into (B, F, D)
        tab_tokens = self.tab_norm(tab.unsqueeze(-1) * self.tab_w.unsqueeze(0) + self.tab_b.unsqueeze(0))
        attended, _ = self.cross_attn(query=tab_tokens, key=ecg_tokens, value=ecg_tokens, need_weights=False)
        cross_summary = self.post_norm(tab_tokens + attended).mean(dim=1)

        g = self.gate(torch.cat([ecg_summary, cross_summary], dim=1))
        fused = g * ecg_summary + (1.0 - g) * cross_summary
        logits = self.classifier(torch.cat([fused, fmap.amax(dim=-1)], dim=1))
        return logits, g.mean(dim=1)


def _download_dat_waveform(filename_lr: str) -> np.ndarray | None:
    """Download a single 100 Hz PTB-XL 12-lead .dat waveform (24 KB, MIT format-16 -> mV)."""
    url = f"{PTBXL_BASE_URL}/{filename_lr}.dat"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CardioGraph-Clinical-ML/2.1"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
        if len(raw) != 24_000:
            return None
        # PTB-XL records100 are 16-bit signed LE, 1000 samples x 12 leads, gain 1000 ADC/mV
        sig = np.frombuffer(raw, dtype="<i2").reshape(1000, 12).T.astype(np.float32) / 1000.0
        # Per-lead z-score normalization (robust to baseline wander / electrode impedance)
        mu = sig.mean(axis=1, keepdims=True)
        sd = np.where(sig.std(axis=1, keepdims=True) < 1e-5, 1.0, sig.std(axis=1, keepdims=True))
        return ((sig - mu) / sd).astype(np.float32)
    except Exception:
        return None


def fetch_or_load_ptbxl_subset(
    cache_path: Path | None = None,
    n_train: int = 800,
    n_val: int = 200,
    n_test: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """Download and cache a patient-grouped, balanced PTB-XL 12-lead ECG cohort (MI/STTC vs NORM)."""
    cache = cache_path or (PROCESSED_DATA_DIR / "ptbxl_ecg_subset.npz")
    meta_cache = cache.with_suffix(".csv")
    if cache.exists() and meta_cache.exists():
        npz = np.load(cache)
        meta = pd.read_csv(meta_cache)
        return {
            "waveforms": npz["waveforms"],
            "meta": meta,
        }

    print("Fetching PTB-XL v1.0.3 clinical metadata from PhysioNet (open-access CC-BY 4.0)...")
    db_df = pd.read_csv(f"{PTBXL_BASE_URL}/ptbxl_database.csv")
    scp_df = pd.read_csv(f"{PTBXL_BASE_URL}/scp_statements.csv", index_col=0)
    diag_map = scp_df[scp_df["diagnostic"] == 1]["diagnostic_class"].to_dict()

    def _classify_record(scp_str: str) -> int:
        codes = ast.literal_eval(scp_str)
        classes = {diag_map[c] for c, likelihood in codes.items() if c in diag_map and float(likelihood) >= 50.0}
        if ("MI" in classes or "STTC" in classes) and "NORM" not in classes:
            return 1
        if classes == {"NORM"}:
            return 0
        return -1

    db_df["target"] = db_df["scp_codes"].map(_classify_record)
    db_df["is_mi"] = db_df["scp_codes"].map(
        lambda s: int(any(diag_map.get(c) == "MI" and float(v) >= 80.0 for c, v in ast.literal_eval(s).items()))
    )
    clean = db_df[db_df["target"].isin([0, 1])].copy()
    # Strictly one ECG per unique patient_id to guarantee zero patient leakage
    clean = clean.drop_duplicates(subset=["patient_id"]).reset_index(drop=True)
    clean["age"] = pd.to_numeric(clean["age"], errors="coerce").clip(18.0, 95.0)
    clean["sex"] = pd.to_numeric(clean["sex"], errors="coerce").fillna(0).astype(float)
    clean["height"] = pd.to_numeric(clean["height"], errors="coerce").clip(130.0, 220.0)
    clean["weight"] = pd.to_numeric(clean["weight"], errors="coerce").clip(35.0, 180.0)
    clean["bmi"] = clean["weight"] / ((clean["height"] / 100.0) ** 2)

    rng = np.random.default_rng(seed)

    def _sample_balanced(df_slice: pd.DataFrame, total_n: int, split_name: str) -> pd.DataFrame:
        half = total_n // 2
        pos = df_slice[df_slice["target"] == 1]
        neg = df_slice[df_slice["target"] == 0]
        p_idx = rng.choice(pos.index.values, size=min(half, len(pos)), replace=False)
        n_idx = rng.choice(neg.index.values, size=min(total_n - len(p_idx), len(neg)), replace=False)
        sub = df_slice.loc[np.concatenate([p_idx, n_idx])].copy()
        sub["split"] = split_name
        return sub

    train_meta = _sample_balanced(clean[clean["strat_fold"].isin(range(1, 9))], n_train, "train")
    val_meta = _sample_balanced(clean[clean["strat_fold"] == 9], n_val, "val")
    test_meta = _sample_balanced(clean[clean["strat_fold"] == 10], n_test, "test")
    selected = pd.concat([train_meta, val_meta, test_meta], ignore_index=True)

    # Assert strict zero patient_id overlap across splits
    tr_ids = set(selected[selected["split"] == "train"]["patient_id"])
    va_ids = set(selected[selected["split"] == "val"]["patient_id"])
    te_ids = set(selected[selected["split"] == "test"]["patient_id"])
    assert tr_ids.isdisjoint(va_ids) and tr_ids.isdisjoint(te_ids) and va_ids.isdisjoint(te_ids)

    print(f"Downloading {len(selected)} real 12-lead 100 Hz ECG waveforms (24 KB each) in parallel...")
    filenames = selected["filename_lr"].tolist()
    with ThreadPoolExecutor(max_workers=20) as pool:
        sigs = list(pool.map(_download_dat_waveform, filenames))

    keep_idx = [i for i, s in enumerate(sigs) if s is not None]
    waveforms = np.stack([sigs[i] for i in keep_idx], axis=0).astype(np.float32)  # (N, 12, 1000)
    meta_out = selected.iloc[keep_idx][
        ["ecg_id", "patient_id", "strat_fold", "split", "filename_lr", "target", "is_mi", *TABULAR_COLS]
    ].reset_index(drop=True)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, waveforms=waveforms)
    meta_out.to_csv(meta_cache, index=False)
    print(f"Cached {len(meta_out)} real 12-lead ECGs -> {cache}")
    return {"waveforms": waveforms, "meta": meta_out}


def _compute_1d_gradcam(model: ECGResNet1D, ecg_single: torch.Tensor) -> np.ndarray:
    """Compute normalized 1D Grad-CAM saliency over the 10-second ECG recording."""
    model.eval()
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def _fwd_hook(_m: nn.Module, _inp: Any, out: torch.Tensor) -> None:
        activations.append(out)

    def _bwd_hook(_m: nn.Module, _gin: Any, gout: Any) -> None:
        gradients.append(gout[0])

    h1 = model.stage3.register_forward_hook(_fwd_hook)
    h2 = model.stage3.register_full_backward_hook(_bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logit = model(ecg_single)
        logit.sum().backward()
        act = activations[0].detach()[0]  # (C, T')
        grad = gradients[0].detach()[0]  # (C, T')
        weights = grad.mean(dim=-1, keepdim=True)
        cam = F.relu((weights * act).sum(dim=0))
        cam_up = F.interpolate(
            cam.view(1, 1, -1), size=ecg_single.shape[-1], mode="linear", align_corners=False
        ).view(-1)
        cam_np = cam_up.cpu().numpy()
        cam_np = (cam_np - cam_np.min()) / max(float(cam_np.max() - cam_np.min()), 1e-8)
        return cam_np
    finally:
        h1.remove()
        h2.remove()


def _train_neural_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_ecg: torch.Tensor,
    val_tab: torch.Tensor,
    val_y: np.ndarray,
    device: torch.device,
    is_multimodal: bool,
    epochs: int = 35,
    lr: float = 1.5e-3,
) -> dict[str, torch.Tensor]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    patience_left = 10

    for _ in range(epochs):
        model.train()
        for b_ecg, b_tab, b_y in train_loader:
            b_ecg = b_ecg.to(device)
            b_tab = b_tab.to(device)
            b_y = b_y.to(device).view(-1, 1)
            # Mild clinical waveform augmentation: small Gaussian sensor noise + random lead amplitude jitter
            noise = torch.randn_like(b_ecg) * 0.03
            scale = 1.0 + (torch.rand((b_ecg.size(0), 12, 1), device=device) - 0.5) * 0.12
            b_ecg_aug = b_ecg * scale + noise

            optimizer.zero_grad(set_to_none=True)
            logits = model(b_ecg_aug, b_tab)[0] if is_multimodal else model(b_ecg_aug)
            loss = loss_fn(logits, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            v_logits = model(val_ecg.to(device), val_tab.to(device))[0] if is_multimodal else model(val_ecg.to(device))
            v_probs = torch.sigmoid(v_logits).cpu().numpy().ravel()
        v_auc = float(roc_auc_score(val_y, v_probs))
        if v_auc > best_auc + 1e-4:
            best_auc = v_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 10
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    return best_state


def _eval_metrics(y_true: np.ndarray, probs: np.ndarray, seed: int = 42) -> dict[str, Any]:
    pred = (probs >= 0.5).astype(int)
    pos = y_true == 1
    neg = y_true == 0
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probs)), 4),
        "roc_auc_ci_95": list(_bootstrap_auc_ci(y_true, probs, n_boot=500, seed=seed)),
        "pr_auc": round(float(average_precision_score(y_true, probs)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "sensitivity": round(float(np.mean(pred[pos] == 1)), 4),
        "specificity": round(float(np.mean(pred[neg] == 0)), 4),
        "brier": round(float(brier_score_loss(y_true, probs)), 4),
        "ece": round(float(compute_ece(probs, y_true)), 4),
    }


def run_ptbxl_multimodal_benchmark(
    n_train: int = 800,
    n_val: int = 200,
    n_test: int = 200,
    epochs: int = 35,
    output_dir: str | Path = ARTIFACTS_DIR,
    seed: int = 42,
) -> dict[str, Any]:
    """Run real 12-lead PTB-XL ECG multimodal fusion benchmark (Official Fold-10 holdout)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = fetch_or_load_ptbxl_subset(n_train=n_train, n_val=n_val, n_test=n_test, seed=seed)
    waveforms: np.ndarray = data["waveforms"]
    meta: pd.DataFrame = data["meta"]

    tr_mask = (meta["split"] == "train").values
    va_mask = (meta["split"] == "val").values
    te_mask = (meta["split"] == "test").values

    # Fit tabular imputer + scaler strictly on Train split (zero leakage)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tab_tr = scaler.fit_transform(imputer.fit_transform(meta.loc[tr_mask, TABULAR_COLS].values)).astype(np.float32)
    X_tab_va = scaler.transform(imputer.transform(meta.loc[va_mask, TABULAR_COLS].values)).astype(np.float32)
    X_tab_te = scaler.transform(imputer.transform(meta.loc[te_mask, TABULAR_COLS].values)).astype(np.float32)

    X_ecg_tr, X_ecg_va, X_ecg_te = waveforms[tr_mask], waveforms[va_mask], waveforms[te_mask]
    y_tr = meta.loc[tr_mask, "target"].values.astype(int)
    y_va = meta.loc[va_mask, "target"].values.astype(int)
    y_te = meta.loc[te_mask, "target"].values.astype(int)

    # 1. Tabular Clinical Metadata Only Baseline
    tab_clf = LogisticRegression(C=1.0, max_iter=500, random_state=seed)
    tab_clf.fit(X_tab_tr, y_tr)
    p_tab_te = tab_clf.predict_proba(X_tab_te)[:, 1]

    # DataLoader for neural models
    tr_ds = TensorDataset(
        torch.from_numpy(X_ecg_tr),
        torch.from_numpy(X_tab_tr),
        torch.from_numpy(y_tr.astype(np.float32)),
    )
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
    va_ecg_t = torch.from_numpy(X_ecg_va)
    va_tab_t = torch.from_numpy(X_tab_va)
    te_ecg_t = torch.from_numpy(X_ecg_te)
    te_tab_t = torch.from_numpy(X_tab_te)

    # 2. Real 12-Lead ECG Only (1D-ResNet)
    torch.manual_seed(seed)
    ecg_model = ECGResNet1D(in_leads=12, d_model=64, dropout=0.20).to(device)
    _train_neural_model(
        ecg_model, tr_loader, va_ecg_t, va_tab_t, y_va, device, is_multimodal=False, epochs=epochs
    )
    ecg_model.eval()
    with torch.inference_mode():
        p_ecg_te = torch.sigmoid(ecg_model(te_ecg_t.to(device))).cpu().numpy().ravel()

    # 3. Multimodal Cross-Attention (Real 12-Lead ECG + Tabular Clinical Metadata)
    torch.manual_seed(seed + 1)
    mm_model = MultimodalECGTransformer(n_tab_features=len(TABULAR_COLS), d_model=64, n_heads=4, dropout=0.20).to(device)
    _train_neural_model(
        mm_model, tr_loader, va_ecg_t, va_tab_t, y_va, device, is_multimodal=True, epochs=epochs
    )
    mm_model.eval()
    with torch.inference_mode():
        mm_logits, mm_gates = mm_model(te_ecg_t.to(device), te_tab_t.to(device))
        p_mm_te = torch.sigmoid(mm_logits).cpu().numpy().ravel()
        mean_ecg_gate = float(mm_gates.mean().item())

    m_tab = _eval_metrics(y_te, p_tab_te, seed=seed)
    m_ecg = _eval_metrics(y_te, p_ecg_te, seed=seed)
    m_mm = _eval_metrics(y_te, p_mm_te, seed=seed)

    z_mm_vs_tab, p_mm_vs_tab = _delong_paired_pvalue(y_te, p_mm_te, p_tab_te)
    z_mm_vs_ecg, p_mm_vs_ecg = _delong_paired_pvalue(y_te, p_mm_te, p_ecg_te)

    # Generate 2-panel publication figure: (A) Ablation CIs + (B) Real Fold-10 MI ECG Grad-CAM
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    te_meta = meta.loc[te_mask].reset_index(drop=True)
    mi_candidates = np.where((te_meta["is_mi"].values == 1) & (p_mm_te > 0.70))[0]
    chosen_idx = int(mi_candidates[0]) if len(mi_candidates) > 0 else int(np.argmax(p_mm_te * (y_te == 1)))
    cam = _compute_1d_gradcam(mm_model.ecg_encoder, te_ecg_t[chosen_idx : chosen_idx + 1].to(device))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.7), dpi=170, gridspec_kw={"width_ratios": [1.0, 1.35]})
    labels = ["Tabular\nDemographics", "12-Lead ECG\n1D-ResNet", "Multimodal\nCross-Attention"]
    aucs = [m_tab["roc_auc"], m_ecg["roc_auc"], m_mm["roc_auc"]]
    lo = [m_tab["roc_auc"] - m_tab["roc_auc_ci_95"][0], m_ecg["roc_auc"] - m_ecg["roc_auc_ci_95"][0], m_mm["roc_auc"] - m_mm["roc_auc_ci_95"][0]]
    hi = [m_tab["roc_auc_ci_95"][1] - m_tab["roc_auc"], m_ecg["roc_auc_ci_95"][1] - m_ecg["roc_auc"], m_mm["roc_auc_ci_95"][1] - m_mm["roc_auc"]]
    colors = ["#94a3b8", "#2b8a9e", "#0f5c6e"]
    bars = ax1.bar(labels, aucs, yerr=[lo, hi], capsize=5, color=colors, width=0.55, edgecolor="#0f1f2d", linewidth=0.6)
    for bar, val, err_hi in zip(bars, aucs, hi, strict=True):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            val + err_hi + 0.014,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )
    ax1.set_ylim(0.50, 1.05)
    ax1.set_ylabel("Fold-10 Holdout ROC-AUC (95% CI)")
    ax1.set_title(f"A. Real PTB-XL 12-Lead ECG Ablation (N={len(y_te)} Holdout)")
    ax1.grid(axis="y", alpha=0.25)

    # Panel B: 4 key ischemic leads (II, V2, V3, V5) with 1D Grad-CAM overlay over 5-second window
    t_sec = np.arange(500) / 100.0
    lead_indices = [1, 7, 8, 10]  # II, V2, V3, V5
    ecg_sample = X_ecg_te[chosen_idx, :, :500]
    cam_win = cam[:500]
    for row_i, li in enumerate(lead_indices):
        offset = (len(lead_indices) - 1 - row_i) * 4.5
        ax2.plot(t_sec, ecg_sample[li] + offset, color="#0f1f2d", linewidth=1.05)
        ax2.text(-0.08, offset, LEAD_NAMES[li], ha="right", va="center", fontsize=8.5, fontweight="bold", color="#0f5c6e")
    ax2.imshow(
        cam_win[np.newaxis, :],
        aspect="auto",
        cmap="YlOrRd",
        alpha=0.40,
        extent=[0.0, 5.0, -2.5, (len(lead_indices) - 1) * 4.5 + 2.5],
    )
    pt_age = int(round(float(te_meta.loc[chosen_idx, "age"]))) if pd.notna(te_meta.loc[chosen_idx, "age"]) else 0
    ax2.set_xlim(-0.35, 5.0)
    ax2.set_yticks([])
    ax2.set_xlabel("Time (seconds, 100 Hz clinical 12-lead ECG)")
    ax2.set_title(
        f"B. Fold-10 MI Patient (Age {pt_age}y, P={p_mm_te[chosen_idx]:.2f}) — 1D Grad-CAM Saliency"
    )
    fig.tight_layout()
    fig_path = Path(output_dir) / "figures" / "ptbxl_multimodal_ecg_benchmark.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)

    summary = {
        "dataset": "PhysioNet PTB-XL v1.0.3 (Real 100 Hz 12-Lead Clinical ECGs, MI/STTC vs NORM)",
        "device": str(device),
        "splits": {
            "n_train_folds_1_8": int(tr_mask.sum()),
            "n_val_fold_9": int(va_mask.sum()),
            "n_test_fold_10": int(te_mask.sum()),
            "patient_overlap": 0,
        },
        "mean_ecg_fusion_gate": round(mean_ecg_gate, 4),
        "delong_multimodal_vs_tabular": {"z": round(z_mm_vs_tab, 3), "p_value": p_mm_vs_tab},
        "delong_multimodal_vs_ecg_only": {"z": round(z_mm_vs_ecg, 3), "p_value": p_mm_vs_ecg},
        "models": {
            "tabular_demographics_only": m_tab,
            "ecg_12lead_resnet1d": m_ecg,
            "multimodal_ecg_cross_attention": m_mm,
        },
    }

    out_dir = Path(output_dir)
    (out_dir / "ptbxl_multimodal_ecg.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md = "\n".join([
        "# Real 12-Lead ECG Multimodal Cross-Attention Benchmark (PhysioNet PTB-XL v1.0.3)",
        "",
        f"Evaluated on **{int(te_mask.sum())}** untouched official **Fold-10** patients "
        f"(`{int(tr_mask.sum())}` Train folds 1–8 / `{int(va_mask.sum())}` Val fold 9, **strict zero `patient_id` overlap**; "
        f"Task: Myocardial Infarction / ST-T Ischemia `MI+STTC` vs. Normal Sinus `NORM`).",
        f"- **Multimodal vs. Tabular Demographics Only:** paired DeLong `z = {summary['delong_multimodal_vs_tabular']['z']}`, `p = {summary['delong_multimodal_vs_tabular']['p_value']:.2e}`",
        f"- **Mean Learned ECG Gate Reliance:** `g_ecg = {summary['mean_ecg_fusion_gate']:.3f}`",
        "",
        "| Architecture | Holdout ROC-AUC [95% CI] | PR-AUC | F1 | Sensitivity | Specificity | Brier ↓ | ECE ↓ |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        f"| **Tabular Demographics Only** | {m_tab['roc_auc']:.4f} [{m_tab['roc_auc_ci_95'][0]:.4f}, {m_tab['roc_auc_ci_95'][1]:.4f}] | {m_tab['pr_auc']:.4f} | {m_tab['f1']:.4f} | {m_tab['sensitivity']:.4f} | {m_tab['specificity']:.4f} | {m_tab['brier']:.4f} | {m_tab['ece']:.4f} |",
        f"| **12-Lead ECG 1D-ResNet** | {m_ecg['roc_auc']:.4f} [{m_ecg['roc_auc_ci_95'][0]:.4f}, {m_ecg['roc_auc_ci_95'][1]:.4f}] | {m_ecg['pr_auc']:.4f} | {m_ecg['f1']:.4f} | {m_ecg['sensitivity']:.4f} | {m_ecg['specificity']:.4f} | {m_ecg['brier']:.4f} | {m_ecg['ece']:.4f} |",
        f"| **Multimodal Cross-Attention (ECG + Tabular)** | **{m_mm['roc_auc']:.4f}** [{m_mm['roc_auc_ci_95'][0]:.4f}, {m_mm['roc_auc_ci_95'][1]:.4f}] | **{m_mm['pr_auc']:.4f}** | **{m_mm['f1']:.4f}** | **{m_mm['sensitivity']:.4f}** | **{m_mm['specificity']:.4f}** | **{m_mm['brier']:.4f}** | **{m_mm['ece']:.4f}** |",
    ]) + "\n"
    (out_dir / "ptbxl_multimodal_ecg.md").write_text(md, encoding="utf-8")
    print(md)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Real 12-lead PTB-XL ECG multimodal benchmark.")
    parser.add_argument("--n-train", type=int, default=800)
    parser.add_argument("--n-val", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--output-dir", type=str, default=str(ARTIFACTS_DIR))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    run_ptbxl_multimodal_benchmark(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        epochs=args.epochs,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
