"""Predictive Uncertainty Estimation using Deep Ensembles.

Reference:
Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017).
Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.
NeurIPS 2017, 30, 6402-6413. arXiv:1612.01474.
"""

from __future__ import annotations
import copy
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class DeepEnsemble(nn.Module):
    """Deep Ensemble for Predictive Uncertainty and Epistemic Risk Quantification.

    Trains M randomly initialized neural network models with proper scoring rules
    to quantify predictive variance and Shannon entropy for each patient.
    """

    def __init__(
        self,
        base_model_fn: Callable[[], nn.Module],
        num_models: int = 5,
        uncertainty_threshold: float = 0.05,
    ):
        super().__init__()
        self.num_models = num_models
        self.uncertainty_threshold = uncertainty_threshold
        self.models = nn.ModuleList([base_model_fn() for _ in range(num_models)])

    def fit(
        self,
        X_train: torch.Tensor | np.ndarray,
        y_train: torch.Tensor | np.ndarray,
        epochs: int = 30,
        lr: float = 0.005,
        batch_size: int = 32,
        weight_decay: float = 1e-4,
        device: str = "cpu",
    ) -> list[list[float]]:
        """Fit all ensemble members with independent seeds and shuffling."""
        if isinstance(X_train, np.ndarray):
            X_tensor = torch.from_numpy(X_train).float()
        else:
            X_tensor = X_train.float()

        if isinstance(y_train, np.ndarray):
            y_tensor = torch.from_numpy(y_train).float().view(-1, 1)
        else:
            y_tensor = y_train.float().view(-1, 1)

        dataset = TensorDataset(X_tensor, y_tensor)
        criterion = nn.BCEWithLogitsLoss()
        all_histories: list[list[float]] = []

        dev = torch.device(device)
        self.to(dev)

        for m_idx, model in enumerate(self.models):
            torch.manual_seed(42 + m_idx * 1007)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

            history: list[float] = []
            model.train()
            for _ in range(epochs):
                epoch_loss = 0.0
                batches = 0
                for bx, by in loader:
                    bx, by = bx.to(dev), by.to(dev)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out.view(-1), by.view(-1))
                    loss.backward()
                    optimizer.step()
                    epoch_loss += float(loss.item())
                    batches += 1
                history.append(epoch_loss / max(1, batches))
            all_histories.append(history)

        return all_histories

    def predict_with_uncertainty(
        self,
        X: torch.Tensor | np.ndarray,
        device: str = "cpu",
    ) -> dict[str, np.ndarray]:
        """Predict class probabilities, epistemic uncertainty, and predictive entropy.

        Returns
        -------
        dict with:
            - 'mean_prob': (N,) ensembled predicted probability
            - 'epistemic_variance': (N,) sample variance across ensemble predictions
            - 'predictive_entropy': (N,) Shannon entropy H(p)
            - 'is_uncertain': (N,) boolean mask of high-uncertainty predictions
        """
        if isinstance(X, np.ndarray):
            X_tensor = torch.from_numpy(X).float()
        else:
            X_tensor = X.float()

        dev = torch.device(device)
        self.to(dev)
        self.eval()

        member_probs: list[np.ndarray] = []
        with torch.no_grad():
            for model in self.models:
                logits = model(X_tensor.to(dev))
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                member_probs.append(probs)

        # Stack into (M, N) matrix
        P = np.stack(member_probs, axis=0)  # shape (M, N)

        mean_prob = np.mean(P, axis=0)
        # Sample variance across model predictions: Var(p)
        variance = np.var(P, axis=0, ddof=1) if self.num_models > 1 else np.zeros_like(mean_prob)

        # Binary Shannon entropy: H(p) = -p log2(p) - (1-p) log2(1-p)
        eps = 1e-8
        p_safe = np.clip(mean_prob, eps, 1.0 - eps)
        entropy = -(p_safe * np.log2(p_safe) + (1.0 - p_safe) * np.log2(1.0 - p_safe))

        is_uncertain = variance >= self.uncertainty_threshold

        return {
            "mean_prob": mean_prob,
            "epistemic_variance": variance,
            "predictive_entropy": entropy,
            "is_uncertain": is_uncertain,
            "member_probs": P,
        }
