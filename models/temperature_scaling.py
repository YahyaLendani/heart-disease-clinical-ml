"""Temperature Scaling for Neural Network Probability Calibration.

Reference:
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
On Calibration of Modern Neural Networks.
ICML 2017. PMLR 70:1321-1330. arXiv:1706.04599.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import brier_score_loss


def compute_ece(
    probs: np.ndarray | torch.Tensor,
    labels: np.ndarray | torch.Tensor,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) as defined in Guo et al. (2017).

    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    probs = probs.flatten()
    labels = labels.flatten()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(labels)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (probs > bin_lower) & (probs <= bin_upper) if i > 0 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            bin_acc = np.mean(labels[in_bin])
            bin_conf = np.mean(probs[in_bin])
            ece += np.abs(bin_acc - bin_conf) * (np.sum(in_bin) / n_samples)

    return float(ece)


class ModelWithTemperature(nn.Module):
    """Wrapper that recalibrates any PyTorch model using Temperature Scaling.

    T is a single learnable scalar parameter optimized on validation holdout logits:
    p_calibrated = sigmoid(logit / T)
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        logits = self.model(*args, **kwargs)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits by temperature."""
        temp = self.temperature.clamp(min=1e-3)
        return logits / temp

    def calibrate(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> dict[str, float]:
        """Learn optimal temperature T using validation logits and BCEWithLogitsLoss.

        Parameters
        ----------
        val_logits : torch.Tensor of shape (N, 1) or (N,)
        val_labels : torch.Tensor of shape (N, 1) or (N,)
        """
        val_logits = val_logits.view(-1, 1).float()
        val_labels = val_labels.view(-1, 1).float()

        # Calculate initial metrics before calibration
        init_probs = torch.sigmoid(val_logits).detach().cpu().numpy()
        labels_np = val_labels.cpu().numpy()
        init_ece = compute_ece(init_probs, labels_np)
        init_brier = float(brier_score_loss(labels_np, init_probs))

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = self.temperature_scale(val_logits)
            loss = criterion(scaled_logits, val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        # Ensure temperature remains strictly positive
        with torch.no_grad():
            self.temperature.clamp_(min=1e-3)

        cal_probs = torch.sigmoid(self.temperature_scale(val_logits)).detach().cpu().numpy()
        cal_ece = compute_ece(cal_probs, labels_np)
        cal_brier = float(brier_score_loss(labels_np, cal_probs))

        return {
            "optimal_temperature": float(self.temperature.item()),
            "ece_before": init_ece,
            "ece_after": cal_ece,
            "brier_before": init_brier,
            "brier_after": cal_brier,
        }
