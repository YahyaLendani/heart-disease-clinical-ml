"""Grad-CAM: Gradient-Weighted Class Activation Mapping for Neural Networks.

Reference:
Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017).
Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.
ICCV 2017, 618-626. arXiv:1610.02391.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """Computes Grad-CAM saliency heatmaps for PyTorch neural models.

    Attaches hooks to the designated target layer, registers activations during
    the forward pass and gradients during the backward pass.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module: nn.Module, grad_input: tuple, grad_output: tuple) -> None:
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        *inputs: torch.Tensor,
        target_class_idx: int = 0,
    ) -> np.ndarray:
        """Compute Grad-CAM attribution weights for given input sample.

        Parameters
        ----------
        *inputs : input tensors passed to model's forward method.
        target_class_idx : int, class index for logit attribution.

        Returns
        -------
        cam : np.ndarray of shape (seq_len,) or (features,) normalized in [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        logits = self.model(*inputs)
        if logits.dim() == 1 or logits.size(-1) == 1:
            target_score = logits.squeeze()
        else:
            target_score = logits[:, target_class_idx].squeeze()

        # Backward pass
        target_score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Failed to capture gradients or activations in target layer.")

        # If 1D convolutional layer: shape is (batch_size, channels, length)
        if self.gradients.dim() == 3:
            # Global average pooling across temporal/spatial dimension
            weights = torch.mean(self.gradients, dim=2, keepdim=True)  # (batch, channels, 1)
            cam = torch.sum(weights * self.activations, dim=1)  # (batch, length)
            cam = F.relu(cam)  # ReLU to highlight positive contributions
        elif self.gradients.dim() == 2:
            # Linear layer activations: shape is (batch, hidden_dim)
            weights = self.gradients
            cam = F.relu(weights * self.activations)
            cam = torch.mean(cam, dim=1, keepdim=True)
        else:
            cam = F.relu(self.gradients * self.activations)

        cam_np = cam.detach().cpu().numpy().flatten()
        # Normalize between 0 and 1
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        return cam_norm

    def close(self) -> None:
        """Remove hooks to prevent memory leaks."""
        self._forward_hook.remove()
        self._backward_hook.remove()
