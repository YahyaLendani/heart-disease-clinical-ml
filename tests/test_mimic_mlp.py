"""Unit tests for MIMIC PyTorch MLP model and custom loss functions."""

import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.heart_mlp import HeartMLP
from training.mimic_train_mlp import BinaryFocalLossWithLogits, get_loss_criterion


class TestBinaryFocalLoss:
    """Test mathematical behavior and properties of BinaryFocalLossWithLogits."""

    def test_loss_decreases_for_well_classified_examples(self):
        """Harder examples should receive higher weight than well-classified ones."""
        criterion = BinaryFocalLossWithLogits(alpha=0.5, gamma=2.0)

        # Well-classified positive: large positive logit for target 1
        logits_easy = torch.tensor([5.0])
        targets_pos = torch.tensor([1.0])
        loss_easy = criterion(logits_easy, targets_pos)

        # Hard-to-classify positive: negative logit for target 1
        logits_hard = torch.tensor([-2.0])
        loss_hard = criterion(logits_hard, targets_pos)

        assert loss_hard.item() > loss_easy.item()
        assert loss_easy.item() >= 0.0

    def test_focal_loss_backward_gradients(self):
        """Loss must support gradient backpropagation."""
        criterion = BinaryFocalLossWithLogits(alpha=0.25, gamma=2.0)
        logits = torch.randn(8, requires_grad=True)
        targets = torch.tensor([0, 1, 0, 0, 1, 0, 0, 0], dtype=torch.float32)

        loss = criterion(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
        assert not torch.isinf(logits.grad).any()


class TestMIMICMLPForward:
    """Test model forward pass and loss criteria initialization."""

    def test_forward_shape_and_logits(self):
        batch_size = 16
        input_dim = 45
        model = HeartMLP(input_dim=input_dim, hidden_dims=(32, 16), dropout=0.2)
        model.eval()

        X = torch.randn(batch_size, input_dim)
        with torch.no_grad():
            out = model(X)

        assert out.shape == (batch_size,)
        assert not torch.isnan(out).any()

    def test_get_loss_criterion_weighted(self):
        device = torch.device("cpu")
        y_train = pd.Series([0] * 90 + [1] * 10)
        criterion = get_loss_criterion("bce_weighted", y_train, device)
        assert isinstance(criterion, torch.nn.BCEWithLogitsLoss)

    def test_get_loss_criterion_focal(self):
        device = torch.device("cpu")
        y_train = pd.Series([0] * 90 + [1] * 10)
        criterion = get_loss_criterion("focal", y_train, device, alpha=0.25, gamma=2.0)
        assert isinstance(criterion, BinaryFocalLossWithLogits)
