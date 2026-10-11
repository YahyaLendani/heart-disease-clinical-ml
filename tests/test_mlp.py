"""Unit tests for the custom PyTorch MLP (HeartMLP with ScratchLinear layers).

Tests verify:
  - Forward pass output shape
  - Gradient flow through all ScratchLinear parameters
  - Parameter updates after optimizer step
  - Mathematical correctness of ScratchLinear (x @ W + b)
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import HeartMLP, ScratchLinear


class TestScratchLinearMath:
    """Verify that ScratchLinear computes x @ W + b exactly."""

    def test_forward_equals_manual_matmul(self):
        """ScratchLinear(x) must equal x @ W + b for a known input."""
        torch.manual_seed(0)
        layer = ScratchLinear(in_features=4, out_features=3)

        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        output = layer(x)

        expected = x @ layer.weight + layer.bias
        assert torch.allclose(output, expected, atol=1e-6), (
            f"ScratchLinear output {output} does not match manual x @ W + b = {expected}"
        )

    def test_forward_batch(self):
        """ScratchLinear handles batched inputs correctly."""
        torch.manual_seed(0)
        layer = ScratchLinear(in_features=5, out_features=2)

        x = torch.randn(8, 5)
        output = layer(x)
        expected = x @ layer.weight + layer.bias
        assert torch.allclose(output, expected, atol=1e-6)
        assert output.shape == (8, 2)


class TestHeartMLPForwardShape:
    """Verify output shape under varying batch sizes and input dimensions."""

    @pytest.mark.parametrize("batch_size", [1, 4, 16, 64])
    def test_output_shape_various_batch_sizes(self, batch_size):
        model = HeartMLP(input_dim=13, hidden_dims=(64, 32), dropout=0.0)
        model.eval()
        x = torch.randn(batch_size, 13)
        logits = model(x)
        assert logits.shape == (batch_size,), (
            f"Expected shape ({batch_size},), got {logits.shape}"
        )

    @pytest.mark.parametrize("input_dim", [5, 13, 50])
    def test_output_shape_various_input_dims(self, input_dim):
        model = HeartMLP(input_dim=input_dim, hidden_dims=(32, 16), dropout=0.0)
        model.eval()
        x = torch.randn(8, input_dim)
        logits = model(x)
        assert logits.shape == (8,)


class TestHeartMLPGradientFlow:
    """Verify that gradients flow through every ScratchLinear layer."""

    def test_gradient_flow_all_layers(self, dummy_model, dummy_tensors):
        """After a backward pass, every ScratchLinear weight and bias must have a gradient."""
        model = dummy_model
        X, y = dummy_tensors

        criterion = nn.BCEWithLogitsLoss()
        model.train()

        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()

        # Check every ScratchLinear layer has gradients
        for name, module in model.named_modules():
            if isinstance(module, ScratchLinear):
                assert module.weight.grad is not None, (
                    f"{name}.weight.grad is None — not attached to computation graph"
                )
                assert module.bias.grad is not None, (
                    f"{name}.bias.grad is None — not attached to computation graph"
                )
                weight_grad_norm = module.weight.grad.abs().sum().item()
                bias_grad_norm = module.bias.grad.abs().sum().item()
                assert weight_grad_norm > 0, (
                    f"{name}.weight.grad is all zeros"
                )
                assert bias_grad_norm > 0, (
                    f"{name}.bias.grad is all zeros"
                )

    def test_all_parameters_have_gradients(self, dummy_model, dummy_tensors):
        """Every trainable parameter in the model must receive a gradient."""
        model = dummy_model
        X, y = dummy_tensors

        criterion = nn.BCEWithLogitsLoss()
        model.train()

        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()

        params_with_grad = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None and param.grad.abs().sum().item() > 0:
                params_with_grad += 1
            else:
                # Only flag if requires_grad is True
                if param.requires_grad:
                    pytest.fail(f"Parameter '{name}' has no gradient or zero gradient")

        assert params_with_grad > 0, "No parameters received gradients at all"


class TestHeartMLPParameterUpdate:
    """Verify that parameters actually change after an optimizer step."""

    def test_parameters_update_after_step(self, dummy_model, dummy_tensors):
        model = dummy_model
        X, y = dummy_tensors

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Record parameters before
        params_before = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
        }

        # Forward + backward + step
        model.train()
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        # Verify parameters changed
        total_change = 0.0
        for name, param in model.named_parameters():
            change = (param.detach() - params_before[name]).abs().sum().item()
            total_change += change

        assert total_change > 0, (
            "Parameters did not update after optimizer.step() — "
            "ScratchLinear layers may not be registered properly"
        )
