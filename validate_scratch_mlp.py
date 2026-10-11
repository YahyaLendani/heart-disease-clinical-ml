from pathlib import Path
import sys

import torch
from torch import nn

try:
    from models import HeartMLP
    from training.train_model import set_seed
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent))
    from models import HeartMLP
    from training.train_model import set_seed


def main() -> None:
    set_seed(42)

    input_dim = 13
    batch_size = 16

    model = HeartMLP(input_dim=input_dim, dropout=0.2)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    features = torch.randn(batch_size, input_dim)
    targets = torch.randint(0, 2, (batch_size,), dtype=torch.float32)

    logits = model(features)
    assert logits.shape == (batch_size,), f"Unexpected output shape: {logits.shape}"

    params_before = [param.detach().clone() for param in model.parameters()]

    optimizer.zero_grad()
    loss = criterion(logits, targets)
    loss.backward()

    grad_norms = []
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad_norms.append(parameter.grad.detach().abs().sum().item())

    assert grad_norms, "No gradients found on model parameters."
    assert any(value > 0 for value in grad_norms), "All gradients are zero."

    optimizer.step()

    params_after = [param.detach().clone() for param in model.parameters()]
    total_change = sum((after - before).abs().sum().item() for before, after in zip(params_before, params_after))
    assert total_change > 0, "Parameters did not update after optimizer step."

    print("Scratch MLP validation passed.")
    print(f"Loss: {loss.item():.6f}")
    print(f"Total parameter change after one step: {total_change:.6f}")


if __name__ == "__main__":
    main()
