import torch
import torch.nn.functional as F
from torch import nn


class ScratchLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        nn.init.zeros_(self.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.weight + self.bias


class HeartMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, int] = (64, 32), dropout: float = 0.25) -> None:
        super().__init__()
        hidden_1, hidden_2 = hidden_dims
        self.dropout = dropout
        self.fc1 = ScratchLinear(input_dim, hidden_1)
        self.bn1 = nn.BatchNorm1d(hidden_1)
        self.fc2 = ScratchLinear(hidden_1, hidden_2)
        self.bn2 = nn.BatchNorm1d(hidden_2)
        self.fc3 = ScratchLinear(hidden_2, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(features)
        if self.training and hidden.shape[0] == 1:
            self.bn1.eval()
            hidden = self.bn1(hidden)
            self.bn1.train()
        else:
            hidden = self.bn1(hidden)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        hidden = self.fc2(hidden)
        if self.training and hidden.shape[0] == 1:
            self.bn2.eval()
            hidden = self.bn2(hidden)
            self.bn2.train()
        else:
            hidden = self.bn2(hidden)
        hidden = F.relu(hidden)
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)

        logits = self.fc3(hidden)
        return logits.squeeze(1)
