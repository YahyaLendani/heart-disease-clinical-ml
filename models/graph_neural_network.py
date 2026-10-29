"""Graph Neural Networks for Clinical Patient Similarity and Ontology Graphs.

References:
1. Kipf, T. N., & Welling, M. (2017).
   Semi-Supervised Classification with Graph Convolutional Networks (GCN).
   ICLR 2017. arXiv:1609.02907.
2. Veličković, P., et al. (2018).
   Graph Attention Networks (GAT).
   ICLR 2018. arXiv:1710.10903.
3. Zhang, J., et al. (2020).
   Graph-BERT: Only Attention is Needed for Learning Graph Representations.
   arXiv:2001.05783.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_knn_patient_graph(
    features: np.ndarray | torch.Tensor,
    k: int = 5,
    metric: str = "cosine",
    include_self_loops: bool = True,
) -> torch.Tensor:
    """Build a vectorized k-Nearest Neighbors patient similarity adjacency matrix.

    Parameters
    ----------
    features : array-like of shape (num_patients, num_features)
    k : int, number of nearest neighbors per patient
    metric : str, 'cosine' or 'euclidean'
    include_self_loops : bool, whether to add self-loops (A + I)

    Returns
    -------
    adj : torch.Tensor of shape (num_patients, num_patients)
    """
    if isinstance(features, np.ndarray):
        X = torch.from_numpy(features).float()
    else:
        X = features.float()

    num_nodes = X.size(0)
    device = X.device

    if num_nodes == 0:
        return torch.empty((0, 0), dtype=torch.float32, device=device)

    if metric == "cosine":
        normed_X = F.normalize(X, p=2, dim=1, eps=1e-8)
        sim = torch.mm(normed_X, normed_X.t())
    elif metric == "euclidean":
        sim = -torch.cdist(X, X, p=2.0)
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Expected 'cosine' or 'euclidean'.")

    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)

    if k > 0 and num_nodes > 1:
        # Mask self-similarity so topk selects strictly k distinct neighbors
        sim.fill_diagonal_(-float("inf"))
        k_eff = min(k, num_nodes - 1)
        _, topk_indices = torch.topk(sim, k=k_eff, dim=1)

        # Vectorized scatter along rows + undirected symmetrization
        adj.scatter_(1, topk_indices, 1.0)
        adj = torch.maximum(adj, adj.t())

    if include_self_loops:
        adj.fill_diagonal_(1.0)
    else:
        adj.fill_diagonal_(0.0)

    return adj


def normalize_adjacency_gcn(adj: torch.Tensor) -> torch.Tensor:
    """Compute symmetric normalized Laplacian: D^{-1/2} A D^{-1/2} in O(N^2) via broadcasting."""
    deg = torch.sum(adj, dim=1)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)
    deg_inv_sqrt = torch.masked_fill(deg_inv_sqrt, torch.isinf(deg_inv_sqrt), 0.0)
    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


class GraphConvolution(nn.Module):
    """Spectral Graph Convolutional Layer (Kipf & Welling, 2017).

    H^{(l+1)} = sigma(D^{-1/2} A D^{-1/2} H^{(l)} W^{(l)})
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        support = torch.mm(x, self.weight)
        output = torch.mm(norm_adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class GraphAttentionLayer(nn.Module):
    """Graph Attention Network (GAT) Layer (Veličković et al., 2018).

    Computes masked self-attention coefficients over local graph neighbors:
    alpha_{ij} = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(in_features, out_features))
        self.a = nn.Parameter(torch.empty(2 * out_features, 1))
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        Wh = torch.mm(h, self.W)  # (N, out_features)

        # Decompose a^T [Wh_i || Wh_j] = (Wh_i) a_src + (Wh_j) a_dst in O(N * F + N^2)
        a_src = self.a[: self.out_features, :]  # (out_features, 1)
        a_dst = self.a[self.out_features :, :]  # (out_features, 1)
        e_src = torch.mm(Wh, a_src)             # (N, 1)
        e_dst = torch.mm(Wh, a_dst)             # (N, 1)
        e = self.leaky_relu(e_src + e_dst.t())  # (N, N)

        # Mask non-edges: zero attention for non-neighbors
        attention = torch.where(adj > 0, e, torch.full_like(e, -9e15))
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)

        h_prime = torch.mm(attention, Wh)
        if self.concat:
            return F.elu(h_prime)
        return h_prime


class GCNClassifier(nn.Module):
    """Multi-layer Graph Convolutional Network Classifier (Kipf & Welling)."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 64,
        out_features: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.gc1 = GraphConvolution(in_features, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, out_features)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.gc1(x, norm_adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.gc2(x, norm_adj)
        return logits


class GATClassifier(nn.Module):
    """Multi-Head Graph Attention Network Classifier (Veličković et al.)."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 32,
        num_heads: int = 2,
        out_features: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.dropout = dropout
        self.heads = nn.ModuleList([
            GraphAttentionLayer(in_features, hidden_dim, dropout=dropout, concat=True)
            for _ in range(num_heads)
        ])
        # Output layer aggregates multi-head features
        self.out_att = GraphAttentionLayer(
            hidden_dim * num_heads, out_features, dropout=dropout, concat=False
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        head_outs = [att(x, adj) for att in self.heads]
        x = torch.cat(head_outs, dim=1)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.out_att(x, adj)
        return logits
