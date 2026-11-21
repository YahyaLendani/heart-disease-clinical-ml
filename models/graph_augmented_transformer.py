"""Graph-Augmented Transformer for Interpretable Clinical Risk Prediction.

Combines two complementary levels of clinical self-attention in a single
end-to-end differentiable, inductively safe architecture:

1. Level 1 — Intra-Patient Feature Transformer (FT-Transformer style):
   Tokenizes both categorical clinical variables (embedding tables) and
   continuous biomarkers (per-feature learnable linear projections), prepends
   a learnable ``[CLS]`` patient summary token, and applies Pre-LayerNorm
   Multi-Head Self-Attention blocks. Extracts ``[CLS]``-to-feature attention
   weights for biomarker-level interpretability.

2. Level 2 — Inter-Patient Cohort Graph Attention (Inductive GAT):
   Refines each patient's ``[CLS]`` embedding by attending over clinically
   similar patients in an inductively masked k-NN cohort graph (test patients
   attend strictly to historical training patients + self-loop, guaranteeing
   zero test-to-test leakage). Extracts patient-to-neighbor attention weights
   for case-based clinical interpretability.

3. Gated Residual Fusion:
   Learns a per-patient sigmoid gate balancing individual tabular features
   against cohort graph context:
       g = sigmoid(W_g [h_tab || h_graph] + b_g)
       h_fused = g * h_graph + (1 - g) * h_tab

References:
- Gorishniy, Y., et al. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS.
- Huang, X., et al. (2020). TabTransformer: Tabular Data Modeling Using Contextual Embeddings.
- Veličković, P., et al. (2018). Graph Attention Networks. ICLR.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors

from .tab_transformer import TransformerEncoderBlock


def build_inductive_knn_indices(
    X_train: np.ndarray,
    X_eval: np.ndarray | None = None,
    k: int = 7,
    metric: str = "cosine",
    n_jobs: int = -1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute strictly inductive top-k training neighbor indices in O(N * k) memory.

    Scales effortlessly to large cohorts (e.g. N = 70,000+ patients) without
    materializing an N x N dense adjacency matrix.

    Zero-leakage invariants:
    - Each training patient retrieves its top-k nearest *other* training patients
      (excluding itself; self-loop is handled explicitly by ``forward_neighbors``).
    - Each eval/test patient retrieves its top-k nearest *training* patients ONLY
      (strictly zero test-to-test or test-to-train leakage).

    Returns
    -------
    train_nbr_idx : np.ndarray of shape ``(n_train, k_eff)``, dtype int64
    eval_nbr_idx : np.ndarray of shape ``(n_eval, k_eff)`` or None
    """
    X_tr = np.asarray(X_train, dtype=np.float32)
    n_tr = len(X_tr)
    k_eff = min(k, max(1, n_tr - 1))

    nn_index = NearestNeighbors(n_neighbors=k_eff + 1, metric=metric, n_jobs=n_jobs)
    nn_index.fit(X_tr)

    # For training points, drop self (which may appear at col 0 or due to exact ties)
    raw_tr_idx = nn_index.kneighbors(X_tr, n_neighbors=k_eff + 1, return_distance=False)
    train_nbr_idx = np.empty((n_tr, k_eff), dtype=np.int64)
    for i in range(n_tr):
        row = raw_tr_idx[i]
        filtered = row[row != i]
        train_nbr_idx[i] = filtered[:k_eff]

    if X_eval is None:
        return train_nbr_idx, None

    X_ev = np.asarray(X_eval, dtype=np.float32)
    eval_nbr_idx = nn_index.kneighbors(X_ev, n_neighbors=k_eff, return_distance=False).astype(np.int64)
    return train_nbr_idx, eval_nbr_idx


def build_inductive_patient_graph(
    X_train: np.ndarray | torch.Tensor,
    X_eval: np.ndarray | torch.Tensor | None = None,
    k: int = 5,
    metric: str = "cosine",
) -> torch.Tensor:
    """Build a strictly inductive k-NN patient similarity adjacency matrix.

    Zero-leakage invariants:
    - Training patients connect ONLY to other training patients (undirected k-NN + self-loop).
    - When ``X_eval`` is provided, evaluation/test patients connect ONLY to their
      top-k nearest training patients (directed train -> eval message passing) plus
      their own self-loop.
    - Train-to-eval block is strictly 0 (eval patients never influence train embeddings).
    - Eval-to-eval off-diagonal block is strictly 0 (zero test-to-test leakage).

    Parameters
    ----------
    X_train : array-like of shape (n_train, n_features)
        Standardized training cohort features.
    X_eval : array-like of shape (n_eval, n_features), optional
        Standardized validation or test cohort features.
    k : int
        Number of nearest training neighbors per patient.
    metric : str
        Similarity metric: ``'cosine'`` or ``'euclidean'``.

    Returns
    -------
    adj : torch.Tensor
        Adjacency matrix of shape ``(n_train, n_train)`` when ``X_eval is None``,
        or ``(n_train + n_eval, n_train + n_eval)`` otherwise.
    """
    X_tr = torch.from_numpy(X_train).float() if isinstance(X_train, np.ndarray) else X_train.float()
    n_tr = X_tr.size(0)
    device = X_tr.device

    if n_tr == 0:
        return torch.empty((0, 0), dtype=torch.float32, device=device)

    # 1. Undirected k-NN graph strictly among training patients
    if metric == "cosine":
        X_tr_norm = F.normalize(X_tr, p=2, dim=1, eps=1e-8)
        sim_tr = torch.mm(X_tr_norm, X_tr_norm.t())
    elif metric == "euclidean":
        sim_tr = -torch.cdist(X_tr, X_tr, p=2.0)
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Expected 'cosine' or 'euclidean'.")

    adj_tr = torch.zeros((n_tr, n_tr), dtype=torch.float32, device=device)
    if k > 0 and n_tr > 1:
        sim_tr.fill_diagonal_(-float("inf"))
        k_eff = min(k, n_tr - 1)
        _, topk_tr = torch.topk(sim_tr, k=k_eff, dim=1)
        adj_tr.scatter_(1, topk_tr, 1.0)
        adj_tr = torch.maximum(adj_tr, adj_tr.t())
    adj_tr.fill_diagonal_(1.0)

    if X_eval is None:
        return adj_tr

    X_ev = (
        torch.from_numpy(X_eval).float().to(device)
        if isinstance(X_eval, np.ndarray)
        else X_eval.float().to(device)
    )
    n_ev = X_ev.size(0)
    n_total = n_tr + n_ev

    adj = torch.zeros((n_total, n_total), dtype=torch.float32, device=device)
    adj[:n_tr, :n_tr] = adj_tr

    if n_ev > 0:
        if metric == "cosine":
            X_ev_norm = F.normalize(X_ev, p=2, dim=1, eps=1e-8)
            sim_ev_tr = torch.mm(X_ev_norm, X_tr_norm.t())
        else:
            sim_ev_tr = -torch.cdist(X_ev, X_tr, p=2.0)

        if k > 0:
            k_ev = min(k, n_tr)
            _, topk_ev = torch.topk(sim_ev_tr, k=k_ev, dim=1)
            adj_ev_tr = torch.zeros((n_ev, n_tr), dtype=torch.float32, device=device)
            adj_ev_tr.scatter_(1, topk_ev, 1.0)
            # Directed: eval nodes attend to train neighbors; train nodes NEVER attend to eval
            adj[n_tr:, :n_tr] = adj_ev_tr

        # Self-loops for eval nodes only (off-diagonal eval-eval remains strictly 0)
        eval_diag = torch.arange(n_tr, n_total, device=device)
        adj[eval_diag, eval_diag] = 1.0

    return adj


class FeatureTokenizer(nn.Module):
    """Unified Feature Tokenizer for categorical and continuous clinical variables.

    Maps each categorical variable via an embedding lookup and each continuous
    biomarker via either a linear projection (``x_j * W_j + b_j``) or
    Periodic Linear-ReLU (PLR) numerical embeddings (Gorishniy et al., NeurIPS 2022)
    which capture sharp clinical cutoffs (e.g. Stage-2 hypertension >= 140 mmHg),
    prepending a learnable ``[CLS]`` summary token.
    """

    def __init__(
        self,
        categorical_dims: Sequence[int],
        continuous_dim: int,
        embed_dim: int = 32,
        use_periodic_embeddings: bool = False,
        n_frequencies: int = 8,
    ) -> None:
        super().__init__()
        self.num_categorical = len(categorical_dims)
        self.continuous_dim = continuous_dim
        self.embed_dim = embed_dim
        self.use_periodic_embeddings = use_periodic_embeddings
        self.n_frequencies = n_frequencies

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim) for cardinality in categorical_dims
        ])

        if continuous_dim > 0:
            self.cont_weight = nn.Parameter(torch.empty(1, continuous_dim, embed_dim))
            self.cont_bias = nn.Parameter(torch.zeros(1, continuous_dim, embed_dim))
            nn.init.xavier_uniform_(self.cont_weight)

            if use_periodic_embeddings:
                # Per-feature learnable frequencies + periodic projection to embed_dim
                self.plr_freqs = nn.Parameter(torch.randn(1, continuous_dim, n_frequencies) * 0.25)
                self.plr_weight = nn.Parameter(
                    torch.empty(continuous_dim, 2 * n_frequencies, embed_dim)
                )
                self.plr_bias = nn.Parameter(torch.zeros(1, continuous_dim, embed_dim))
                nn.init.xavier_uniform_(self.plr_weight)
            else:
                self.register_parameter("plr_freqs", None)
                self.register_parameter("plr_weight", None)
                self.register_parameter("plr_bias", None)
        else:
            self.register_parameter("cont_weight", None)
            self.register_parameter("cont_bias", None)
            self.register_parameter("plr_freqs", None)
            self.register_parameter("plr_weight", None)
            self.register_parameter("plr_bias", None)

    def forward(
        self,
        x_cat: torch.Tensor | None = None,
        x_cont: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Tokenize tabular inputs into a sequence of shape ``(N, 1 + F, embed_dim)``."""
        batch_size = x_cat.size(0) if x_cat is not None else (
            x_cont.size(0) if x_cont is not None else 0
        )
        if batch_size == 0:
            raise ValueError("At least one of x_cat or x_cont must be provided with N > 0.")

        tokens: list[torch.Tensor] = [self.cls_token.expand(batch_size, -1, -1)]

        if self.num_categorical > 0 and x_cat is not None:
            cat_tokens = torch.stack(
                [self.cat_embeddings[i](x_cat[:, i]) for i in range(self.num_categorical)],
                dim=1,
            )
            tokens.append(cat_tokens)

        if self.continuous_dim > 0 and x_cont is not None:
            assert self.cont_weight is not None and self.cont_bias is not None
            # Linear skip term: (N, C, 1) * (1, C, D) + (1, C, D) -> (N, C, D)
            cont_tokens = x_cont.unsqueeze(-1) * self.cont_weight + self.cont_bias

            if self.use_periodic_embeddings:
                assert self.plr_freqs is not None and self.plr_weight is not None and self.plr_bias is not None
                # Periodic Linear-ReLU (PLR): [sin(2*pi*c*x), cos(2*pi*c*x)] -> Linear -> ReLU
                angles = 2.0 * torch.pi * x_cont.unsqueeze(-1) * self.plr_freqs  # (N, C, K)
                periodic = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (N, C, 2K)
                plr_out = F.relu(
                    torch.einsum("nck,ckd->ncd", periodic, self.plr_weight) + self.plr_bias
                )
                cont_tokens = cont_tokens + plr_out

            tokens.append(cont_tokens)

        return torch.cat(tokens, dim=1)


class InterpretableMultiHeadGATLayer(nn.Module):
    """Factored O(N*F + N^2) Multi-Head Graph Attention Layer returning attention weights."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.15,
        alpha: float = 0.2,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.W = nn.Linear(embed_dim, embed_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(-1), gain=1.414)
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(-1), gain=1.414)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute multi-head graph attention over patient embeddings.

        Parameters
        ----------
        h : torch.Tensor of shape ``(N, embed_dim)``
        adj : torch.Tensor of shape ``(N, N)``

        Returns
        -------
        h_out : torch.Tensor of shape ``(N, embed_dim)``
        mean_attn : torch.Tensor of shape ``(N, N)``
            Head-averaged normalized neighbor attention weights.
        """
        n_nodes = h.size(0)
        # (N, H, D_h)
        Wh = self.W(h).view(n_nodes, self.num_heads, self.head_dim)

        # Factored attention logits: (H, N, 1) + (H, 1, N) -> (H, N, N)
        e_src = torch.einsum("nhd,hd->hn", Wh, self.a_src).unsqueeze(-1)
        e_dst = torch.einsum("nhd,hd->hn", Wh, self.a_dst).unsqueeze(-2)
        e = self.leaky_relu(e_src + e_dst)

        mask = (adj > 0).unsqueeze(0)  # (1, N, N)
        e_masked = e.masked_fill(~mask, -1e9)
        attn = F.softmax(e_masked, dim=-1)  # (H, N, N)
        attn_dropped = F.dropout(attn, p=self.dropout, training=self.training)

        # Aggregate neighbor values: (H, N, N) x (H, N, D_h) -> (H, N, D_h)
        Wh_t = Wh.permute(1, 0, 2)
        h_agg = torch.bmm(attn_dropped, Wh_t).permute(1, 0, 2).contiguous()
        h_out = F.elu(self.out_proj(h_agg.view(n_nodes, self.embed_dim)))

        mean_attn = attn.mean(dim=0)  # (N, N)
        return h_out, mean_attn

    def forward_neighbors(
        self,
        h_query: torch.Tensor,
        h_neighbors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scalable O(B * k * D) inductive ego-graph attention for large cohorts (N >> 5,000).

        Computes the exact same Multi-Head Graph Attention formula as ``forward``,
        but restricted to each query patient's self-loop plus its ``k`` retrieved
        training neighbors instead of materializing an ``(N, N)`` dense matrix.

        Parameters
        ----------
        h_query : torch.Tensor of shape ``(B, embed_dim)``
        h_neighbors : torch.Tensor of shape ``(B, k, embed_dim)``

        Returns
        -------
        h_out : torch.Tensor of shape ``(B, embed_dim)``
        mean_attn : torch.Tensor of shape ``(B, 1 + k)``
            Head-averaged attention weights over ``[self, nbr_1, ..., nbr_k]``.
        """
        batch_size = h_query.size(0)
        # Concatenate self-loop at index 0 with k retrieved neighbors: (B, 1 + k, D)
        h_ego = torch.cat([h_query.unsqueeze(1), h_neighbors], dim=1)
        ego_len = h_ego.size(1)

        Wh_q = self.W(h_query).view(batch_size, self.num_heads, self.head_dim)
        Wh_ego = self.W(h_ego).view(batch_size, ego_len, self.num_heads, self.head_dim)

        # Factored ego-graph attention logits: (B, H, 1) + (B, H, 1 + k) -> (B, H, 1 + k)
        e_src = torch.einsum("bhd,hd->bh", Wh_q, self.a_src).unsqueeze(-1)
        e_dst = torch.einsum("bkhd,hd->bhk", Wh_ego, self.a_dst)
        e = self.leaky_relu(e_src + e_dst)

        attn = F.softmax(e, dim=-1)  # (B, H, 1 + k)
        attn_dropped = F.dropout(attn, p=self.dropout, training=self.training)

        # Aggregate values: (B, H, 1 + k) x (B, 1 + k, H, D_h) -> (B, H, D_h)
        h_agg = torch.einsum("bhk,bkhd->bhd", attn_dropped, Wh_ego).contiguous()
        h_out = F.elu(self.out_proj(h_agg.view(batch_size, self.embed_dim)))

        mean_attn = attn.mean(dim=1)  # (B, 1 + k)
        return h_out, mean_attn


class GraphAugmentedClinicalTransformer(nn.Module):
    """Graph-Augmented Transformer for Interpretable Clinical Risk Prediction.

    Unifies intra-patient feature-token self-attention with inter-patient
    inductive cohort graph attention via gated residual fusion.
    """

    def __init__(
        self,
        categorical_dims: Sequence[int],
        continuous_dim: int,
        embed_dim: int = 32,
        num_transformer_heads: int = 4,
        num_transformer_layers: int = 2,
        num_gat_heads: int = 4,
        num_gat_layers: int = 2,
        mlp_hidden_dim: int = 32,
        dropout: float = 0.2,
        num_classes: int = 1,
        use_periodic_embeddings: bool = False,
        n_frequencies: int = 8,
        use_label_propagation: bool = False,
    ) -> None:
        super().__init__()
        self.num_categorical = len(categorical_dims)
        self.continuous_dim = continuous_dim
        self.num_features = self.num_categorical + continuous_dim
        self.embed_dim = embed_dim
        self.use_periodic_embeddings = use_periodic_embeddings
        self.use_label_propagation = use_label_propagation

        # Level 1: Intra-Patient Feature Tokenizer (with optional PLR embeddings) + Transformer
        self.tokenizer = FeatureTokenizer(
            categorical_dims=categorical_dims,
            continuous_dim=continuous_dim,
            embed_dim=embed_dim,
            use_periodic_embeddings=use_periodic_embeddings,
            n_frequencies=n_frequencies,
        )
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
            )
            for _ in range(num_transformer_layers)
        ])
        self.tab_norm = nn.LayerNorm(embed_dim)

        # Optional UniMP Masked Label Propagation embedding (0=Healthy, 1=CVD+, 2=Masked/Self)
        if use_label_propagation:
            self.label_embedding = nn.Embedding(3, embed_dim)
            nn.init.normal_(self.label_embedding.weight, std=0.02)
        else:
            self.label_embedding = None

        # Level 2: Inter-Patient Inductive Cohort Graph Attention
        self.gat_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_gat_layers)])
        self.gat_layers = nn.ModuleList([
            InterpretableMultiHeadGATLayer(
                embed_dim=embed_dim,
                num_heads=num_gat_heads,
                dropout=dropout,
            )
            for _ in range(num_gat_layers)
        ])

        # Gated Residual Fusion: balances tabular vs. cohort graph evidence
        self.gate_proj = nn.Linear(2 * embed_dim, embed_dim)
        self.fusion_norm = nn.LayerNorm(embed_dim)

        # Clinical Risk Prediction Head
        self.risk_head = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_classes),
        )

    def encode_tabular(
        self,
        x_cat: torch.Tensor | None,
        x_cont: torch.Tensor | None,
        feature_noise_std: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Level-1 Intra-Patient Feature Transformer.

        Returns
        -------
        h_tab : torch.Tensor of shape ``(N, embed_dim)``
        feat_attn : torch.Tensor of shape ``(N, num_features)``
        """
        if self.training and feature_noise_std > 0.0 and x_cont is not None:
            x_cont = x_cont + torch.randn_like(x_cont) * feature_noise_std

        tokens = self.tokenizer(x_cat=x_cat, x_cont=x_cont)  # (N, 1 + F, D)
        cls_feature_attns: list[torch.Tensor] = []

        for block in self.transformer_blocks:
            tokens, attn_weights = block(tokens)  # attn_weights: (N, H, 1+F, 1+F)
            cls_to_feat = attn_weights[:, :, 0, 1:].mean(dim=1)  # (N, F)
            cls_feature_attns.append(cls_to_feat)

        h_tab = self.tab_norm(tokens[:, 0, :])  # (N, D)
        feat_attn = torch.stack(cls_feature_attns, dim=0).mean(dim=0)
        feat_attn = feat_attn / feat_attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return h_tab, feat_attn

    def forward(
        self,
        x_cat: torch.Tensor | None,
        x_cont: torch.Tensor | None,
        adj: torch.Tensor,
        return_explanations: bool = False,
        feature_noise_std: float = 0.0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Full-graph forward pass with optional dual-level interpretability outputs.

        Parameters
        ----------
        x_cat : torch.Tensor or None, shape ``(N, num_categorical)``
        x_cont : torch.Tensor or None, shape ``(N, continuous_dim)``
        adj : torch.Tensor, shape ``(N, N)``
            Inductive patient similarity adjacency matrix.
        return_explanations : bool
            When True, returns ``(logits, explanations)`` containing:
            - ``feature_attn``: ``(N, num_features)`` normalized ``[CLS]``-to-feature attention
            - ``graph_attn``: ``(N, N)`` mean patient-to-neighbor graph attention
            - ``gate``: ``(N, embed_dim)`` per-patient graph reliance gate in ``(0, 1)``
        feature_noise_std : float
            Optional train-only Gaussian jitter on continuous biomarkers for regularization.
        """
        # Level 1: Intra-patient Feature Transformer
        h_tab, feat_attn = self.encode_tabular(
            x_cat=x_cat, x_cont=x_cont, feature_noise_std=feature_noise_std
        )

        # Level 2: Inter-patient Cohort Graph Attention
        h_graph = h_tab
        graph_attns: list[torch.Tensor] = []
        for norm, gat in zip(self.gat_norms, self.gat_layers, strict=True):
            delta, g_attn = gat(norm(h_graph), adj)
            h_graph = h_graph + delta
            graph_attns.append(g_attn)

        # Gated Residual Fusion
        gate = torch.sigmoid(self.gate_proj(torch.cat([h_tab, h_graph], dim=-1)))
        h_fused = self.fusion_norm(gate * h_graph + (1.0 - gate) * h_tab)
        logits = self.risk_head(h_fused)

        if not return_explanations:
            return logits

        mean_graph_attn = torch.stack(graph_attns, dim=0).mean(dim=0)  # (N, N)
        explanations = {
            "feature_attn": feat_attn,
            "graph_attn": mean_graph_attn,
            "gate": gate,
        }
        return logits, explanations

    def forward_minibatch(
        self,
        x_cat: torch.Tensor | None,
        x_cont: torch.Tensor | None,
        nbr_x_cat: torch.Tensor | None = None,
        nbr_x_cont: torch.Tensor | None = None,
        nbr_h_tab: torch.Tensor | None = None,
        nbr_labels: torch.Tensor | None = None,
        label_mask_prob: float = 0.0,
        return_explanations: bool = False,
        feature_noise_std: float = 0.0,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Scalable O(B * k) mini-batch inductive forward pass for large clinical cohorts.

        Parameters
        ----------
        x_cat : torch.Tensor or None, shape ``(B, num_categorical)``
        x_cont : torch.Tensor or None, shape ``(B, continuous_dim)``
        nbr_x_cat : torch.Tensor or None, shape ``(B, k, num_categorical)``
            Categorical features of each query patient's ``k`` training neighbors.
        nbr_x_cont : torch.Tensor or None, shape ``(B, k, continuous_dim)``
            Continuous features of each query patient's ``k`` training neighbors.
        nbr_h_tab : torch.Tensor or None, shape ``(B, k, embed_dim)``
            Optional pre-encoded neighbor embeddings (avoids re-encoding neighbors).
        nbr_labels : torch.Tensor or None, shape ``(B, k)``
            Optional historical training outcomes (0 or 1) for UniMP masked label
            propagation. Query self-loop is ALWAYS masked (label id = 2) so zero
            target leakage can occur.
        label_mask_prob : float
            During training, randomly masks neighbor labels to 2 with this probability
            so the model learns balanced reliance on features and cohort outcomes.
        return_explanations : bool
            When True, returns ``(logits, explanations)`` where ``neighbor_attn``
            has shape ``(B, 1 + k)`` (index 0 = self, indices 1..k = cohort neighbors).
        """
        h_tab, feat_attn = self.encode_tabular(
            x_cat=x_cat, x_cont=x_cont, feature_noise_std=feature_noise_std
        )
        batch_size = h_tab.size(0)

        if nbr_h_tab is None:
            if nbr_x_cat is None and nbr_x_cont is None:
                raise ValueError("Provide either nbr_h_tab or (nbr_x_cat / nbr_x_cont).")
            k_nbrs = nbr_x_cat.size(1) if nbr_x_cat is not None else nbr_x_cont.size(1)  # type: ignore[union-attr]
            flat_cat = nbr_x_cat.reshape(batch_size * k_nbrs, -1) if nbr_x_cat is not None else None
            flat_cont = (
                nbr_x_cont.reshape(batch_size * k_nbrs, -1) if nbr_x_cont is not None else None
            )
            nbr_flat_h, _ = self.encode_tabular(
                x_cat=flat_cat, x_cont=flat_cont, feature_noise_std=0.0
            )
            nbr_h = nbr_flat_h.view(batch_size, k_nbrs, self.embed_dim)
        else:
            nbr_h = nbr_h_tab

        # UniMP Masked Label Propagation: inject neighbor outcome embeddings (self strictly masked)
        h_graph = h_tab
        if self.label_embedding is not None:
            if nbr_labels is None:
                eff_labels = torch.full(
                    (batch_size, nbr_h.size(1)), 2, dtype=torch.long, device=h_tab.device
                )
            else:
                eff_labels = nbr_labels.long().clamp(0, 2)
                if self.training and label_mask_prob > 0.0:
                    drop_mask = torch.rand_like(eff_labels.float()) < label_mask_prob
                    eff_labels = torch.where(drop_mask, torch.full_like(eff_labels, 2), eff_labels)
            nbr_h = nbr_h + self.label_embedding(eff_labels)
            # Query node always receives masked label token (id=2 -> zero self-label leakage)
            self_mask_tok = torch.full((batch_size,), 2, dtype=torch.long, device=h_tab.device)
            h_graph = h_graph + self.label_embedding(self_mask_tok)

        # Level 2: Scalable Ego-Graph Multi-Head Attention over [self, nbr_1..nbr_k]
        neighbor_attns: list[torch.Tensor] = []
        for norm, gat in zip(self.gat_norms, self.gat_layers, strict=True):
            delta, n_attn = gat.forward_neighbors(norm(h_graph), norm(nbr_h))
            h_graph = h_graph + delta
            neighbor_attns.append(n_attn)

        gate = torch.sigmoid(self.gate_proj(torch.cat([h_tab, h_graph], dim=-1)))
        h_fused = self.fusion_norm(gate * h_graph + (1.0 - gate) * h_tab)
        logits = self.risk_head(h_fused)

        if not return_explanations:
            return logits

        mean_nbr_attn = torch.stack(neighbor_attns, dim=0).mean(dim=0)  # (B, 1 + k)
        explanations = {
            "feature_attn": feat_attn,
            "neighbor_attn": mean_nbr_attn,
            "gate": gate,
        }
        return logits, explanations

    @torch.inference_mode()
    def explain_prediction(
        self,
        x_cat: torch.Tensor | None,
        x_cont: torch.Tensor | None,
        adj: torch.Tensor,
        patient_idx: int,
        feature_names: Sequence[str],
        top_k_neighbors: int = 5,
    ) -> dict[str, Any]:
        """Generate a dual-level clinical explanation for a single patient.

        Returns
        -------
        dict with keys:
        - ``risk_probability`` : float in [0, 1]
        - ``graph_reliance_gate`` : float in [0, 1] (mean reliance on cohort graph vs tabular)
        - ``feature_importance`` : dict[str, float] sorted descending by attention weight
        - ``top_cohort_neighbors`` : list[dict] of top-k neighbor indices & attention weights
        """
        was_training = self.training
        self.eval()
        logits, expl = self.forward(x_cat, x_cont, adj, return_explanations=True)
        if was_training:
            self.train()

        prob = float(torch.sigmoid(logits[patient_idx].squeeze()).item())
        gate_val = float(expl["gate"][patient_idx].mean().item())

        feat_weights = expl["feature_attn"][patient_idx].cpu().numpy()
        if len(feature_names) != len(feat_weights):
            raise ValueError(
                f"feature_names length ({len(feature_names)}) must match "
                f"num_features ({len(feat_weights)})."
            )
        feature_importance = dict(
            sorted(
                ((name, float(w)) for name, w in zip(feature_names, feat_weights, strict=True)),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        # Top cohort neighbors excluding self-loop
        neighbor_weights = expl["graph_attn"][patient_idx].clone()
        neighbor_weights[patient_idx] = -1.0
        k_eff = min(top_k_neighbors, max(0, neighbor_weights.numel() - 1))
        top_neighbors: list[dict[str, float | int]] = []
        if k_eff > 0:
            vals, idxs = torch.topk(neighbor_weights, k=k_eff)
            for idx_t, val_t in zip(idxs.tolist(), vals.tolist(), strict=True):
                if val_t > 0.0:
                    top_neighbors.append({
                        "neighbor_idx": int(idx_t),
                        "attention_weight": float(val_t),
                    })

        return {
            "patient_idx": int(patient_idx),
            "risk_probability": prob,
            "graph_reliance_gate": gate_val,
            "feature_importance": feature_importance,
            "top_cohort_neighbors": top_neighbors,
        }
