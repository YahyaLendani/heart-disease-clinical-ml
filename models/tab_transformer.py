"""TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

Reference:
Huang, X., Khetan, A., Cvitkovic, M., & Karnin, Z. (2020).
TabTransformer: Tabular Data Modeling Using Contextual Embeddings.
arXiv:2012.06678.

Also implements the Multi-Head Attention mechanism from:
Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Scaled Dot-Product Multi-Head Self-Attention.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, num_tokens, embed_dim)

        Returns
        -------
        output : torch.Tensor of shape (batch_size, num_tokens, embed_dim)
        attn_weights : torch.Tensor of shape (batch_size, num_heads, num_tokens, num_tokens)
        """
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.attn_dropout(attn_weights)

        context = torch.matmul(attn_weights_dropped, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        output = self.out_dropout(self.out_proj(context))
        return output, attn_weights


class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block with Pre-LayerNorm architecture."""

    def __init__(self, embed_dim: int, num_heads: int = 4, ffn_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        ffn_dim = ffn_dim or (embed_dim * 4)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads=num_heads, dropout=dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed)
        x = x + attn_out

        x = x + self.ffn(self.norm2(x))
        return x, attn_weights


class TabTransformer(nn.Module):
    """TabTransformer architecture for tabular data with categorical and continuous features.

    Categorical features are mapped to embeddings, contextualized via Transformer
    blocks, concatenated with normalized continuous features, and passed to an MLP head.
    """

    def __init__(
        self,
        categorical_dims: Sequence[int],
        continuous_dim: int,
        embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_hidden_dims: Sequence[int] = (64, 32),
        dropout: float = 0.1,
        num_classes: int = 1,
    ):
        super().__init__()
        self.num_categorical = len(categorical_dims)
        self.continuous_dim = continuous_dim
        self.embed_dim = embed_dim

        # Categorical embedding tables (one per categorical variable)
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim) for cardinality in categorical_dims
        ])

        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads=num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        # Continuous feature normalization
        if continuous_dim > 0:
            self.cont_norm = nn.LayerNorm(continuous_dim)
        else:
            self.cont_norm = None

        # MLP prediction head
        in_mlp_dim = (self.num_categorical * embed_dim) + continuous_dim
        mlp_layers: list[nn.Module] = []
        curr_dim = in_mlp_dim
        for h_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(curr_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            curr_dim = h_dim
        mlp_layers.append(nn.Linear(curr_dim, num_classes))
        self.mlp_head = nn.Sequential(*mlp_layers)

    def forward(
        self,
        x_cat: torch.Tensor | None = None,
        x_cont: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass.

        Parameters
        ----------
        x_cat : torch.Tensor of shape (batch_size, num_categorical) with integer indices.
        x_cont : torch.Tensor of shape (batch_size, continuous_dim) with float values.
        return_attn : bool, whether to return attention weight matrices.
        """
        all_attn_weights: list[torch.Tensor] = []
        parts: list[torch.Tensor] = []

        if self.num_categorical > 0 and x_cat is not None:
            # Map each categorical variable to embedding vector: (batch_size, 1, embed_dim)
            cat_embeds = [
                self.embeddings[i](x_cat[:, i]).unsqueeze(1)
                for i in range(self.num_categorical)
            ]
            # Concatenate along token sequence dimension: (batch_size, num_cat, embed_dim)
            x_cat_seq = torch.cat(cat_embeds, dim=1)

            for layer in self.layers:
                x_cat_seq, attn_weights = layer(x_cat_seq)
                if return_attn:
                    all_attn_weights.append(attn_weights)

            # Flatten contextual embeddings: (batch_size, num_cat * embed_dim)
            flat_cat = x_cat_seq.reshape(x_cat_seq.size(0), -1)
            parts.append(flat_cat)

        if self.continuous_dim > 0 and x_cont is not None:
            assert self.cont_norm is not None
            parts.append(self.cont_norm(x_cont))

        if not parts:
            raise ValueError("At least one of x_cat or x_cont must be provided.")

        fused = torch.cat(parts, dim=1)
        logits = self.mlp_head(fused)

        if return_attn:
            return logits, all_attn_weights
        return logits
