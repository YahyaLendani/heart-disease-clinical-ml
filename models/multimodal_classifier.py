"""Multimodal Clinical Classifier: ResNet, Vision Transformer (ViT), and Multimodal Fusion.

References:
1. Baltrušaitis, T., Ahuja, C., & Morency, L.-P. (2018).
   Multimodal Machine Learning: A Survey and Taxonomy. IEEE TPAMI.
   arXiv:1705.09406.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016).
   Deep Residual Learning for Image Recognition (ResNet). CVPR 2016.
   arXiv:1512.03385.
3. Dosovitskiy, A., et al. (2020).
   An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale (ViT).
   ICLR 2021. arXiv:2010.11929.
"""

from __future__ import annotations
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet1DBlock(nn.Module):
    """1D Residual Convolutional Block (He et al., 2016).

    y = ReLU(F(x) + residual(x))
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)


class ViTPatchEncoder(nn.Module):
    """Vision / Diagnostic Signal Transformer Patch Encoder (Dosovitskiy et al., 2020).

    Divides a continuous signal/waveform into non-overlapping patches, maps them via
    linear projection, adds position embeddings and a [CLS] token, and processes via
    Transformer self-attention.
    """

    def __init__(
        self,
        signal_length: int = 128,
        patch_size: int = 16,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if signal_length % patch_size != 0:
            raise ValueError(f"signal_length ({signal_length}) must be divisible by patch_size ({patch_size})")

        self.num_patches = signal_length // patch_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Linear projection of flattened patches
        self.patch_proj = nn.Linear(patch_size, embed_dim)

        # [CLS] token and 1D positional encodings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_signal: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_signal : torch.Tensor of shape (batch_size, signal_length)

        Returns
        -------
        cls_rep : torch.Tensor of shape (batch_size, embed_dim)
        """
        batch_size = x_signal.size(0)
        # Reshape into patches: (batch_size, num_patches, patch_size)
        patches = x_signal.view(batch_size, self.num_patches, self.patch_size)
        projected = self.patch_proj(patches)  # (batch_size, num_patches, embed_dim)

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, projected], dim=1)  # (batch_size, num_patches + 1, embed_dim)
        tokens = tokens + self.pos_embed

        encoded = self.transformer(tokens)
        encoded = self.norm(encoded)
        # Return representation of [CLS] token
        return encoded[:, 0]


class MultimodalFusionClassifier(nn.Module):
    """Multimodal Clinical Classifier combining tabular EHR and diagnostic waveforms.

    Implements the 3 primary fusion paradigms identified in Baltrušaitis et al. (2018):
    1. 'early': Feature concatenation before joint MLP head.
    2. 'late': Decision fusion averaging predictions from unimodal branches.
    3. 'cross_attention': Tabular features attend to visual/signal representations.
    """

    def __init__(
        self,
        tabular_dim: int,
        signal_length: int = 128,
        patch_size: int = 16,
        embed_dim: int = 64,
        fusion_strategy: str = "early",
        num_classes: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.fusion_strategy = fusion_strategy
        self.embed_dim = embed_dim

        # Tabular representation branch
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Signal / Vision Transformer branch
        self.signal_vit = ViTPatchEncoder(
            signal_length=signal_length,
            patch_size=patch_size,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # 1D Residual feature extractor option
        self.resnet_block = ResNet1DBlock(in_channels=1, out_channels=16)

        if fusion_strategy == "early":
            self.head = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
            )
        elif fusion_strategy == "late":
            self.tab_head = nn.Linear(embed_dim, num_classes)
            self.sig_head = nn.Linear(embed_dim, num_classes)
        elif fusion_strategy == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=2, dropout=dropout, batch_first=True
            )
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, num_classes),
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

    def forward(self, x_tab: torch.Tensor, x_sig: torch.Tensor) -> torch.Tensor:
        tab_repr = self.tabular_mlp(x_tab)  # (batch_size, embed_dim)
        sig_repr = self.signal_vit(x_sig)  # (batch_size, embed_dim)

        if self.fusion_strategy == "early":
            fused = torch.cat([tab_repr, sig_repr], dim=1)
            return self.head(fused)

        if self.fusion_strategy == "late":
            logits_tab = self.tab_head(tab_repr)
            logits_sig = self.sig_head(sig_repr)
            return (logits_tab + logits_sig) / 2.0

        if self.fusion_strategy == "cross_attention":
            # Tabular token attends to signal token
            query = tab_repr.unsqueeze(1)
            key_val = sig_repr.unsqueeze(1)
            attn_out, _ = self.cross_attn(query, key_val, key_val)
            fused = attn_out.squeeze(1)
            return self.head(fused)

        raise RuntimeError("Unreachable")
