"""Shared model architectures and building blocks."""

from .feature_engineering import add_cardiovascular_features
from .graph_neural_network import (
    GATClassifier,
    GCNClassifier,
    GraphAttentionLayer,
    GraphConvolution,
    build_knn_patient_graph,
    normalize_adjacency_gcn,
)
from .heart_mlp import HeartMLP, ScratchLinear
from .multimodal_classifier import MultimodalFusionClassifier, ResNet1DBlock, ViTPatchEncoder
from .tab_transformer import MultiHeadSelfAttention, TabTransformer, TransformerEncoderBlock
from .temperature_scaling import ModelWithTemperature, compute_ece

__all__ = [
    "HeartMLP",
    "ScratchLinear",
    "add_cardiovascular_features",
    "TabTransformer",
    "MultiHeadSelfAttention",
    "TransformerEncoderBlock",
    "GraphConvolution",
    "GraphAttentionLayer",
    "GCNClassifier",
    "GATClassifier",
    "build_knn_patient_graph",
    "normalize_adjacency_gcn",
    "ResNet1DBlock",
    "ViTPatchEncoder",
    "MultimodalFusionClassifier",
    "ModelWithTemperature",
    "compute_ece",
]
