"""Shared model architectures and building blocks (names are imported lazily).

``add_cardiovascular_features`` only needs pandas, so tree-model scripts that
use it no longer import PyTorch.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "HeartMLP": ".heart_mlp",
    "ScratchLinear": ".heart_mlp",
    "add_cardiovascular_features": ".feature_engineering",
    "TabTransformer": ".tab_transformer",
    "MultiHeadSelfAttention": ".tab_transformer",
    "TransformerEncoderBlock": ".tab_transformer",
    "GraphConvolution": ".graph_neural_network",
    "GraphAttentionLayer": ".graph_neural_network",
    "GCNClassifier": ".graph_neural_network",
    "GATClassifier": ".graph_neural_network",
    "build_knn_patient_graph": ".graph_neural_network",
    "normalize_adjacency_gcn": ".graph_neural_network",
    "ResNet1DBlock": ".multimodal_classifier",
    "ViTPatchEncoder": ".multimodal_classifier",
    "MultimodalFusionClassifier": ".multimodal_classifier",
    "ModelWithTemperature": ".temperature_scaling",
    "compute_ece": ".temperature_scaling",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
