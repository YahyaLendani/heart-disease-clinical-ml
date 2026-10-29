"""Training pipelines and runners (names are imported lazily, see ``evaluation``)."""

from __future__ import annotations

import importlib

_EXPORTS = {
    "DeepEnsemble": ".deep_ensembles",
    "SoftVotingEnsemble": ".ensemble_models",
    "StackingEnsemble": ".ensemble_models",
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
