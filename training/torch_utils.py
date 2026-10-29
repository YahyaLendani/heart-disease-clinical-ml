"""Small, fast PyTorch helpers for training MLPs on a few hundred rows.

With datasets this small, ``torch.utils.data.DataLoader`` spends more time
in Python bookkeeping (sampler, collate, per-item indexing) than the model
spends computing. Keeping the whole split in one tensor and slicing it with a
shuffled index is several times faster and gives the same mini-batches
semantics (shuffle, batch size, optional ``drop_last``).
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Iterator

import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> torch.Generator:
    """Seed Python, NumPy and PyTorch; return a dedicated generator for shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def to_tensor(values, device: torch.device | str = "cpu", dtype=torch.float32) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.to(device=device, dtype=dtype)
    # torch.tensor copies, so read-only pandas buffers are safe to convert.
    return torch.tensor(np.asarray(values), dtype=dtype, device=device)


def iterate_minibatches(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield ``(xb, yb)`` slices of tensors that already live on the target device.

    A trailing batch of size 1 is always skipped when training with
    BatchNorm-style layers would otherwise see a single sample.
    """
    n = X.shape[0]
    if shuffle:
        order = torch.randperm(n, generator=generator).to(X.device)
    else:
        order = torch.arange(n, device=X.device)
    stop = n - (n % batch_size) if drop_last and n > batch_size else n
    for start in range(0, stop, batch_size):
        idx = order[start : start + batch_size]
        if idx.numel() == 1 and n > 1:
            continue
        yield X[idx], y[idx]


@torch.inference_mode()
def predict_logits(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Full-batch inference in eval mode (the whole split fits in memory)."""
    model.eval()
    return model(X)


def clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Cheap snapshot of model weights (detached copies, same device)."""
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


@contextlib.contextmanager
def single_thread():
    """Run a block with one PyTorch intra-op thread, then restore the previous setting.

    Folds always train this way, in the main process or in a worker, so the
    floating-point summation order (and therefore the results) does not
    depend on ``--n-jobs``. For models this small one thread is as fast.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)
