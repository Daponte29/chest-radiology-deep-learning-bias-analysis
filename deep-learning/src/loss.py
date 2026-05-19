from __future__ import annotations

import torch
import torch.nn as nn


def build_criterion(name: str = "cross_entropy") -> nn.Module:
    """Return a loss function by name."""
    registry = {
        "cross_entropy": nn.CrossEntropyLoss(),
        "bce": nn.BCEWithLogitsLoss(),
        "mse": nn.MSELoss(),
        "l1": nn.L1Loss(),
        # TODO: add custom losses below
    }
    if name not in registry:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(registry)}")
    return registry[name]


# TODO: add custom loss classes here, e.g.:
# class FocalLoss(nn.Module): ...
