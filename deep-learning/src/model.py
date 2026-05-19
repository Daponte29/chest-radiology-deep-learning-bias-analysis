from __future__ import annotations

import torch
import torch.nn as nn
from src.config import ModelConfig


class MyModel(nn.Module):
    """TODO: define your architecture."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # TODO: define layers
        # self.backbone = ...
        # self.head = ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement forward pass
        raise NotImplementedError


def build_model(cfg: ModelConfig) -> MyModel:
    return MyModel(cfg)
