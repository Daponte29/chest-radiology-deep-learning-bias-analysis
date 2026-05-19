from __future__ import annotations

import torch


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return (preds.argmax(dim=1) == targets).float().mean().item()


# TODO: add task-specific metrics below, e.g.:
# def f1_score(...): ...
# def iou(...): ...
# def rmse(...): ...


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Aggregate all metrics into one dict — extend as needed."""
    return {
        "accuracy": accuracy(preds, targets),
        # TODO: add more
    }
