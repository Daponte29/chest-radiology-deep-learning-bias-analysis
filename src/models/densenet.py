"""DenseNet models adapted for multi-label chest X-ray classification.

Design notes
------------
- The classifier head outputs raw logits (no Sigmoid).
- Use nn.BCEWithLogitsLoss() as your loss function during training.
  It fuses sigmoid + BCE in one numerically stable operation.
- At inference, wrap logits with torch.sigmoid() to get probabilities,
  then threshold (e.g. >= 0.5) to get binary predictions per label.

CheXpert label order (14 classes)
----------------------------------
0  No Finding          7  Pleural Effusion
1  Enlarged CM         8  Pleural Other
2  Cardiomegaly        9  Fracture
3  Lung Opacity        10 Support Devices
4  Lung Lesion         11 Edema
5  Consolidation       12 Atelectasis
6  Pneumonia           13 Pneumothorax
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models


NUM_CHEXPERT_CLASSES = 14

_DENSENET_VARIANTS: dict[str, callable] = {
    "densenet121": tv_models.densenet121,
    "densenet169": tv_models.densenet169,
    "densenet201": tv_models.densenet201,
}


class DenseNetClassifier(nn.Module):
    """DenseNet backbone with a linear classifier head for multi-label classification.

    Outputs **raw logits** — no sigmoid applied.  Pair with
    ``nn.BCEWithLogitsLoss`` during training and apply ``torch.sigmoid``
    at inference time.

    Args:
        num_classes (int): Number of output labels. Defaults to 14 for CheXpert.
        pretrained (bool): Load ImageNet pre-trained weights.
        variant (str): One of 'densenet121', 'densenet169', 'densenet201'.
        dropout_p (float): Dropout probability before the linear head.
                           0.0 disables dropout entirely.
    """

    def __init__(
        self,
        num_classes: int = NUM_CHEXPERT_CLASSES,
        pretrained: bool = True,
        variant: str = "densenet121",
        dropout_p: float = 0.0,
    ):
        super().__init__()
        if variant not in _DENSENET_VARIANTS:
            raise ValueError(
                f"Unknown variant '{variant}'. Choose from {list(_DENSENET_VARIANTS)}"
            )

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = _DENSENET_VARIANTS[variant](weights=weights)

        in_features = backbone.classifier.in_features

        # Raw logits — BCEWithLogitsLoss handles the sigmoid internally.
        head_layers: list[nn.Module] = []
        if dropout_p > 0.0:
            head_layers.append(nn.Dropout(p=dropout_p))
        head_layers.append(nn.Linear(in_features, num_classes))
        backbone.classifier = nn.Sequential(*head_layers)

        self.backbone = backbone
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (batch_size, num_classes)."""
        return self.backbone(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities — use at inference, not during training."""
        return torch.sigmoid(self.forward(x))

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions (0/1) per label using a probability threshold."""
        return (self.predict_proba(x) >= threshold).long()