"""Loss factory for training modules."""

from __future__ import annotations

from torch import nn


def get_classification_loss() -> nn.Module:
    """Return default loss for multiclass classification."""
    return nn.CrossEntropyLoss()
