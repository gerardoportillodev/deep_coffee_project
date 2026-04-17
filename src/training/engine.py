"""Training/evaluation step engine."""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


def run_inference(model: nn.Module, dataloader: Iterable, device: str) -> tuple[list[int], list[int]]:
    """Run model inference and return true/predicted labels."""
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            predictions = torch.argmax(logits, dim=1).cpu().tolist()

            y_pred.extend(predictions)
            y_true.extend(labels.tolist())

    return y_true, y_pred
