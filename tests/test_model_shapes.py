"""Shape tests for core model outputs."""

from __future__ import annotations

import torch

from src.models.mlp import CoffeeMLP


def test_mlp_output_shape() -> None:
    model = CoffeeMLP(image_size=128, num_classes=4)
    batch = torch.randn(8, 3, 128, 128)
    logits = model(batch)
    assert logits.shape == (8, 4)


def test_count_parameters_positive() -> None:
    model = CoffeeMLP(image_size=128, num_classes=4)
    assert model.count_parameters() > 0
