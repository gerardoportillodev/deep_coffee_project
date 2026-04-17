"""Baseline MLP model for Stage 1 classification."""

from __future__ import annotations

import torch.nn as nn


class CoffeeMLP(nn.Module):
    """Multilayer perceptron baseline for image classification."""

    def __init__(
        self,
        image_size: int,
        num_classes: int,
        hidden_dims: tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = 3 * image_size * image_size

        layers: list[nn.Module] = [nn.Flatten()]
        previous_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def count_parameters(self) -> int:
        """Return number of trainable parameters."""
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
