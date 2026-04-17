"""Inference utilities for deployment and demo applications."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from PIL import Image

from configs.config import Config
from src.data.transforms import get_eval_transforms
from src.models.mlp import CoffeeMLP


def load_model(config: Config, checkpoint_path: Path | None = None) -> CoffeeMLP:
    """Load Stage 1 baseline model with trained checkpoint."""
    model = CoffeeMLP(
        image_size=config.data.image_size,
        num_classes=len(config.data.class_names),
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
    )
    model_path = checkpoint_path or (config.paths.models_dir / "stage1_mlp_best.pt")
    state_dict = torch.load(model_path, map_location=config.device)
    model.load_state_dict(state_dict)
    model.to(config.device)
    model.eval()
    return model


def predict_image(
    image_path: Path,
    model: CoffeeMLP,
    class_names: Sequence[str],
    image_size: int,
    device: str,
) -> dict[str, float | str]:
    """Predict class probabilities for a single image."""
    transform = get_eval_transforms(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1).squeeze(0)

    top_index = int(torch.argmax(probabilities).item())
    return {
        "predicted_class": class_names[top_index],
        "confidence": float(probabilities[top_index].item()),
    }
