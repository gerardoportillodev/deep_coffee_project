"""Evaluate Stage 1 model checkpoint on test split."""

from __future__ import annotations

from pathlib import Path

import torch

from configs.config import get_config
from src.data.dataset import get_dataloaders
from src.evaluation.metrics import (
    compute_classification_metrics,
    save_confusion_matrix_figure,
    save_metrics_json,
)
from src.models.mlp import CoffeeMLP
from src.training.engine import run_inference
from src.utils.seed import set_seed


def main() -> None:
    cfg = get_config()
    cfg.ensure_directories()
    set_seed(cfg.seed)

    dataloaders = get_dataloaders(cfg)

    model = CoffeeMLP(
        image_size=cfg.data.image_size,
        num_classes=len(cfg.data.class_names),
        hidden_dims=cfg.model.hidden_dims,
        dropout=cfg.model.dropout,
    )

    checkpoint_path = cfg.paths.models_dir / "stage1_mlp_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. Run scripts/train_stage1.py first."
        )

    state_dict = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state_dict)
    model.to(cfg.device)

    y_true, y_pred = run_inference(model, dataloaders["test"], cfg.device)
    metrics = compute_classification_metrics(y_true, y_pred, cfg.data.class_names)

    results_path = cfg.paths.results_dir / "stage1_test_metrics.json"
    confusion_matrix_path = cfg.paths.figures_dir / "stage1_confusion_matrix.png"

    save_metrics_json(metrics, results_path)
    save_confusion_matrix_figure(y_true, y_pred, cfg.data.class_names, confusion_matrix_path)

    print("Stage 1 evaluation complete.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")
    print(f"Saved metrics: {results_path}")
    print(f"Saved confusion matrix: {confusion_matrix_path}")


if __name__ == "__main__":
    main()
