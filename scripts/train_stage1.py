"""Train Stage 1 baseline MLP model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from configs.config import get_config
from src.data.dataset import get_dataloaders
from src.evaluation.metrics import save_training_curves
from src.models.mlp import CoffeeMLP
from src.training.trainer import Trainer
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
    print(f"Trainable parameters: {model.count_parameters():,}")

    optimizer = Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=cfg.device,
        checkpoint_path=cfg.paths.models_dir / "stage1_mlp_best.pt",
        history_path=cfg.paths.results_dir / "stage1_training_history.json",
        early_stopping_patience=cfg.training.early_stopping_patience,
        scheduler_factor=cfg.training.scheduler_factor,
        scheduler_patience=cfg.training.scheduler_patience,
    )

    output = trainer.fit(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        epochs=cfg.training.epochs,
    )

    save_training_curves(
        history=output.history,
        output_path=cfg.paths.figures_dir / "stage1_training_curves.png",
    )

    print(f"Best checkpoint: {output.best_checkpoint_path}")
    print("Stage 1 training complete.")


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
