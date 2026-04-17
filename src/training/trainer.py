"""Training orchestration for Stage 1 baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass(slots=True)
class TrainingOutput:
    """Container for training outputs."""

    history: dict[str, list[float]]
    best_checkpoint_path: Path


class Trainer:
    """Model trainer with early stopping, scheduler, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        device: str,
        checkpoint_path: Path,
        history_path: Path,
        early_stopping_patience: int = 7,
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 3,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.history_path = history_path
        self.early_stopping_patience = early_stopping_patience
        self.scheduler = ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
        )

    def _run_epoch(self, dataloader: DataLoader, training: bool) -> tuple[float, float]:
        mode = "train" if training else "val"
        self.model.train(training)

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for inputs, labels in tqdm(dataloader, desc=f"{mode} epoch", leave=False):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            if training:
                self.optimizer.zero_grad()

            with torch.set_grad_enabled(training):
                logits = self.model(inputs)
                loss = self.criterion(logits, labels)
                if training:
                    loss.backward()
                    self.optimizer.step()

            batch_size = labels.size(0)
            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_examples += batch_size
            total_loss += loss.item() * batch_size

        avg_loss = total_loss / max(total_examples, 1)
        avg_acc = total_correct / max(total_examples, 1)
        return avg_loss, avg_acc

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
    ) -> TrainingOutput:
        """Train model for a given number of epochs."""
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            val_loss, val_acc = self._run_epoch(val_loader, training=False)

            self.scheduler.step(val_loss)

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                print("Early stopping triggered.")
                break

        with self.history_path.open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)

        return TrainingOutput(history=history, best_checkpoint_path=self.checkpoint_path)
