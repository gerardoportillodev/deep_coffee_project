"""Plot wrappers for evaluation module."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from src.evaluation.metrics import save_confusion_matrix_figure, save_training_curves

__all__ = ["save_confusion_matrix_figure", "save_training_curves", "save_default_eval_plots"]


def save_default_eval_plots(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    history: dict[str, list[float]],
    figures_dir: Path,
) -> None:
    """Save default confusion matrix and training curves."""
    save_confusion_matrix_figure(y_true, y_pred, class_names, figures_dir / "confusion_matrix.png")
    save_training_curves(history, figures_dir / "training_curves.png")
