"""Error analysis helpers."""

from __future__ import annotations

from typing import Sequence


def collect_misclassified_indices(y_true: Sequence[int], y_pred: Sequence[int]) -> list[int]:
    """Return indices where predictions differ from targets."""
    return [idx for idx, (truth, pred) in enumerate(zip(y_true, y_pred, strict=True)) if truth != pred]
