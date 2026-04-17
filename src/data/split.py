"""Data splitting helpers."""

from __future__ import annotations

from collections.abc import Sequence

from src.data.dataset import SampleRecord, split_train_val_samples

__all__ = ["SampleRecord", "split_train_val_samples", "split_samples"]


def split_samples(
    samples: Sequence[SampleRecord],
    val_size: float,
    random_state: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Thin wrapper for train/validation splitting utility."""
    return split_train_val_samples(samples, val_size=val_size, random_state=random_state)
