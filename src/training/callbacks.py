"""Callback primitives for future stage expansion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EarlyStoppingState:
    """Simple state container for early stopping."""

    best_score: float = float("inf")
    patience_counter: int = 0
