"""Path convenience functions."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return repository root path."""
    return Path(__file__).resolve().parents[2]
