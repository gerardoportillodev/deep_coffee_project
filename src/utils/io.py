"""I/O utility helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(payload: dict[str, Any], path: Path) -> None:
    """Save dictionary as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file into dictionary."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)
