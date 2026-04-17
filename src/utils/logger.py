"""Logging configuration helper."""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml


def configure_logging(config_path: Path) -> None:
    """Configure Python logging from YAML config file."""
    if not config_path.exists():
        logging.basicConfig(level=logging.INFO)
        return

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    logging.config.dictConfig(config)
