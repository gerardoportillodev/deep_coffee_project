"""Configuration tests."""

from __future__ import annotations

from configs.config import Config


def test_config_defaults() -> None:
    cfg = Config()
    assert cfg.stage_name == "stage1_mlp"
    assert len(cfg.data.class_names) == 4
    assert cfg.training.epochs > 0
    assert cfg.data.image_size > 0
