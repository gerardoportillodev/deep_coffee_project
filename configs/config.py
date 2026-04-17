"""Central project configuration using dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass(slots=True)
class PathsConfig:
    """Filesystem paths for data, artifacts, and reports."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    data_root: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    train_dir: Path = field(init=False)
    test_dir: Path = field(init=False)
    interim_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_root = self.project_root / "data"
        self.raw_data_dir = self.data_root / "raw" / "coffee_beans"
        self.train_dir = self.raw_data_dir / "train"
        self.test_dir = self.raw_data_dir / "test"
        self.interim_dir = self.data_root / "interim"
        self.processed_dir = self.data_root / "processed"
        self.figures_dir = self.project_root / "figures"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.reports_dir = self.project_root / "reports"


@dataclass(slots=True)
class DataConfig:
    """Data-related configuration values."""

    class_names: tuple[str, ...] = ("dark", "green", "light", "medium")
    image_size: int = 128
    val_size: float = 0.2
    num_workers: int = 0
    pin_memory: bool = True


@dataclass(slots=True)
class ModelConfig:
    """Baseline MLP architecture settings."""

    hidden_dims: tuple[int, ...] = (512, 256)
    dropout: float = 0.3


@dataclass(slots=True)
class TrainingConfig:
    """Hyperparameters and optimization settings."""

    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    early_stopping_patience: int = 7


@dataclass(slots=True)
class Config:
    """Master configuration object for the deep coffee project."""

    stage_name: str = "stage1_mlp"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    def ensure_directories(self) -> None:
        """Create writable project artifact directories if missing."""
        dirs = [
            self.paths.interim_dir,
            self.paths.processed_dir,
            self.paths.figures_dir,
            self.paths.models_dir,
            self.paths.results_dir,
            self.paths.reports_dir,
        ]
        for directory in dirs:
            directory.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration into a JSON-friendly dict."""
        config_dict = asdict(self)

        def _convert(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_convert(item) for item in value]
            return value

        return _convert(config_dict)


def get_config() -> Config:
    """Return default project configuration."""
    return Config()
