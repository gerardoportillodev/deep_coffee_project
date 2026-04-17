"""Dataset and DataLoader utilities for coffee bean classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from configs.config import Config
from src.data.transforms import get_eval_transforms, get_train_transforms

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(slots=True)
class SampleRecord:
    """Represents one image sample and its label index."""

    image_path: Path
    label: int


class CoffeeBeansDataset(Dataset):
    """Custom dataset for coffee bean images."""

    def __init__(
        self,
        samples: Sequence[SampleRecord],
        transform: Callable | None = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, sample.label



def collect_samples(split_dir: Path, class_names: Sequence[str]) -> list[SampleRecord]:
    """Collect labeled image files for a given split directory."""
    records: list[SampleRecord] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            continue
        for file_path in class_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                records.append(SampleRecord(image_path=file_path, label=class_index))

    if not records:
        raise FileNotFoundError(
            f"No images found in {split_dir}. Expected class folders: {', '.join(class_names)}"
        )
    return records


def split_train_val_samples(
    train_samples: Sequence[SampleRecord],
    val_size: float,
    random_state: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Create stratified train/validation split from training samples."""
    labels = [sample.label for sample in train_samples]
    indices = np.arange(len(train_samples))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_size,
        random_state=random_state,
        shuffle=True,
        stratify=labels,
    )

    train_split = [train_samples[int(i)] for i in train_idx]
    val_split = [train_samples[int(i)] for i in val_idx]
    return train_split, val_split


def get_dataloaders(cfg: Config) -> dict[str, DataLoader]:
    """Build train/validation/test dataloaders using project config."""
    class_names = cfg.data.class_names
    train_samples = collect_samples(cfg.paths.train_dir, class_names)
    test_samples = collect_samples(cfg.paths.test_dir, class_names)

    train_split, val_split = split_train_val_samples(
        train_samples,
        val_size=cfg.data.val_size,
        random_state=cfg.seed,
    )

    train_dataset = CoffeeBeansDataset(
        samples=train_split,
        transform=get_train_transforms(cfg.data.image_size),
    )
    val_dataset = CoffeeBeansDataset(
        samples=val_split,
        transform=get_eval_transforms(cfg.data.image_size),
    )
    test_dataset = CoffeeBeansDataset(
        samples=test_samples,
        transform=get_eval_transforms(cfg.data.image_size),
    )

    common_loader_kwargs = {
        "batch_size": cfg.training.batch_size,
        "num_workers": cfg.data.num_workers,
        "pin_memory": cfg.data.pin_memory and cfg.device == "cuda",
    }

    dataloaders = {
        "train": DataLoader(train_dataset, shuffle=True, **common_loader_kwargs),
        "val": DataLoader(val_dataset, shuffle=False, **common_loader_kwargs),
        "test": DataLoader(test_dataset, shuffle=False, **common_loader_kwargs),
    }
    return dataloaders
