"""Exploratory data analysis utilities for coffee bean dataset."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.data.dataset import CoffeeBeansDataset, SampleRecord, collect_samples


def class_distribution(split_dir: Path, class_names: Sequence[str]) -> pd.DataFrame:
    """Return class count dataframe for a split directory."""
    samples = collect_samples(split_dir, class_names)
    counts = Counter(sample.label for sample in samples)
    rows = [{"class_name": class_name, "count": counts[idx]} for idx, class_name in enumerate(class_names)]
    return pd.DataFrame(rows)


def plot_class_distribution(distribution: pd.DataFrame, output_path: Path) -> None:
    """Save class distribution bar chart."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.bar(distribution["class_name"], distribution["count"])
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_sample_images(
    samples: Sequence[SampleRecord],
    class_names: Sequence[str],
    output_path: Path,
    max_images: int = 8,
) -> None:
    """Save image grid preview from sample records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = list(samples[:max_images])
    columns = min(4, len(selected))
    rows = int(np.ceil(len(selected) / columns)) if columns else 1

    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, axis in enumerate(axes):
        if idx >= len(selected):
            axis.axis("off")
            continue
        record = selected[idx]
        image = Image.open(record.image_path).convert("RGB")
        axis.imshow(image)
        axis.set_title(class_names[record.label])
        axis.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)


def compute_rgb_statistics(samples: Sequence[SampleRecord], max_images: int = 200) -> dict[str, float]:
    """Compute per-channel mean and std from a subset of images."""
    subset = list(samples[:max_images])
    channels = []
    for sample in subset:
        image = np.array(Image.open(sample.image_path).convert("RGB"), dtype=np.float32) / 255.0
        channels.append(image.reshape(-1, 3))

    if not channels:
        return {"mean_r": 0.0, "mean_g": 0.0, "mean_b": 0.0, "std_r": 0.0, "std_g": 0.0, "std_b": 0.0}

    stacked = np.concatenate(channels, axis=0)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    return {
        "mean_r": float(mean[0]),
        "mean_g": float(mean[1]),
        "mean_b": float(mean[2]),
        "std_r": float(std[0]),
        "std_g": float(std[1]),
        "std_b": float(std[2]),
    }


def plot_color_histograms(samples: Sequence[SampleRecord], output_path: Path, max_images: int = 100) -> None:
    """Save RGB histogram visualization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subset = list(samples[:max_images])

    if not subset:
        return

    red_vals: list[np.ndarray] = []
    green_vals: list[np.ndarray] = []
    blue_vals: list[np.ndarray] = []

    for sample in subset:
        image = np.array(Image.open(sample.image_path).convert("RGB"))
        red_vals.append(image[:, :, 0].ravel())
        green_vals.append(image[:, :, 1].ravel())
        blue_vals.append(image[:, :, 2].ravel())

    plt.figure(figsize=(8, 5))
    plt.hist(np.concatenate(red_vals), bins=50, color="red", alpha=0.4, label="R")
    plt.hist(np.concatenate(green_vals), bins=50, color="green", alpha=0.4, label="G")
    plt.hist(np.concatenate(blue_vals), bins=50, color="blue", alpha=0.4, label="B")
    plt.title("RGB Histograms")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def run_stage1_eda(train_dir: Path, class_names: Sequence[str], figures_dir: Path) -> dict[str, float]:
    """Generate default EDA artifacts for Stage 1."""
    samples = collect_samples(train_dir, class_names)
    distribution = class_distribution(train_dir, class_names)
    plot_class_distribution(distribution, figures_dir / "stage1_class_distribution.png")
    plot_sample_images(samples, class_names, figures_dir / "stage1_sample_images.png")
    plot_color_histograms(samples, figures_dir / "stage1_rgb_histograms.png")
    return compute_rgb_statistics(samples)
