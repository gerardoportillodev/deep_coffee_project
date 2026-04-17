"""Dataset behavior tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.data.dataset import collect_samples


def test_collect_samples_with_mock_image(tmp_path: Path) -> None:
    root = tmp_path / "train"
    class_dir = root / "dark"
    class_dir.mkdir(parents=True)

    image_path = class_dir / "sample.png"
    Image.new("RGB", (16, 16), color=(120, 90, 60)).save(image_path)

    samples = collect_samples(root, ["dark", "green", "light", "medium"])
    assert len(samples) == 1
    assert samples[0].label == 0
