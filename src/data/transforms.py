"""Image transformation pipelines for train and evaluation."""

from __future__ import annotations

from torchvision import transforms


def get_train_transforms(image_size: int) -> transforms.Compose:
    """Return augmentation pipeline for training data."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transforms(image_size: int) -> transforms.Compose:
    """Return deterministic preprocessing for validation/test data."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
