"""ImageFolder-style dataset and transforms for ResNet training/inference."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_dataloaders(
    processed_dir: str, image_size: int, batch_size: int, num_workers: int
) -> dict[str, DataLoader]:
    """Expects processed_dir/{train,val,test}/<class_name>/*.jpg layout."""
    loaders = {}
    for split in ("train", "val", "test"):
        split_dir = Path(processed_dir) / split
        if not split_dir.exists():
            continue
        dataset = datasets.ImageFolder(
            root=str(split_dir),
            transform=build_transforms(image_size, train=(split == "train")),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
