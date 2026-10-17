from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_transform(augment: bool = False) -> transforms.Compose:
    operations = [
        transforms.Resize((28, 28)),
    ]
    if augment:
        operations.extend(
            [
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    return transforms.Compose(operations)


def load_dataset(
    train: bool,
    data_dir: str | Path | None = None,
    augment: bool = False,
    download: bool = True,
) -> datasets.MNIST:
    root = Path(data_dir) if data_dir else PROJECT_ROOT / "data"
    return datasets.MNIST(
        root=str(root),
        train=train,
        download=download,
        transform=build_transform(augment=augment),
    )


def split_indices(
    dataset_size: int, validation_fraction: float = 0.1, seed: int = 42
) -> tuple[list[int], list[int]]:
    if dataset_size < 2:
        raise ValueError("At least two training examples are required.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    validation_size = max(1, int(dataset_size * validation_fraction))
    if validation_size >= dataset_size:
        raise ValueError("The validation split must leave at least one training example.")
    indices = torch.randperm(
        dataset_size, generator=torch.Generator().manual_seed(seed)
    ).tolist()
    return indices[validation_size:], indices[:validation_size]


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
    validation_fraction: float = 0.1,
) -> tuple[DataLoader, DataLoader]:
    augmented_train = load_dataset(train=True, data_dir=data_dir, augment=True)
    clean_train = load_dataset(train=True, data_dir=data_dir)
    train_indices, validation_indices = split_indices(
        len(clean_train), validation_fraction=validation_fraction, seed=seed
    )
    train_dataset = Subset(augmented_train, train_indices)
    validation_dataset = Subset(clean_train, validation_indices)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, validation_loader
