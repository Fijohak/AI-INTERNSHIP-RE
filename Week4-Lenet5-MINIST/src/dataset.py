"""
MNIST dataset loading and preprocessing

Uses torchvision built-in MNIST with automatic download and normalization.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# Pre-computed MNIST mean and std
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


def get_transforms():
    """Get train/test transforms"""
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ])
    return train_transform, test_transform


def get_datasets(data_root: str = "./data"):
    """Download and return train/test datasets"""
    train_transform, test_transform = get_transforms()

    train_dataset = datasets.MNIST(
        root=data_root, train=True, download=True, transform=train_transform,
    )
    test_dataset = datasets.MNIST(
        root=data_root, train=False, download=True, transform=test_transform,
    )
    return train_dataset, test_dataset


def get_dataloaders(
    batch_size: int = 64,
    data_root: str = "./data",
    val_split: float = 0.0,
    num_workers: int = 0,
):
    """
    Return DataLoaders (train / val / test)

    Args:
        batch_size: Batch size
        data_root: Dataset storage path
        val_split: Validation split ratio (0.0 = no split)
        num_workers: Data loading workers

    Returns:
        dict: {"train_loader", "val_loader", "test_loader"}
    """
    train_dataset, test_dataset = get_datasets(data_root)

    if val_split > 0.0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers,
        )
    else:
        val_loader = None

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    }
