"""Dataset loaders for CIFAR-10 / CIFAR-100.

The evaluation protocol follows common practise for knowledge distillation on
CIFAR: random crop with 4-pixel padding plus horizontal flip for training, and
channel-wise normalization using the dataset statistics.
"""
import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR_STATS = {
    'cifar100': ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    'cifar10': ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}

DATASETS = {'cifar100': datasets.CIFAR100, 'cifar10': datasets.CIFAR10}


def build_transforms(dataset):
    mean, std = CIFAR_STATS[dataset]
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf


def build_dataloaders(dataset='cifar100', data_root='./data', batch_size=128,
                      num_workers=4):
    """Return ``(train_loader, test_loader)`` for the requested CIFAR dataset."""
    if dataset not in DATASETS:
        raise KeyError(f'unknown dataset {dataset!r}; available: {sorted(DATASETS)}')
    train_tf, test_tf = build_transforms(dataset)
    cls = DATASETS[dataset]
    os.makedirs(data_root, exist_ok=True)

    train_set = cls(root=data_root, train=True, download=True, transform=train_tf)
    test_set = cls(root=data_root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
