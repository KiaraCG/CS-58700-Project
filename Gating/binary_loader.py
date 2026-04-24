"""
binary_loader.py
================
Simple binary DataLoader: bird=1, digit=0.
No task strings, no multi-head routing — just (image, label) pairs.

Usage:
    from binary_loader import get_binary_loaders
    train_loader, test_loader = get_binary_loaders(args)
"""

import os
import random

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
from torchvision import datasets, transforms


BIRD_DIR    = "/scratch/scholar/shams3/inatbirds/birds_100classes"
RANDOM_SEED = 42
IMG_SIZE    = 64


# ── Transforms ────────────────────────────────────────────────────────────────

def _bird_transform(rotate=False, reflect=False):
    ops = []
    if rotate:  ops.append(transforms.Lambda(lambda x: transforms.functional.rotate(x, 90)))
    if reflect: ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(ops)


def _mnist_transform(rotate=False, reflect=False):
    ops = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.Grayscale(num_output_channels=3),
    ]
    if rotate:  ops.append(transforms.Lambda(lambda x: transforms.functional.rotate(x, 90)))
    if reflect: ops.append(transforms.RandomHorizontalFlip(p=1.0))
    ops += [
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(ops)


# ── Datasets ──────────────────────────────────────────────────────────────────

class BirdDataset(Dataset):
    """Birds from ImageFolder — label = 1."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, _ = self.subset[idx]          # ignore species label
        return self.transform(x), 1      # binary label: bird = 1


class DigitDataset(Dataset):
    """MNIST digits — label = 0."""
    def __init__(self, train: bool, transform):
        self._ds = datasets.MNIST(
            './data', train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        x, _ = self._ds[idx]             # ignore digit label
        return x, 0                      # binary label: digit = 0


# ── Factory ───────────────────────────────────────────────────────────────────

def get_loaders(args):
    """
    Returns (train_loader, test_loader).
    Each batch: (images [B,3,64,64], labels [B])  — labels are 0 or 1.
    """
    batch_size     = args.batch_size
    rotate_images  = getattr(args, 'rotate_images',  False)
    reflect_images = getattr(args, 'reflect_images', False)

    # Train transforms: always clean (augment only at test time for probing)
    bird_train_tf  = _bird_transform()
    mnist_train_tf = _mnist_transform()

    # Test transforms: optionally augmented for equivariance probing
    bird_test_tf   = _bird_transform(rotate=rotate_images,  reflect=reflect_images)
    mnist_test_tf  = _mnist_transform(rotate=rotate_images, reflect=reflect_images)

    # ── Birds: 80/20 split ────────────────────────────────────────────────────
    full_birds = datasets.ImageFolder(BIRD_DIR)
    train_size = int(0.8 * len(full_birds))
    test_size  = len(full_birds) - train_size
    generator  = torch.Generator().manual_seed(RANDOM_SEED)
    bird_train_sub, bird_test_sub = random_split(
        full_birds, [train_size, test_size], generator=generator
    )

    bird_train = BirdDataset(bird_train_sub, bird_train_tf)
    bird_test  = BirdDataset(bird_test_sub,  bird_test_tf)

    # ── MNIST: subsample to match bird counts ─────────────────────────────────
    def subsample(ds, n):
        idx = random.Random(RANDOM_SEED).sample(range(len(ds)), min(n, len(ds)))
        return Subset(ds, idx)

    mnist_train = subsample(DigitDataset(train=True,  transform=mnist_train_tf), len(bird_train))
    mnist_test  = subsample(DigitDataset(train=False, transform=mnist_test_tf),  len(bird_test))

    # ── Combine ───────────────────────────────────────────────────────────────
    train_dataset = ConcatDataset([bird_train, mnist_train])
    test_dataset  = ConcatDataset([bird_test,  mnist_test])

    print(f"[binary] train {len(train_dataset)} ({len(bird_train)} birds + {len(mnist_train)} digits) | "
          f"test {len(test_dataset)} ({len(bird_test)} birds + {len(mnist_test)} digits)")

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    return train_loader, test_loader