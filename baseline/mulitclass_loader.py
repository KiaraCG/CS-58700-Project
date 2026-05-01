"""
multiclass_loader.py
====================
Multi-class DataLoader for joint bird + digit classification.

Returns batches of (image, label, task) where:
    task = "bird"  → label = species index (0 to n_bird_classes-1)
    task = "digit" → label = digit (0-9)

Usage:
    from multiclass_loader import get_multiclass_loaders
    train_loader, test_loader, n_bird_classes = get_multiclass_loaders(args)
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split
from torchvision import datasets, transforms


BIRD_DIR    = "/scratch/scholar/shams3/inatbirds/birds_train_small/bird_train"
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
    """Birds — keeps original species label, tags task='bird'."""
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y, "bird"


class DigitDataset(Dataset):
    """MNIST digits — keeps original digit label (0-9), tags task='digit'."""
    def __init__(self, train: bool, transform):
        self._ds = datasets.MNIST(
            './data', train=train, download=True, transform=transform
        )

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, idx):
        x, y = self._ds[idx]
        return x, y, "digit"


# ── Factory ───────────────────────────────────────────────────────────────────

def get_multiclass_loaders(args):
    """
    Returns (train_loader, test_loader, n_bird_classes).

    Each batch: (images [B,3,64,64], labels [B], tasks [B])
        tasks[i] = 'bird'  → labels[i] is species index
        tasks[i] = 'digit' → labels[i] is digit 0-9
    """
    batch_size     = args.batch_size
    rotate_images  = getattr(args, 'rotate_images',  False)
    reflect_images = getattr(args, 'reflect_images', False)

    # Train: always clean — augment only at test time for equivariance probing
    bird_train_tf  = _bird_transform()
    mnist_train_tf = _mnist_transform()

    # Test: optionally rotated/reflected for equivariance probing
    bird_test_tf   = _bird_transform(rotate=rotate_images,  reflect=reflect_images)
    mnist_test_tf  = _mnist_transform(rotate=rotate_images, reflect=reflect_images)


# ── Birds: 80/20 split (Filtered to 10 classes) ────────────────────────────
    full_birds     = datasets.ImageFolder(BIRD_DIR)
    n_bird_classes = 10 # Force it to exactly 10 classes
    
    # 1. Filter out everything except classes 0 through 9
    valid_indices  = [i for i, label in enumerate(full_birds.targets) if label < n_bird_classes]
    filtered_birds = Subset(full_birds, valid_indices)

    # 2. Calculate sizes based on the FILTERED dataset, not the full one
    train_size     = int(0.8 * len(filtered_birds))
    test_size      = len(filtered_birds) - train_size
    
    # 3. Split the filtered dataset
    generator      = torch.Generator().manual_seed(RANDOM_SEED)
    bird_train_sub, bird_test_sub = random_split(
        filtered_birds, [train_size, test_size], generator=generator
    )

    # 4. Wrap with your custom dataset class/transforms
    bird_train = BirdDataset(bird_train_sub, bird_train_tf)
    bird_test  = BirdDataset(bird_test_sub,  bird_test_tf)
    print(f"[birds] {n_bird_classes} classes | "
          f"train {len(bird_train)} | test {len(bird_test)}")


    # ── MNIST: stratified subsample to match bird counts ─────────────────────
    def subsample(ds, n):
        rng = random.Random(RANDOM_SEED)
        # Group indices by class label
        class_indices = {}
        for i in range(len(ds)):
            _, label, _ = ds[i]
            class_indices.setdefault(label, []).append(i)
        n_classes = len(class_indices)
        per_class = n // n_classes
        remainder = n %  n_classes
        chosen = []
        for cls, idxs in sorted(class_indices.items()):
            # Give one extra sample to the first `remainder` classes
            k = per_class + (1 if cls < remainder else 0)
            chosen.extend(rng.sample(idxs, min(k, len(idxs))))
        return Subset(ds, chosen)

    mnist_train = subsample(
        DigitDataset(train=True,  transform=mnist_train_tf), len(bird_train)
    )
    mnist_test  = subsample(
        DigitDataset(train=False, transform=mnist_test_tf),  len(bird_test)
    )

    # ── Combine ───────────────────────────────────────────────────────────────
    train_dataset = ConcatDataset([bird_train, mnist_train])
    test_dataset  = ConcatDataset([bird_test,  mnist_test])

    print(f"[multiclass] train {len(train_dataset)} "
          f"({len(bird_train)} birds / {len(mnist_train)} digits) | "
          f"test {len(test_dataset)} "
          f"({len(bird_test)} birds / {len(mnist_test)} digits) | "
          f"{n_bird_classes} bird classes")

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    return train_loader, test_loader, n_bird_classes