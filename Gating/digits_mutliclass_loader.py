"""
svhn_mnist_loader.py
====================
Multi-task DataLoader: SVHN (rotation-sensitive) + MNIST (rotation-invariant).

Returns batches of (image, label, task) where:
    task = "svhn"  → label = digit 0-9, images are upright street numbers
    task = "mnist" → label = digit 0-9, images are randomly rotated handwritten digits

Gate hypothesis:
    SVHN  samples → g_rot should → 0  (orientation matters for street numbers)
    MNIST samples → g_rot should → 1  (invariant to rotation for handwritten digits)

Both datasets have ~60-70k images and 10 classes — balanced and learnable.
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import datasets, transforms


RANDOM_SEED = 42
IMG_SIZE    = 32   # SVHN is 32×32 natively; resize MNIST to match

SVHN_MEAN  = (0.4377, 0.4438, 0.4728)
SVHN_STD   = (0.1980, 0.2010, 0.1970)

# Use SVHN stats for both since MNIST will be converted to 3-channel
NORM = transforms.Normalize(SVHN_MEAN, SVHN_STD)


# ── Datasets ──────────────────────────────────────────────────────────────────

class SVHNDataset(Dataset):
    """
    SVHN — rotation sensitive.
    Clean images, no rotation augmentation.
    Tag: task = 'svhn'
    """
    def __init__(self, split: str, reflect=False):
        ops = [transforms.ToTensor(), NORM]
        if reflect:
            ops.insert(0, transforms.RandomHorizontalFlip(p=1.0))
        self._ds = datasets.SVHN(
            './data', split=split, download=True,
            transform=transforms.Compose(ops)
        )

    def __len__(self):  return len(self._ds)

    def __getitem__(self, idx):
        x, y = self._ds[idx]
        return x, y, 'svhn'


class MNISTDataset(Dataset):
    """
    MNIST — rotation invariant.
    Random 0/90/180/270° rotation applied during TRAINING to teach invariance.
    At test time: optionally rotate to probe robustness.
    Tag: task = 'mnist'
    """
    def __init__(self, train: bool, rotate_train=True, rotate_test=False,
                 reflect=False):
        ops = [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.Grayscale(num_output_channels=3),
        ]

        if train and rotate_train:
            # Random 90° increments during training — teaches invariance
            ops.append(transforms.Lambda(
                lambda x: transforms.functional.rotate(x, random.choice([0, 90, 180, 270]))
            ))
        elif not train and rotate_test:
            # Fixed 90° rotation at test time for probing
            ops.append(transforms.Lambda(
                lambda x: transforms.functional.rotate(x, 90)
            ))

        if reflect:
            ops.append(transforms.RandomHorizontalFlip(p=1.0))

        ops += [transforms.ToTensor(), NORM]

        self._ds = datasets.MNIST(
            './data', train=train, download=True,
            transform=transforms.Compose(ops)
        )

    def __len__(self):  return len(self._ds)

    def __getitem__(self, idx):
        x, y = self._ds[idx]
        return x, y, 'mnist'


# ── Factory ───────────────────────────────────────────────────────────────────

def get_svhn_mnist_loaders(args):
    """
    Returns (train_loader, test_loader, n_classes=10).

    Each batch: (images [B,3,32,32], labels [B], tasks [B])
        tasks[i] = 'svhn'  → rotation-sensitive street digit
        tasks[i] = 'mnist' → rotation-invariant handwritten digit
    """
    batch_size     = args.batch_size
    rotate_images  = getattr(args, 'rotate_images',  False)
    reflect_images = getattr(args, 'reflect_images', False)

    # ── Build datasets ────────────────────────────────────────────────────────
    svhn_train  = SVHNDataset('train', reflect=False)
    svhn_test   = SVHNDataset('test',  reflect=reflect_images)

    # MNIST: rotated during training to teach invariance
    # At test time: optionally rotate to probe robustness
    mnist_train = MNISTDataset(train=True,  rotate_train=True,
                               rotate_test=False, reflect=False)
    mnist_test  = MNISTDataset(train=False, rotate_train=False,
                               rotate_test=rotate_images, reflect=reflect_images)

    # ── Balance: subsample larger dataset to match smaller ────────────────────
    def subsample(ds, n):
        idx = random.Random(RANDOM_SEED).sample(range(len(ds)), min(n, len(ds)))
        return Subset(ds, idx)

    n_train = min(len(svhn_train), len(mnist_train)) // 2
    n_test  = min(len(svhn_test),  len(mnist_test)) // 2

    svhn_train  = subsample(svhn_train,  n_train)
    mnist_train = subsample(mnist_train, n_train)
    svhn_test   = subsample(svhn_test,   n_test)
    mnist_test  = subsample(mnist_test,  n_test)

    train_dataset = ConcatDataset([svhn_train, mnist_train])
    test_dataset  = ConcatDataset([svhn_test,  mnist_test])

    print(f"[svhn+mnist] train {len(train_dataset)} "
          f"({len(svhn_train)} SVHN + {len(mnist_train)} MNIST) | "
          f"test {len(test_dataset)} "
          f"({len(svhn_test)} SVHN + {len(mnist_test)} MNIST)")
    print(f"  MNIST train augmentation: random 90° rotations (teaches invariance)")
    print(f"  SVHN  train augmentation: none (preserves orientation sensitivity)")

    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)

    return train_loader, test_loader, 10   # both tasks have 10 classes