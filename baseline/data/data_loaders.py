import random

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from birds_loader import get_bird_data_loaders


# ---------------------------------------------------------------------------
# ColorMNIST — IRM benchmark variant
# (Arjovsky et al., "Invariant Risk Minimization", 2019)
#
# Construction:
#   - Binary task: digit < 5  →  label 0,  digit >= 5  →  label 1
#     (but we keep the original 10-class labels for per-class reporting)
#   - A spurious binary color is assigned:  color = label XOR noise
#     • train env 1: noise flipped with prob 0.1  (color corr. 0.9 w/ label)
#     • train env 2: noise flipped with prob 0.2  (color corr. 0.8 w/ label)
#     • test  env  : noise flipped with prob 0.9  (color corr. 0.1 w/ label)
#   - Red channel  ← digit image  if color==1,  else 0
#   - Green channel← digit image  if color==0,  else 0
#   - Blue channel ← 0
#
# The standard IRM paper trains on env1+env2 and tests on the test env.
# Here we expose train / test loaders that match that split.
# ---------------------------------------------------------------------------

def _make_environment(images, labels, e, rng):
    """
    images : (N, 28, 28) uint8 tensor in [0, 255]
    labels : (N,)        long  tensor  (0..9)
    e      : float       — probability of flipping the color assignment
    rng    : torch.Generator

    Returns a list of dicts {'image': (3,28,28) float, 'label': int}
    """
    # Binary label: digit >= 5
    binary_labels = (labels >= 5).long()

    # Spurious color: XOR with Bernoulli noise
    noise = torch.bernoulli(torch.full(binary_labels.shape, e, dtype=torch.float), generator=rng)
    colors = (binary_labels ^ noise.long())  # 0 = green, 1 = red

    # Normalise images to [0,1]
    images = images.float() / 255.0  # (N, 28, 28)

    # Build RGB: stack two colour channels
    red = images * colors.float().unsqueeze(-1).unsqueeze(-1)
    green = images * (1 - colors).float().unsqueeze(-1).unsqueeze(-1)
    blue = torch.zeros_like(images)

    rgb = torch.stack([red, green, blue], dim=1)  # (N, 3, 28, 28)

    # Normalise each channel independently (IRM paper uses no normalisation,
    # but we apply the standard MNIST stats extended to 3 channels for
    # consistency with the rest of this codebase)
    mean = torch.tensor([0.1307, 0.1307, 0.0]).view(3, 1, 1)
    std = torch.tensor([0.3081, 0.3081, 1.0]).view(3, 1, 1)  # std=1 keeps blue=0
    rgb = (rgb - mean) / std

    return [rgb[i] for i in range(len(labels))], [labels[i].item() for i in range(len(labels))]


class ColorMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        # Store as tensors, not a list of dicts
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            image = self.transform(image)
        return image, label


def get_colormnist_loaders(batch_size, data_root='./data', test_transform=None):
    """
    Returns (train_loader, test_loader) for the IRM ColorMNIST benchmark.
    Train = env1 (e=0.1) + env2 (e=0.2) concatenated.
    Test  = test  env  (e=0.9).

    test_transform: optional callable applied to each test image tensor (3, H, W)
    """
    raw_train = datasets.MNIST(data_root, train=True, download=True)
    raw_test = datasets.MNIST(data_root, train=False, download=True)

    rng = torch.Generator()
    rng.manual_seed(0)

    # Split train set in half for the two environments
    n = len(raw_train)
    half = n // 2
    idx = torch.randperm(n, generator=rng)
    idx1, idx2 = idx[:half], idx[half:]

    train_images = raw_train.data  # (60000, 28, 28) uint8
    train_labels = raw_train.targets  # (60000,)        long

    # env1 = _make_environment(train_images[idx1], train_labels[idx1], 0.1, rng)
    # env2 = _make_environment(train_images[idx2], train_labels[idx2], 0.2, rng)
    # train_samples = env1 + env2
    #
    test_images = raw_test.data
    test_labels = raw_test.targets
    # test_samples = _make_environment(test_images, test_labels, 0.9, rng)

    # -----------
    env1_images, env1_labels = _make_environment(train_images[idx1], train_labels[idx1], 0.1, rng)
    env2_images, env2_labels = _make_environment(train_images[idx2], train_labels[idx2], 0.2, rng)

    train_images_all = env1_images + env2_images
    train_labels_all = env1_labels + env2_labels

    test_images_all, test_labels_all = _make_environment(test_images, test_labels, 0.9, rng)

    train_loader = DataLoader(ColorMNISTDataset(
        train_images_all, train_labels_all),
        batch_size=batch_size, shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster transfer to GPU
        persistent_workers=True  # Keeps workers alive between epochs
    )
    test_loader = DataLoader(ColorMNISTDataset(
        test_images_all, test_labels_all, transform=test_transform),
        batch_size=batch_size, shuffle=False,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster transfer to GPU
        persistent_workers=True  # Keeps workers alive between epochs
    )
    return train_loader, test_loader


def random_flip(x):
    a = random.random()
    if a < 0.33:
        return torch.flip(x, dims=[-1])
    if a < 0.67:
        return torch.flip(x, dims=[-2])
    return x


def get_loaders(args):
    dataset = args.dataset.lower()

    if dataset == 'mnist':
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_transform = train_transform

        if args.reflect_images:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(random_flip),
                transforms.Normalize((0.1307,), (0.3081,))
            ])

        train_loader = DataLoader(
            datasets.MNIST('./data', train=True, download=True, transform=train_transform),
            batch_size=args.batch_size, shuffle=True,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster transfer to GPU
            persistent_workers=True  # Keeps workers alive between epochs
        )
        test_loader = DataLoader(
            datasets.MNIST('./data', train=False, transform=test_transform),
            batch_size=args.batch_size, shuffle=False,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster transfer to GPU
            persistent_workers=True  # Keeps workers alive between epochs
        )
        in_channels = 1
        n_classes = 10

    elif dataset == 'svhn':
        # SVHN: 32×32 RGB, 10 classes (digit 0–9; torchvision maps label 10 → 0)
        mean = (0.4377, 0.4438, 0.4728)
        std = (0.1980, 0.2010, 0.1970)
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        test_transform = train_transform
        if args.reflect_images:
            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(random_flip),
                transforms.Normalize(mean, std)
            ])

        train_loader = DataLoader(
            datasets.SVHN('./data', split='train', download=True, transform=train_transform),
            batch_size=args.batch_size, shuffle=True,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster transfer to GPU
            persistent_workers=True  # Keeps workers alive between epochs
        )
        test_loader = DataLoader(
            datasets.SVHN('./data', split='test', download=True, transform=test_transform),
            batch_size=args.batch_size, shuffle=False,
            num_workers=4,  # Parallel data loading
            pin_memory=True,  # Faster transfer to GPU
            persistent_workers=True  # Keeps workers alive between epochs
        )
        in_channels = 3
        n_classes = 10

    elif dataset == 'colormnist':
        cm_test_transform = transforms.Lambda(random_flip) if args.reflect_images else None
        train_loader, test_loader = get_colormnist_loaders(args.batch_size, test_transform=cm_test_transform)
        in_channels = 3
        n_classes = 10
    elif dataset == 'inaturalist':
        return get_bird_data_loaders(args)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_loader, test_loader, in_channels, n_classes
