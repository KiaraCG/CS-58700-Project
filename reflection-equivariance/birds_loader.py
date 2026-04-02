import os

import kagglehub
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

bird_data_dir = "/Users/kiarachau/PycharmProjects/data/inatbirds"


def download_data():
    path = kagglehub.dataset_download("sharansmenon/inat2021birds")

    os.makedirs(bird_data_dir, exist_ok=True)

    import shutil

    for item in os.listdir(path):
        shutil.move(os.path.join(path, item), bird_data_dir)

    print("Dataset downloaded to:", bird_data_dir)


def get_bird_data_loaders(args):
    data_dir = bird_data_dir
    batch_size = args.batch_size
    reflect_images = getattr(args, 'reflect_images', False)

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    test_transform = train_transform
    if reflect_images:
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

    # --- Load full dataset ---
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    # --- Split dataset (80% train / 20% test) ---
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # --- Apply test transforms ---
    test_dataset.dataset.transform = test_transform

    # --- Create DataLoaders ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # --- Metadata ---
    in_channels = 3
    n_classes = len(full_dataset.classes)

    return train_loader, test_loader, in_channels, n_classes
