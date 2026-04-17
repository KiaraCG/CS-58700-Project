import os
import shutil

import kagglehub
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

bird_data_dir = "/scratch/scholar/jchaugar/inatbirds/birds_train_small"


def download_data():
    path = kagglehub.dataset_download("sharansmenon/inat2021birds")

    os.makedirs(bird_data_dir, exist_ok=True)

    for item in os.listdir(path):
        shutil.move(os.path.join(path, item), bird_data_dir)

    print("Dataset downloaded to:", bird_data_dir)


def resize_images():
    from PIL import Image

    temp_dir = bird_data_dir + "_resized_tmp"
    target_size = (224, 224)  # desired size for CNNs

    # Make temporary output folder
    os.makedirs(temp_dir, exist_ok=True)

    # Walk through input folder (supports class subfolders)
    for root, dirs, files in os.walk(bird_data_dir):
        rel_path = os.path.relpath(root, bird_data_dir)
        save_dir = os.path.join(temp_dir, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                save_path = os.path.join(save_dir, file)

                with Image.open(img_path) as img:
                    img = img.convert("RGB")  # ensure 3 channels
                    img = img.resize(target_size)  # resize
                    img.save(save_path)

    # --- Delete original folder ---
    shutil.rmtree(bird_data_dir)

    # --- Rename resized folder to original name ---
    os.rename(temp_dir, bird_data_dir)

    print("Resizing complete. Original images replaced with resized images.")


class TransformDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y


def get_bird_data_loaders(args):
    batch_size = args.batch_size
    reflect_images = getattr(args, 'reflect_images', False)
    rotate_images = getattr(args, 'rotate_images', False)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    test_transform = train_transform
    if reflect_images:
        test_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])
    if rotate_images:
        test_transform = transforms.Compose([
            transforms.RandomRotation(180),   # full 360 range
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))
        ])

    # --- Load full dataset ---
    full_dataset = datasets.ImageFolder(bird_data_dir)

    # --- Split dataset (80% train / 20% test) ---
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # --- Apply test transforms ---
    train_dataset = TransformDataset(train_dataset, transform=train_transform)
    test_dataset = TransformDataset(test_dataset, transform=test_transform)

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
