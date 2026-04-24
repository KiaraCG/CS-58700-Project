import os
import shutil

import kagglehub
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms


def download_data(bird_data_dir):
    path = kagglehub.dataset_download("sharansmenon/inat2021birds")

    os.makedirs(bird_data_dir, exist_ok=True)

    for item in os.listdir(path):
        shutil.move(os.path.join(path, item), bird_data_dir)

    print("Dataset downloaded to:", bird_data_dir)


def resize_images(bird_data_dir):
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


def get_bird_data_loaders(args, mean, std, data_dir):
    batch_size = args.batch_size
    reflect_images = getattr(args, 'reflect_images', False)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transform = train_transform
    if reflect_images:
        test_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

    # --- Load full dataset ---
    full_dataset = NumericImageFolder(data_dir)

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

    #### TODO RESTRICT TO THE ESTABLISHED 10 CLASSES 

    return train_loader, test_loader, in_channels, n_classes


class NumericImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        classes = [d for d in classes if d.isdigit() and 0 <= int(d) <= 19]

        classes.sort(key=lambda x: int(x))

        class_to_idx = {cls_name: int(cls_name) for cls_name in classes}

        return classes, class_to_idx
