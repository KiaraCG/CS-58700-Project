import os
import shutil
import random
 
import kagglehub
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
 
bird_data_dir = "/scratch/scholar/shams3/inatbirds/birds_train_small"
 
IMAGES_PER_CLASS = 50   # → 40 train + 10 test after 80/20 split
RANDOM_SEED      = 42
 
 
# ─────────────────────────────────────────────
# Download + subsample
# ─────────────────────────────────────────────
 
def download_data():
    path = kagglehub.dataset_download("sharansmenon/inat2021birds")
 
    os.makedirs(bird_data_dir, exist_ok=True)
 
    for item in os.listdir(path):
        shutil.move(os.path.join(path, item), bird_data_dir)
 
    print("Dataset downloaded to:", bird_data_dir)
 
    # Keep only IMAGES_PER_CLASS images per class right after downloading
    _subsample_classes(bird_data_dir, IMAGES_PER_CLASS)
 
 
def _subsample_classes(root: str, n: int, seed: int = RANDOM_SEED) -> None:
    """
    For every class sub-folder in *root*, keep only *n* randomly chosen
    images and delete the rest in-place.
 
    With n=50 and an 80/20 train/test split this gives:
        train → 40 images / class
        test  → 10 images / class
    """
    rng = random.Random(seed)
    image_extensions = {".jpg", ".jpeg", ".png"}
 
    class_dirs = sorted(
        [d for d in os.scandir(root) if d.is_dir()],
        key=lambda e: e.name,
    )
 
    if not class_dirs:
        print("[subsample] No class sub-folders found — nothing to trim.")
        return
 
    total_kept = total_removed = 0
    for cls in class_dirs:
        images = [
            f.path for f in os.scandir(cls.path)
            if os.path.splitext(f.name)[1].lower() in image_extensions
        ]
        if len(images) <= n:
            total_kept += len(images)
            continue                       # already small enough, leave as-is
 
        rng.shuffle(images)
        for p in images[n:]:
            os.remove(p)
        total_kept    += n
        total_removed += len(images) - n
 
    n_classes = len(class_dirs)
    print(f"[subsample] {n_classes} classes | kept {total_kept} "
          f"({n}/class) | removed {total_removed}")
 
 
# ─────────────────────────────────────────────
# Resize (optional — run once after download)
# ─────────────────────────────────────────────
 
def resize_images():
    temp_dir    = bird_data_dir + "_resized_tmp"
    target_size = (224, 224)
 
    os.makedirs(temp_dir, exist_ok=True)
 
    for root, dirs, files in os.walk(bird_data_dir):
        rel_path = os.path.relpath(root, bird_data_dir)
        save_dir = os.path.join(temp_dir, rel_path)
        os.makedirs(save_dir, exist_ok=True)
 
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path  = os.path.join(root, file)
                save_path = os.path.join(save_dir, file)
                with Image.open(img_path) as img:
                    img = img.convert("RGB").resize(target_size)
                    img.save(save_path)
 
    shutil.rmtree(bird_data_dir)
    os.rename(temp_dir, bird_data_dir)
    print("Resizing complete.")
 
 
# ─────────────────────────────────────────────
# Dataset / DataLoader
# ─────────────────────────────────────────────
 
class TransformDataset(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform
 
    def __len__(self):
        return len(self.subset)
 
    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return self.transform(x), y
 
 
def get_bird_data_loaders(args):
    batch_size     = args.batch_size
    reflect_images = getattr(args, 'reflect_images', False)
 
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225)),
    ])
 
    test_transform = train_transform
    if reflect_images:
        test_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])
 
    # Load full dataset (already subsampled to 50/class on disk)
    full_dataset = datasets.ImageFolder(bird_data_dir)
    n_classes    = len(full_dataset.classes)
 
    # 80/20 split → ~40 train / ~10 test per class
    train_size = int(0.8 * len(full_dataset))
    test_size  = len(full_dataset) - train_size
    import torch
    generator  = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, test_subset = random_split(
        full_dataset, [train_size, test_size], generator=generator
    )
 
    train_dataset = TransformDataset(train_subset, transform=train_transform)
    test_dataset  = TransformDataset(test_subset,  transform=test_transform)
 
    print(f"[birds] {n_classes} classes | "
          f"train {len(train_dataset)} | test {len(test_dataset)}")
 
    loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
 
    train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
    test_loader  = DataLoader(test_dataset,  shuffle=False, **loader_kwargs)
 
    in_channels = 3
    return train_loader, test_loader, in_channels, n_classes
 