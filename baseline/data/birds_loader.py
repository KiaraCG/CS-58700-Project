import os
import shutil
import random
 
import torch
import kagglehub
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import datasets, transforms
 
bird_data_dir = "/scratch/scholar/shams3/inatbirds/birds_train_small/bird_train"
# bird_data_dir = "/scratch/scholar/shams3/inatbirds"
 

RANDOM_SEED      = 42
 
 
# ─────────────────────────────────────────────
# Download + subsample
# ─────────────────────────────────────────────
 
def download_data():
    path = kagglehub.dataset_download("sharansmenon/inatbirds100k")
 
    os.makedirs(bird_data_dir, exist_ok=True)
 
    for item in os.listdir(path):
        shutil.move(os.path.join(path, item), bird_data_dir)
 
    print("Dataset downloaded to:", bird_data_dir)
 
    # Keep only IMAGES_PER_CLASS images per class right after downloading
    _keep_n_classes(bird_data_dir, n_classes=10, seed=RANDOM_SEED)
 
 
def _keep_n_classes(root: str, n_classes: int = 100, seed: int = RANDOM_SEED) -> None:
    """
    Keep only n_classes random class folders, delete the rest.
    All images within kept classes are preserved.
    """
    import shutil, random

    keep_classes = set("03111_Animalia_Chordata_Aves_Accipitriformes_Accipitridae_Accipiter_badius",          "04562_Animalia_Chordata_Aves_Strigiformes_Tytonidae_Tyto_alba",
"03932_Animalia_Chordata_Aves_Passeriformes_Meliphagidae_Meliphaga_lewinii",           "04567_Animalia_Chordata_Aves_Suliformes_Anhingidae_Anhinga_rufa",
"04492_Animalia_Chordata_Aves_Procellariiformes_Procellariidae_Calonectris_diomedea",  "04577_Animalia_Chordata_Aves_Suliformes_Phalacrocoracidae_Phalacrocorax_carbo",
"04505_Animalia_Chordata_Aves_Psittaciformes_Psittacidae_Amazona_viridigenalis",       "04588_Animalia_Chordata_Aves_Suliformes_Sulidae_Sula_sula",
"04555_Animalia_Chordata_Aves_Strigiformes_Strigidae_Megascops_kennicottii",           "04596_Animalia_Chordata_Aves_Trogoniformes_Trogonidae_Trogon_rufus")

    rng = random.Random(seed)
    all_classes = sorted([d for d in os.scandir(root) if d.is_dir()],
                         key=lambda e: e.name)

    if len(all_classes) <= n_classes:
        print(f"[keep_classes] Only {len(all_classes)} classes found, keeping all.")
        return

    keep    = set(e.name for e in rng.sample(all_classes, n_classes))
    removed = 0
    for cls in all_classes:
        if cls.name not in keep:
            shutil.rmtree(cls.path)
            removed += 1

    total_images = sum(
        len(os.listdir(os.path.join(root, c))) for c in keep
    )
    print(f"[keep_classes] Kept {n_classes} classes ({total_images} images) | "
          f"removed {removed} classes")
 
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
 
    # 1. Load the massive dataset (All classes)
    full_dataset = datasets.ImageFolder(bird_data_dir)
    n_classes    = 10
 
    # 2. FILTERING STEP: Find the indices for only the first 10 classes (labels 0 to 9)
    # This prevents the CUDA out-of-bounds error!
    valid_indices = [i for i, label in enumerate(full_dataset.targets) if label < n_classes]
    
    # 3. Create a smaller dataset using only those 10 classes
    filtered_dataset = Subset(full_dataset, valid_indices)

    # 4. Now do your 80/20 split on the FILTERED dataset, not the full one
    train_size = int(0.8 * len(filtered_dataset))
    test_size  = len(filtered_dataset) - train_size

    generator  = torch.Generator().manual_seed(RANDOM_SEED)
    train_subset, test_subset = random_split(
        filtered_dataset, [train_size, test_size], generator=generator
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