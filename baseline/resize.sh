# !/bin/bash

module load conda/2024.09
conda activate CS587

python -c "
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor

root = '/scratch/scholar/shams3/inatbirds/birds_train_small/bird_train'

def resize_file(path):
    with Image.open(path) as img:
        img = img.convert('RGB').resize((64, 64))
        img.save(path)

paths = [
    f.path
    for cls in os.scandir(root)
    for f in os.scandir(cls.path)
    if f.name.lower().endswith(('.jpg','.jpeg','.png'))
]

print(f'Resizing {len(paths)} images...')
with ThreadPoolExecutor(max_workers=4) as ex:
    ex.map(resize_file, paths)
print('Done.')
"

python -c "
from PIL import Image
import os
cls = next(os.scandir('/scratch/scholar/shams3/inatbirds/birds_train_small/bird_train'))
f = next(os.scandir(cls.path))
print(Image.open(f.path).size)
"