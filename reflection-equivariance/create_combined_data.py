import os

from tqdm import tqdm


def process_mnist_inplace(root_dir):
    import cv2
    mnist_folders = [str(i) for i in range(10)]

    for folder in mnist_folders:
        folder_path = os.path.join(root_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Skipping {folder}: Path not found.")
            continue

        print(f"Processing folder: {folder}")
        for filename in tqdm(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, filename)

            # Read image
            img = cv2.imread(file_path)
            if img is None: continue

            # 1. Resize to 224x224
            # 2. Convert to 3-channel (cv2.imread usually loads as 3-ch BGR by default,
            # but we force it to ensure consistency)
            if len(img.shape) == 2:  # If it was loaded as grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)

            # Overwrite the original file
            cv2.imwrite(file_path, img)
