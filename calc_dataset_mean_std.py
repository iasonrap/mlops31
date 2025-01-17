from pathlib import Path
import torch
from PIL import Image
import numpy as np

def preprocess(input_folder: Path, batch_size: int = 128) -> None:
    all_image_paths = []
    image_path_targets = []
    for animal_subfolder in input_folder.iterdir():
        if animal_subfolder.is_dir():
            images = list(animal_subfolder.glob("*.*"))
            images = [img for img in images if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]]
            all_image_paths.extend(images)
            image_path_targets.extend([animal_subfolder.name] * len(images))

    # Calculate running mean and standard deviation of the dataset
    running_sum = 0.0
    running_square_sum = 0.0
    n_pixels = 0

    # Process images in batches
    for i in range(0, len(all_image_paths), batch_size):
        batch_paths = all_image_paths[i:i + batch_size]
        batch = []

        for img_path in batch_paths:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize((224, 224), Image.Resampling.BILINEAR)
                img_array = np.array(img)  # Convert to NumPy array
                img_tensor = torch.tensor(img_array, dtype=torch.float32) / 255.0  # Normalize to [0, 1]
                batch.append(img_tensor)

        # Stack images into a single tensor
        batch_tensor = torch.stack(batch)

        # Update running sum and pixel count
        n_pixels += batch_tensor.numel()
        running_sum += batch_tensor.sum().item()
        running_square_sum += (batch_tensor ** 2).sum().item()

    # Calculate mean and standard deviation
    mean = running_sum / n_pixels
    std = ((running_square_sum / n_pixels) - (mean ** 2)) ** 0.5 

    print(f"Dataset Mean: {mean}")
    print(f"Dataset Std: {std}")

if __name__ == "__main__":
    input_folder = Path("data/raw")
    output_folder = Path("data/processed")
    preprocess(input_folder, output_folder)