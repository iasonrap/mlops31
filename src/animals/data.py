from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from torchvision.io import read_image
import kagglehub
import shutil
from PIL import Image
import torchvision.transforms as T
import random
from math import floor
import numpy as np
import os

class AnimalsDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, targets, transform=None):
        self.image_paths = image_paths
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.targets[idx], dtype=torch.long)  # Convert label to tensor
        

        return image, label

def calculate_mean_std(input_folder: Path, batch_size: int = 128) -> None:
    all_image_paths = []
    for animal_subfolder in input_folder.iterdir():
        if animal_subfolder.is_dir():
            images = list(animal_subfolder.glob("*.*"))
            images = [img for img in images if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]]
            all_image_paths.extend(images)

    # Calculate running mean and standard deviation of the dataset
    running_sum = torch.zeros(3)
    running_square_sum = torch.zeros(3)
    n_pixels = 0

    # Process images in batches
    for i in range(0, len(all_image_paths), batch_size):
        batch_paths = all_image_paths[i:i + batch_size]
        batch = []

        for img_path in batch_paths:
            try:
                with Image.open(img_path) as img:
                    # Convert image to RGB and resize
                    img = img.convert("RGB")
                    img = img.resize((224, 224), Image.Resampling.BILINEAR)

                    # Convert to NumPy array and then to PyTorch tensor
                    img_array = np.array(img) 
                    
                    # Normalize to [0, 1]
                    img_array = img_array / 255.0

                    img_tensor = torch.tensor(img_array, dtype=torch.float32).permute(2, 0, 1)  # Channels first
                    batch.append(img_tensor)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

        if not batch:
            continue  # Skip empty batches

        # Stack images into a single tensor
        batch_tensor = torch.stack(batch)  # Shape: [batch_size, 3, H, W]

        # Update running sums and pixel count
        n_pixels += batch_tensor.size(0) * batch_tensor.size(2) * batch_tensor.size(3)  # Total pixels
        running_sum += batch_tensor.sum(dim=[0, 2, 3])  # Sum over batch, height, width
        running_square_sum += (batch_tensor ** 2).sum(dim=[0, 2, 3])

    # Calculate mean and standard deviation for each channel
    mean = running_sum / n_pixels
    std = torch.sqrt(running_square_sum / n_pixels - mean ** 2)
    print(f"Mean: {mean}, Std: {std}")
    return mean, std


def download(dataset: str) -> None:
    """Preprocess the raw data and save it to the output folder."""
    # Download latest version
    if os.path.exists("data/raw/raw-img"):
        print("Dataset exists, skipping download.")
        return
    path = kagglehub.dataset_download(dataset)

    # Move filed from downloaded folder to data/raw
    source_folder = Path(path+"/raw-img")
    destination_folder = Path(str(Path.cwd())+"/data/raw/raw-img")
    if destination_folder.is_dir():
        shutil.rmtree(destination_folder)
    shutil.move(str(source_folder), str(destination_folder))

    # Remove the downloaded folder
    shutil.rmtree(source_folder.parent.parent.parent.parent)

    print(f"Folder moved from '{source_folder}' to '{destination_folder}'")

def split_dataset(input_folder: Path, split_ratios=(0.8, 0.1, 0.1), mean=None, std=None) -> tuple[AnimalsDataset, AnimalsDataset, AnimalsDataset]:
    """
    Split the dataset into train, test, and validation PyTorch Datasets.

    Args:
        input_folder (Path): Path to the folder containing images and subfolders.
        split_ratios (tuple): Ratios for splitting the dataset.

    Returns:
        tuple: Train, test, and validation datasets as PyTorch TensorDatasets.
    """
    
    all_image_paths = []
    all_targets = []
    for animal_subfolder in input_folder.iterdir():
        if animal_subfolder.is_dir():
            images = list(animal_subfolder.glob("*.*"))
            images = [img for img in images if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]]
            all_image_paths.extend(images)
            all_targets.extend([animal_subfolder.name] * len(images))

    # Shuffle images and targets for random distribution

    zipped = list(zip(all_image_paths, all_targets))
    random.shuffle(zipped)
    all_image_paths, all_targets = zip(*zipped)
    

    

    # Calculate split sizes
    total = len(all_image_paths)
    train_count = floor(total * split_ratios[0])
    test_count = floor(total * split_ratios[1])
    #val_count = floor(total * split_ratios[2])  # Remaining for validation

    # Split the image paths
    train_images, train_targets = all_image_paths[:train_count], all_targets[:train_count]
    test_images, test_targets = all_image_paths[train_count:train_count + test_count], all_targets[train_count:train_count + test_count]
    val_images, val_targets = all_image_paths[train_count + test_count:], all_targets[train_count + test_count:]

    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(all_targets)))}
    train_targets = [label_to_idx[label] for label in train_targets]
    test_targets = [label_to_idx[label] for label in test_targets]
    val_targets = [label_to_idx[label] for label in val_targets]

    if mean is None or std is None:
        mean, std = calculate_mean_std(input_folder)
    
    transform_list = [T.Resize((224, 224)), T.ToTensor(), T.Normalize(mean=mean, std=std)]
    transform = T.Compose(transform_list)

    train_dataset = AnimalsDataset(train_images, train_targets, transform)
    test_dataset = AnimalsDataset(test_images, test_targets, transform)
    val_dataset = AnimalsDataset(val_images, val_targets, transform)

    return train_dataset, test_dataset, val_dataset

if __name__ == "__main__":
    dataset = "alessiocorrado99/animals10"
    download(dataset)

    input_folder = Path(str(Path.cwd())+"/data/raw/raw-img")

    train_dataset, test_dataset, val_dataset = split_dataset(input_folder)

    print(f"Train images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
