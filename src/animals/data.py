import os
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from google.cloud import storage
from PIL import Image
from sklearn.model_selection import train_test_split


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
        batch_paths = all_image_paths[i : i + batch_size]
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
        running_square_sum += (batch_tensor**2).sum(dim=[0, 2, 3])

    # Calculate mean and standard deviation for each channel
    mean = running_sum / n_pixels
    std = torch.sqrt(running_square_sum / n_pixels - mean**2)
    print(f"Mean: {mean}, Std: {std}")
    return mean, std


def download(bucket_name, destination_folder) -> None:
    """Preprocess the raw data and save it to the output folder."""
    # Download latest version
    if os.path.exists("data/raw/raw-img"):
        print("Dataset exists, skipping download.")
        return

    storage_client = storage.Client()

    # Get the bucket
    bucket = storage_client.bucket(bucket_name)

    # List all blobs in the bucket
    blobs = bucket.list_blobs(prefix="raw/")

    # Ensure the destination folder exists
    os.makedirs(destination_folder, exist_ok=True)

    # Download each blob
    count = 0
    for blob in blobs:
        print(blob.name)
        if str(blob.name).endswith("/") or str(blob.name).endswith(".zip"):
            continue
        # print(blob.name)
        destination_file_name = os.path.join(destination_folder, blob.name)  # Append blob name to destination folder
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)  # Ensure subdirectories exist
        # print(destination_file_name)
        blob.download_to_filename(destination_file_name)  # Save blob to the file path
        # print(f"Downloaded '{blob.name}' to '{destination_file_name}'.")


def split_dataset(
    input_folder: Path, split_ratios=(0.8, 0.1, 0.1), mean=None, std=None
) -> tuple[AnimalsDataset, AnimalsDataset, AnimalsDataset]:
    """
    Split the dataset into train, test, and validation PyTorch Datasets.

    Args:
        input_folder (Path): Path to the folder containing images and subfolders.
        split_ratios (tuple): Ratios for splitting the dataset.
        mean (torch.Tensor): Mean for normalization (RGB channels).
        std (torch.Tensor): Std for normalization (RGB channels).

    Returns:
        tuple: Train, test, and validation datasets as PyTorch Datasets.
    """
    # Collect image paths and labels
    all_image_paths = []
    all_targets = []
    for animal_subfolder in input_folder.iterdir():
        if animal_subfolder.is_dir():
            images = [
                img
                for img in animal_subfolder.glob("*.*")
                if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
            ]
            all_image_paths.extend(images)
            all_targets.extend([animal_subfolder.name] * len(images))

    # Map labels to indices
    label_to_idx = {label: idx for idx, label in enumerate(sorted(set(all_targets)))}
    all_targets = [label_to_idx[label] for label in all_targets]

    # Split data using train_test_split
    train_ratio, val_ratio, test_ratio = split_ratios
    train_images, temp_images, train_targets, temp_targets = train_test_split(
        all_image_paths, all_targets, test_size=(1 - train_ratio), stratify=all_targets, random_state=42
    )
    val_images, test_images, val_targets, test_targets = train_test_split(
        temp_images,
        temp_targets,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=temp_targets,
        random_state=42,
    )

    # Calculate mean and std if not provided
    if mean is None or std is None:
        mean, std = calculate_mean_std(input_folder)

    # Define transformations
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )

    # Create datasets
    train_dataset = AnimalsDataset(train_images, train_targets, transform)
    val_dataset = AnimalsDataset(val_images, val_targets, transform)
    test_dataset = AnimalsDataset(test_images, test_targets, transform)

    return train_dataset, test_dataset, val_dataset


if __name__ == "__main__":
    bucket_name = "31animals"  # Replace with your bucket name
    destination_folder = str(Path.cwd().parent.parent) + "/data/"  # Replace with your desired local folder
    download(bucket_name, destination_folder)

    input_folder = Path(str(Path.cwd()) + "/data/raw/raw-img")

    train_dataset, test_dataset, val_dataset = split_dataset(input_folder)

    print(f"Train images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
