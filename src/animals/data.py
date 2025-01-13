from pathlib import Path
import torch
from torch.utils.data import TensorDataset
from torchvision.io import read_image
import kagglehub
import shutil
from PIL import Image, ImageOps
import random
from math import floor

def download(dataset: str) -> None:
    """Preprocess the raw data and save it to the output folder."""
    # Download latest version
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

def process_images(input_folder: Path, output_folder: Path, size=(128, 128), normalize=True):
    """
    Process images by normalizing, resizing, and applying random rotation.

    Args:
        input_folder (Path): Path to the folder containing images and subfolders.
        output_folder (Path): Path to save processed images.
        size (tuple): Desired size for resizing (width, height).
        normalize (bool): Whether to normalize pixel values to [0, 1].
    """
    if not input_folder.exists():
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist.")
    
    output_folder.mkdir(parents=True, exist_ok=True)
    for img_path in input_folder.rglob("*.*"):  # Recursively iterate through files
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            continue  # Skip non-image files
        
        try:
            # Open the image
            with Image.open(img_path) as img:
                # Convert RGBA AND CMYK to RGB
                if img.mode == 'RGBA':
                    img = img.convert('RGB')
                elif img.mode == 'CMYK':
                    img = img.convert('RGB')
                # Resize the image
                img_resized = img.resize(size, Image.Resampling.LANCZOS)

                # Apply random rotation
                angle = random.randint(0, 360)
                img_rotated = img.rotate(angle, expand=True)

                # Normalize if required
                if normalize:
                    img_normalized = ImageOps.autocontrast(img_rotated)
                else:
                    img_normalized = img_rotated

                # Resize the image
                img_resized = img_normalized.resize(size, Image.Resampling.LANCZOS)

                # Determine output path
                relative_path = img_path.relative_to(input_folder)
                output_path = output_folder / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Save processed image
                img_resized.save(output_path, icc_profile=None)
                #print(f"Processed and saved: {output_path}")

        except Exception as e:
            print(f"Error processing file {img_path}: {e}")
    print("Process completed.")

def split_dataset(input_folder: Path, split_ratios=(0.8, 0.1, 0.1)):
    """
    Split the dataset into train, test, and validation PyTorch Datasets.

    Args:
        input_folder (Path): Path to the folder containing images and subfolders.
        split_ratios (tuple): Ratios for splitting the dataset.

    Returns:
        tuple: Train, test, and validation datasets as PyTorch TensorDatasets.
    """
    
    all_image_paths = []
    for animal_subfolder in input_folder.iterdir():
        if animal_subfolder.is_dir():
            images = list(animal_subfolder.glob("*.*"))
            images = [img for img in images if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]]
            all_image_paths.extend(images)

    # Shuffle images for random distribution
    random.shuffle(all_image_paths)

    # Calculate split sizes
    total = len(all_image_paths)
    train_count = floor(total * split_ratios[0])
    test_count = floor(total * split_ratios[1])
    val_count = floor(total * split_ratios[2])  # Remaining for validation

    # print(train_count,"  ,  ",test_count,"  ,  ",val_count)
    # Split the image paths
    train_images = all_image_paths[:train_count]
    test_images = all_image_paths[train_count+1:train_count + test_count]
    val_images = all_image_paths[val_count:]

    # Helper function to convert image paths to tensors
    def paths_to_tensor_dataset(image_paths):
        tensors = []
        for img_path in image_paths:
            img = read_image(str(img_path))
            if img.size(0) == 1:
                img = img.repeat(3, 1, 1)
            tensors.append(img)
        #tensors = [read_image(str(img_path)) for img_path in image_paths]
        #print(tensors)
        return TensorDataset(torch.stack(tensors)) if tensors else None

    # Convert splits into PyTorch TensorDatasets
    train_dataset = paths_to_tensor_dataset(train_images)
    test_dataset = paths_to_tensor_dataset(test_images)
    val_dataset = paths_to_tensor_dataset(val_images)

    return train_dataset, test_dataset, val_dataset

# Example usage
dataset = "alessiocorrado99/animals10"
download(dataset)

input_folder = Path(str(Path.cwd())+"/data/raw/raw-img")  # Replace with your input folder path
output_folder = Path(str(Path.cwd())+"/data/processed/proc")  # Replace with your output folder path
process_images(input_folder, output_folder, size=(128, 128), normalize=True)

train_paths, test_paths, val_paths = split_dataset(output_folder)

print(f"Train images: {len(train_paths)}")
print(f"Test images: {len(test_paths)}")
print(f"Validation images: {len(val_paths)}")