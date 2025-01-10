from pathlib import Path

import typer
from torch.utils.data import Dataset

import kagglehub
import shutil

from PIL import Image, ImageOps
import random

class MyDataset(Dataset):
    """My custom dataset."""

    def __init__(self) -> None:
        self.data_path = Path.cwd()

    def __len__(self) -> int:
        """Return the length of the dataset."""

    def __getitem__(self, index: int):
        """Return a given sample from the dataset."""

    def preprocess(self) -> None:
        """Preprocess the raw data and save it to the output folder."""
        # # Download latest version
        # path = kagglehub.dataset_download("alessiocorrado99/animals10")

        # # Move filed from downloaded folder to data/raw
        # source_folder = Path(path+"/raw-img")
        # destination_folder = Path(str(Path.cwd())+"/data/raw/raw-img")
        # if destination_folder.is_dir():
        #     shutil.rmtree(destination_folder)
        # shutil.move(str(source_folder), str(destination_folder))

        # # Remove the downloaded folder
        # shutil.rmtree(source_folder.parent.parent.parent.parent)

        # print(f"Folder moved from '{source_folder}' to '{destination_folder}'")

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
                        # Resize the image
                        img_resized = img.resize(size, Image.Resampling.LANCZOS)
                        #img_resized = img.resize(size)
                        # Apply random rotation
                        angle = random.randint(0, 360)
                        img_rotated = img_resized.rotate(angle, expand=True)

                        # Normalize if required
                        if normalize:
                            img_normalized = ImageOps.autocontrast(img_rotated)
                        else:
                            img_normalized = img_rotated
                        # Determine output path
                        relative_path = img_path.relative_to(input_folder)
                        output_path = output_folder / relative_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        # Save processed image
                        img_normalized.save(output_path)
                        print(f"Processed and saved: {output_path}")
                except Exception as e:
                    print(f"Error processing file {img_path}: {e}")

        # Example usage
        input_folder = Path(str(Path.cwd())+"/data/raw/raw-img")  # Replace with your input folder path
        output_folder = Path(str(Path.cwd())+"/data/processed/proc")  # Replace with your output folder path
        process_images(input_folder, output_folder, size=(128, 128), normalize=True)

        

def preprocess() -> None:
    print("Preprocessing data...")
    dataset = MyDataset()
    dataset.preprocess()


if __name__ == "__main__":
    typer.run(preprocess)
