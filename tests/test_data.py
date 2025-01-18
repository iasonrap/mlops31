from data import AnimalsDataset, calculate_mean_std, split_dataset
import torch
from PIL import Image
import tempfile
from pathlib import Path

def test_animals_dataset_initialization():
    image_paths = ["path1.jpg", "path2.jpg"]
    targets = [0, 1]
    dataset = AnimalsDataset(image_paths, targets)
    assert len(dataset) == 2, f"Expected dataset length 2, but got {len(dataset)}"
    assert dataset.image_paths == image_paths, "Image paths not set correctly"
    assert dataset.targets == targets, "Targets not set correctly"

def test_animals_dataset_getitem():
    # Create temporary image files
    with tempfile.TemporaryDirectory() as temp_dir:
        img_path = f"{temp_dir}/test_image.jpg"
        image = Image.new("RGB", (100, 100))
        image.save(img_path)

        image_paths = [img_path]
        targets = [0]
        dataset = AnimalsDataset(image_paths, targets)

        img, label = dataset[0]
        assert isinstance(img, torch.Tensor), "Image is not converted to a tensor"
        assert label == 0, f"Expected label 0, but got {label}"

def test_calculate_mean_std():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create some test images
        for i in range(5):
            img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
            img.save(f"{temp_dir}/image_{i}.jpg")

        mean, std = calculate_mean_std(Path(temp_dir), batch_size=2)
        assert len(mean) == 3, "Mean should have 3 channels (RGB)"
        assert len(std) == 3, "Standard deviation should have 3 channels (RGB)"
        assert (mean >= 0).all() and (mean <= 1).all(), "Mean values should be in [0, 1] range"
        assert (std >= 0).all(), "Standard deviation should be non-negative"

def test_split_dataset():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create labeled subfolders with images
        labels = ["cat", "dog"]
        for label in labels:
            label_dir = Path(temp_dir) / label
            label_dir.mkdir()
            for i in range(5):
                img = Image.new("RGB", (100, 100), color=(i * 50, i * 50, i * 50))
                img.save(label_dir / f"image_{i}.jpg")

        train_dataset, test_dataset, val_dataset = split_dataset(Path(temp_dir), split_ratios=(0.6, 0.2, 0.2))
        assert len(train_dataset) == 6, f"Expected 6 train images, got {len(train_dataset)}"
        assert len(test_dataset) == 2, f"Expected 2 test images, got {len(test_dataset)}"
        assert len(val_dataset) == 2, f"Expected 2 validation images, got {len(val_dataset)}"