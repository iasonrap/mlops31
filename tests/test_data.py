from src.animals.data import AnimalsDataset, calculate_mean_std, split_dataset
import torch
from pathlib import Path

project_root = Path(__file__).resolve()
image_path = project_root / "data" / "raw" / "raw-img"


def test_animals_dataset_initialization():
    image_paths = ["src/animals/data/raw/raw-img/cane/OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg",
                   "src/animals/data/raw/raw-img/cane/OIP-__Yu1XH3iAC10OzGQFpC-AHaE8.jpeg"]
    targets = [0, 1]
    dataset = AnimalsDataset(image_paths, targets)
    assert len(dataset) == 2, f"Expected dataset length 2, but got {len(dataset)}"
    assert dataset.image_paths == image_paths, "Image paths not set correctly"
    assert dataset.targets == targets, "Targets not set correctly"


def test_calculate_mean_std():
    mean, std = calculate_mean_std(image_path, batch_size=2)

    # Assert mean and std are tensors
    assert isinstance(mean, torch.Tensor), "Mean should be a torch.Tensor"
    assert isinstance(std, torch.Tensor), "Standard deviation should be a torch.Tensor"

    # Assert mean and std have the correct shape
    assert mean.shape == (3,), "Mean should have 3 channels (RGB)"
    assert std.shape == (3,), "Standard deviation should have 3 channels (RGB)"

    # Assert mean values are within the [0, 1] range
    assert torch.all(mean >= 0) and torch.all(mean <= 1), "Mean values should be in the [0, 1] range"

    # Assert standard deviation values are non-negative
    assert torch.all(std >= 0), "Standard deviation values should be non-negative"


def test_split_dataset():
    train_dataset, test_dataset, val_dataset = split_dataset(image_path, split_ratios=(0.8, 0.1, 0.1))
    assert len(train_dataset) == 20943, f"Expected 20943 train images, got {len(train_dataset)}"
    assert len(test_dataset) == 2618, f"Expected 2618 test images, got {len(test_dataset)}"
    assert len(val_dataset) == 2618, f"Expected 2618 validation images, got {len(val_dataset)}"


def main():
    test_animals_dataset_initialization()
    test_calculate_mean_std()
    test_split_dataset()


if __name__ == "__main__":
    main()
