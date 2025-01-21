from src.animals.model import AnimalModel
import os
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# Get the current working directory
current_dir = os.getcwd()
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Append "src/animals/conf" to the parent directory
config_path = os.path.relpath(os.path.join(parent_dir, "src/animals/conf"), current_dir)


def test_model_output_shape():
    """Test that the output shape matches the expected one for a batch of images"""
    GlobalHydra.instance().clear()  # Reset Hydra state between tests
    with initialize(config_path=config_path, version_base="1.1"):
        cfg = compose(config_name="config.yaml")
        model = AnimalModel(cfg.hyperparameters.model_name, cfg.hyperparameters.num_classes)
        x = torch.randn(1, 3, 224, 224)
        y = model(x)
        assert y.shape == (1, cfg.hyperparameters.num_classes), f"Expected output shape (1, {cfg.hyperparameters.num_classes}), but got {y.shape}"


def test_model_with_different_input_size():
    """Test that the model works with a different input size (e.g. 28x128)"""
    GlobalHydra.instance().clear()  # Reset Hydra state between tests
    with initialize(config_path=config_path, version_base="1.1"):
        cfg = compose(config_name="config.yaml")
        model = AnimalModel(cfg.hyperparameters.model_name, cfg.hyperparameters.num_classes)
        x = torch.randn(1, 3, 128, 128)
        y = model(x)
        assert y.shape == (1, cfg.hyperparameters.num_classes), f"Expected output shape (1, {cfg.hyperparameters.num_classes}), but got {y.shape}"


def test_model_with_invalid_input():
    """Test the model with invalid input (e.g., incorrect number of channels)"""
    GlobalHydra.instance().clear()  # Reset Hydra state between tests
    with initialize(config_path=config_path, version_base="1.1"):
        cfg = compose(config_name="config.yaml")
        model = AnimalModel(cfg.hyperparameters.model_name, cfg.hyperparameters.num_classes)
        x = torch.randn(1, 1, 224, 224)
        try:
            y = model(x)
            assert False, "Expected an error due to invalid input channels"
        except Exception as e:
            assert isinstance(e, RuntimeError), f"Expected a RuntimeError, but got {type(e)}"


def main():
    test_model_output_shape()
    test_model_with_different_input_size()
    test_model_with_invalid_input()


if __name__ == "__main__":
    main()
