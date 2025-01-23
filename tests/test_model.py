from src.animals.model import AnimalModel
from hydra import initialize, compose
import torch


def test_model_output_shape():
    """Test that the output shape matches the expected one for a batch of images."""
    # Initialize Hydra and compose the configuration
    config_path = "../src/animals/conf"
    with initialize(config_path=config_path, version_base="1.1"):
        cfg = compose(config_name="config.yaml")

    # Create the model using the configuration
    model = AnimalModel(cfg.hyperparameters.model_name, cfg.hyperparameters.num_classes)

    # Create a dummy input tensor
    x = torch.randn(1, 3, 224, 224)
    y = model(x)

    # Assert the output shape is correct
    expected_shape = (1, cfg.hyperparameters.num_classes)
    assert y.shape == expected_shape, f"Expected output shape {expected_shape}, but got {y.shape}"


def test_model_with_different_input_size():
    """Test that the model works with a different input size (e.g. 128x128)."""
    # Initialize Hydra and compose the configuration
    config_path = "../src/animals/conf"
    with initialize(config_path=config_path, version_base="1.1"):
        cfg = compose(config_name="config.yaml")

    # Create the model using the configuration
    model = AnimalModel(cfg.hyperparameters.model_name, cfg.hyperparameters.num_classes)

    # Create a dummy input tensor with a different size
    x = torch.randn(1, 3, 128, 128)
    y = model(x)

    # Assert the output shape is correct
    expected_shape = (1, cfg.hyperparameters.num_classes)
    assert y.shape == expected_shape, f"Expected output shape {expected_shape}, but got {y.shape}"


def test_model_with_invalid_input():
    """Test the model with invalid input (e.g., incorrect number of channels)."""
    # Initialize Hydra and compose the configuration
    config_path = "../src/animals/conf"  # Update this path if necessary
    with initialize(config_path=config_path, version_base="1.1"):
        cfg = compose(config_name="config.yaml")

    # Create the model using the configuration
    model = AnimalModel(cfg.hyperparameters.model_name, cfg.hyperparameters.num_classes)

    # Create a dummy input tensor with an invalid number of channels
    x = torch.randn(1, 1, 224, 224)

    # Test for expected exception
    try:
        y = model(x)
        assert False, f"Expected an error due to invalid input channels, but got output {y}"
    except Exception as e:
        # Ensure the raised exception is the expected type
        assert isinstance(e, RuntimeError), f"Expected a RuntimeError, but got {type(e)}"


def main():
    test_model_output_shape()
    test_model_with_different_input_size()
    test_model_with_invalid_input()


if __name__ == "__main__":
    main()
