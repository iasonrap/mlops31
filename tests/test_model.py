from animals.model import AnimalModel
import hydra
import torch


@hydra.main(version_base="1.1", config_path="../src/animals/conf", config_name="config.yaml")
def test_model_output_shape(cfg):
    """Test that the output shape matches the expected one for a batch of images"""
    model = AnimalModel(cfg.hyperparameters.model_name, cfg.hyperparameters.num_classes)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    assert y.shape == (1, cfg.hyperparameters.num_classes), f"Expected output shape (1, {cfg.hyperparameters.num_classes}), but got {y.shape}"


@hydra.main(version_base="1.1", config_path="../src/animals/conf", config_name="config.yaml")
def test_model_with_different_input_size(cfg):
    """Test that the model works with a different input size (e.g. 28x128)"""
    model = AnimalModel(cfg.hyperparameters.model_name, cfg.hyperparameters.num_classes)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    assert y.shape == (1, cfg.hyperparameters.num_classes), f"Expected output shape (1, {cfg.hyperparameters.num_classes}), but got {y.shape}"


@hydra.main(version_base="1.1", config_path="../src/animals/conf", config_name="config.yaml")
def test_model_with_invalid_input(cfg):
    """Test the model with invalid input (e.g., incorrect number of channels)"""
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
