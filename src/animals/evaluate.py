import torch
import typer
from hydra import initialize, compose
from model import AnimalModel
from data import split_dataset
from pathlib import Path
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(cfg, model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)
    start_time = time.time()
    model = AnimalModel(cfg.hyperparameters.model_name, cfg.hyperparameters.num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint), strict=False)
    model.to(DEVICE)
    _, test_set, _ = split_dataset(Path("data/raw/raw-img/"), 
                                    mean=torch.tensor([0.5177, 0.5003, 0.4126]), std=torch.tensor([0.2659, 0.2610, 0.2785]))

    test_dataloader = torch.utils.data.DataLoader(test_set, cfg.hyperparameters.batch_size, shuffle=False)

    model.eval()
    correct, total = 0, 0

    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        output = model(img)
        correct += (output.argmax(1) == target).float().sum().item()
        total += target.size(0)
    stop_time = time.time()
    print(f"Time taken: {stop_time - start_time}")
    print(f"Accuracy: {correct / total}")


# Initialize Typer app
app = typer.Typer()

# Define the evaluate command
@app.command()
def evaluate_command(
    model_checkpoint: str = typer.Option(..., help="Path to the model checkpoint")
):
    """Wrapper for evaluation."""
    # Initialize Hydra and load the config
    with initialize(version_base="1.1", config_path="conf"):
        cfg = compose(config_name="config.yaml")

    # Call the evaluation function
    evaluate(cfg, model_checkpoint)

if __name__ == "__main__":
    app()