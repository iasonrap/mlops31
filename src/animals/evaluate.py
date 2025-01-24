import requests
import time
from pathlib import Path

import pandas as pd
import torch
import typer
from hydra import compose, initialize
from model import AnimalModel

from data import split_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
animals_classes = {
    0: "dog",
    1: "horse",
    2: "elephant",
    3: "butterfly",
    4: "chicken",
    5: "cat",
    6: "cow",
    7: "sheep",
    8: "spider",
    9: "squirrel",
}


def evaluate(cfg, model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)
    start_time = time.time()
    model = AnimalModel(cfg.hyperparameters.model_name, cfg.hyperparameters.num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint), strict=False)
    model.to(DEVICE)
    _, test_set, _ = split_dataset(
        Path("data/raw/raw-img/"),
        mean=torch.tensor([0.5177, 0.5003, 0.4126]),
        std=torch.tensor([0.2659, 0.2610, 0.2785]),
    )

    test_dataloader = torch.utils.data.DataLoader(test_set, cfg.hyperparameters.batch_size, shuffle=False)

    model.eval()
    correct, total = 0, 0

    results = []

    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        output = model(img)
        predictions = output.argmax(1)
        correct += (predictions == target).float().sum().item()
        total += target.size(0)

        softmaxxed = output.softmax(dim=-1)

        for i in range(len(target)):
            results.append(
                {
                    "target": animals_classes[target[i].item()],
                    **{animal: softmaxxed[i, j].item() for j, animal in animals_classes.items()},
                }
            )

    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/evaluation_results.csv", index=False)

    stop_time = time.time()
    print(f"Time taken: {stop_time - start_time}")
    print(f"Accuracy: {correct / total}")

    bucket_name = "gcp_monitoring_animals"
    object_name = "evaluation_results.csv"
    file_path = "reports/evaluation_results.csv"

    # Construct the URL for the GCS object
    url = f"https://storage.googleapis.com/{bucket_name}/reference/{object_name}"

    # Open the file and upload it
    with open(file_path, "rb") as file_data:
        headers = {
            "Content-Type": "text/csv",  # Ensure the correct content type
        }
        response = requests.put(url, data=file_data, headers=headers)

    # Check the response
    if response.status_code == 200:
        print("File uploaded successfully.")
    else:
        print(f"Failed to upload file. Status code: {response.status_code}")
        print("Response:", response.text)


# Initialize Typer app
app = typer.Typer()


# Define the evaluate command
@app.command()
def evaluate_command(model_checkpoint: str = typer.Option(..., help="Path to the model checkpoint")):
    """Wrapper for evaluation."""
    # Initialize Hydra and load the config
    with initialize(version_base="1.1", config_path="conf"):
        cfg = compose(config_name="config.yaml")

    # Call the evaluation function
    evaluate(cfg, model_checkpoint)


if __name__ == "__main__":
    app()
