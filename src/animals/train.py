from pathlib import Path
import os
import hydra
import torch
import torch.nn as nn
from model import AnimalModel
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import typer
from dotenv import load_dotenv
from data import split_dataset
from google.cloud import storage
import wandb

print("Current working directory:", os.getcwd())
print("Contents of the directory:", os.listdir(os.getcwd()))

data_dir = Path("data/raw/raw-img/")
if not data_dir.exists():
    raise FileNotFoundError(f"Directory {data_dir} does not exist.")

load_dotenv()
api_key = os.getenv("WANDB_API_KEY")
entity = os.getenv("WANDB_ENTITY")
project = os.getenv("WANDB_PROJECT")

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    #print("Authenticating...")
    storage_client = storage.Client()
    #print("Finding bucket...")
    bucket = storage_client.bucket(bucket_name)
    #print("Creating blob...")
    blob = bucket.blob(destination_blob_name)
    #print("Uploading file...")
    blob.upload_from_filename(source_file_name)

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )

torch.manual_seed(42)

wandb.login(key=api_key)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(cfg) -> None:
    """Train the model."""
    print(f"Training the model with lr={cfg.optimizer.lr}, batch_size={cfg.hyperparameters.batch_size}, epochs={cfg.hyperparameters.epochs}")
    model = AnimalModel(cfg.hyperparameters.model_name, cfg.hyperparameters.num_classes).to(DEVICE)

    run = wandb.init(project=project, entity=entity, config={"lr": cfg.optimizer.lr, "batch_size": cfg.hyperparameters.batch_size, "epochs": cfg.hyperparameters.epochs})
    # Set working dir
    os.chdir(hydra.utils.get_original_cwd())
    train_dataset, _, val_dataset = split_dataset(Path(hydra.utils.get_original_cwd() + "/data/raw/raw-img"), 
                                                  mean=torch.tensor([0.5177, 0.5003, 0.4126]), std=torch.tensor([0.2659, 0.2610, 0.2785]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg.hyperparameters.batch_size, shuffle=False)

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    stats = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        preds, targets = [], []

        for i, (images, labels) in enumerate(train_dataloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            stats['train_loss'].append(loss.item())

            acc = (outputs.argmax(1) == labels).float().mean().item()
            stats['train_acc'].append(acc)

            preds.append(outputs.detach().cpu())
            targets.append(labels.detach().cpu())
            
            wandb.log({"train_loss": loss.item(), "train_accuracy_batches": acc})

            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{cfg.hyperparameters.epochs}, Step {i}/{len(train_dataloader)}, Loss: {loss.item()}, Acc: {acc}")

        # Perform eval
        model.eval()
        val_preds, val_targets = [], []
        total_val_loss, total_val_acc = 0.0, 0.0  # Initialize accumulators
        num_batches = 0  # To track the number of batches

        with torch.no_grad():
            for _, (images, labels) in enumerate(val_dataloader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss = loss_fn(outputs, labels)

                # Accumulate loss and accuracy
                total_val_loss += val_loss.item()
                val_acc = (outputs.argmax(1) == labels).float().mean().item()
                total_val_acc += val_acc
                num_batches += 1

                # Append predictions and targets for further analysis
                val_preds.append(outputs.detach().cpu())
                val_targets.append(labels.detach().cpu())

            # Compute average loss and accuracy
            avg_val_loss = total_val_loss / num_batches
            avg_val_acc = total_val_acc / num_batches

            # Log the average metrics to wandb
            wandb.log({"val_loss": avg_val_loss, "val_accuracy_epochs": avg_val_acc})

        print(
            f"Epoch {epoch+1}/{cfg.hyperparameters.epochs}, Train Loss: {loss.item()}, Train Acc: {acc:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}"
        )

        preds = torch.cat(preds, 0)
        targets = torch.cat(targets, 0)
        val_preds = torch.cat(val_preds, 0)
        val_targets = torch.cat(val_targets, 0)

    print("Training complete.")

    final_train_acc = accuracy_score(targets, preds.argmax(1))
    final_val_acc = accuracy_score(val_targets, val_preds.argmax(1))
    final_train_f1 = f1_score(targets, preds.argmax(1), average="weighted")
    final_val_f1 = f1_score(val_targets, val_preds.argmax(1), average="weighted")

    print("Saving model locally...")
    torch.save(model.state_dict(), "models/AnimalModel.pth")

    print("Saving model to google cloud bucket...")
    try:
        upload_blob("31animals", "models/AnimalModel.pth", "models/AnimalModel.pth")
    except Exception as e:
        print("Failed to upload model to google cloud bucket. Error:", e)

    artifact = wandb.Artifact(
        name="animals_resnet_model",
        type="model",
        description="A model trained to classify pictures of animals in 10 different classes",
        metadata={
            "train_accuracy": final_train_acc,
            "train_f1": final_train_f1,
            "val_accuracy": final_val_acc,
            "val_f1": final_val_f1,
            "num_parameters": sum(p.numel() for p in model.parameters()),
        },
    )

    artifact.add_file("models/AnimalModel.pth")
    run.log_artifact(artifact)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Plot the training loss as a line
    axs[0].plot(stats['train_loss'], label='Train Loss', linestyle='-', color='blue')
    # Add validation loss as scatter points
    axs[0].scatter(range(len(stats['val_loss'])), stats['val_loss'], label='Val Loss', color='red', marker='x')
    axs[0].set_title('Loss')
    axs[0].legend()

    # Plot the training accuracy as a line
    axs[1].plot(stats['train_acc'], label='Train Acc', linestyle='-', color='blue')
    # Add validation accuracy as scatter points
    axs[1].scatter(range(len(stats['val_acc'])), stats['val_acc'], label='Val Acc', color='red', marker='x')
    axs[1].set_title('Accuracy')
    axs[1].legend()

    # Save the figure
    fig.savefig("reports/figures/training_plot.png")

    wandb.log({"training_plot": wandb.Image("reports/figures/training_plot.png")})


@hydra.main(version_base="1.1", config_path="conf", config_name="config.yaml")
def main(cfg):
    typer.run(train(cfg))


if __name__ == '__main__':
    main()    
    