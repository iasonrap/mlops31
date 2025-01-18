import hydra
import torch
import torch.nn as nn
from model import AnimalModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import typer
from dotenv import load_dotenv
load_dotenv()

torch.seed()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(cfg) -> None:
    """Train the model."""
    print(f"Training the model with lr={cfg.optimizer.lr}, batch_size={cfg.hyperparameters.batch_size}, epochs={cfg.hyperparameters.epochs}")
    model = AnimalModel(cfg.hyperparameters.model_name, cfg.hyperparameters.num_classes).to(DEVICE)

    # Load the data
    train_loader, val_loader, _ #TODO: Load the data

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    stats = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(cfg.hyperparameters.epochs):
        model.train()
        preds, targets = [], []

        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            stats['train_loss'].append(loss.item())

            acc = (outputs.argmax(1) == labels).float().item()
            stats['train_acc'].append(acc)

            preds.append(outputs.detach().cpu())
            targets.append(labels.detach().cpu())
            

        # Perform eval
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss = loss_fn(outputs, labels)

                stats['val_loss'].append(val_loss.item())

                acc = (outputs.argmax(1) == labels).float().mean().item()
                stats['val_acc'].append(acc)

                val_preds.append(outputs.detach().cpu())
                val_targets.append(labels.detach().cpu())

        print(f"Epoch {epoch+1}/{cfg.hyperparameters.epochs}, Train Loss: {loss.item()}, Train Acc: {acc}, Val Loss: {val_loss.item()}, Val Acc: {acc}")

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

    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    axs[0].plot(stats['train_loss'], label='Train Loss')
    axs[0].plot(stats['val_loss'], label='Val Loss')
    axs[0].set_title('Loss')
    axs[0].legend()

    axs[1].plot(stats['train_acc'], label='Train Acc')
    axs[1].plot(stats['val_acc'], label='Val Acc')
    axs[1].set_title('Accuracy')
    axs[1].legend()

    plt.savefig("reports/figures/training_plot.png")

@hydra.main(version_base="1.1", config_path="conf", config_name="config.yaml")
def main(cfg):
    typer.run(train(cfg))


if __name__ == '__main__':
    main()    
    