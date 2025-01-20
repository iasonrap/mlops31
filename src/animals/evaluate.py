import torch
import typer
from model import AnimalModel
from data import split_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model_checkpoint: str, give_res: bool = False) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    print(model_checkpoint)

    model = AnimalModel().to(DEVICE)
    model.load_state_dict(torch.load(model_checkpoint))
    model.to(DEVICE)
    _, test_set, _ = split_dataset()

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    model.eval()
    correct, total = 0, 0

    for img, target in test_dataloader:
        img, target = img.to(DEVICE), target.to(DEVICE)
        output = model(img)
        correct += (output.argmax(1) == target).float().sum().item()
        total += target.size(0)
        
    print(f"Accuracy: {correct / total}")
    if give_res:
        return correct / total


if __name__ == "__main__":
    typer.run(evaluate)
