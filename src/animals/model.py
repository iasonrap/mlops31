import hydra
import timm
import torch
import torch.nn as nn


class AnimalModel(nn.Module):
    def __init__(self, model_name, num_classes):
        """Initialize the model with the number of classes"""
        super(AnimalModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


@hydra.main(version_base="1.1", config_path="conf", config_name="config.yaml")
def main(cfg):
    model = AnimalModel(cfg.hyperparameters.model_name, cfg.hyperparameters.num_classes)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)


if __name__ == "__main__":
    main()
