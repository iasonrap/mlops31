import hydra
import torch
import timm
import torch.nn as nn

@hydra.main(config_name="conf/config.yaml")
class AnimalModel(nn.Module):
    def __init__(self, cfg):
        """Initialize the model with the number of classes"""
        super(AnimalModel, self).__init__()
        self.model = timm.create_model(cfg.hyperparameters.model_name, pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, cfg.hyperparameters.num_classes)
        print(cfg.hyperparameters.model_name)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = AnimalModel()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)