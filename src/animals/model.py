import torch
import timm
import torch.nn as nn

class AnimalModel(nn.Module):
    def __init__(self, num_classes):
        """Initialize the model with the number of classes"""
        super(AnimalModel, self).__init__()
        self.model = timm.create_model('resnet18', pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    model = AnimalModel()
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y.shape)