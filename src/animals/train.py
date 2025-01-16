import torch
import torch.nn as nn


from model import AnimalModel

torch.seed()

model = AnimalModel(10)
