from loss_basins.models.model_wrapper import ModelWrapper
import torch as t
import torch.nn as nn


class _MnistConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 12, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.linear_layers = nn.Sequential(nn.Linear(12 * 4 * 4, 10))

        self.layers = nn.Sequential(
            *self.conv_layers, nn.Flatten(-3), *self.linear_layers
        )

    def forward(self, x):
        return self.layers(x)


def MnistConv(**args):
    return ModelWrapper(_MnistConv(), **args)
