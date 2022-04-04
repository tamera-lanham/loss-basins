from loss_basins.utils import ExperimentationMixin, FreezableModule
import torch as t
import torch.nn as nn


class MnistConv(FreezableModule, nn.Module, ExperimentationMixin):
    def __init__(self, training_params=None):
        super().__init__()
        self.conv_layers = [
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 12, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        ]
        self.linear_layers = [nn.Linear(12 * 4 * 4, 10)]

        self.layers = nn.Sequential(
            *self.conv_layers, nn.Flatten(-3), *self.linear_layers
        )

        self.init_mixin(training_params)

    def forward(self, x):
        return self.layers(x)
