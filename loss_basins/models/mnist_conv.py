import torch as t
import torch.nn as nn


class MnistConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 12, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(-3),
            nn.Linear(12 * 4 * 4, 10),
        )

    def forward(self, x):
        return self.layers(x)
