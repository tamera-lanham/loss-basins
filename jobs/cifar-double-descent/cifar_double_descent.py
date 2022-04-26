from dataclasses import dataclass
from typing import Callable
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar


import os
from dataclasses import dataclass, field, asdict
import argparse

import matplotlib.pyplot as plt
import matplotlib.colors
import time
import copy

from loss_basins.models.lightning import ResNetLightning, SaveModelState


# Build a ResNet
# Load data (and transform) (and add label noise)
# Train, saving things along the way
#

# Hyperparams
batch_size = 128
label_noise = 0.2
device_type = "gpu"
epochs = 5


@dataclass
class Parameters:
    batch_size: int = 128
    label_noise: float = 0.2
    device_type: str = "cpu"

    loss_fn: Callable = F.cross_entropy
    optimizer_class: type = t.optim.SGD
    optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 1e-3})

    epochs: int = 5


class CIFAR10_label_noise(torchvision.datasets.CIFAR10):
    def __init__(self, label_noise=0, **kwargs):
        super().__init__(**kwargs)
        self.label_noise = label_noise

        self.noise_idxs = np.where(
            np.random.rand(super().__len__()) < self.label_noise
        )[0]

        self.noise_vals = np.empty(len(self.noise_idxs), dtype=int)

        self.num_classes = 10

        for i, idx in enumerate(self.noise_idxs):

            new_label = np.random.randint(self.num_classes - 1)

            if new_label == (super().__getitem__(int(idx))[1]):
                # if a randomized thing is assigned the same thing, then like don't do that
                new_label = self.num_classes - 1

            self.noise_vals[i] = new_label

    def __getitem__(self, idx):

        if idx in self.noise_idxs:

            new_label = self.noise_vals[self.noise_idxs == idx][0]

            return (super().__getitem__(idx)[0], new_label)

        else:

            return super().__getitem__(idx)


def get_dataloaders(batch_size, label_noise, data_path="./_data/cifar-10"):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset_ln = CIFAR10_label_noise(
        label_noise=label_noise,
        root=data_path,
        train=True,
        download=True,
        transform=transform_train,
    )
    trainloader_ln = t.utils.data.DataLoader(
        trainset_ln, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_path, train=False, download=True, transform=transform_test
    )
    testloader = t.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    return trainloader_ln, testloader


def train(params: Parameters, train_loader, val_loader, output_path: str):

    model = ResNetLightning(
        params.loss_fn,
        params.optimizer_class,
        params.optimizer_kwargs,
    )

    progress_bar = TQDMProgressBar()
    checkpoint = SaveModelState(output_path, every_n_epochs=1)

    trainer = pl.Trainer(
        accelerator=params.device_type,
        logger=False,
        num_processes=1,
        callbacks=[progress_bar, checkpoint],
        enable_checkpointing=False,
        max_epochs=params.epochs,
    )

    trainer.fit(model, train_loader, val_loader)

    return model, trainer


if __name__ == "__main__":
    params = Parameters(device_type="gpu", epochs=100)

    trainloader, testloader = get_dataloaders(
        batch_size=params.batch_size,
        label_noise=params.label_noise,
    )

    train(params, trainloader, testloader, output_path="./_data/cifar_double_descent")
