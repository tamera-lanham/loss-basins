from dataclasses import dataclass
import numpy as np
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from loss_basins.models import ResNetLightning
from loss_basins.training_jobs.training_job import TrainingJob, Metadata


@dataclass
class CifarDoubleDescentJobMetadata(Metadata):
    description: str = "Deep double descent replication using ResNet on CIFAR-10"
    epochs: int = 5
    label_noise: float = 0.2
    batch_size: int = 128
    resnet_width: int = 64


class CIFAR10_label_noise(CIFAR10):
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


class CifarDoubleDescentJob(TrainingJob):
    def model(self, metadata: CifarDoubleDescentJobMetadata):

        return ResNetLightning(k=metadata.resnet_width)

    def data_loaders(self, metadata: CifarDoubleDescentJobMetadata):

        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        data_path = "./_data/CIFAR10/"
        trainset_ln = CIFAR10_label_noise(
            label_noise=metadata.label_noise,
            root=data_path,
            train=True,
            download=True,
            transform=transform_train,
        )
        trainloader_ln = DataLoader(
            trainset_ln, batch_size=metadata.batch_size, shuffle=True, num_workers=1
        )

        testset = CIFAR10(
            root=data_path, train=False, download=True, transform=transform_test
        )
        testloader = DataLoader(
            testset, batch_size=metadata.batch_size, shuffle=False, num_workers=1
        )

        return trainloader_ln, testloader

    def trainer(
        self, init_metadata: CifarDoubleDescentJobMetadata, default_trainer_kwargs: dict
    ) -> Trainer:

        return Trainer(**default_trainer_kwargs, max_epochs=init_metadata.epochs)
