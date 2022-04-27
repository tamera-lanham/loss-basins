from dataclasses import dataclass
from typing import Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar

from loss_basins.callbacks import SaveModelState
from loss_basins.data import mnist_loader
from loss_basins.models import MnistLightning
from loss_basins.training_jobs.training_job import TrainingJob, Metadata


@dataclass
class ExampleJobMetadata(Metadata):
    lr: float = 1e-3
    epochs: int = 3


class ExampleTrainingJob(TrainingJob):
    def model(self, metadata: ExampleJobMetadata):
        return MnistLightning()

    def data_loaders(self, metadata: ExampleJobMetadata):
        train_loader = mnist_loader(100, train=True, num_workers=8)
        val_loader = mnist_loader(100, train=False, num_workers=1)
        return train_loader, val_loader

    def trainer(self, init_metadata: ExampleJobMetadata):

        trainer = pl.Trainer(
            accelerator="cpu",
            logger=False,
            num_processes=1,
            enable_checkpointing=False,
            max_epochs=init_metadata.epochs,
        )

        return trainer

    def run_init(self, init_metadata: ExampleJobMetadata, trainer: pl.Trainer):
        train_loader, val_loader = self.data_loaders(init_metadata)
        model = self.model(init_metadata)

        trainer.fit(model, train_loader, val_loader)
