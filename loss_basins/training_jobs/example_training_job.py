from dataclasses import dataclass
import pytorch_lightning as pl
import torch as t
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from loss_basins.models import LightningModel
from loss_basins.training_jobs.training_job import TrainingJob, Metadata


class IdentityDataset(Dataset):
    def __init__(self, shape, n_batches):
        self.data = [t.randn(shape) for _ in range(n_batches)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        X = self.data[idx]
        return (X, X)


@dataclass
class ExampleJobMetadata(Metadata):
    n_inputs: int = 16
    batch_size: int = 32
    n_batches: int = 500
    epochs: int = 5


class ExampleTrainingJob(TrainingJob):
    def model(self, metadata: ExampleJobMetadata) -> pl.LightningModule:

        input_size = metadata.n_inputs
        hidden_size = metadata.n_inputs * 4
        output_size = metadata.n_inputs

        torch_model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        return LightningModel(torch_model, loss_fn=nn.MSELoss())

    def data_loaders(self, metadata: ExampleJobMetadata):

        train_loader = DataLoader(
            IdentityDataset(
                (metadata.batch_size, metadata.n_inputs), metadata.n_batches
            )
        )
        val_loader = DataLoader(
            IdentityDataset(
                (metadata.batch_size, metadata.n_inputs), metadata.n_batches // 5
            )
        )

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
