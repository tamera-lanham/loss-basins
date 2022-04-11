import os
from pathlib import Path
import pytorch_lightning as pl
from typing import Callable, Iterable, Optional
import torch as t
from torch import nn
from pytorch_lightning.callbacks import Callback
from loss_basins.models.cifar_conv import ConvNet
from loss_basins.models.mnist_conv import MnistConv


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        torch_module,
        loss_fn=nn.functional.cross_entropy,
        optimizer_class=t.optim.SGD,
        optimizer_kwargs=None,
    ):
        super().__init__()
        self.model = torch_module

        self.loss_fn = loss_fn
        self.optimizer_class = optimizer_class

        default_opt_kwargs = {"lr": 1e-3}
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer_kwargs = {**default_opt_kwargs, **optimizer_kwargs}

    def forward(self, x):
        self.model(x)

    def _get_loss(self, batch):
        X, y = batch
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        return loss

    def training_step(self, batch):
        loss = self._get_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss = self._get_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_kwargs
        )
        return optimizer


def MnistConvLightning(**kwargs):
    return LightningModel(MnistConv(), **kwargs)


def CifarConvLightning(**kwargs):
    return LightningModel(ConvNet(), **kwargs)


class SaveModelState(Callback):
    def __init__(
        self,
        output_path,
        every_n_epochs=None,
        every_n_batches=None,
        save_init=True,
        single_file=False,
    ):
        self.output_path = Path(output_path)
        self.every_n_epochs = every_n_epochs
        self.every_n_batches = every_n_batches
        self.save_init = save_init
        self.single_file = single_file

        self.data = []

        if not self.single_file:
            if not self.output_path.exists():
                os.makedirs(self.output_path)

    def _save_state(self, trainer, pl_module, epoch=None, batch=0):
        callback_metrics = trainer.callback_metrics
        if epoch is None:
            epoch = trainer.current_epoch

        data = {
            "epoch": epoch,
            "batch": batch,
            **callback_metrics,
            "state_dict": pl_module.state_dict(),
        }

        self._save_state_data(data)

    def _save_state_data(self, data):
        if self.single_file:
            self.data.append(data)
        else:
            filename = f"epoch_{data.get('epoch')}_batch_{data.get('batch')}.pt"
            t.save(data, self.output_path.joinpath(filename))

    def on_fit_end(self, _, __):
        if self.single_file:
            t.save(self.data, self.output_path)

    def on_fit_start(self, trainer, pl_module):
        if self.save_init:
            self._save_state(trainer, pl_module, epoch=-1)

    def on_train_batch_end(self, trainer, pl_module, _, __, batch_idx):
        if self.every_n_batches and batch_idx % self.every_n_batches == 0:
            self._save_state(
                trainer, pl_module, epoch=trainer.current_epoch, batch=batch_idx
            )

    def on_train_epoch_end(self, trainer, pl_module):
        if self.every_n_epochs and trainer.current_epoch % self.every_n_epochs == 0:
            self._save_state(trainer, pl_module, epoch=trainer.current_epoch)
