import pytorch_lightning as pl
import torch as t
from torch import nn
import torch.nn.functional as F
from loss_basins.models.cifar_conv import ConvNet
from loss_basins.models.mnist_conv import MnistConv
from loss_basins.models.resnet import make_resnet18k


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
        return self.model(x)

    def _get_loss(self, batch):
        X, y = batch
        y_pred = self.model(X)
        loss = self.loss_fn(y_pred, y)
        return loss, y_pred

    def training_step(self, batch, batch_idx):
        loss, _ = self._get_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        loss, y_pred = self._get_loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss, y_pred

    def configure_optimizers(self):
        optimizer = self.optimizer_class(
            self.model.parameters(), **self.optimizer_kwargs
        )
        return optimizer


def ResNetLightning(
    loss_fn=None, optimizer_class=None, optimizer_kwargs=None, **kwargs
):

    if loss_fn is None:
        loss_fn = F.cross_entropy
    if optimizer_class is None:
        optimizer_class = t.optim.SGD

    return LightningModel(
        make_resnet18k(**kwargs),
        loss_fn,
        optimizer_class,
        optimizer_kwargs,
    )


def MnistLightning(**kwargs):
    return LightningModel(MnistConv(), **kwargs)


def CifarConvLightning(**kwargs):
    return LightningModel(ConvNet(), **kwargs)
