import torch as t
from loss_basins.models.lightning import ResNetLightning

model = t.load("./_data/cifar_double_descent/epoch_0_batch_0.pt")
