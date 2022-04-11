from loss_basins.models.mnist_conv import MnistConv
from loss_basins.training import TrainingRun
from loss_basins.data import mnist
import torch as t
from tqdm import tqdm

model = MnistConv()
data = mnist(32)

training_run = TrainingRun(model, data)


losses = training_run.to_convergence(0.2)
