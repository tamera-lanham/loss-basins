from loss_basins.models.mnist_conv import MnistConv
from loss_basins.utils import TrainingParams
from loss_basins.data import mnist
import torch as t
from tqdm import tqdm

model = MnistConv()
model.training_params = TrainingParams()

data = mnist(10)

losses, activations = model.train_to_convergence(data, 0.05)
print(losses[-1])
