from loss_basins.models import MnistConv
from loss_basins.utils import TrainingParams
from loss_basins.data import mnist
import torch as t
import torch.nn as nn


def normalize(vec, epsilon):
    return epsilon * t.nn.functional.normalize(vec, dim=0)


data = mnist(10)
training_params = TrainingParams()
model = MnistConv(training_params)
model.train_to_convergence(data, 0.05)

model.freeze()
param_vector = model.params_to_vector()

perturbation_norm = 20
perturbation = normalize(t.randn(param_vector.shape), perturbation_norm)
perturbation = nn.Parameter(perturbation)

new_params = param_vector + perturbation
model.params_from_vector(new_params)

n_steps = 500
optimizer = t.optim.SGD([perturbation], lr=training_params.lr)
loss_fn = training_params.loss_function()
losses = []
for i in range(n_steps):

    perturbation.data = normalize(perturbation, perturbation_norm)
    new_params = param_vector + perturbation
    model.params_from_vector(new_params)

    optimizer.zero_grad()
    X, y = next(data)
    y_hat = model(X)
    loss = loss_fn(y_hat, y)
    loss.backward()
    optimizer.step()
    losses.append(loss.detach())

    if i % 10 == 0:
        print(loss)
