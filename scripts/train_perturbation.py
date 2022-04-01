from loss_basins.models import MnistConv
from loss_basins.utils import TrainingParams, convert_to_freezable
from loss_basins.data import mnist
import torch as t
import torch.nn as nn


def normalize(vec, epsilon):
    return epsilon * t.nn.functional.normalize(vec)


data = mnist(10)
training_params = TrainingParams()
model = convert_to_freezable(MnistConv(training_params))
model.train_to_convergence(data, 0.05)

param_vector = model.params_to_vector()
param_vector
model.freeze()


# optimizer = t.optim.SGD([perturbation])


# for i in range(100):
#     normed_perturbation = normalize(perturbation, perturbation_norm)
#     perturbed_model = addToModel(model, normed_perturbation)
