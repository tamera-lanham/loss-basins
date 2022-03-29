from loss_basins.models.mnist_conv import MnistConv, TrainingParams
from loss_basins.data import mnist
import torch as t
import torch.nn as nn


def normalize(vec, epsilon):
    return epsilon * t.nn.functional.normalize(vec)


data = mnist(10)
training_params = TrainingParams()
model = MnistConv(training_params)
model.train_to_convergence(data, 0.05)
model.freeze()

basin_center = model.get_params()

perturbation_norm = 0.001
perturbation = t.rand(basin_center.shape)
perturbation = t.autograd.Variable(perturbation)

new_weights = basin_center + perturbation
new_model = MnistConv()
new_model.initialize(new_weights)

params = list(new_model.model.parameters())
edited_layers = []
for i, layer in enumerate(new_model.model.layers):
    if hasattr(layer, "weight"):
        print(layer.weight.shape)

        perturbation = nn.Parameter(t.rand(layer.weight.shape))
        edited_weight = layer.weight + perturbation
        layer.weight = edited_weight

    if hasattr(layer, "bias"):
        print(layer.bias.shape)

        perturbation = nn.Parameter(t.rand(layer.bias.shape))
        edited_bias = layer.bias + perturbation
        layer.bias = edited_bias


# optimizer = t.optim.SGD([perturbation])


# for i in range(100):
#     normed_perturbation = normalize(perturbation, perturbation_norm)
#     perturbed_model = addToModel(model, normed_perturbation)
