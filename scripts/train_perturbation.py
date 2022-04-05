from loss_basins.models import MnistConv
from loss_basins.training import TrainingRun
from loss_basins.data import mnist
import torch as t
import torch.nn as nn

from loss_basins.utils import (
    FreezableModule,
    params_from_vector,
    params_to_vector,
    freezable_to_normal,
)
from loss_basins.utils.utils import save_model


def normalize(vec, epsilon):
    return epsilon * t.nn.functional.normalize(vec, dim=0)


data = mnist(10)
model = MnistConv()
training_run = TrainingRun(model, data)
training_run.to_convergence(threshold=0.2)

f_model = FreezableModule.convert(model).freeze()
param_vector = params_to_vector(f_model)

perturbation_norm = 20
perturbation = normalize(t.randn(param_vector.shape), perturbation_norm)
perturbation = nn.Parameter(perturbation)

new_params = param_vector + perturbation
params_from_vector(model, new_params)


class PerturbationTrainingRun(TrainingRun):
    def one_step(self, X: t.Tensor, y: t.Tensor) -> float:

        perturbation.data = normalize(perturbation, perturbation_norm)
        new_params = param_vector + perturbation
        params_from_vector(self.model, new_params)

        return super().one_step(X, y)


training_run = PerturbationTrainingRun(f_model, data, params_to_optimize=[perturbation])
losses = training_run.to_convergence(0.2)


save_model(freezable_to_normal(f_model, MnistConv()))
