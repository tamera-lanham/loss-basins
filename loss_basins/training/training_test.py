from cgi import test
from loss_basins.training import *
from loss_basins.data import T, identity_normal, mnist
import torch as t
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 50),
            nn.ReLU(),
            nn.Linear(50, 5),
        )

    def forward(self, x):
        return self.layers(x)


def test_get_activations():
    model = Model()
    data = identity_normal((10, 5))
    training_params = TrainingParams(loss_function=nn.MSELoss(), get_activations=True)
    training_run = TrainingRun(model, data, training_params)

    losses, activations = training_run.for_n_steps(2)

    assert all([isinstance(a, t.Tensor) for a in activations[0][1:]])
    assert all([a.shape[0] == 10 for a in activations[0][1:]])
    assert not any(
        [
            a0.allclose(a1)
            for (a0, a1) in zip(activations[0], activations[1])
            if a0 is not t.nan
        ]
    )


def test_train_to_convergence():
    model = Model()
    data = identity_normal((100, 5))
    training_params = TrainingParams(loss_function=nn.MSELoss())
    training_run = TrainingRun(model, data, training_params)

    losses = training_run.to_convergence(threshold=0.2, window_size=10)
    assert sum(losses[:10]) > sum(losses[-10:])
    assert sum(losses[-10:]) / 10 <= 0.2


if __name__ == "__main__":
    test_train_to_convergence()
    test_get_activations()
