from loss_basins.data import identity_normal
from loss_basins.utils.utils import *
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(5, 50),
    nn.ReLU(),
    nn.Linear(50, 5),
)


def test_activations():

    data = identity_normal((10, 5))
    batches = list(data.batches(2))
    activations_holder = Activations(model)

    (X0, y0), (X1, y1) = batches
    model(X0)
    activations_0 = activations_holder.get()

    model(X1)
    activations_1 = activations_holder.get()

    assert all(a.shape[0] == 10 for a in activations_0)
    assert all(a0.shape == a1.shape for (a0, a1) in zip(activations_0, activations_1))
    assert not any(a0.allclose(a1) for (a0, a1) in zip(activations_0, activations_1))

    activations_holder.remove_hooks()
    model(X0)
    activations_2 = activations_holder.get()
    assert all(a1.allclose(a2) for (a1, a2) in zip(activations_1, activations_2))
    assert not any(a0.allclose(a2) for (a0, a2) in zip(activations_0, activations_2))


def test_param_vector():
    params = [p.clone() for p in model.parameters()]
    vector = params_to_vector(model)

    vec_rand = t.rand(vector.shape)
    params_from_vector(model, vec_rand)
    vector_2 = params_to_vector(model)
    assert not vector.allclose(vector_2)
    assert vector_2.equal(vec_rand)

    params_from_vector(model, vector)
    assert all([p1.allclose(p2) for (p1, p2) in zip(params, model.parameters())])


if __name__ == "__main__":
    test_activations()
    test_param_vector()
