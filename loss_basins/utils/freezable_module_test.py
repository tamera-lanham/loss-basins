from loss_basins.utils.freezable_module import *


class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(nn.Conv2d(1, 4, 5), nn.ReLU(), nn.MaxPool2d(2))
        self.flatten = nn.Flatten(-3)
        self.linear_layers = nn.Linear(36, 10)

    def forward(self, x):
        y1 = self.conv_layers(x)
        y2 = self.flatten(y1)
        y3 = self.linear_layers(y2)
        return y3


def test_convert_and_freeze():
    cases = [  # List of tuples (module, input)
        (nn.Conv2d(1, 3, 5), t.randn((1, 1, 10, 10))),
        (nn.Linear(10, 10), t.randn(3, 10)),
    ]
    for module, x in cases:
        y1 = module(x)
        frozen_module = convert_to_freezable(module)
        y2 = frozen_module(x)
        frozen_module.freeze()
        y3 = frozen_module(x)
        assert y1.allclose(y2)
        assert y1.allclose(y3)


def test_convert_recursive():
    orig_model = TestModel()
    model = convert_to_freezable(orig_model)
    model.freeze()

    assert not model is orig_model
    assert not any(isinstance(m, FreezableModule) for m in orig_model.modules())
    assert all(isinstance(m, FreezableModule) for m in model.modules())
    assert all(m.is_frozen() for m in model.modules())

    x = t.randn(3, 1, 10, 10)
    y = orig_model(x)
    y2 = model(x)

    assert y.allclose(y2)


def test_load_non_recursive():

    x = t.randn((3, 1, 12, 15))
    conv = FreezableConv2d(1, 3, 5)
    y = conv(x)

    new_conv = FreezableConv2d(1, 3, 5)
    assert not y.allclose(new_conv(x))

    new_conv.freeze()
    new_conv.load_parameters(conv.parameters())
    y2 = new_conv(x)
    assert y.allclose(y2)


def test_load_recursive():
    x = t.randn(3, 1, 10, 10)
    orig_model = TestModel()
    y = orig_model(x)

    random_model = TestModel()
    frozen_model = convert_to_freezable(random_model).freeze()
    assert all(
        p1.allclose(p2)
        for (p1, p2) in zip(random_model.parameters(), frozen_model.parameters())
    )
    assert not y.allclose(frozen_model(x))

    frozen_model.load_parameters(orig_model.parameters())

    assert all(
        p1.allclose(p2)
        for (p1, p2) in zip(orig_model.parameters(), frozen_model.parameters())
    )
    assert y.allclose(frozen_model(x))


def test_keep_grad_fn():

    x = t.randn(5, 10)
    lin = FreezableLinear(10, 10)
    y = lin(x)

    new_params = [param * 3 for param in lin.parameters()]
    lin.freeze()
    lin.load_parameters(new_params)
    y2 = lin(x)

    assert all(param.grad_fn.name() == "MulBackward0" for param in lin.parameters())
    assert y2.allclose(y * 3, atol=1e-4)


def test_train_perturbation():

    x = t.randn((1, 1, 10, 10))
    conv = FreezableConv2d(1, 3, 5)
    y = conv(x)

    perturbations = [nn.Parameter(t.randn(param.shape)) for param in conv.parameters()]
    new_params = [
        param + perturb for param, perturb in zip(conv.parameters(), perturbations)
    ]
    conv.freeze()
    conv.load_parameters(new_params)

    initial_params = [p.detach().clone() for p in conv.parameters()]
    initial_perturbs = [p.detach().clone() for p in perturbations]

    y2 = conv(x)
    loss = nn.functional.mse_loss(y, y2)
    optimizer = t.optim.SGD(perturbations, lr=1e-1)
    loss.backward()
    optimizer.step()

    for initial_param, param in zip(initial_params, conv.parameters()):
        assert initial_param.allclose(param)

    for initial_perturb, perturb in zip(initial_perturbs, perturbations):
        assert not initial_perturb.allclose(perturb)


if __name__ == "__main__":
    test_convert_and_freeze()
    test_convert_recursive()
    test_load_non_recursive()
    test_load_recursive()
    test_keep_grad_fn()
    test_train_perturbation()
