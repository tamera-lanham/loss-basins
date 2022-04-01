from collections import OrderedDict
from copy import deepcopy
from typing import Iterator, Tuple
import torch as t
import torch.nn as nn


class FrozenMixin(nn.Module):
    def __init__(self):
        super().__init__()

    def is_frozen(self) -> bool:
        if not hasattr(self, "_frozen"):
            self._frozen = False
        return self._frozen

    def freeze(self, recurse=True):
        self.frozen_params = {
            name: param.detach().clone()
            for name, param in self.named_parameters(recurse=False)
        }
        if not self.is_frozen():
            self._frozen = True

        if recurse:
            for child_module in self.modules():
                if child_module is not self:
                    child_module.freeze(False)
        return self

    def named_parameters(self, recurse: bool = True) -> Iterator[t.Tensor]:
        if self.is_frozen():
            get_members_fn = lambda module: module.frozen_params.items()
        else:
            get_members_fn = lambda module: module._parameters.items()

        gen = self._named_members(get_members_fn, recurse=recurse)
        for elem in gen:
            yield elem

    def _load_named_parameters(self, parameters: Iterator[Tuple[str, t.Tensor]]):
        if not self.is_frozen():
            raise RuntimeError(
                "Cannot load parameters into a module that has not been frozen."
            )

        existing_params = self.frozen_params.items()
        new_params = []
        for (input_param_name, input_param), (
            existing_param_name,
            existing_param,
        ) in zip(parameters, existing_params):
            assert input_param_name == existing_param_name
            assert input_param.shape == existing_param.shape
            assert input_param.dtype == existing_param.dtype

            new_params.append([input_param_name, input_param])

        self.frozen_params = dict(new_params)

    def load_named_parameters(self, parameters: Iterator[Tuple[str, t.Tensor]]):
        child_params = {}
        self_params = []
        for name, param in parameters:
            if "." in name:
                child_name, rest_of_names = name.split(".", 1)
                if child_name not in child_params:
                    child_params[child_name] = []
                child_params[child_name].append((rest_of_names, param))
            else:
                self_params.append((name, param))

        for child_name, child_module in self.named_children():
            if child_name in child_params:
                if not isinstance(child_module, FrozenMixin):
                    raise RuntimeError(
                        f"Trying to load parameters into non-FrozenMixin module {child_module}"
                    )
                child_module.load_named_parameters(child_params[child_name])

        self._load_named_parameters(self_params)

    def load_parameters(self, parameters: Iterator[t.Tensor]):
        named_parameters = zip(
            (name for name, _ in self.named_parameters()), parameters
        )
        self.load_named_parameters(named_parameters)


class Conv2d(nn.Conv2d, FrozenMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def convert(cls, nn_conv2d: nn.Conv2d):
        instance = cls(
            in_channels=nn_conv2d.in_channels,
            out_channels=nn_conv2d.out_channels,
            kernel_size=nn_conv2d.kernel_size,
            stride=nn_conv2d.stride,
            padding=nn_conv2d.padding,
            dilation=nn_conv2d.dilation,
            groups=nn_conv2d.groups,
            bias=nn_conv2d.bias is not None,
            device=getattr(nn_conv2d, "device", None),
            dtype=getattr(nn_conv2d, "dtype", None),
        )
        instance.load_state_dict(nn_conv2d.state_dict())
        return instance

    def forward(self, x):
        if self.is_frozen():
            return nn.functional.conv2d(
                x,
                self.frozen_params.get("weight"),
                self.frozen_params.get("bias"),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        return super().forward(x)


class Linear(nn.Linear, FrozenMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def convert(cls, nn_linear: nn.Linear):
        instance = cls(
            in_features=nn_linear.in_features,
            out_features=nn_linear.out_features,
            bias=nn_linear.bias is not None,
            device=getattr(nn_linear, "device", None),
            dtype=getattr(nn_linear, "dtype", None),
        )
        instance.load_state_dict(nn_linear.state_dict())
        return instance

    def forward(self, x):
        if self.is_frozen():
            return nn.functional.linear(
                x, self.frozen_params.get("weight"), self.frozen_params.get("bias")
            )

        return super().forward(x)


class Sequential(nn.Sequential, FrozenMixin):
    pass


def convert(module: nn.Module) -> FrozenMixin:
    """Convert this module and all its submodules to a FrozenMixin version (where appropriate)"""
    module_mapping = {nn.Conv2d: Conv2d, nn.Linear: Linear}

    if type(module) in module_mapping:
        return module_mapping[type(module)].convert(module)

    if isinstance(module, nn.Sequential):
        # Assumes that no other modules have been added outside of _modules
        return Sequential(
            OrderedDict(
                (name, convert(child)) for name, child in module.named_children()
            )
        )

    module = deepcopy(module)
    module.__class__ = type(
        "FrozenModule", (module.__class__, FrozenMixin), {}
    )  # type: ignore

    for name, child in module.named_children():
        if not hasattr(module, name):
            raise RuntimeError(
                f"Could not find child attribute {name} in module {module}."
            )
        setattr(module, name, convert(child))

    return module


###########################################################################


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
        frozen_module = convert(module)
        y2 = frozen_module(x)
        frozen_module.freeze()
        y3 = frozen_module(x)
        assert y1.allclose(y2)
        assert y1.allclose(y3)


def test_convert_recursive():
    orig_model = TestModel()
    model = convert(orig_model)
    model.freeze()

    assert not model is orig_model
    assert not any(isinstance(m, FrozenMixin) for m in orig_model.modules())
    assert all(isinstance(m, FrozenMixin) for m in model.modules())
    assert all(m.is_frozen() for m in model.modules())

    x = t.randn(3, 1, 10, 10)
    y = orig_model(x)
    y2 = model(x)

    assert y.allclose(y2)


def test_load_non_recursive():

    x = t.randn((3, 1, 12, 15))
    conv = Conv2d(1, 3, 5)
    y = conv(x)

    new_conv = Conv2d(1, 3, 5)
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
    frozen_model = convert(random_model).freeze()
    assert all(
        p1.allclose(p2)
        for (p1, p2) in zip(random_model.parameters(), frozen_model.parameters())
    )
    assert not y.allclose(frozen_model(x))

    frozen_model.load_parameters(orig_model.parameters())
    assert y.allclose(frozen_model(x))


def test_keep_grad_fn():

    x = t.randn(5, 10)
    lin = Linear(10, 10)
    y = lin(x)

    new_params = [param * 3 for param in lin.parameters()]
    lin.freeze()
    lin.load_parameters(new_params)
    y2 = lin(x)

    assert all(param.grad_fn.name() == "MulBackward0" for param in lin.parameters())
    assert y2.allclose(y * 3, atol=1e-4)


def test_train_perturbation():

    x = t.randn((1, 1, 10, 10))
    conv = Conv2d(1, 3, 5)
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


if True:
    test_convert_and_freeze()
    test_convert_recursive()
    test_load_non_recursive()
    test_load_recursive()
    test_keep_grad_fn()
    test_train_perturbation()
