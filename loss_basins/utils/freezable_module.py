from copy import deepcopy
from typing import Iterator, Tuple
import torch as t
import torch.nn as nn


class FreezableModule:
    def is_frozen(self) -> bool:
        if not hasattr(self, "_frozen"):
            self._frozen = False
        return self._frozen

    def freeze(self, recurse=True):
        self.convert_submodules_to_freezable()

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
                if not isinstance(child_module, FreezableModule):
                    raise RuntimeError(
                        f"Trying to load parameters into non-FreezableModule module {child_module}"
                    )
                child_module.load_named_parameters(child_params[child_name])

        self._load_named_parameters(self_params)

    def load_parameters(self, parameters: Iterator[t.Tensor]):
        named_parameters = zip(
            (name for name, _ in self.named_parameters()), parameters
        )
        self.load_named_parameters(named_parameters)

    @classmethod
    def convert(cls, module: nn.Module) -> "FreezableModule":
        module_mapping = {nn.Conv2d: FreezableConv2d, nn.Linear: FreezableLinear}

        if isinstance(module, FreezableModule):
            return module

        if type(module) in module_mapping:
            return module_mapping[type(module)].convert(module)

        if next(module.parameters(recurse=False), None) is not None:
            param_str = ", ".join([n for (n, p) in module.named_parameters()])
            raise RuntimeError(
                f"Unable to convert to FreezableModule: module {module.__class__} with parameters {param_str}"
            )

        module = deepcopy(module)
        module.__class__ = type(
            "FrozenModule", (FreezableModule, module.__class__), {}
        )  # type: ignore
        return module

    def convert_submodules_to_freezable(self):
        """Convert all submodules to FreezableModules"""

        for name, child in self.named_children():
            if not hasattr(self, name):
                raise RuntimeError(
                    f"Could not find child module {name} as an attribute on module {self}."
                )

            if not isinstance(getattr(self, name), nn.Module):
                raise RuntimeError(
                    f"Module {self} attribute {name} was type {type(name)}, not torch.nn.Module."
                )

            setattr(self, name, self.convert(child))
            getattr(self, name).convert_submodules_to_freezable()


class FreezableConv2d(FreezableModule, nn.Conv2d):
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


class FreezableLinear(FreezableModule, nn.Linear):
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
