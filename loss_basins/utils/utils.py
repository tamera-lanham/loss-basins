from loss_basins.utils.freezable_module import FreezableModule
import torch as t
import torch.nn as nn
from typing import Callable, Union


class Activations:
    def __init__(self, model: nn.Module):
        n_modules = len(list(model.modules()))
        self._activations = [t.nan] * n_modules
        self._hook_handles = [None] * n_modules
        self.set_hooks(model)

    def set_hooks(self, model: nn.Module):
        def outer_hook(i: int) -> Callable:
            def hook(module, input, output):
                self._activations[i] = output.detach().clone()

            return hook

        for i, module in enumerate(model.modules()):
            handle = module.register_forward_hook(outer_hook(i))
            self._hook_handles[i] = handle

    def get(self) -> list[Union[t.Tensor, float]]:
        return self._activations.copy()

    def remove_hooks(self):
        [handle.remove() for handle in self._hook_handles]

    def __del__(self):
        self.remove_hooks()


def params_to_vector(model) -> t.Tensor:
    flat_params = [param.flatten() for param in model.parameters()]
    return t.cat(flat_params)


def params_from_vector(model, vector: t.Tensor):
    assert len(vector.shape) == 1

    if isinstance(model, FreezableModule) and model.is_frozen():
        shapes = [param.shape for param in model.parameters()]
        widths = [t.tensor(shape).prod().item() for shape in shapes]
        vector_split = vector.split(widths)
        new_params = [
            split.reshape(shape) for split, shape in zip(vector_split, shapes)
        ]

        model.load_parameters(new_params)

    else:
        nn.utils.vector_to_parameters(vector, model.parameters())


def n_params(model):
    return sum([p.numel() for p in model.parameters()])
