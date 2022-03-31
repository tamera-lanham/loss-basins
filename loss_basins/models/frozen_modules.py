from turtle import forward
from typing import Tuple, Union
import torch as t
import torch.nn as nn


class FrozenModule:
    def load(self, parameters):
        self.parameters = parameters


class Conv2d(FrozenModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple, None] = 1,
        padding: Union[int, Tuple, str] = 0,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.params = []

        kernel_size = nn.modules.utils._pair(kernel_size)
        k = groups / (in_channels * t.tensor(kernel_size).prod())

        weight_shape = (
            out_channels,
            in_channels // groups,
            kernel_size[0],
            kernel_size[1],
        )
        weight = (t.rand(weight_shape, device=device) - 0.5) * 2 * k.sqrt()
        weight.requires_grad = True
        self.params.append(weight)

        if bias:
            bias_shape = (out_channels,)
            bias_ = (t.rand(bias_shape, device=device) - 0.5) * 2 * k.sqrt()
            bias_.requires_grad = True
            self.params.append(bias_)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return nn.functional.conv2d(
            x,
            self.params[0],
            self.params[1],
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def from_existing(module: nn.Module) -> FrozenModule:
    if isinstance(module, nn.Conv2d):
        device = module.device if hasattr(module, "device") else None
        dtype = module.dtype if hasattr(module, "dtype") else None
        return Conv2d(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None,
            device,
            dtype,
        )

    raise NotImplementedError()


def main():

    test_layers = [nn.Conv2d(2, 6, 5, 2, 1, 2, 2)]

    for test_layer in test_layers:
        layer = from_existing(test_layer)

        assert [
            test_p.allclose(p)
            for test_p, p in zip(test_layer.parameters(), layer.params)
        ]


if __name__ == "__main__":
    main()
