from typing import Any, Callable, Iterator
import torch as t
import torchvision


def random_normal(
    input_shape: tuple[int], output_shape: tuple[int], n_batches=1000
) -> Iterator[tuple[t.Tensor, t.Tensor]]:
    for _ in n_batches:
        inputs = t.randn(input_shape)
        outputs = t.randn(output_shape)
        yield inputs, outputs


def identity_normal(
    shape: tuple[int], n_batches=1000
) -> Iterator[tuple[t.Tensor, t.Tensor]]:
    for _ in n_batches:
        data = t.randn(shape)
        yield data, data


def fake_mnist(batch_size: int, train=True) -> Iterator[tuple[t.Tensor, t.Tensor]]:

    input_batch = lambda batch_size: t.rand((batch_size, 1, 28, 28), dtype=t.float32)
    output_batch = lambda batch_size: t.randint(0, 10, (batch_size,), dtype=t.float32)

    total_samples = 60000 if train else 10000
    n_standard_batches = total_samples // batch_size

    for _ in range(n_standard_batches):
        yield input_batch(batch_size), output_batch(batch_size)

    final_batch_size = total_samples % batch_size
    if final_batch_size:
        yield input_batch(final_batch_size), output_batch(final_batch_size)


def _mnist_loader(batch_size: int, train=True) -> Iterator[tuple[t.Tensor, t.Tensor]]:

    dataset = torchvision.datasets.MNIST(
        root="./_data",
        train=train,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    loader = t.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def mnist(batch_size: int, train=True) -> Iterator[tuple[t.Tensor, t.Tensor]]:
    data = _mnist_loader(batch_size, train=train)

    for X, y in data:
        yield X, y
