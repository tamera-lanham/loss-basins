from typing import Any, Callable, Iterator, Optional, Tuple, TypeVar, Union
import torch as t
import torchvision

T = TypeVar("T")  # Typically Tuple[t.Tensor, t.Tensor] for (X, y) data


class Dataset:
    def __init__(
        self,
        generator_fn: Callable[[Any], Iterator[T]],
        *generator_args,
        **generator_kwargs
    ):
        # generator_fn is expected to generate one epoch of data in batches
        self.generator_fn = generator_fn
        self.args = generator_args
        self.kwargs = generator_kwargs

    def __call__(
        self, n_epochs: Optional[int] = None, include_epoch_count=True
    ) -> Union[Iterator[Tuple[int, T]], Iterator[T]]:
        if n_epochs is not None:
            return self.multi_epoch(n_epochs, include_epoch_count)
        return self.inifinite_epochs(include_epoch_count)

    def single_epoch(self) -> Iterator[T]:
        return self.generator_fn(*self.args, **self.kwargs)

    def multi_epoch(
        self, n_epochs: int, include_epoch_count=True
    ) -> Union[Iterator[Tuple[int, T]], Iterator[T]]:
        for i in range(n_epochs):
            for x in self.single_epoch():
                if include_epoch_count:
                    yield i, x
                else:
                    yield x

    def inifinite_epochs(
        self, include_epoch_count=True
    ) -> Union[Iterator[Tuple[int, T]], Iterator[T]]:
        i = 0
        while True:
            for x in self.single_epoch():
                if include_epoch_count:
                    yield i, x
                else:
                    yield x
            i += 1


def random_normal(
    input_shape: tuple[int], output_shape: tuple[int], n_batches=1000
) -> Dataset:
    kwargs = locals()

    def epoch_generator(**kwargs) -> Iterator[tuple[t.Tensor, t.Tensor]]:
        for _ in range(n_batches):
            inputs = t.randn(input_shape)
            outputs = t.randn(output_shape)
            yield inputs, outputs

    return Dataset(epoch_generator, **kwargs)


def identity_normal(
    shape: tuple[int], n_batches=1000
) -> Iterator[tuple[t.Tensor, t.Tensor]]:
    kwargs = locals()

    def epoch_generator(**kwargs) -> Iterator[tuple[t.Tensor, t.Tensor]]:
        for _ in range(n_batches):
            data = t.randn(shape)
            yield data, data

    return Dataset(epoch_generator, **kwargs)


def fake_mnist(batch_size: int, train=True):
    kwargs = locals()

    def epoch_generator(**kwargs) -> Iterator[tuple[t.Tensor, t.Tensor]]:

        input_batch = lambda batch_size: t.rand(
            (batch_size, 1, 28, 28), dtype=t.float32
        )
        output_batch = lambda batch_size: t.randint(
            0, 10, (batch_size,), dtype=t.float32
        )

        total_samples = 60000 if train else 10000
        n_standard_batches = total_samples // batch_size

        for _ in range(n_standard_batches):
            yield input_batch(batch_size), output_batch(batch_size)

        final_batch_size = total_samples % batch_size
        if final_batch_size:
            yield input_batch(final_batch_size), output_batch(final_batch_size)

    return Dataset(epoch_generator, **kwargs)


def _mnist_loader(batch_size: int, train=True) -> Iterator[tuple[t.Tensor, t.Tensor]]:

    dataset = torchvision.datasets.MNIST(
        root="./_data",
        train=train,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    loader = t.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def mnist(batch_size: int, train=True):
    kwargs = locals()

    def epoch_generator(**kwargs) -> Iterator[tuple[t.Tensor, t.Tensor]]:
        data = _mnist_loader(batch_size, train=train)

        for X, y in data:
            yield X, y

    return Dataset(epoch_generator, **kwargs)
