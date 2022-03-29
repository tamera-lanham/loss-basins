import torch as t
import torchvision


def random_normal(input_shape: tuple[int], output_shape: tuple[int]):
    while True:
        inputs = t.randn(input_shape)
        outputs = t.randn(output_shape)
        yield inputs, outputs


def identity_normal(shape: tuple[int]):
    while True:
        data = t.randn(shape)
        yield data, data


def fake_mnist(batch_size):
    input_shape, input_dtype = (batch_size, 1, 28, 28), t.float32
    output_shape, ouptut_dtype = (batch_size,), t.int64

    while True:
        inputs = t.rand(input_shape, dtype=input_dtype)
        outputs = t.randint(0, 10, output_shape, dtype=ouptut_dtype)
        yield inputs, outputs


def _mnist_loader(batch_size, train=True):

    dataset = torchvision.datasets.MNIST(
        root="./_data",
        train=train,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    loader = t.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


def mnist(batch_size, train=True):
    while True:
        data = _mnist_loader(batch_size, train=True)

        for X, y in data:
            yield X, y
