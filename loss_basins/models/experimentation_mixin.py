import dataclasses
import torch as t
import torch.nn as nn
from tqdm import tqdm
from typing import Callable, Generator, Iterator, Optional, Tuple


@dataclasses.dataclass
class TrainingParams:
    device: str = "cpu"
    lr: float = 1e-3
    loss_function: Callable = t.nn.CrossEntropyLoss


class ExperimentationMixin:
    def init_mixin(self, training_params: Optional[TrainingParams] = None) -> None:
        self.training_params = training_params
        self._activations = []
        self._set_activation_hooks()

    def _set_activation_hooks(self):
        def hook(model, input, output):
            self._activations.append(output.detach().clone())

        for layer in self.layers:
            layer.register_forward_hook(hook)

    def _train_generator(
        self, data_iter: Iterator[Tuple[t.Tensor, t.Tensor]]
    ) -> Generator[Tuple[t.Tensor, t.Tensor], None, None]:
        # Returns iterator of (loss, activations)
        # data is an infinite iterator of (inputs, targets)
        if not self.training_params:
            raise AttributeError("Model has no training_params")

        params = self.training_params
        self.to(params.device)
        optimizer = t.optim.SGD(self.parameters(), params.lr)
        loss_fn = params.loss_function()

        for i, (X, y) in enumerate(data_iter):
            X.to(params.device), y.to(params.device)

            optimizer.zero_grad()
            y_pred = self.forward(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            yield loss.detach(), self._activations.copy()

    def train_n_steps(
        self, data_iter: Iterator[Tuple[t.Tensor, t.Tensor]], n_steps: int
    ) -> Tuple[list[t.Tensor], list[t.Tensor]]:
        train_iter = self._train_generator(data_iter)

        losses, activations = [], []
        for _ in tqdm(range(n_steps)):
            loss, activation = next(train_iter)
            losses.append(loss)
            activations.append(activation)

        return losses, activations

    def train_to_convergence(
        self,
        data_iter: Iterator[Tuple[t.Tensor, t.Tensor]],
        threshold: float = 0.1,
        window_size: int = 10,
    ) -> Tuple[list[t.Tensor], list[t.Tensor]]:
        train_iter = self._train_generator(data_iter)

        losses, activations = [], []
        with tqdm() as pbar:
            while True:
                loss, activation = next(train_iter)
                losses.append(loss)
                activations.append(activation)

                loss_window = losses[-window_size:]
                loss_range = max(loss_window) - min(loss_window)

                pbar.update()
                pbar.set_postfix(
                    {
                        "loss": "%.4f" % float(loss.detach()),
                        "loss_range": "%.4f" % loss_range,
                    }
                )

                if len(loss_window) == window_size and loss_range <= threshold:
                    break

        return losses, activations

    def forward_no_grad(self, data: t.tensor) -> Tuple[t.Tensor, t.Tensor]:
        with t.no_grad():
            outputs = self.forward(data)
        return outputs, self._activations.copy()

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad = False

        assert [not p.requires_grad for p in self.parameters()]

    def get_params(self) -> t.Tensor:
        return t.cat([p.flatten() for p in self.parameters()]).clone().detach()

    def set_params(self, param_vector: t.Tensor) -> None:
        nn.utils.vector_to_parameters(param_vector, self.parameters())

    def add_to_params(self, param_vector):
        self.initialize(self.get_params() + param_vector)
