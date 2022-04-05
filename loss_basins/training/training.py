from loss_basins.data import Dataset
from loss_basins.utils import Activations
from dataclasses import dataclass, field
import torch as t
import torch.nn as nn
from tqdm import tqdm
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple, Union


@dataclass
class TrainingParams:
    device: str = "cpu"
    lr: float = 1e-3
    optimizer: type = t.optim.SGD
    loss_function: Callable = nn.CrossEntropyLoss()
    get_activations: bool = False
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)


class _IterRunner:
    def __init__(self, get_activations: bool):
        self.get_activations = get_activations
        self.losses = []
        self.activations = []

    def run(self, train_iter, pbar=None):
        for loss, activation in train_iter:
            self.losses.append(loss)
            if self.get_activations:
                self.activations.append(activation)
            if pbar:
                pbar.set_postfix({"loss": f"{loss:.4f}"})

    def outputs(self):
        if self.get_activations:
            return self.losses, self.activations
        return self.losses


class TrainingRun:
    def __init__(
        self,
        model: nn.Module,
        data: Dataset,
        training_params: Optional[TrainingParams] = None,
        params_to_optimize: Optional[Iterable[t.Tensor]] = None,
    ):
        self.model = model
        self.data = data
        self.params = TrainingParams() if training_params is None else training_params

        if params_to_optimize is None:
            params_to_optimize = self.model.parameters()
        self.optimizer = self.params.optimizer(
            params_to_optimize, lr=self.params.lr, **self.params.optimizer_kwargs
        )

        if self.params.get_activations:
            self.activations = Activations(model)
        else:
            self.activations = None

    def one_step(self, X: t.Tensor, y: t.Tensor) -> float:

        X.to(self.params.device), y.to(self.params.device)
        self.optimizer.zero_grad()
        y_pred = self.model.forward(X)
        loss = self.params.loss_function(y_pred, y)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _train_generator(
        self, data_iter: Iterator[Tuple[t.Tensor, t.Tensor]]
    ) -> Iterator[Tuple[float, Optional[t.Tensor]]]:
        # Returns iterator of (loss, activations)

        self.model.to(self.params.device)

        for X, y in data_iter:
            loss = self.one_step(X, y)

            if self.params.get_activations:
                yield loss, self.activations.get()
            else:
                yield loss, None

    def for_n_steps(
        self, n_steps: int
    ) -> Union[list[float], Tuple[list[float], list[t.Tensor]]]:
        iter_runner = _IterRunner(self.params.get_activations)

        train_iter = self._train_generator(self.data.batches(n_steps))
        with tqdm(train_iter, total=n_steps) as pbar:
            iter_runner.run(pbar, pbar)

        return iter_runner.outputs()

    def for_n_epochs(
        self, n_epochs: int
    ) -> Union[list[float], Tuple[list[float], list[t.Tensor]]]:
        iter_runner = _IterRunner(self.params.get_activations)

        with tqdm(range(n_epochs), total=n_epochs) as pbar:
            for epoch in pbar:
                train_iter = self._train_generator(self.data.one_epoch())
                iter_runner.run(train_iter, pbar)

        return iter_runner.outputs()

    def to_convergence(
        self, threshold: float = 0.1, window_size: int = 10
    ) -> Tuple[list[float], list[t.Tensor]]:

        train_iter = self._train_generator(self.data.inifinite_epochs(False))

        losses, activations = [], []
        with tqdm(train_iter) as pbar:
            for i, (loss, activation) in enumerate(pbar):
                losses.append(loss)
                activations.append(activation)
                pbar.set_postfix({"loss": f"{loss:.4f}"})

                avg_loss = sum(losses[-window_size:]) / min(i + 1, window_size)
                if i >= window_size and avg_loss <= threshold:
                    break

        if self.params.get_activations:
            return losses, activations
        return losses
