from abc import ABC, abstractmethod
import dataclasses
import torch as t
from typing import Callable, Generator, Iterator, Optional, Tuple


@dataclasses.dataclass
class TrainingParams:
    device: str = "cpu"
    lr: float = 1e-3
    loss_function: Callable = t.nn.CrossEntropyLoss


class ModelBase(ABC):
    @abstractmethod
    def __init__(self, training_params: Optional[TrainingParams]):
        raise NotImplementedError()

    @abstractmethod
    def _train_generator(
        self, data_iter: Iterator[Tuple[t.Tensor, t.Tensor]]
    ) -> Generator[Tuple[t.Tensor, t.Tensor], None, None]:
        # Train for n_steps, data is an infinite iterator of (inputs, targets)
        raise NotImplementedError()

    @abstractmethod
    def forward(self, data: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        # Get activations for one step, data is a tensor
        raise NotImplementedError()

    @abstractmethod
    def initialize(self, param_vector: t.Tensor):
        # Packs params into model correctly
        raise NotImplementedError()

    @abstractmethod
    def get_params(self) -> t.Tensor:
        # Unpack params from model consistent with above
        raise NotImplementedError()

    @abstractmethod
    def add_to_params(self, param_vector: t.Tensor):
        # self.initialize(self.get_params() + param_vector)
        raise NotImplementedError()
