from xml.dom.minidom import Attr
from loss_basins.models.model_base import ModelBase, TrainingParams
import torch as t
import torch.nn as nn


class _MnistConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 12, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.linear_layers = nn.Sequential(nn.Linear(12 * 4 * 4, 10))

        self.layers = nn.Sequential(
            *self.conv_layers, nn.Flatten(-3), *self.linear_layers
        )

    def forward(self, x):
        return self.layers(x)


class MnistConv(ModelBase):
    def __init__(self, training_params=None):
        self.training_params = training_params
        self.model = _MnistConv()

        self._activations = []
        self._set_activation_hooks()

    def _set_activation_hooks(self):
        def hook(model, input, output):
            self._activations.append(output.detach().clone())

        for layer in self.model.layers:
            layer.register_forward_hook(hook)

    def train_generator(self, data_iter):
        # Returns iterator of (loss, activations)
        # data is an infinite iterator of (inputs, targets)
        if not self.training_params:
            raise AttributeError("Model has no training_params")

        params = self.training_params
        self.model.to(params.device)
        optimizer = t.optim.SGD(self.model.parameters(), params.lr)
        loss_fn = t.nn.CrossEntropyLoss()

        for i, (X, y) in enumerate(data_iter):
            X.to(params.device), y.to(params.device)

            optimizer.zero_grad()
            y_pred = self.model(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            yield loss.detach(), self._activations.copy()

    def train(self, data_iter, n_steps):
        train_iter = self.train_generator(data_iter)

        losses, activations = [], []
        for _ in range(n_steps):
            loss, activation = next(train_iter)
            losses.append(loss)
            activations.append(activation)

        return losses, activations

    def forward(self, data):
        with t.no_grad():
            outputs = self.model(data)
        return outputs, self._activations.copy()

    def get_params(self):
        return t.cat([p.flatten() for p in self.model.parameters()]).clone().detach()

    def initialize(self, param_vector):
        nn.utils.vector_to_parameters(param_vector, self.model.parameters())

    def add_to_params(self, param_vector):
        self.initialize(self.get_params() + param_vector)
