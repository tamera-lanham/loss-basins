from loss_basins.data import identity_normal
from loss_basins.utils.utils import load_model, save_model
import torch.nn as nn
import torch as t


class ExampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(5, 50)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(50, 5)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


model = ExampleModel()
# save_model(original_model)
# model = load_model()

data = identity_normal((10, 5))
X, y = list(data.batches(1))[0]
traced_model = t.jit.trace(model, X)

scripted_model = t.jit.script(model)

saved_model_dir = "_data/saved_models/"

save_model(traced_model)
# scripted_model.save(saved_model_dir + "scripted.pt")
