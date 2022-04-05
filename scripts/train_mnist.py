from loss_basins.models.mnist_conv import MnistConv
from loss_basins.training import TrainingRun
from loss_basins.data import mnist
import torch as t
from tqdm import tqdm

model = MnistConv()
data = mnist(32)

training_run = TrainingRun(model, data)


losses = training_run.to_convergence(0.2)

test_data = mnist(100, test=True).one_epoch()
n_correct, total = 0, 0
for X, y in test_data:
    with t.no_grad():
        logits = model(X)
        n_correct += (logits.argmax(dim=1) == y).sum()
        total += X.shape[0]

acc = (100 * n_correct / total).item()
print(f"{acc:.2f}% accuracy")
