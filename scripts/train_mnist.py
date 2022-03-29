from loss_basins.models.mnist_conv import MnistConv, TrainingParams
from loss_basins.data import mnist
import torch as t

model = MnistConv(training_params=TrainingParams())

vec = model.get_params().clone().detach()

vec_rand = t.rand(vec.shape)
t.nn.utils.vector_to_parameters(vec_rand, model.model.parameters())

vec_2 = model.get_params()
assert not vec_2.equal(vec)
assert vec_2.equal(vec_rand)


data = mnist(10)
model.freeze()


def set_rg(tensor):
    tensor.requires_grad = True


data_rg = [(set_rg(X), y) for i, (X, y) in enumerate(data) if i <= 1000]
losses, activations = model.train(data_rg, 1000)
# losses, activations = model.train_to_convergence(data, 0.05)
print(losses[-1])


vec_3 = model.get_params()
assert not vec_3.equal(vec_2)
