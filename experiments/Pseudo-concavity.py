from loss_basins.models.mnist_conv import MnistConv, TrainingParams
from loss_basins.data import mnist
import torch as t

training_params=TrainingParams()
data = mnist(batch_size=10)

# Generate an initialization at random
model = MnistConv(training_params)

# Store the init
init_params = model.get_params()

# Train to convergence
losses, _ = model.train_to_convergence(data)

# Get final params
final_params = model.get_params()

# Dir = (final - init) / n_steps

current_model = initial_model.clone().detach()
for i in range(n_steps):
    # current_model += dir
    # Copy the model
    # Train to convergence, log the initial and final loss
    # Store the parameter vector