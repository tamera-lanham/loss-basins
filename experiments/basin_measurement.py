from loss_basins.models.mnist_conv import MnistConv
import torch as t

def basinDimensionality(model):
  N = nParams(model)
  min = 0
  max = N
  while min != max:
    k = min + (max-min)//2
    result = testDim(model, k, N)
    if result == "no solution":
      max = k-1
    elif result == "solution":
      min = k
    else:
      print("grey area -- parallel at " + k)
      return k
  return k


def testDim(model, k, N):
    pass

def normalize(vec, epsilon=0.001):
    return epsilon*t.nn.functional.normalize(vec)

def intersect(mdl, k, N, off):
    model = mdl.clone().detach()
    
    # Freeze weights

    # Generate offset vector (small random vector in full space). 
    # Note -- the offset size should be a somewhat smaller epsilon than the one used for normalizing the perturbation
    offset = None

    if off:
        # Add offset to model
        pass
    
    # Generate perturbation vector (vector of dimension N+1-k); this is what we will optimize to see if it ends up in the low 2nd deriv subspace.
    perturbation = None
    
    #perturbed_model = MnistConv().initialize(model.get_params() + perturbation)
    
    # Train
    for ...:
        # Normalize perturbation and multiply by epsilon
        # Add perturbation to 
        perturbed = addToModel(model.freeze(), normalize(perturbation))


