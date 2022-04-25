import os
from pathlib import Path
import torch as t

from loss_basins.utils.utils import load_model

traced_model = load_model()

# traced_model = t.jit.load(saved_model_dir + "traced.pt")
