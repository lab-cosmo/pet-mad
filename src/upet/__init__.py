__version__ = "0.2.0"

import warnings

import torch

from ._models import get_upet, save_upet


# hides a harmless warning from nvalchemi's neighbor list implmentation
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The .grad attribute of a Tensor that is not a leaf Tensor",
)

# Disable static fusion. Besides the fact that atomistic batches have variable
# sizes, statically fused CUDA kernels cannot allocate new tensors at runtime,
# causing "Global alloc not supported yet" errors (cuda 13+) at the time of writing
torch.jit.set_fusion_strategy([("DYNAMIC", 10)])


__all__ = ["get_upet", "save_upet"]
