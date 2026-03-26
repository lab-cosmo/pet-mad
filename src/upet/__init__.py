__version__ = "0.2.2"

import warnings

import torch
from warp._src import utils as wp_utils

from ._models import get_upet, list_upet, save_upet


# hides a harmless warning from nvalchemi's neighbor list implmentation
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="The .grad attribute of a Tensor that is not a leaf Tensor",
)

# we want to suppress a further warning from nvalchemi's usage of warp
# warp uses an internal warn() helper: we wrap it.
_orig_warn = wp_utils.warn


def _warn_filtered(message, category=None, stacklevel=1):
    if category is DeprecationWarning and "warp.vec" in str(message):
        return
    return _orig_warn(message, category=category, stacklevel=stacklevel)


wp_utils.warn = _warn_filtered

# Disable static fusion. Besides the fact that atomistic batches have variable
# sizes, statically fused CUDA kernels cannot allocate new tensors at runtime,
# causing "Global alloc not supported yet" errors (cuda 13+) at the time of writing
torch.jit.set_fusion_strategy([("DYNAMIC", 10)])

__all__ = ["get_upet", "list_upet", "save_upet"]
