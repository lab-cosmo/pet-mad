import warnings

import torch

from ._models import get_upet, list_upet, save_upet
from ._scm_version import __version__  # noqa: F401
from ._weighted_sum import create_weighted_sum_checkpoint, extract_wrapped_checkpoint


warnings.filterwarnings(
    "ignore", category=DeprecationWarning, message="warp.config.quiet is deprecated"
)

# Disable static fusion. Besides the fact that atomistic batches have variable
# sizes, statically fused CUDA kernels cannot allocate new tensors at runtime,
# causing "Global alloc not supported yet" errors (cuda 13+) at the time of writing
torch.jit.set_fusion_strategy([("DYNAMIC", 10)])

__all__ = [
    "get_upet",
    "list_upet",
    "save_upet",
    "create_weighted_sum_checkpoint",
    "extract_wrapped_checkpoint",
]
