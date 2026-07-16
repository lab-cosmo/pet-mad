# SPDX-License-Identifier: BSD-3-Clause
"""Internal helpers for :mod:`upet.nvalchemi.wrapper`."""

from __future__ import annotations

import contextlib
import sys
import types
import warnings
from typing import Any

import torch


def ensure_hostlist_stub() -> None:
    """Stub out the ``hostlist`` module if it isn't installed.

    ``metatrain.pet.__init__`` imports ``metatrain.pet.trainer``, which
    imports ``metatrain.utils.distributed.slurm``, which pulls in
    ``hostlist`` — a SLURM-only helper that most environments don't have
    installed. Stubbing it out lets the import chain resolve so the
    pure-torch ``PETBackend`` code paths remain usable without SLURM.
    """
    if "hostlist" not in sys.modules:
        try:
            import hostlist  # noqa: F401
        except ImportError:
            sys.modules["hostlist"] = types.ModuleType("hostlist")


# ---------------------------------------------------------------------------
# Hyper-parameter normalisation
# ---------------------------------------------------------------------------

REQUIRED_HYPERS: tuple[str, ...] = (
    "cutoff",
    "cutoff_function",
    "cutoff_width",
    "d_pet",
    "d_node",
    "d_head",
    "d_feedforward",
    "num_heads",
    "num_gnn_layers",
    "num_attention_layers",
    "normalization",
    "activation",
    "attention_temperature",
    "transformer_type",
    "featurizer_type",
)

# The single-block, scalar ``energy`` output shape passed to
# ``PETBackend.add_output``. The block key ``energy___0`` and shape ``[1]``
# are what ``metatrain.pet.model.PET._add_output`` derives for a standard
# scalar energy target.
ENERGY_OUTPUT_SHAPES: dict[str, list[int]] = {"energy___0": [1]}


def normalize_hypers(hypers: dict[str, Any]) -> dict[str, Any]:
    """Validate *hypers* and fill in the keys ``PETBackend`` reads directly.

    :param hypers: Hyper-parameter dict pulled from the checkpoint's
        ``model_hypers`` (or constructed by hand).
    :return: Normalised copy, with ``num_neighbors_adaptive`` /
        ``adaptive_cutoff_method`` / ``cutoff_width_adaptive`` /
        ``system_conditioning`` defaulted when absent.
    :raises ValueError: When a required key is missing.
    """
    missing = [key for key in REQUIRED_HYPERS if key not in hypers]
    if missing:
        raise ValueError(f"PET hypers are missing required keys: {missing}")
    normalized = dict(hypers)
    normalized.setdefault("num_neighbors_adaptive", None)
    normalized.setdefault("adaptive_cutoff_method", "grid")
    normalized.setdefault("cutoff_width_adaptive", 1.0)
    normalized.setdefault("system_conditioning", False)
    return normalized


# ---------------------------------------------------------------------------
# State-dict filtering helpers
# ---------------------------------------------------------------------------

_HEAD_PREFIXES: tuple[str, ...] = (
    "node_heads.",
    "edge_heads.",
    "node_last_layers.",
    "edge_last_layers.",
)


def filter_state_dict(raw_sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Filter a (upgraded) PET checkpoint state dict down to the backend.

    The metatrain checkpoint layout nests the pure-torch core under a
    ``backend.`` prefix and keeps the additive composition model, scaler,
    long-range featurizer and ``finetune_config`` outside it. This keeps
    only the ``backend.*`` keys (with the prefix stripped) and, among
    those, only the ``energy`` readout heads/last-layers, dropping any
    other output (e.g. ``non_conservative_forces``) so the result loads
    into an energy-only ``PETBackend`` with ``strict=True``.

    :param raw_sd: State dict from the upgraded checkpoint's
        ``model_state_dict``.
    :return: Filtered state dict, keyed for ``PETBackend.load_state_dict``.
    """
    filtered: dict[str, torch.Tensor] = {}
    for key, value in raw_sd.items():
        if not key.startswith("backend."):
            continue
        name = key[len("backend.") :]
        if name.startswith(_HEAD_PREFIXES):
            output_name = name.split(".")[1]
            if output_name != "energy":
                continue
        filtered[name] = value
    return filtered


def decode_tensor_map_values(buffer: torch.Tensor) -> torch.Tensor:
    """Decode a serialized ``TensorMap`` buffer to a flat tensor.

    The metatrain composition / scaler modules store their weights as
    ``TensorMap`` byte buffers (uint8 tensors). Decoding with
    ``metatensor.torch.load_buffer`` returns a ``TensorMap``; the single
    block's ``values`` tensor carries the actual numeric weights.

    :param buffer: ``uint8`` tensor holding the serialized TensorMap.
    :return: The decoded ``values`` tensor from the single block.
    """
    import metatensor.torch as mts

    tensor_map = mts.load_buffer(buffer)
    return tensor_map.block(0).values


@contextlib.contextmanager
def ignore_nonleaf_grad_warning():
    """Silence the benign non-leaf ``.grad`` warning emitted under compile.

    When Dynamo's builder wraps a grad-tracking (non-leaf, ``requires_grad``)
    tensor as a graph input, it reads its ``.grad`` and PyTorch emits a
    harmless ``UserWarning``. The autograd graph is left intact, so forces
    via ``autograd.grad`` still work.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*grad attribute of a Tensor that is not a leaf Tensor.*",
            category=UserWarning,
        )
        yield
