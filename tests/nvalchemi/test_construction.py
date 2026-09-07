"""Construction of :class:`~upet.nvalchemi.UPETWrapper` and its static config.

Covers what the wrapper looks like straight after ``__init__``: the
``PETBackend`` it builds, the buffers it registers, the ``ModelConfig``
capabilities it advertises, and the properties derived from the hypers.
"""

from __future__ import annotations

import pytest
import torch


pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed; skipping")

from _helpers import (  # noqa: E402
    ATOMIC_NUMBERS,
    CUTOFF,
    D_NODE,
    D_PET,
    tiny_hypers,
)
from metatrain.pet.modules.backend import PETBackend  # noqa: E402
from nvalchemi.models.base import NeighborListFormat  # noqa: E402

from upet.nvalchemi import UPETWrapper  # noqa: E402


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_wrapper_builds_backend(self, wrapper):
        # The wrapper builds and owns a metatrain PETBackend from hypers.
        assert isinstance(wrapper.backend, PETBackend)
        assert wrapper.backend.d_node == D_NODE
        assert wrapper.backend.d_pet == D_PET
        # The single scalar energy output is registered.
        assert "energy" in wrapper.backend.node_last_layers

    def test_default_model_config(self, wrapper):
        assert "forces" in wrapper.model_config.active_outputs
        assert "stress" in wrapper.model_config.active_outputs

    def test_composition_buffer_shape(self, wrapper):
        assert wrapper.composition_energy.shape == (len(ATOMIC_NUMBERS),)

    def test_scale_buffer_scalar(self, wrapper):
        assert wrapper.scale_energy.shape == ()

    def test_buffers_not_in_state_dict(self, wrapper):
        # composition_energy / scale_energy are non-persistent.
        sd_keys = wrapper.state_dict().keys()
        assert not any(k.endswith("composition_energy") for k in sd_keys)
        assert not any(k.endswith("scale_energy") for k in sd_keys)

    def test_validate_hypers_rejects_missing(self):
        bad = tiny_hypers()
        del bad["cutoff"]
        with pytest.raises(ValueError, match="missing required keys"):
            UPETWrapper(
                atomic_types=ATOMIC_NUMBERS,
                hypers=bad,
                composition_energy=torch.zeros(len(ATOMIC_NUMBERS)),
                scale_energy=torch.tensor(1.0),
            )


# ---------------------------------------------------------------------------
# ModelConfig capability checks
# ---------------------------------------------------------------------------


class TestModelConfigCapabilities:
    def test_forces_via_autograd(self, wrapper):
        assert "forces" in wrapper.model_config.autograd_outputs

    def test_stress_via_autograd(self, wrapper):
        assert "stress" in wrapper.model_config.autograd_outputs

    def test_outputs_include_energies_forces_stresses(self, wrapper):
        cfg = wrapper.model_config
        assert "energy" in cfg.outputs
        assert "forces" in cfg.outputs
        assert "stress" in cfg.outputs

    def test_autograd_inputs(self, wrapper):
        assert "positions" in wrapper.model_config.autograd_inputs

    def test_supports_pbc(self, wrapper):
        assert wrapper.model_config.supports_pbc is True

    def test_embedding_shapes_available(self, wrapper):
        shapes = wrapper.embedding_shapes
        assert "node_embeddings" in shapes
        assert "graph_embeddings" in shapes

    def test_neighbor_config_coo(self, wrapper):
        nc = wrapper.model_config.neighbor_config
        assert nc is not None
        assert nc.format == NeighborListFormat.COO
        assert nc.cutoff == pytest.approx(CUTOFF)
        assert nc.half_list is False

    def test_needs_pbc_false(self, wrapper):
        assert wrapper.model_config.needs_pbc is False


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_cutoff(self, wrapper):
        assert wrapper.cutoff == pytest.approx(CUTOFF)
        assert isinstance(wrapper.cutoff, float)

    def test_embedding_shapes(self, wrapper):
        # Embeddings concat node + cutoff-weighted edge features (one readout
        # layer for the feedforward featurizer): d_node + d_pet.
        shapes = wrapper.embedding_shapes
        assert shapes["node_embeddings"] == (D_NODE + D_PET,)
        assert shapes["graph_embeddings"] == (D_NODE + D_PET,)

    def test_model_dtype(self, wrapper):
        assert wrapper._model_dtype == torch.float32
