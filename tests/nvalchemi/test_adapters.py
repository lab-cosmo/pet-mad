"""Translation between `nvalchemi.data.Batch` and the `PETBackend` tensors.

``adapt_input`` flattens a batch into the concatenated plain tensors
``PETBackend.preprocess`` consumes; ``adapt_output`` wraps the backend's
results back into ``ModelOutputs``.
"""

from __future__ import annotations

import pytest
import torch


pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed; skipping")

from _helpers import make_water  # noqa: E402


# ---------------------------------------------------------------------------
# adapt_input
# ---------------------------------------------------------------------------


class TestAdaptInput:
    def test_required_keys_present(self, wrapper, single_batch):
        # adapt_input returns the concatenated, plain-tensor structure
        # representation consumed by PETBackend.preprocess.
        inp = wrapper.adapt_input(single_batch)
        for key in (
            "positions",
            "centers",
            "neighbors",
            "species",
            "cells",
            "cell_shifts",
            "system_indices",
        ):
            assert key in inp, f"Missing key: {key}"

    def test_species_are_raw_atomic_numbers(self, wrapper, single_batch):
        # H2O = [8, 1, 1]; the species->index map is applied inside the backend,
        # so adapt_input passes raw atomic numbers through.
        inp = wrapper.adapt_input(single_batch)
        assert inp["species"].tolist() == [8, 1, 1]

    def test_centers_neighbors_from_neighbor_list(self, wrapper, single_batch):
        inp = wrapper.adapt_input(single_batch)
        assert inp["centers"].tolist() == [0, 1, 0, 2, 1, 2]
        assert inp["neighbors"].tolist() == [1, 0, 2, 0, 2, 1]

    def test_cells_shape(self, wrapper, single_batch):
        # Non-PBC batch → identity cell [B, 3, 3].
        inp = wrapper.adapt_input(single_batch)
        assert inp["cells"].shape == (1, 3, 3)

    def test_positions_requires_grad_when_forces_active(self, wrapper, single_batch):
        wrapper.model_config.active_outputs = {"energy", "forces"}
        wrapper.adapt_input(single_batch)
        assert single_batch.positions.requires_grad

    def test_positions_no_grad_energy_only(self, wrapper, single_batch):
        wrapper.model_config.active_outputs = {"energy"}
        wrapper.adapt_input(single_batch)
        assert not single_batch.positions.requires_grad

    def test_atomic_data_promoted_to_batch(self, wrapper):
        data = make_water()
        inp = wrapper.adapt_input(data)
        assert inp["species"].shape == (3,)

    def test_multi_batch_shapes(self, wrapper, multi_batch):
        inp = wrapper.adapt_input(multi_batch)
        # 6 atoms total across 2 water molecules; 2 cells.
        assert inp["species"].shape == (6,)
        assert inp["cells"].shape == (2, 3, 3)
        assert inp["system_indices"].tolist() == [0, 0, 0, 1, 1, 1]

    def test_pbc_runs(self, wrapper, pbc_batch):
        inp = wrapper.adapt_input(pbc_batch)
        assert inp["positions"].shape == (3, 3)
        assert inp["cells"].shape == (1, 3, 3)


# ---------------------------------------------------------------------------
# adapt_output
# ---------------------------------------------------------------------------


class TestAdaptOutput:
    def test_energy_key_in_output(self, wrapper, single_batch):
        raw = {"energy": torch.randn(1, 1)}
        out = wrapper.adapt_output(raw, single_batch)
        assert "energy" in out

    def test_energies_shape(self, wrapper, single_batch):
        raw = {"energy": torch.randn(1, 1)}
        out = wrapper.adapt_output(raw, single_batch)
        assert out["energy"].shape == (1, 1)

    def test_1d_energy_unsqueezed(self, wrapper, single_batch):
        raw = {"energy": torch.randn(1)}
        out = wrapper.adapt_output(raw, single_batch)
        assert out["energy"].shape == (1, 1)

    def test_forces_passed_through(self, wrapper, single_batch):
        wrapper.model_config.active_outputs = {"energy", "forces"}
        raw = {"energy": torch.randn(1, 1), "forces": torch.randn(3, 3)}
        out = wrapper.adapt_output(raw, single_batch)
        assert out["forces"].shape == (3, 3)

    def test_stress_passed_through(self, wrapper, single_batch):
        wrapper.model_config.active_outputs = {"energy", "forces", "stress"}
        raw = {
            "energy": torch.randn(1, 1),
            "forces": torch.randn(3, 3),
            "stress": torch.randn(1, 3, 3),
        }
        out = wrapper.adapt_output(raw, single_batch)
        assert out["stress"].shape == (1, 3, 3)
