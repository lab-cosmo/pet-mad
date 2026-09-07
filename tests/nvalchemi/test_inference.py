"""`UPETWrapper.forward` and `compute_embeddings`.

Energies, autograd forces and stress off a real (if tiny) backend, plus
the node / graph embedding entry points.
"""

from __future__ import annotations

import pytest
import torch


pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed; skipping")

from _helpers import (  # noqa: E402
    ATOMIC_NUMBERS,
    D_NODE,
    D_PET,
    make_two_atoms,
    make_water,
    tiny_hypers,
)
from nvalchemi.data import AtomicData, Batch  # noqa: E402

from upet.nvalchemi import UPETWrapper  # noqa: E402


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


class TestForward:
    def test_energies_shape_single(self, wrapper, single_batch):
        out = wrapper.forward(single_batch)
        assert out["energy"].shape == (1, 1)

    def test_energies_shape_multi(self, wrapper, multi_batch):
        out = wrapper.forward(multi_batch)
        assert out["energy"].shape == (2, 1)

    def test_energies_dtype(self, wrapper, single_batch):
        out = wrapper.forward(single_batch)
        assert out["energy"].dtype == wrapper._model_dtype

    def test_forces_shape(self, wrapper, single_batch):
        out = wrapper.forward(single_batch)
        assert out["forces"].shape == (3, 3)

    def test_forces_shape_multi(self, wrapper, multi_batch):
        out = wrapper.forward(multi_batch)
        assert out["forces"].shape == (6, 3)

    def test_no_forces_when_disabled(self, wrapper, single_batch):
        wrapper.model_config.active_outputs = {"energy"}
        out = wrapper.forward(single_batch)
        assert out.get("forces") is None

    def test_atomic_data_input(self, wrapper):
        data = make_water()
        out = wrapper.forward(data)
        assert out["energy"].shape == (1, 1)

    def test_pbc_stress_shape(self, wrapper, pbc_batch):
        out = wrapper.forward(pbc_batch)
        assert out["stress"].shape == (1, 3, 3)

    def test_forces_match_finite_difference(self, wrapper):
        """Conservative forces agree with a numerical gradient.

        Uses a small two-atom system so the finite-difference evaluation is
        cheap, and a ``float64`` copy of the wrapper so the FD comparison
        isn't dominated by float32 rounding.
        """
        torch.manual_seed(0)
        w = UPETWrapper(
            atomic_types=ATOMIC_NUMBERS,
            hypers=tiny_hypers(),
            composition_energy=torch.zeros(len(ATOMIC_NUMBERS), dtype=torch.float64),
            scale_energy=torch.tensor(1.0, dtype=torch.float64),
        )
        w.backend = w.backend.to(torch.float64)
        data = make_two_atoms()
        data["positions"] = data.positions.to(torch.float64)
        batch = Batch.from_data_list([data])
        out = w.forward(batch)
        analytic_force = out["forces"][0, 0].item()

        eps = 1e-4
        base_pos = torch.tensor([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float64)

        def energy_at(pos):
            d = AtomicData(
                positions=pos,
                atomic_numbers=torch.tensor([1, 1], dtype=torch.long),
                neighbor_list=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                neighbor_list_shifts=torch.zeros(2, 3, dtype=torch.long),
            )
            b = Batch.from_data_list([d])
            w.model_config.active_outputs = {"energy"}
            return w.forward(b)["energy"].item()

        # Restore autograd-output expectation after the energy-only sanity calls.
        w.model_config.active_outputs = {"energy", "forces", "stress"}

        pos_p = base_pos.clone()
        pos_p[0, 0] += eps
        pos_m = base_pos.clone()
        pos_m[0, 0] -= eps
        fd = -(energy_at(pos_p) - energy_at(pos_m)) / (2 * eps)

        assert analytic_force == pytest.approx(fd, abs=1e-3)


# ---------------------------------------------------------------------------
# compute_embeddings
# ---------------------------------------------------------------------------


class TestComputeEmbeddings:
    # Embeddings concat node + cutoff-weighted edge features: d_node + d_pet.
    _EMB_DIM = D_NODE + D_PET

    def test_node_embeddings_shape(self, wrapper, single_batch):
        result = wrapper.compute_embeddings(single_batch)
        assert result.node_embeddings.shape == (3, self._EMB_DIM)

    def test_graph_embeddings_shape(self, wrapper, single_batch):
        result = wrapper.compute_embeddings(single_batch)
        assert result.graph_embeddings.shape == (1, self._EMB_DIM)

    def test_graph_embeddings_shape_multi(self, wrapper, multi_batch):
        result = wrapper.compute_embeddings(multi_batch)
        assert result.graph_embeddings.shape == (2, self._EMB_DIM)

    def test_graph_embeddings_is_sum_of_node_embeddings(self, wrapper, single_batch):
        result = wrapper.compute_embeddings(single_batch)
        expected_graph = result.node_embeddings.sum(dim=0)
        assert torch.allclose(result.graph_embeddings[0], expected_graph)

    def test_does_not_mutate_model_config(self, wrapper, single_batch):
        wrapper.model_config.active_outputs = {"energy", "forces", "stress"}
        wrapper.compute_embeddings(single_batch)
        assert "forces" in wrapper.model_config.active_outputs
        assert "stress" in wrapper.model_config.active_outputs

    def test_atomic_data_input(self, wrapper):
        data = make_water()
        result = wrapper.compute_embeddings(data)
        assert result.node_embeddings.shape == (3, self._EMB_DIM)

    def test_no_grad_on_positions_after_embeddings(self, wrapper, single_batch):
        wrapper.compute_embeddings(single_batch)
        assert not single_batch.positions.requires_grad
