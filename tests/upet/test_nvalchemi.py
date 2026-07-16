"""Tests for :class:`~upet.nvalchemi.UPETWrapper`.

All tests in this module require the optional ``nvalchemi`` extra (and the
``metatrain`` / ``metatomic`` / ``metatensor`` packages that ride along with
it) and are automatically skipped when it isn't installed. Install with::

    pip install "upet[nvalchemi]"
"""

from __future__ import annotations

import sys
import types

import pytest
import torch


# metatrain.pet.__init__ imports metatrain.pet.trainer, which imports
# metatrain.utils.distributed.slurm, which pulls in `hostlist` — a SLURM-only
# helper most environments don't have installed. Stub it out so the import
# chain resolves and these tests can still exercise PET's pure-torch code
# paths. (upet.nvalchemi.wrapper does the same thing internally, but only
# once it is imported, so this stays self-contained regardless of import
# order.)
if "hostlist" not in sys.modules:
    try:
        import hostlist  # noqa: F401
    except ImportError:
        sys.modules["hostlist"] = types.ModuleType("hostlist")

# Skip the entire module when the optional nvalchemi-toolkit / metatrain
# dependencies aren't installed.
pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed; skipping")

from metatrain.pet.modules.backend import PETBackend  # noqa: E402
from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.models.base import NeighborListFormat  # noqa: E402
from nvalchemi.neighbors import compute_neighbors  # noqa: E402

from upet._models import _resolve_and_download_checkpoint  # noqa: E402
from upet.nvalchemi import UPETWrapper  # noqa: E402


def _fetch_checkpoint_path(model: str, version: str) -> str:
    """Resolve + download a UPET checkpoint, returning its local path.

    Mirrors what :meth:`UPETWrapper.from_checkpoint` does internally when
    given a combined ``<model>-<size>`` name, for tests that need the raw
    checkpoint file directly (e.g. to compare against the original
    metatrain model).
    """
    model_name, size = model.rsplit("-", 1)
    _, _, path = _resolve_and_download_checkpoint(model_name, size, version)
    return path


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_ATOMIC_NUMBERS = [1, 6, 8]  # H, C, O — keeps species_to_species_index small
_CUTOFF = 5.0
_D_NODE = 16
_D_PET = 8


def _tiny_hypers() -> dict:
    """Return a minimal PET hyper-parameter dict for fast unit tests."""
    return {
        "cutoff": _CUTOFF,
        "cutoff_width": 0.5,
        "cutoff_function": "Bump",
        "d_pet": _D_PET,
        "d_node": _D_NODE,
        "d_head": 8,
        "d_feedforward": 8,
        "num_heads": 2,
        "num_gnn_layers": 1,
        "num_attention_layers": 1,
        "normalization": "RMSNorm",
        "activation": "SwiGLU",
        "attention_temperature": 1.0,
        "transformer_type": "PreLN",
        "featurizer_type": "feedforward",
        "num_neighbors_adaptive": None,
        "adaptive_cutoff_method": "grid",
        "system_conditioning": False,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_water(device: str = "cpu") -> AtomicData:
    """Single H2O molecule with a pre-computed full edge list (no PBC)."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    numbers = torch.tensor([8, 1, 1], dtype=torch.long, device=device)
    neighbor_list = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
        dtype=torch.long,
        device=device,
    )
    neighbor_list_shifts = torch.zeros(
        neighbor_list.shape[0], 3, dtype=torch.long, device=device
    )
    return AtomicData(
        positions=positions,
        atomic_numbers=numbers,
        neighbor_list=neighbor_list,
        neighbor_list_shifts=neighbor_list_shifts,
    )


def _make_two_atoms(device: str = "cpu") -> AtomicData:
    """Two H atoms at (0, 0, 0) and (1.1, 0, 0) with a symmetric edge pair."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]], dtype=torch.float32, device=device
    )
    numbers = torch.tensor([1, 1], dtype=torch.long, device=device)
    neighbor_list = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)
    neighbor_list_shifts = torch.zeros(2, 3, dtype=torch.long, device=device)
    return AtomicData(
        positions=positions,
        atomic_numbers=numbers,
        neighbor_list=neighbor_list,
        neighbor_list_shifts=neighbor_list_shifts,
    )


def _make_pbc_water(device: str = "cpu") -> AtomicData:
    """H2O in a 10 Å cubic periodic box."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    numbers = torch.tensor([8, 1, 1], dtype=torch.long, device=device)
    neighbor_list = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
        dtype=torch.long,
        device=device,
    )
    cell = (torch.eye(3, dtype=torch.float32, device=device) * 10.0).unsqueeze(0)
    neighbor_list_shifts = torch.zeros(6, 3, dtype=torch.long, device=device)
    pbc = torch.tensor([[True, True, True]], device=device)
    return AtomicData(
        positions=positions,
        atomic_numbers=numbers,
        neighbor_list=neighbor_list,
        cell=cell,
        neighbor_list_shifts=neighbor_list_shifts,
        pbc=pbc,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wrapper() -> UPETWrapper:
    """UPETWrapper with zero composition and unit scaler."""
    torch.manual_seed(0)
    composition = torch.zeros(len(_ATOMIC_NUMBERS), dtype=torch.float32)
    scale = torch.tensor(1.0, dtype=torch.float32)
    return UPETWrapper(
        atomic_types=_ATOMIC_NUMBERS,
        hypers=_tiny_hypers(),
        composition_energy=composition,
        scale_energy=scale,
    )


@pytest.fixture
def single_batch() -> Batch:
    return Batch.from_data_list([_make_water()])


@pytest.fixture
def multi_batch() -> Batch:
    """Two H2O molecules as a batched system (B=2, N=6)."""
    return Batch.from_data_list([_make_water(), _make_water()])


@pytest.fixture
def pbc_batch() -> Batch:
    return Batch.from_data_list([_make_pbc_water()])


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_wrapper_builds_backend(self, wrapper):
        # The wrapper builds and owns a metatrain PETBackend from hypers.
        assert isinstance(wrapper.backend, PETBackend)
        assert wrapper.backend.d_node == _D_NODE
        assert wrapper.backend.d_pet == _D_PET
        # The single scalar energy output is registered.
        assert "energy" in wrapper.backend.node_last_layers

    def test_default_model_config(self, wrapper):
        assert "forces" in wrapper.model_config.active_outputs
        assert "stress" in wrapper.model_config.active_outputs

    def test_composition_buffer_shape(self, wrapper):
        assert wrapper.composition_energy.shape == (len(_ATOMIC_NUMBERS),)

    def test_scale_buffer_scalar(self, wrapper):
        assert wrapper.scale_energy.shape == ()

    def test_buffers_not_in_state_dict(self, wrapper):
        # composition_energy / scale_energy are non-persistent.
        sd_keys = wrapper.state_dict().keys()
        assert not any(k.endswith("composition_energy") for k in sd_keys)
        assert not any(k.endswith("scale_energy") for k in sd_keys)

    def test_validate_hypers_rejects_missing(self):
        bad = _tiny_hypers()
        del bad["cutoff"]
        with pytest.raises(ValueError, match="missing required keys"):
            UPETWrapper(
                atomic_types=_ATOMIC_NUMBERS,
                hypers=bad,
                composition_energy=torch.zeros(len(_ATOMIC_NUMBERS)),
                scale_energy=torch.tensor(1.0),
            )

    def test_residual_featurizer_builds(self):
        # Both 'feedforward' and 'residual' featurizers are now supported.
        torch.manual_seed(0)
        hypers = _tiny_hypers()
        hypers["featurizer_type"] = "residual"
        w = UPETWrapper(
            atomic_types=_ATOMIC_NUMBERS,
            hypers=hypers,
            composition_energy=torch.zeros(len(_ATOMIC_NUMBERS)),
            scale_energy=torch.tensor(1.0),
        )
        # Residual featurization keeps one readout layer per GNN layer.
        assert w.backend.num_readout_layers == hypers["num_gnn_layers"]


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
        assert nc.cutoff == pytest.approx(_CUTOFF)
        assert nc.half_list is False

    def test_needs_pbc_false(self, wrapper):
        assert wrapper.model_config.needs_pbc is False


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_cutoff(self, wrapper):
        assert wrapper.cutoff == pytest.approx(_CUTOFF)
        assert isinstance(wrapper.cutoff, float)

    def test_embedding_shapes(self, wrapper):
        # Embeddings concat node + cutoff-weighted edge features (one readout
        # layer for the feedforward featurizer): d_node + d_pet.
        shapes = wrapper.embedding_shapes
        assert shapes["node_embeddings"] == (_D_NODE + _D_PET,)
        assert shapes["graph_embeddings"] == (_D_NODE + _D_PET,)

    def test_model_dtype(self, wrapper):
        assert wrapper._model_dtype == torch.float32


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
        data = _make_water()
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
        data = _make_water()
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
            atomic_types=_ATOMIC_NUMBERS,
            hypers=_tiny_hypers(),
            composition_energy=torch.zeros(len(_ATOMIC_NUMBERS), dtype=torch.float64),
            scale_energy=torch.tensor(1.0, dtype=torch.float64),
        )
        w.backend = w.backend.to(torch.float64)
        data = _make_two_atoms()
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
    _EMB_DIM = _D_NODE + _D_PET

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
        data = _make_water()
        result = wrapper.compute_embeddings(data)
        assert result.node_embeddings.shape == (3, self._EMB_DIM)

    def test_no_grad_on_positions_after_embeddings(self, wrapper, single_batch):
        wrapper.compute_embeddings(single_batch)
        assert not single_batch.positions.requires_grad


# ---------------------------------------------------------------------------
# torch.compile of the backend building blocks
# ---------------------------------------------------------------------------


class TestCompiledBackend:
    """The compiled backend (preprocess/calculate_features/predict) matches eager."""

    @staticmethod
    def _compile_backend(w: UPETWrapper) -> None:
        """Compile the three backend methods in place, as `from_checkpoint` does."""
        w.eval()
        for p in w.parameters():
            p.requires_grad = False
        w.backend.preprocess = torch.compile(w.backend.preprocess, fullgraph=True)
        w.backend.calculate_features = torch.compile(
            w.backend.calculate_features, fullgraph=True
        )
        w.backend.predict = torch.compile(w.backend.predict, fullgraph=True)
        w._compiled = True

    @staticmethod
    def _make_pair() -> tuple[UPETWrapper, UPETWrapper]:
        """Build identical eager + compiled float64 wrappers."""
        torch.manual_seed(0)
        kwargs = dict(
            atomic_types=_ATOMIC_NUMBERS,
            hypers=_tiny_hypers(),
            composition_energy=torch.zeros(len(_ATOMIC_NUMBERS), dtype=torch.float64),
            scale_energy=torch.tensor(1.0, dtype=torch.float64),
        )
        eager = UPETWrapper(**kwargs)
        eager.backend = eager.backend.to(torch.float64)

        torch.manual_seed(0)
        compiled = UPETWrapper(**kwargs)
        compiled.backend = compiled.backend.to(torch.float64)
        compiled.load_state_dict(eager.state_dict())
        TestCompiledBackend._compile_backend(compiled)
        return eager, compiled

    @staticmethod
    def _water64() -> Batch:
        d = _make_water()
        d["positions"] = d.positions.to(torch.float64)
        return Batch.from_data_list([d])

    @staticmethod
    def _pbc_water64() -> Batch:
        d = _make_pbc_water()
        d["positions"] = d.positions.to(torch.float64)
        d["cell"] = d.cell.to(torch.float64)
        return Batch.from_data_list([d])

    def test_compiled_energy_matches_eager(self):
        eager, compiled = self._make_pair()
        eager.model_config.active_outputs = {"energy"}
        compiled.model_config.active_outputs = {"energy"}
        e_eager = eager.forward(self._water64())["energy"]
        e_compiled = compiled.forward(self._water64())["energy"]
        torch.testing.assert_close(e_compiled.detach(), e_eager.detach())

    def test_compiled_forces_match_eager(self):
        # Forces via autograd backprop through the compiled backend and match
        # eager (no PBC → no stress / strain path).
        eager, compiled = self._make_pair()
        eager.model_config.active_outputs = {"energy", "forces"}
        compiled.model_config.active_outputs = {"energy", "forces"}
        f_eager = eager.forward(self._water64())["forces"]
        f_compiled = compiled.forward(self._water64())["forces"]
        torch.testing.assert_close(f_compiled.detach(), f_eager.detach())

    def test_compiled_forces_and_stress_match_eager(self):
        # Forces + stress share a single backward (autograd_forces_and_stresses),
        # which is what keeps the compiled graph compatible with torch.compile's
        # donated-buffer optimization.
        eager, compiled = self._make_pair()
        eager.model_config.active_outputs = {"energy", "forces", "stress"}
        compiled.model_config.active_outputs = {"energy", "forces", "stress"}
        out_eager = eager.forward(self._pbc_water64())
        out_compiled = compiled.forward(self._pbc_water64())
        torch.testing.assert_close(
            out_compiled["forces"].detach(), out_eager["forces"].detach()
        )
        torch.testing.assert_close(
            out_compiled["stress"].detach(), out_eager["stress"].detach()
        )


# ---------------------------------------------------------------------------
# export_model
# ---------------------------------------------------------------------------


class TestExportModel:
    def test_export_snapshot(self, wrapper, tmp_path):
        path = tmp_path / "pet.pt"
        wrapper.export_model(path)
        assert path.exists()
        loaded = torch.load(path, weights_only=False)
        assert isinstance(loaded, dict)
        assert "backend_state_dict" in loaded
        assert "hypers" in loaded
        assert "atomic_types" in loaded
        assert "composition_energy" in loaded
        assert "scale_energy" in loaded

    def test_export_state_dict(self, wrapper, tmp_path):
        path = tmp_path / "pet_sd.pt"
        wrapper.export_model(path, as_state_dict=True)
        assert path.exists()
        sd = torch.load(path, weights_only=True)
        assert isinstance(sd, dict)
        assert any("gnn_layers" in k for k in sd.keys())

    def test_reload_snapshot_into_new_backend(self, wrapper, tmp_path):
        path = tmp_path / "pet.pt"
        wrapper.export_model(path)
        snapshot = torch.load(path, weights_only=False)

        new_backend = PETBackend(snapshot["hypers"], snapshot["atomic_types"])
        new_backend.add_output("energy", {"energy___0": [1]})
        new_backend.load_state_dict(snapshot["backend_state_dict"], strict=True)
        # Check every parameter matches.
        for key in wrapper.backend.state_dict():
            assert torch.allclose(
                new_backend.state_dict()[key], wrapper.backend.state_dict()[key]
            )


# ---------------------------------------------------------------------------
# Integration tests — real PET checkpoint
# ---------------------------------------------------------------------------


# Fetched from the `lab-cosmo/upet` HuggingFace repo (see UPETWrapper.from_checkpoint).
_CHECKPOINT_MODEL = "pet-mad-xs"
_CHECKPOINT_VERSION = "1.5.0"  # 'grid' adaptive cutoff
# Not yet published on HuggingFace; must be provided locally to run this test.
_SOLVER_CHECKPOINT_PATH = "pet-mad-xs-v1.6.0.ckpt"  # 'solver' adaptive cutoff
_CUTOFF = 7.5


def _crystal(device: str = "cpu") -> AtomicData:
    return AtomicData(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], dtype=torch.float32
        ).to(device=device),
        atomic_numbers=torch.tensor([6, 6], dtype=torch.long).to(device=device),
        cell=torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).to(device=device),
        pbc=torch.tensor([True] * 3, dtype=torch.bool).reshape(1, 3).to(device=device),
    )


def _batch(dtype: torch.dtype = torch.float32) -> Batch:
    data = _crystal()
    batch = Batch.from_data_list([data])
    compute_neighbors(batch, cutoff=_CUTOFF, format=NeighborListFormat.COO)
    return batch


@pytest.fixture(scope="session")
def real_wrapper_cpu():
    """Fetch the pet-mad-xs checkpoint from HuggingFace once per session."""
    try:
        return UPETWrapper.from_checkpoint(
            model=_CHECKPOINT_MODEL,
            version=_CHECKPOINT_VERSION,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    except Exception as e:
        pytest.skip(f"Could not fetch/load PET checkpoint: {e}")


class TestRealCheckpoint:
    """Integration tests against the pet-mad-xs-v1.5.0 checkpoint."""

    def test_is_upet_wrapper(self, real_wrapper_cpu):
        assert isinstance(real_wrapper_cpu, UPETWrapper)

    def test_model_config_matches_wrapper(self, real_wrapper_cpu):
        cfg = real_wrapper_cpu.model_config
        assert "forces" in cfg.autograd_outputs
        assert "stress" in cfg.autograd_outputs
        assert "energy" in cfg.outputs
        assert cfg.neighbor_config is not None
        assert cfg.neighbor_config.format == NeighborListFormat.COO

    def test_cutoff_positive(self, real_wrapper_cpu):
        assert real_wrapper_cpu.cutoff > 0.0

    def test_inference_energy_finite(self, real_wrapper_cpu):
        batch = _batch()
        out = real_wrapper_cpu.forward(batch)
        e = out["energy"]
        assert e.shape == (1, 1)
        assert torch.isfinite(e).all()

    def test_inference_forces_shape(self, real_wrapper_cpu):
        batch = _batch()
        out = real_wrapper_cpu.forward(batch)
        assert out["forces"].shape == (2, 3)
        assert torch.isfinite(out["forces"]).all()

    def test_inference_stress_shape(self, real_wrapper_cpu):
        """Non-PBC system produces a stress tensor of zeros with shape [B, 3, 3]."""
        batch = _batch()
        out = real_wrapper_cpu.forward(batch)
        assert out["stress"].shape == (1, 3, 3)
        assert torch.isfinite(out["stress"]).all()

    def test_batch_determinism(self, real_wrapper_cpu):
        """Same input → same output across two consecutive calls."""
        b1 = _batch()
        b2 = _batch()
        e1 = real_wrapper_cpu.forward(b1)["energy"]
        e2 = real_wrapper_cpu.forward(b2)["energy"]
        assert torch.allclose(e1, e2, atol=1e-6)

    def test_batched_matches_single(self, real_wrapper_cpu):
        """Two-water batch energies equal the single-water energy."""
        single = real_wrapper_cpu.forward(_batch())
        multi_batch = Batch.from_data_list([_crystal(), _crystal()])
        compute_neighbors(multi_batch, cutoff=_CUTOFF, format=NeighborListFormat.COO)
        multi = real_wrapper_cpu.forward(multi_batch)
        assert torch.allclose(
            multi["energy"][0], single["energy"][0], atol=1e-4, rtol=1e-4
        )
        assert torch.allclose(
            multi["energy"][1], single["energy"][0], atol=1e-4, rtol=1e-4
        )

    def test_compute_embeddings_run(self, real_wrapper_cpu):
        batch = _batch()
        result = real_wrapper_cpu.compute_embeddings(batch)
        assert result.node_embeddings.shape[0] == 2
        assert result.graph_embeddings.shape == (1, result.node_embeddings.shape[1])

    def test_embeddings_match_metatrain_features(self):
        """compute_embeddings reproduces metatrain PET's per-atom ``feature`` output.

        The metatrain ``feature`` output is built by
        ``metatrain.pet.model.PET._get_output_features`` as the concatenation of
        the per-layer node features with the cutoff-weighted, neighbor-summed
        per-layer edge features — exactly what :meth:`UPETWrapper.compute_embeddings`
        computes. This verifies they agree value-for-value (in float64).
        """
        ase = pytest.importorskip("ase")
        from metatomic.torch import ModelOutput, systems_to_torch
        from metatrain.pet import PET
        from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

        try:
            checkpoint_path = _fetch_checkpoint_path(
                _CHECKPOINT_MODEL, _CHECKPOINT_VERSION
            )
        except Exception as e:
            pytest.skip(f"Could not fetch PET checkpoint: {e}")

        # metatrain PET (float64) with its native "feature" output.
        raw = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        wrapped = raw.get("wrapped_model_checkpoint", raw)
        PET.upgrade_checkpoint(wrapped)
        pet = PET.load_checkpoint(wrapped, context="export").to(torch.float64).eval()

        data = _crystal()
        atoms = ase.Atoms(
            positions=data.positions.numpy(),
            numbers=data.atomic_numbers.numpy(),
            cell=data.cell.squeeze().numpy(),
            pbc=[True, True, True],
        )
        system = systems_to_torch(atoms, dtype=torch.float64)
        system = get_system_with_neighbor_lists(system, pet.requested_neighbor_lists())
        mt_feature = (
            pet([system], {"feature": ModelOutput(sample_kind="atom")})["feature"]
            .block()
            .values.detach()
        )

        # nvalchemi embeddings (float64) on the same structure.
        wrapper = UPETWrapper.from_checkpoint(
            checkpoint_path=checkpoint_path, dtype=torch.float64
        )
        batch = Batch.from_data_list([_crystal()])
        batch["positions"] = batch.positions.to(torch.float64)
        batch["cell"] = batch.cell.to(torch.float64)
        compute_neighbors(batch, cutoff=_CUTOFF, format=NeighborListFormat.COO)
        nv_embeddings = wrapper.compute_embeddings(batch).node_embeddings.detach()

        assert nv_embeddings.shape == mt_feature.shape
        torch.testing.assert_close(nv_embeddings, mt_feature, atol=1e-8, rtol=1e-8)

    def test_export_and_reload(self, real_wrapper_cpu, tmp_path):
        path = tmp_path / "pet_snapshot.pt"
        real_wrapper_cpu.export_model(path)
        snapshot = torch.load(path, weights_only=False)
        new_backend = PETBackend(snapshot["hypers"], snapshot["atomic_types"])
        new_backend.add_output("energy", {"energy___0": [1]})
        new_backend.load_state_dict(snapshot["backend_state_dict"], strict=True)

    def test_grid_compile_raises(self):
        """Compiling a 'grid' adaptive-cutoff model is rejected up front.

        Autograd backward through the compiled grid cutoff aborts at the C++
        level (always with ``fullgraph=True``, intermittently without), so
        ``from_checkpoint`` raises a clear error rather than letting it crash at
        the first backward. Both the plain and ``fullgraph=True`` requests raise.
        """
        try:
            checkpoint_path = _fetch_checkpoint_path(
                _CHECKPOINT_MODEL, _CHECKPOINT_VERSION
            )
        except Exception as e:
            pytest.skip(f"Could not fetch PET checkpoint: {e}")

        for extra in ({}, {"fullgraph": True}):
            with pytest.raises(ValueError, match="grid"):
                UPETWrapper.from_checkpoint(
                    checkpoint_path=checkpoint_path,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                    compile_model=True,
                    **extra,
                )

    def test_solver_compile_fullgraph_forces_stress(self):
        """A 'solver' model (pet-mad >= v1.6.0) compiles with ``fullgraph=True``.

        Verifies the ``compile_kwargs`` forwarding and that autograd forces +
        stress through the fullgraph-compiled solver backbone match eager.
        """
        import os
        import warnings

        if not os.path.exists(_SOLVER_CHECKPOINT_PATH):
            pytest.skip(f"Checkpoint {_SOLVER_CHECKPOINT_PATH} not found.")

        eager = UPETWrapper.from_checkpoint(
            checkpoint_path=_SOLVER_CHECKPOINT_PATH,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )
        assert eager.backend.adaptive_cutoff_method.lower() == "solver"
        out_eager = eager.forward(_batch(torch.float64))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            compiled = UPETWrapper.from_checkpoint(
                checkpoint_path=_SOLVER_CHECKPOINT_PATH,
                device=torch.device("cpu"),
                dtype=torch.float64,
                compile_model=True,
                fullgraph=True,
            )
            assert compiled._compiled is True
            out_compiled = compiled.forward(_batch(torch.float64))

        torch.testing.assert_close(
            out_compiled["forces"].detach(), out_eager["forces"].detach()
        )
        torch.testing.assert_close(
            out_compiled["stress"].detach(), out_eager["stress"].detach()
        )

    def test_metatrain_model_compatibility(self, real_wrapper_cpu):
        """UPETWrapper predictions match those from the original metatrain model."""
        ase = pytest.importorskip("ase")
        from metatomic.torch import NeighborListOptions, systems_to_torch
        from metatrain.utils.data.target_info import get_energy_target_info
        from metatrain.utils.evaluate_model import evaluate_model
        from metatrain.utils.io import load_model
        from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

        try:
            checkpoint_path = _fetch_checkpoint_path(
                _CHECKPOINT_MODEL, _CHECKPOINT_VERSION
            )
        except Exception as e:
            pytest.skip(f"Could not fetch PET checkpoint: {e}")

        mt_model = load_model(checkpoint_path)
        data = _crystal()
        atoms = ase.Atoms(
            positions=data.positions.cpu().numpy(),
            numbers=data.atomic_numbers.cpu().numpy(),
            cell=data.cell.cpu().squeeze().numpy() if data.cell is not None else None,
            pbc=data.pbc.cpu().squeeze().numpy() if data.pbc is not None else None,
        )
        system = systems_to_torch(atoms)
        neighbor_options = NeighborListOptions(
            cutoff=_CUTOFF, full_list=True, strict=True
        )
        system = get_system_with_neighbor_lists(system, [neighbor_options])
        targets = {
            "energy": get_energy_target_info(
                "energy",
                {"unit": "eV"},
                add_position_gradients=True,
                add_strain_gradients=True,
            )
        }
        mt_output = evaluate_model(
            mt_model, [system], targets=targets, is_training=False
        )

        mt_energy = mt_output["energy"].block().values.detach().squeeze()
        mt_forces = -mt_output["energy"].block().gradient("positions").values.squeeze()
        mt_stress = (
            -mt_output["energy"].block().gradient("strain").values.squeeze()
            / atoms.get_volume()
        )

        nv_output = real_wrapper_cpu.forward(_batch())
        nv_energy = nv_output["energy"].detach().squeeze().item()
        nv_forces = nv_output["forces"].detach()
        nv_stress = nv_output["stress"].detach().squeeze()

        assert abs(nv_energy - mt_energy) < 1e-4, (
            f"Energy mismatch: nvalchemi={nv_energy}, metatrain={mt_energy}"
        )
        assert torch.allclose(nv_forces, mt_forces, atol=1e-4, rtol=1e-4), (
            f"Force mismatch: max |dF|={float((nv_forces - mt_forces).abs().max())}"
        )

        # nvalchemi's shared `autograd_stresses` returns *tensile-positive*
        # Cauchy stress (+1/V * dE/dstrain), the opposite sign to metatrain's
        # strain gradient convention (mt_stress = -1/V * dE/dstrain), so
        # nv_stress == -mt_stress. The looser tolerance absorbs float32 noise in
        # the near-zero off-diagonal terms (metatrain returns the raw,
        # unsymmetrized strain gradient while nvalchemi applies a symmetric
        # strain); the dominant diagonal stress matches to < 1e-2 / ~324.
        assert torch.allclose(nv_stress, -mt_stress, atol=2e-2, rtol=1e-4), (
            f"Stress mismatch: max |dS|={float((nv_stress + mt_stress).abs().max())}"
        )
