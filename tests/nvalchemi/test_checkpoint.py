"""Integration tests against a real PET checkpoint.

These download ``pet-mad-xs`` from HuggingFace and compare
:class:`~upet.nvalchemi.UPETWrapper` against the original metatrain
model, so they are network-bound and skip when the fetch fails.
"""

from __future__ import annotations

import ase
import numpy as np
import pytest
import torch
from metatomic.torch import ModelOutput

from upet.ase import UPETCalculator


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


# Fetched from the `lab-cosmo/upet` HuggingFace repo (see UPETWrapper.from_checkpoint).
_CHECKPOINT_MODEL = "pet-mad-xs"
_CHECKPOINT_VERSION = "1.5.0"  # 'grid' adaptive cutoff
# Not yet published on HuggingFace; must be provided locally to run this test.
_SOLVER_CHECKPOINT_PATH = "pet-mad-xs-v1.6.0.ckpt"  # 'solver' adaptive cutoff
CHECKPOINT_CUTOFF = 7.5  # the cutoff pet-mad-xs was trained with


def _crystal(device: str = "cpu") -> AtomicData:
    return AtomicData(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]], dtype=torch.float32
        ).to(device=device),
        atomic_numbers=torch.tensor([6, 6], dtype=torch.long).to(device=device),
        cell=3.0 * torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).to(device=device),
        pbc=torch.tensor([True] * 3, dtype=torch.bool).reshape(1, 3).to(device=device),
    )


def _batch(dtype: torch.dtype = torch.float32) -> Batch:
    data = _crystal()
    batch = Batch.from_data_list([data])
    compute_neighbors(batch, cutoff=CHECKPOINT_CUTOFF, format=NeighborListFormat.COO)
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


def test_is_upet_wrapper(real_wrapper_cpu):
    assert isinstance(real_wrapper_cpu, UPETWrapper)


def test_wrapper_has_correct_config(real_wrapper_cpu):
    cfg = real_wrapper_cpu.model_config
    assert "forces" in cfg.autograd_outputs
    assert "stress" in cfg.autograd_outputs
    assert "energy" in cfg.outputs
    assert real_wrapper_cpu.cutoff > 0.0
    assert cfg.neighbor_config is not None
    assert cfg.neighbor_config.format == NeighborListFormat.COO


def test_inference_targets_finite(real_wrapper_cpu):
    batch = _batch()
    out = real_wrapper_cpu.forward(batch)
    e = out["energy"]
    f = out["forces"]
    s = out["stress"]
    assert e.shape == (1, 1)
    assert torch.isfinite(e).all()
    assert f.shape == (2, 3)
    assert torch.isfinite(f).all()
    assert s.shape == (1, 3, 3)
    assert torch.isfinite(s).all()


def test_batch_determinism(real_wrapper_cpu):
    """Same input → same output across two consecutive calls."""
    b1 = _batch()
    b2 = _batch()
    out_1 = real_wrapper_cpu.forward(b1)
    out_2 = real_wrapper_cpu.forward(b2)
    assert torch.allclose(out_1["energy"], out_2["energy"], atol=1e-8)
    assert torch.allclose(out_1["forces"], out_2["forces"], atol=1e-8)
    assert torch.allclose(out_1["stress"], out_2["stress"], atol=1e-8)


def test_batched_matches_single(real_wrapper_cpu):
    """Two-water batch energies equal the single-water energy."""
    single = real_wrapper_cpu.forward(_batch())
    multi_batch = Batch.from_data_list([_crystal(), _crystal()])
    compute_neighbors(
        multi_batch, cutoff=CHECKPOINT_CUTOFF, format=NeighborListFormat.COO
    )
    multi = real_wrapper_cpu.forward(multi_batch)
    assert torch.allclose(multi["energy"][0], single["energy"][0], atol=1e-8)
    assert torch.allclose(multi["energy"][1], single["energy"][0], atol=1e-8)


def test_compute_embeddings_run(real_wrapper_cpu):
    batch = _batch()
    result = real_wrapper_cpu.compute_embeddings(batch)
    assert result.node_embeddings.shape[0] == 2
    assert result.graph_embeddings.shape == (1, result.node_embeddings.shape[1])


def test_embeddings_match_ase_features():
    """compute_embeddings reproduces metatrain PET's per-atom ``feature`` output.

    The metatrain ``feature`` output is built by
    ``metatrain.pet.model.PET._get_output_features`` as the concatenation of
    the per-layer node features with the cutoff-weighted, neighbor-summed
    per-layer edge features — exactly what :meth:`UPETWrapper.compute_embeddings`
    computes. This verifies they agree value-for-value (in float64).
    """

    checkpoint_path = _fetch_checkpoint_path(_CHECKPOINT_MODEL, _CHECKPOINT_VERSION)
    ase_calculator = UPETCalculator(
        checkpoint_path=checkpoint_path, dtype=torch.float64
    )
    outputs = {"features": ModelOutput()}

    data = _crystal()
    atoms = ase.Atoms(
        positions=data.positions.numpy(),
        numbers=data.atomic_numbers.numpy(),
        cell=data.cell.squeeze().numpy(),
        pbc=[True, True, True],
    )
    ase_output = ase_calculator._base_calculator.run_model(atoms, outputs=outputs)
    ase_feature = ase_output["features"].block().values

    batch = Batch.from_data_list([data])
    # nvalchemi embeddings (float64) on the same structure.
    nv_wrapper = UPETWrapper.from_checkpoint(
        checkpoint_path=checkpoint_path, dtype=torch.float64
    )
    batch.positions = batch.positions.to(dtype=torch.float64)
    batch.cell = batch.cell.to(dtype=torch.float64)
    compute_neighbors(batch, cutoff=CHECKPOINT_CUTOFF, format=NeighborListFormat.COO)
    nv_embeddings = nv_wrapper.compute_embeddings(batch).graph_embeddings.detach()

    assert nv_embeddings.shape == ase_feature.shape
    torch.testing.assert_close(nv_embeddings, ase_feature, atol=1e-8, rtol=1e-8)


def test_export_and_reload(real_wrapper_cpu, tmp_path):
    path = tmp_path / "pet_snapshot.pt"
    real_wrapper_cpu.export_model(path)
    snapshot = torch.load(path, weights_only=False)
    new_backend = PETBackend(snapshot["hypers"], snapshot["atomic_types"])
    new_backend.add_output("energy", {"energy___0": [1]})
    new_backend.load_state_dict(snapshot["backend_state_dict"], strict=True)


def test_metatrain_model_compatibility(real_wrapper_cpu):
    """UPETWrapper predictions match those from the original metatrain model."""
    checkpoint_path = _fetch_checkpoint_path(_CHECKPOINT_MODEL, _CHECKPOINT_VERSION)
    ase_calculator = UPETCalculator(
        checkpoint_path=checkpoint_path, dtype=torch.float64
    )

    data = _crystal()
    atoms = ase.Atoms(
        positions=data.positions.cpu().numpy(),
        numbers=data.atomic_numbers.cpu().numpy(),
        cell=data.cell.cpu().squeeze().numpy() if data.cell is not None else None,
        pbc=data.pbc.cpu().squeeze().numpy() if data.pbc is not None else None,
    )
    atoms.calc = ase_calculator

    ase_energy = atoms.get_potential_energy()
    ase_forces = atoms.get_forces()
    ase_stress = atoms.get_stress(voigt=False)

    nv_output = real_wrapper_cpu.forward(_batch())
    nv_energy = nv_output["energy"].detach().squeeze().item()
    nv_forces = nv_output["forces"].detach().numpy()
    nv_stress = nv_output["stress"].detach().squeeze().numpy()

    assert abs(nv_energy - ase_energy) < 1e-5, (
        f"Energy mismatch: nvalchemi={nv_energy}, ase={ase_energy}"
    )
    assert np.allclose(nv_forces, ase_forces, atol=1e-5), (
        f"Force mismatch: max |dF|={float((nv_forces - ase_forces).abs().max())}"
    )
    assert np.allclose(nv_stress, ase_stress, atol=1e-5), (
        f"Stress mismatch: max |dS|={float((nv_stress - ase_stress).abs().max())}"
    )
