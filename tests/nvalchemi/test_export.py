"""`UPETWrapper.export_model` snapshots and state dicts."""

from __future__ import annotations

import pytest
import torch


pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed; skipping")

from metatrain.pet.modules.backend import PETBackend  # noqa: E402


def test_export_snapshot(wrapper, tmp_path):
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


def test_export_state_dict(wrapper, tmp_path):
    path = tmp_path / "pet_sd.pt"
    wrapper.export_model(path, as_state_dict=True)
    assert path.exists()
    sd = torch.load(path, weights_only=True)
    assert isinstance(sd, dict)
    assert any("gnn_layers" in k for k in sd.keys())


def test_reload_snapshot_into_new_backend(wrapper, tmp_path):
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
