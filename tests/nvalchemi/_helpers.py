"""Shared constants, hyper-parameters and `AtomicData` builders.

Imported by the ``test_wrapper_*`` modules *after* their
``pytest.importorskip("nvalchemi")`` guard, so importing ``nvalchemi`` at
module scope here is safe.
"""

from __future__ import annotations

import torch
from nvalchemi.data import AtomicData


ATOMIC_NUMBERS = [1, 6, 8]  # H, C, O — keeps species_to_species_index small
CUTOFF = 5.0
D_NODE = 16
D_PET = 8


def tiny_hypers() -> dict:
    """Return a minimal PET hyper-parameter dict for fast unit tests."""
    return {
        "cutoff": CUTOFF,
        "cutoff_width": 0.5,
        "cutoff_function": "Bump",
        "d_pet": D_PET,
        "d_node": D_NODE,
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


def make_water(device: str = "cpu") -> AtomicData:
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


def make_two_atoms(device: str = "cpu") -> AtomicData:
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


def make_pbc_water(device: str = "cpu") -> AtomicData:
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
