"""Molecular dynamics through nvalchemi's integrators.

Ten steps of Langevin NVT on a random periodic system, driven by
:class:`upet.nvalchemi.UPETWrapper`. This is the end-to-end check that the
wrapper survives a real dynamics loop: repeated ``compute`` calls with a
neighbor list rebuilt from moving positions, autograd forces flowing into
the integrator, and no state left behind on the batch between steps.
"""

from __future__ import annotations

import pytest
import torch


pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed; skipping")

from _helpers import ATOMIC_NUMBERS  # noqa: E402
from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.dynamics import (  # noqa: E402
    DynamicsStage,
    NVTLangevin,
    initialize_velocities,
)
from nvalchemi.hooks import NeighborListHook  # noqa: E402
from nvalchemi.neighbors import compute_neighbors  # noqa: E402


N_ATOMS = 12
BOX = 10.0  # Å; > 2 * cutoff, so the minimum-image convention holds
MIN_SEPARATION = 1.5  # Å; keeps the random draw off the repulsive wall
TEMPERATURE = 300.0  # K
TIMESTEP = 0.5  # fs
FRICTION = 0.01  # 1/fs
N_STEPS = 10


def _random_system(seed: int = 0) -> AtomicData:
    """Random cubic periodic system of ``N_ATOMS`` drawn from `ATOMIC_NUMBERS`.

    Positions are rejection-sampled so that no pair sits closer than
    ``MIN_SEPARATION`` under the minimum-image convention — an accidental
    near-overlap would produce forces large enough to blow the trajectory
    up, which would say nothing about the wrapper.

    ``velocities``, ``forces`` and ``energy`` are allocated up front:
    ``BaseDynamics.compute`` writes model outputs into existing batch
    fields via ``copy_`` and silently skips the ones that are absent.
    """
    generator = torch.Generator().manual_seed(seed)
    positions = torch.empty(0, 3)
    while positions.shape[0] < N_ATOMS:
        candidate = torch.rand(1, 3, generator=generator) * BOX
        if positions.shape[0] > 0:
            delta = positions - candidate
            delta -= BOX * torch.round(delta / BOX)  # minimum image
            if delta.norm(dim=-1).min() < MIN_SEPARATION:
                continue
        positions = torch.cat([positions, candidate])

    species = torch.tensor(ATOMIC_NUMBERS)[
        torch.randint(0, len(ATOMIC_NUMBERS), (N_ATOMS,), generator=generator)
    ]
    return AtomicData(
        positions=positions,
        atomic_numbers=species,
        cell=(torch.eye(3) * BOX).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
        velocities=torch.zeros(N_ATOMS, 3),
        forces=torch.zeros(N_ATOMS, 3),
        energy=torch.zeros(1, 1),
    )


def test_nvt_langevin_ten_steps(wrapper):
    """10 steps of NVT Langevin dynamics keep the trajectory finite and moving."""
    wrapper.model_config.active_outputs = {"energy", "forces"}

    batch = Batch.from_data_list([_random_system()])
    initial_positions = batch.positions.clone()

    initialize_velocities(
        batch.velocities,
        batch.atomic_masses,
        torch.full((1,), TEMPERATURE),
        batch.batch_idx.int(),
        random_seed=0,
    )

    # Seed the neighbor list and the forces of the first BAOAB half-kick;
    # from here on the hook rebuilds the list before every model call.
    neighbor_config = wrapper.model_config.neighbor_config
    compute_neighbors(batch, config=neighbor_config, format=neighbor_config.format)
    dynamics = NVTLangevin(
        model=wrapper,
        dt=TIMESTEP,
        temperature=TEMPERATURE,
        friction=FRICTION,
        random_seed=0,
        hooks=[
            NeighborListHook(config=neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE)
        ],
    )
    dynamics.compute(batch)

    final = dynamics.run(batch, n_steps=N_STEPS)

    assert dynamics.step_count == N_STEPS
    assert final.positions.shape == (N_ATOMS, 3)
    assert torch.isfinite(final.positions).all()
    assert torch.isfinite(final.velocities).all()
    assert torch.isfinite(final.forces).all()
    assert torch.isfinite(final.energy).all()

    # The atoms moved, but not by an implausible amount: ten 0.5 fs steps at
    # 300 K displace an atom by well under an ångström.
    displacement = (final.positions - initial_positions).norm(dim=-1)
    assert displacement.max() > 0.0
    assert displacement.max() < 1.0

    # The thermostat keeps the kinetic temperature in a physical range; the
    # band is wide because the model here is untrained.
    kinetic = 0.5 * (batch.atomic_masses.unsqueeze(-1) * final.velocities**2).sum()
    temperature = 2.0 * kinetic / (3.0 * N_ATOMS * 8.617333262e-5)
    assert 0.0 < temperature < 10.0 * TEMPERATURE

    # The dynamics loop leaves the batch's positions as a plain leaf tensor —
    # `UPETWrapper.forward` must not leak its gradient-enabled clone back onto
    # the batch, or the next step's autograd setup would fail.
    assert final.positions.is_leaf
    assert not final.positions.requires_grad
