"""
NVT molecular dynamics
======================================

:py:class:`~nvalchemi.dynamics.NVTLangevin` samples the canonical ensemble
via a Langevin thermostat. This example runs it on a small water cluster
with :py:class:`~upet.nvalchemi.UPETWrapper` supplying conservative forces.

.. note::

   This example requires the optional ``nvalchemi`` extra:
   ``pip install "upet[nvalchemi]"``.
"""

import torch
from ase.build import molecule
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage, NVTLangevin, initialize_velocities
from nvalchemi.hooks import NeighborListHook
from nvalchemi.neighbors import compute_neighbors

from upet.nvalchemi import UPETWrapper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UPETWrapper.from_checkpoint(model="pet-mad-xs", version="1.5.0", device=device)

# Three water molecules spaced out into a loose cluster.
cluster = molecule("H2O")
for i in range(1, 3):
    shifted = molecule("H2O")
    shifted.translate([3.5 * i, 0.0, 0.0])
    cluster += shifted

data = AtomicData.from_atoms(cluster, device=device)

# The integrators write model outputs back into the batch fields that already
# exist and silently skip the ones that don't, while `AtomicData.from_atoms`
# only fills the fields the `Atoms` object itself carries -- so the output
# buffers have to be allocated up front.
data.energy = torch.zeros(1, 1, device=device)
data.forces = torch.zeros_like(data.positions)

batch = Batch.from_data_list([data], device=device)
compute_neighbors(batch, config=model.model_config.neighbor_config)

temperature = torch.full((batch.num_graphs,), 300.0, device=device)
initialize_velocities(
    batch.velocities, batch.atomic_masses, temperature, batch.batch_idx.int()
)

# `NeighborListHook` rebuilds the neighbor list during the run, once an atom
# has moved further than the `skin` buffer.
nvt = NVTLangevin(
    model=model,
    dt=0.1,
    temperature=300.0,
    friction=0.5,
    n_steps=200,
    hooks=[
        NeighborListHook(
            model.model_config.neighbor_config,
            skin=0.5,
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
    ],
)

# The first integrator step already reads the forces, so the batch needs one
# model evaluation before the loop starts.
nvt.compute(batch)
batch = nvt.run(batch)

print(f"Ran {nvt.step_count} NVT steps")
print(f"Energy : {batch.energy.item():+.4f} eV")
