"""
NVE molecular dynamics
======================================

:py:class:`~nvalchemi.dynamics.NVE` integrates Newton's equations of motion
with the velocity Verlet algorithm, conserving the total energy. This
example runs it on a silicon supercell with
:py:class:`~upet.nvalchemi.UPETWrapper` supplying conservative forces.

.. note::

   This example requires the optional ``nvalchemi`` extra:
   ``pip install "upet[nvalchemi]"``.
"""

import torch
from ase.build import bulk
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVE, DynamicsStage, initialize_velocities
from nvalchemi.hooks import NeighborListHook
from nvalchemi.neighbors import compute_neighbors

from upet.nvalchemi import UPETWrapper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UPETWrapper.from_checkpoint(model="pet-mad-xs", version="1.5.0", device=device)

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond").repeat((2, 2, 2))
data = AtomicData.from_atoms(atoms, device=device)

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
nve = NVE(
    model=model,
    dt=1.0,
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
nve.compute(batch)
batch = nve.run(batch)

print(f"Ran {nve.step_count} NVE steps")
print(f"Energy : {batch.energy.item():+.4f} eV")
