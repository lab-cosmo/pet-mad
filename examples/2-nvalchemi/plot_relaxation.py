"""
Geometry optimization (FIRE)
===============================================

:py:class:`~nvalchemi.dynamics.FIRE` drives atomic positions toward a local
energy minimum using the Fast Inertial Relaxation Engine algorithm, with
:py:class:`~upet.nvalchemi.UPETWrapper` supplying forces and
:py:class:`~nvalchemi.dynamics.ConvergenceHook` stopping the run early once
the maximum force norm drops below a threshold.

.. note::

   This example requires the optional ``nvalchemi`` extra:
   ``pip install "upet[nvalchemi]"``.
"""

import torch
from ase.build import bulk
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE, ConvergenceHook, DynamicsStage
from nvalchemi.hooks import NeighborListHook
from nvalchemi.neighbors import compute_neighbors

from upet.nvalchemi import UPETWrapper


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UPETWrapper.from_checkpoint(model="pet-mad-xs", version="1.5.0", device=device)

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
atoms.rattle(0.1, seed=0)
data = AtomicData.from_atoms(atoms, device=device)

# The integrators write model outputs back into the batch fields that already
# exist and silently skip the ones that don't, while `AtomicData.from_atoms`
# only fills the fields the `Atoms` object itself carries -- so the output
# buffers have to be allocated up front.
data.energy = torch.zeros(1, 1, device=device)
data.forces = torch.zeros_like(data.positions)

batch = Batch.from_data_list([data], device=device)
compute_neighbors(batch, config=model.model_config.neighbor_config)

fire = FIRE(
    model=model,
    dt=0.1,
    n_steps=300,
    convergence_hook=ConvergenceHook.from_fmax(0.02),
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
fire.compute(batch)
batch = fire.run(batch)

print(f"Relaxed after {fire.step_count} steps")
print(f"Energy : {batch.energy.item():+.4f} eV")
print(f"Fmax   : {torch.linalg.vector_norm(batch.forces, dim=-1).max():.4f} eV/Å")
