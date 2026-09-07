"""
NPT molecular dynamics
======================================

:py:class:`~nvalchemi.dynamics.NPT` samples the isothermal-isobaric ensemble
via a Martyna-Tobias-Klein barostat coupled to a Nosé-Hoover thermostat, so
both the atomic positions and the simulation cell evolve. This example runs
it on a silicon supercell at 300 K and 1 bar, with
:py:class:`~upet.nvalchemi.UPETWrapper` supplying forces and stress.

.. note::

   This example requires the optional ``nvalchemi`` extra:
   ``pip install "upet[nvalchemi]"``.
"""

import torch
from ase import units
from ase.build import bulk
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NPT, DynamicsStage, initialize_velocities
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
data.stress = torch.zeros(1, 3, 3, device=device)

batch = Batch.from_data_list([data], device=device)
compute_neighbors(batch, config=model.model_config.neighbor_config)

temperature = torch.full((batch.num_graphs,), 300.0, device=device)
initialize_velocities(
    batch.velocities, batch.atomic_masses, temperature, batch.batch_idx.int()
)

# nvalchemi's pressure follows the same eV/Å³ convention as its Cauchy
# stress output, matching ASE's internal unit system for stress.
bar = 1e-4 * units.GPa

npt = NPT(
    model=model,
    dt=1.0,
    temperature=300.0,
    pressure=1.0 * bar,
    barostat_time=100.0,
    thermostat_time=25.0,
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
npt.compute(batch)
batch = npt.run(batch)

print(f"Ran {npt.step_count} NPT steps")
print(f"Energy : {batch.energy.item():+.4f} eV")
print(f"Volume : {torch.linalg.det(batch.cell).abs().item():.2f} Å³")
