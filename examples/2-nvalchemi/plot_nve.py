"""
NVE molecular dynamics
======================================

:py:class:`~nvalchemi.dynamics.NVE` integrates Newton's equations of motion
with the velocity Verlet algorithm, conserving the total energy. This
example runs it on a silicon supercell with
:py:class:`~upet.nvalchemi.UPETWrapper` supplying conservative forces.

.. note::

   This example requires the optional ``nvalchemi`` extra:
   ``pip install "upet[nvalchemi]"``. If it isn't installed, the example
   prints an install hint and exits without failing.
"""

import torch
from ase.build import bulk


try:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import NVE, initialize_velocities
    from nvalchemi.neighbors import compute_neighbors

    from upet.nvalchemi import UPETWrapper

    NVALCHEMI_AVAILABLE = True
except ImportError:
    NVALCHEMI_AVAILABLE = False

if NVALCHEMI_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UPETWrapper.from_checkpoint(
        model="pet-mad-xs", version="1.5.0", device=device
    )

    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond").repeat((2, 2, 2))
    batch = Batch.from_data_list(
        [AtomicData.from_atoms(atoms, device=device)], device=device
    )
    compute_neighbors(batch, config=model.model_config.neighbor_config)

    temperature = torch.full((batch.num_graphs,), 300.0, device=device)
    initialize_velocities(
        batch.velocities, batch.atomic_masses, temperature, batch.batch_idx.int()
    )

    nve = NVE(model=model, dt=1.0, n_steps=200)
    batch = nve.run(batch)

    print(f"Ran {nve.step_count} NVE steps")
    print(f"Energy : {batch.energy.item():+.4f} eV")
else:
    print(
        "This example requires the optional 'nvalchemi' extra: "
        "pip install 'upet[nvalchemi]'"
    )
