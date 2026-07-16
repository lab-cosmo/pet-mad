"""
NVT molecular dynamics
======================================

:py:class:`~nvalchemi.dynamics.NVTLangevin` samples the canonical ensemble
via a Langevin thermostat. This example runs it on a small water cluster
with :py:class:`~upet.nvalchemi.UPETWrapper` supplying conservative forces.

.. note::

   This example requires the optional ``nvalchemi`` extra:
   ``pip install "upet[nvalchemi]"``. If it isn't installed, the example
   prints an install hint and exits without failing.
"""

import torch
from ase.build import molecule


try:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import NVTLangevin, initialize_velocities
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

    # Three water molecules spaced out into a loose cluster.
    cluster = molecule("H2O")
    for i in range(1, 3):
        shifted = molecule("H2O")
        shifted.translate([3.5 * i, 0.0, 0.0])
        cluster += shifted

    batch = Batch.from_data_list(
        [AtomicData.from_atoms(cluster, device=device)], device=device
    )
    compute_neighbors(batch, config=model.model_config.neighbor_config)

    temperature = torch.full((batch.num_graphs,), 300.0, device=device)
    initialize_velocities(
        batch.velocities, batch.atomic_masses, temperature, batch.batch_idx.int()
    )

    nvt = NVTLangevin(model=model, dt=0.1, temperature=300.0, friction=0.5, n_steps=200)
    batch = nvt.run(batch)

    print(f"Ran {nvt.step_count} NVT steps")
    print(f"Energy : {batch.energy.item():+.4f} eV")
else:
    print(
        "This example requires the optional 'nvalchemi' extra: "
        "pip install 'upet[nvalchemi]'"
    )
