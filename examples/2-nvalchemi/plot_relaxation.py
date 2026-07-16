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
   ``pip install "upet[nvalchemi]"``. If it isn't installed, the example
   prints an install hint and exits without failing.
"""

import torch
from ase.build import bulk


try:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics import FIRE, ConvergenceHook
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

    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.rattle(0.1, seed=0)
    batch = Batch.from_data_list(
        [AtomicData.from_atoms(atoms, device=device)], device=device
    )
    compute_neighbors(batch, config=model.model_config.neighbor_config)

    fire = FIRE(
        model=model,
        dt=0.1,
        n_steps=300,
        convergence_hook=ConvergenceHook.from_fmax(0.02),
    )
    batch = fire.run(batch)

    print(f"Relaxed after {fire.step_count} steps")
    print(f"Energy : {batch.energy.item():+.4f} eV")
    print(f"Fmax   : {torch.linalg.vector_norm(batch.forces, dim=-1).max():.4f} eV/Å")
else:
    print(
        "This example requires the optional 'nvalchemi' extra: "
        "pip install 'upet[nvalchemi]'"
    )
