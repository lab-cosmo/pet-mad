"""
Batched evaluation
=================================

This example evaluates energy, forces, and stress for several structures at
once. Each ASE ``Atoms`` object is converted to an
:py:class:`~nvalchemi.data.AtomicData` instance with
:py:meth:`~nvalchemi.data.AtomicData.from_atoms`, and the resulting list is
collated into a single multi-graph :py:class:`~nvalchemi.data.Batch` with
:py:meth:`~nvalchemi.data.Batch.from_data_list`. A single forward pass through
:py:class:`~upet.nvalchemi.UPETWrapper` then evaluates all structures
together, which is substantially more efficient than looping over structures
one at a time (e.g. with the ASE calculator, see :ref:`usage_ase`).

.. note::

   This example requires the optional ``nvalchemi`` extra:
   ``pip install "upet[nvalchemi]"``. If it isn't installed, the example
   prints an install hint and exits without failing.
"""

import torch
from ase.build import bulk


try:
    from nvalchemi.data import AtomicData, Batch
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

    # %%
    # Building a batch of different structures
    # -------------------------------------------
    # Three diamond-structure crystals with different compositions and cell
    # sizes. ``Batch.from_data_list`` handles the ragged atom counts
    # transparently.

    structures = {
        "Si": bulk("Si", cubic=True, a=5.43, crystalstructure="diamond"),
        "C": bulk("C", cubic=True, a=3.57, crystalstructure="diamond"),
        "Ge": bulk("Ge", cubic=True, a=5.66, crystalstructure="diamond"),
    }
    data_list = [
        AtomicData.from_atoms(atoms, device=device) for atoms in structures.values()
    ]
    batch = Batch.from_data_list(data_list, device=device)
    print(f"Batch: {batch.num_graphs} systems, {batch.num_nodes} atoms total")

    # %%
    # Neighbor list and a single batched forward pass
    # ---------------------------------------------------
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    outputs = model(batch)

    # %%
    # Per-system results
    # -------------------
    # ``outputs["energy"]`` has shape ``[num_graphs, 1]``; forces are stacked
    # over all atoms in the batch, ordered the same way as ``data_list``.
    energies = outputs["energy"].squeeze(-1).detach().cpu()
    for name, energy in zip(structures.keys(), energies, strict=True):
        print(f"  {name:>2s}: E = {energy.item():+.4f} eV")
else:
    print(
        "This example requires the optional 'nvalchemi' extra: "
        "pip install 'upet[nvalchemi]'"
    )
