"""
Basics: single-structure evaluation
=================================================

:py:class:`~upet.nvalchemi.UPETWrapper` wraps a UPET / PET-MAD checkpoint as
an `nvalchemi-toolkit <https://github.com/NVIDIA/nvalchemi-toolkit>`_
``BaseModelMixin`` model, so it can be driven through nvalchemi's batched
:py:class:`~nvalchemi.data.Batch` data pipeline instead of ASE ``Atoms``.
This is the entry point for large-scale batched inference (see
:doc:`plot_batched_eval`) and for nvalchemi's GPU-accelerated MD
integrators (see :doc:`plot_nvt`).

This example builds a bulk silicon cell as an ASE ``Atoms`` object,
converts it to a single-graph :py:class:`~nvalchemi.data.Batch` with
:py:meth:`~nvalchemi.data.AtomicData.from_atoms`, and evaluates energy,
forces, and stress with :py:class:`~upet.nvalchemi.UPETWrapper`.

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

# %%
# Loading a checkpoint
# --------------------
# :py:meth:`~upet.nvalchemi.UPETWrapper.from_checkpoint` fetches a named
# UPET model from HuggingFace (here the small ``pet-mad-xs`` model), or
# loads a local checkpoint file when ``checkpoint_path`` is given instead.

if NVALCHEMI_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UPETWrapper.from_checkpoint(
        model="pet-mad-xs", version="1.5.0", device=device
    )

    # %%
    # From ASE ``Atoms`` to a single-graph ``Batch``
    # ------------------------------------------------
    # :py:meth:`~nvalchemi.data.AtomicData.from_atoms` builds an
    # :py:class:`~nvalchemi.data.AtomicData` instance directly from an ASE
    # ``Atoms`` object, carrying over positions, atomic numbers, cell and
    # PBC. :py:meth:`~nvalchemi.data.Batch.from_data_list` then promotes it
    # to a single-graph batch.

    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    data = AtomicData.from_atoms(atoms, device=device)
    batch = Batch.from_data_list([data], device=device)

    # %%
    # Neighbor list and evaluation
    # ------------------------------
    # :py:func:`~nvalchemi.neighbors.compute_neighbors` is the one-shot
    # convenience function for populating a batch's neighbor list outside
    # a dynamics loop; ``model.model_config.neighbor_config`` already
    # encodes the cutoff and list format the model expects.
    compute_neighbors(batch, config=model.model_config.neighbor_config)

    outputs = model(batch)

    print(f"Energy : {outputs['energy'].item():+.4f} eV")
    print(f"Forces :\n{outputs['forces'].detach().cpu().numpy()}")
    print(f"Stress :\n{outputs['stress'].squeeze(0).detach().cpu().numpy()}")
else:
    print(
        "This example requires the optional 'nvalchemi' extra: "
        "pip install 'upet[nvalchemi]'"
    )
