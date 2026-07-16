.. _usage_nvalchemi:

nvalchemi-toolkit
=================

UPET models can be driven through `nvalchemi-toolkit
<https://github.com/NVIDIA/nvalchemi-toolkit>`_, NVIDIA's GPU-native
toolkit for batched inference and molecular dynamics.
:py:class:`~upet.nvalchemi.UPETWrapper` wraps any UPET / PET-MAD checkpoint
as an ``nvalchemi-toolkit`` ``BaseModelMixin`` model, so it can be driven
through nvalchemi's batched :py:class:`~nvalchemi.data.Batch` data pipeline
and its ``FIRE`` / ``NVE`` / ``NVTLangevin`` / ``NPT`` integrators.

Requires the optional ``nvalchemi`` extra (see :ref:`installation`):

.. code-block:: bash

   pip install "upet[nvalchemi]"


Usage
-----

Single-structure evaluation
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Convert an ASE ``Atoms`` object to an :py:class:`~nvalchemi.data.AtomicData`
instance with :py:meth:`~nvalchemi.data.AtomicData.from_atoms`, promote it
to a single-graph :py:class:`~nvalchemi.data.Batch`, compute its neighbor
list, and evaluate:

.. code-block:: python

   import torch
   from ase.build import bulk
   from nvalchemi.data import AtomicData, Batch
   from nvalchemi.neighbors import compute_neighbors
   from upet.nvalchemi import UPETWrapper

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = UPETWrapper.from_checkpoint(model="pet-mad-s", version="1.5.0", device=device)

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   data = AtomicData.from_atoms(atoms, device=device)
   batch = Batch.from_data_list([data], device=device)
   compute_neighbors(batch, config=model.model_config.neighbor_config)

   outputs = model(batch)
   energy = outputs["energy"]
   forces = outputs["forces"]
   stress = outputs["stress"]

``model.model_config.neighbor_config`` carries the cutoff and neighbor-list
format the model expects, so :py:func:`~nvalchemi.neighbors.compute_neighbors`
never needs the cutoff repeated by hand.

Batched evaluation
^^^^^^^^^^^^^^^^^^

Passing several structures at once only requires collecting one
``AtomicData`` per structure and collating them with
:py:meth:`~nvalchemi.data.Batch.from_data_list`; a single forward pass then
evaluates all of them together:

.. code-block:: python

   structures = [
       bulk("Si", cubic=True, a=5.43, crystalstructure="diamond"),
       bulk("C", cubic=True, a=3.57, crystalstructure="diamond"),
       bulk("Ge", cubic=True, a=5.66, crystalstructure="diamond"),
   ]
   data_list = [AtomicData.from_atoms(atoms, device=device) for atoms in structures]
   batch = Batch.from_data_list(data_list, device=device)
   compute_neighbors(batch, config=model.model_config.neighbor_config)

   outputs = model(batch)
   energies = outputs["energy"]  # one row per structure, shape [3, 1]

``Batch.from_data_list`` handles the differing atom counts transparently;
``outputs["energy"]`` comes back with one row per input structure, and
``outputs["forces"]`` is stacked over all atoms in the batch in the same
order as ``data_list``.


Examples
--------

Runnable end-to-end workflows built on top of ``UPETWrapper``, driving
nvalchemi's ``FIRE``, ``NVE``, ``NVTLangevin``, and ``NPT`` integrators:

* :doc:`/generated_examples/2-nvalchemi/plot_basics` —
  single-structure energy / forces / stress evaluation.
* :doc:`/generated_examples/2-nvalchemi/plot_batched_eval` —
  batched evaluation of several structures in a single forward pass.
* :doc:`/generated_examples/2-nvalchemi/plot_relaxation` —
  geometry optimization with the ``FIRE`` integrator and a force-based
  :py:class:`~nvalchemi.dynamics.ConvergenceHook`.
* :doc:`/generated_examples/2-nvalchemi/plot_nve` —
  microcanonical (NVE) molecular dynamics via velocity Verlet.
* :doc:`/generated_examples/2-nvalchemi/plot_nvt` —
  canonical (NVT) molecular dynamics with a Langevin thermostat.
* :doc:`/generated_examples/2-nvalchemi/plot_npt` —
  isothermal-isobaric (NPT) molecular dynamics with a Nosé-Hoover
  thermostat and Martyna-Tobias-Klein barostat.
