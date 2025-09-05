PET-MAD: Universal Models for Advanced Atomistic Simulations
=============================================================

.. image:: ../static/pet-mad-logo-with-text.svg
   :width: 400
   :align: center
   :alt: PET-MAD Logo

PET-MAD is a universal interatomic potential for advanced materials modeling across the periodic table. This model is based on the **Point Edge Transformer (PET)** model trained on the **Massive Atomic Diversity (MAD) Dataset** and is capable of predicting energies and forces in complex atomistic simulations.

In addition, it contains **PET-MAD-DOS** - a universal model for predicting the density of states (DOS) of materials, as well as their Fermi levels and bandgaps. **PET-MAD-DOS** uses a slightly modified **PET** architecture, and the same **MAD** dataset.

Key Features
------------

- **Universality**: PET-MAD models are generally-applicable, and can be used for predicting energies and forces, as well as the density of states, Fermi levels, and bandgaps for a wide range of materials and molecules.
- **Accuracy**: PET-MAD models achieve high accuracy in various types of atomistic simulations of organic and inorganic systems, comparable with system-specific models, while being fast and efficient.
- **Efficiency**: PET-MAD models are highly computationally efficient and have low memory usage, making them suitable for large-scale simulations.
- **Infrastructure**: Various MD engines are available for diverse research and application needs.
- **HPC Compatibility**: Efficient in HPC environments for extensive simulations.

Supported Simulation Engines
-----------------------------

PET-MAD integrates with the following atomistic simulation engines:

- **Atomic Simulation Environment (ASE)**
- **LAMMPS** (including KOKKOS support)
- **i-PI**
- **OpenMM** (coming soon)
- **GROMACS** (coming soon)

Quick Start
-----------

Install PET-MAD using pip:

.. code-block:: bash

   pip install pet-mad

Use PET-MAD with ASE:

.. code-block:: python

   from pet_mad.calculator import PETMADCalculator
   from ase.build import bulk

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   calculator = PETMADCalculator(version="latest", device="cpu")
   atoms.calc = calculator

   energy = atoms.get_potential_energy()
   forces = atoms.get_forces()

Documentation Contents
----------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/usage_examples

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/ase_interface
   tutorials/lammps_interface
   tutorials/dos_calculations
   tutorials/uncertainty_quantification
   tutorials/dataset_exploration
   tutorials/batched_evaluation

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/calculator
   api/models
   api/explore
   api/utils

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   citation
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
