.. _quickstart:

Quick start
===========

Install UPET with ``pip``:

.. code-block:: bash

   pip install upet

Then run a single-point energy and force evaluation on a bulk silicon cell
using the ASE-compatible :py:class:`~upet.calculator.UPETCalculator`:

.. code-block:: python

   from upet.calculator import UPETCalculator
   from ase.build import bulk

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   calculator = UPETCalculator(model="pet-mad-s", version="1.5.0", device="cpu")
   atoms.calc = calculator

   energy = atoms.get_potential_energy()
   forces = atoms.get_forces()

For DOS calculations, you can use the :py:class:`~upet.calculator.PETMADDOSCalculator`:

.. code-block:: python

   from upet.calculator import PETMADDOSCalculator
   from ase.build import bulk

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   calculator = PETMADDOSCalculator(version="latest", device="cpu")
   energies, predicted_DOS = pet_mad_dos_calculator.calculate_dos(atoms)

where the energies refer to the energy grid that the DOS is projected on and has units
of eV and the DOS refers to the density of states values and has units of states/eV. 

In both examples, the first call downloads the model checkpoint from the `HuggingFace
repository <https://huggingface.co/lab-cosmo/upet>`_ and caches it locally,
so subsequent calls are fast.

Next steps
----------

- See :ref:`installation` for the full list of installation methods (PyPI,
  GitHub, ``uv``, pinning a specific version).
- See :ref:`usage` for the complete feature surface: ASE workflows,
  batched evaluation with ``metatrain``, LAMMPS, i-PI, TorchSim, and
  GROMACS.
- See :ref:`models` for the list of available pre-trained models and
  their recommended use cases.
