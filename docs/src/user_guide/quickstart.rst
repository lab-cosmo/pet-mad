#############
 Quick Start
#############

This guide will help you get started with PET-MAD quickly.

**********************
 Basic Usage with ASE
**********************

The simplest way to use PET-MAD is through the ASE (Atomic Simulation
Environment) interface:

.. code:: python

   from pet_mad.calculator import PETMADCalculator
   from ase.build import bulk

   # Create a silicon crystal
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")

   # Initialize the PET-MAD calculator
   calculator = PETMADCalculator(version="latest", device="cpu")
   atoms.calc = calculator

   # Calculate energy and forces
   energy = atoms.get_potential_energy()
   forces = atoms.get_forces()

   print(f"Energy: {energy:.3f} eV")
   print(f"Forces shape: {forces.shape}")

******************
 Available Models
******************

PET-MAD provides several pre-trained models:

-  **v1.0.2**: Stable PET-MAD model (recommended for production use)
-  **v1.1.0**: Development version with non-conservative forces and
   stresses
-  **latest**: Points to the latest stable version (v1.0.2)

.. code:: python

   # Use specific version
   calculator = PETMADCalculator(version="v1.0.2", device="cpu")

   # Use latest stable version (default)
   calculator = PETMADCalculator(version="latest", device="cpu")

******************
 GPU Acceleration
******************

To use GPU acceleration (if available):

.. code:: python

   import torch

   device = "cuda" if torch.cuda.is_available() else "cpu"
   calculator = PETMADCalculator(version="latest", device=device)

**************************************
 Density of States (DOS) Calculations
**************************************

PET-MAD-DOS can predict electronic density of states:

.. code:: python

   from pet_mad.calculator import PETMADDOSCalculator
   from ase.build import bulk

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   dos_calculator = PETMADDOSCalculator(version="latest", device="cpu")

   # Calculate DOS
   energies, dos = dos_calculator.calculate_dos(atoms)

   # Calculate bandgap and Fermi level
   bandgap = dos_calculator.calculate_bandgap(atoms)
   fermi_level = dos_calculator.calculate_efermi(atoms)

   print(f"Bandgap: {bandgap:.3f} eV")
   print(f"Fermi level: {fermi_level:.3f} eV")

**************
 What's Next?
**************

-  Learn about :doc:`usage_examples` for more advanced use cases
-  Explore :doc:`../tutorials/ase_interface` for detailed ASE usage
-  Check out :doc:`../tutorials/lammps_interface` for LAMMPS integration
-  See :doc:`../tutorials/uncertainty_quantification` for uncertainty
   estimation
