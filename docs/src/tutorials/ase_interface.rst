########################
 ASE Interface Tutorial
########################

This tutorial covers the comprehensive usage of PET-MAD with the Atomic
Simulation Environment (ASE).

**************
 Introduction
**************

The ASE interface is the primary way to use PET-MAD for single-point
calculations and molecular dynamics simulations. It provides a familiar
interface for users already working with ASE.

*************
 Basic Setup
*************

.. code:: python

   from pet_mad.calculator import PETMADCalculator
   from ase.build import bulk, molecule
   import torch

   # Choose device
   device = "cuda" if torch.cuda.is_available() else "cpu"
   print(f"Using device: {device}")

   # Initialize calculator
   calculator = PETMADCalculator(version="latest", device=device)

***************************
 Single-Point Calculations
***************************

Energy and Forces
=================

.. code:: python

   from ase.build import bulk

   # Create a silicon supercell
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   atoms = atoms.repeat((2, 2, 2))  # 64-atom supercell

   atoms.calc = calculator

   # Calculate properties
   energy = atoms.get_potential_energy()
   forces = atoms.get_forces()
   stress = atoms.get_stress()

   print(f"Total energy: {energy:.3f} eV")
   print(f"Energy per atom: {energy/len(atoms):.3f} eV/atom")
   print(f"Max force: {np.max(np.linalg.norm(forces, axis=1)):.3f} eV/Å")
   print(f"Stress tensor shape: {stress.shape}")

Molecular Systems
=================

.. code:: python

   from ase.build import molecule
   from ase.visualize import view

   # Water molecule
   water = molecule("H2O")
   water.calc = calculator

   energy = water.get_potential_energy()
   forces = water.get_forces()

   print(f"Water energy: {energy:.3f} eV")

   # Methane molecule
   methane = molecule("CH4")
   methane.calc = calculator

   ch4_energy = methane.get_potential_energy()
   print(f"Methane energy: {ch4_energy:.3f} eV")

***********************
 Geometry Optimization
***********************

.. code:: python

   from ase.build import molecule
   from ase.optimize import BFGS
   from ase.constraints import FixAtoms
   import numpy as np

   # Create a slightly distorted water molecule
   water = molecule("H2O")
   positions = water.get_positions()
   positions += np.random.normal(0, 0.1, positions.shape)  # Add noise
   water.set_positions(positions)

   water.calc = calculator

   # Optimize geometry
   optimizer = BFGS(water, trajectory="water_opt.traj")
   optimizer.run(fmax=0.01)  # Optimize until forces < 0.01 eV/Å

   final_energy = water.get_potential_energy()
   print(f"Optimized energy: {final_energy:.3f} eV")

Surface Calculations
====================

.. code:: python

   from ase.build import surface, add_adsorbate
   from ase.optimize import BFGS

   # Create a Si(100) surface
   slab = surface("Si", (1, 0, 0), 4, vacuum=10.0)
   slab = slab.repeat((2, 2, 1))  # Larger surface

   # Add hydrogen adsorbate
   add_adsorbate(slab, "H", 2.0, position="ontop")

   slab.calc = calculator

   # Optimize only the adsorbate and top layer
   constraint = FixAtoms(indices=range(len(slab) - 8))  # Fix bottom atoms
   slab.set_constraint(constraint)

   optimizer = BFGS(slab, trajectory="surface_opt.traj")
   optimizer.run(fmax=0.05)

********************
 Molecular Dynamics
********************

NVE Dynamics
============

.. code:: python

   from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
   from ase.md.verlet import VelocityVerlet
   from ase import units
   import numpy as np

   # Create system
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   atoms = atoms.repeat((3, 3, 3))  # 216 atoms
   atoms.calc = calculator

   # Set initial velocities for 300K
   MaxwellBoltzmannDistribution(atoms, temperature_K=300)

   # Create MD object
   md = VelocityVerlet(atoms, timestep=1.0 * units.fs)


   def print_energy(a=atoms):
       epot = a.get_potential_energy() / len(a)
       ekin = a.get_kinetic_energy() / len(a)
       print(
           f"Energy per atom: Epot = {epot:.3f} eV  Ekin = {ekin:.3f} eV  "
           f"Etot = {epot+ekin:.3f} eV  T = {ekin/(1.5*units.kB):.1f} K"
       )


   # Run MD
   for i in range(100):
       md.run(10)  # Run 10 steps
       print_energy()

NVT Dynamics
============

.. code:: python

   from ase.md.langevin import Langevin
   from ase import units

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   atoms = atoms.repeat((2, 2, 2))
   atoms.calc = calculator

   # Set initial velocities
   MaxwellBoltzmannDistribution(atoms, temperature_K=300)

   # Langevin thermostat
   md = Langevin(atoms, timestep=1.0 * units.fs, temperature_K=300, friction=0.01)

   # Run simulation with trajectory output
   from ase.io import Trajectory

   traj = Trajectory("nvt_md.traj", "w", atoms)

   for i in range(1000):
       md.run(1)
       if i % 10 == 0:
           traj.write()
           print_energy()

   traj.close()

*******************
 Advanced Features
*******************

Non-Conservative Forces
=======================

.. code:: python

   # For faster MD simulations (requires v1.1.0+)
   fast_calculator = PETMADCalculator(
       version="v1.1.0", device=device, non_conservative=True
   )

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   atoms.calc = fast_calculator

   # This is 2-3x faster but requires careful MD setup
   energy = atoms.get_potential_energy()
   forces = atoms.get_forces()

.. warning::

   Non-conservative forces can lead to instabilities in MD simulations.
   Use with caution and consider using smaller timesteps or additional
   stabilization techniques.

Uncertainty Quantification
==========================

.. code:: python

   # Enable uncertainty estimation
   uq_calculator = PETMADCalculator(
       version="v1.0.2", device=device, calculate_uncertainty=True, calculate_ensemble=True
   )

   atoms = molecule("H2O")
   atoms.calc = uq_calculator

   energy = atoms.get_potential_energy()
   uncertainty = atoms.calc.get_energy_uncertainty()
   ensemble = atoms.calc.get_energy_ensemble()

   print(f"Energy: {energy:.3f} ± {uncertainty:.3f} eV")
   print(f"Ensemble std: {np.std(ensemble):.3f} eV")

Rotational Averaging
====================

.. code:: python

   # Useful for molecules and clusters
   rot_calculator = PETMADCalculator(
       version="latest", device=device, rotational_average_order=14  # Lebedev grid
   )

   # This averages predictions over molecular rotations
   methane = molecule("CH4")
   methane.calc = rot_calculator

   energy = methane.get_potential_energy()
   forces = methane.get_forces()

******************
 Performance Tips
******************

Memory Management
=================

.. code:: python

   import torch

   # For large systems, use mixed precision
   calculator = PETMADCalculator(
       version="latest", device="cuda", dtype=torch.float32  # Saves GPU memory
   )

   # Clear GPU cache when needed
   torch.cuda.empty_cache()

Batch Processing
================

For multiple single-point calculations, use batched evaluation:

.. code:: python

   # Create multiple structures
   structures = []
   for i in range(50):
       atoms = bulk("Si", cubic=True, a=5.43 + i * 0.01, crystalstructure="diamond")
       structures.append(atoms)

   # Batch evaluation is much faster
   batch_size = 10
   all_energies = []

   for i in range(0, len(structures), batch_size):
       batch = structures[i : i + batch_size]
       results = calculator.compute_energy(batch)
       all_energies.extend(results["energy"])

*****************
 Troubleshooting
*****************

Common Issues
=============

#. **Out of Memory**: Reduce batch size or use ``dtype=torch.float32``
#. **Unsupported Elements**: PET-MAD supports elements 1-86 except
   Astatine (85)
#. **Slow Performance**: Ensure you're using GPU if available

.. code:: python

   # Check supported elements
   supported_z = list(range(1, 87))
   supported_z.remove(85)  # Remove Astatine

   # Check if structure contains unsupported elements
   atomic_numbers = atoms.get_atomic_numbers()
   unsupported = set(atomic_numbers) - set(supported_z)
   if unsupported:
       print(f"Unsupported elements: {unsupported}")

Debugging
=========

.. code:: python

   import logging

   # Enable debug logging
   logging.basicConfig(level=logging.DEBUG)

   # This will show detailed information about calculations
   calculator = PETMADCalculator(version="latest", device="cpu")

******************************
 Integration with Other Tools
******************************

With ASE Databases
==================

.. code:: python

   from ase.db import connect

   # Store results in ASE database
   db = connect("results.db")

   structures = [bulk("Si"), bulk("C", crystalstructure="diamond")]

   for atoms in structures:
       atoms.calc = calculator
       energy = atoms.get_potential_energy()

       db.write(atoms, energy=energy, formula=atoms.get_chemical_formula())

With Phonopy
============

.. code:: python

   from phonopy import Phonopy
   from phonopy.structure.atoms import PhonopyAtoms
   import numpy as np

   # Create phonopy calculation
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")

   # Convert to phonopy format
   phonopy_atoms = PhonopyAtoms(
       symbols=atoms.get_chemical_symbols(),
       positions=atoms.get_positions(),
       cell=atoms.get_cell(),
   )

   # Create supercell for phonon calculation
   phonon = Phonopy(phonopy_atoms, [[2, 0, 0], [0, 2, 0], [0, 0, 2]])

   # This would require implementing force calculations for displaced structures
   # using the PET-MAD calculator
