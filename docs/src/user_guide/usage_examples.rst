################
 Usage Examples
################

This page provides comprehensive examples of using PET-MAD for various
applications.

*************************************
 Basic Energy and Force Calculations
*************************************

.. code:: python

   from pet_mad.calculator import PETMADCalculator
   from ase.build import bulk, molecule
   import torch

   # Use GPU if available
   device = "cuda" if torch.cuda.is_available() else "cpu"
   calculator = PETMADCalculator(version="latest", device=device)

   # Silicon crystal
   si_crystal = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   si_crystal.calc = calculator

   energy = si_crystal.get_potential_energy()
   forces = si_crystal.get_forces()
   stress = si_crystal.get_stress()

   print(f"Silicon crystal energy: {energy:.3f} eV")
   print(f"Max force magnitude: {max(np.linalg.norm(forces, axis=1)):.3f} eV/Å")

   # Water molecule
   water = molecule("H2O")
   water.calc = calculator

   water_energy = water.get_potential_energy()
   water_forces = water.get_forces()

   print(f"Water molecule energy: {water_energy:.3f} eV")

**************************************
 Non-Conservative Forces and Stresses
**************************************

For faster calculations, you can use non-conservative (direct)
prediction of forces and stresses:

.. code:: python

   from pet_mad.calculator import PETMADCalculator
   from ase.build import bulk

   # Enable non-conservative forces (requires v1.1.0 or higher)
   calculator = PETMADCalculator(version="v1.1.0", device="cpu", non_conservative=True)

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   atoms.calc = calculator

   # Forces and stresses are now predicted directly (2-3x faster)
   energy = atoms.get_potential_energy()
   forces = atoms.get_forces()
   stress = atoms.get_stress()

.. note::

   Non-conservative forces provide significant speedup but require
   additional care during MD simulations to avoid instabilities.

****************************
 Uncertainty Quantification
****************************

PET-MAD can estimate the uncertainty of its predictions:

.. code:: python

   from pet_mad.calculator import PETMADCalculator
   from ase.build import bulk

   # Enable uncertainty quantification (requires v1.0.2)
   calculator = PETMADCalculator(
       version="v1.0.2", device="cpu", calculate_uncertainty=True, calculate_ensemble=True
   )

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   atoms.calc = calculator

   energy = atoms.get_potential_energy()

   # Get uncertainty estimates
   energy_uncertainty = atoms.calc.get_energy_uncertainty()
   energy_ensemble = atoms.calc.get_energy_ensemble()

   print(f"Energy: {energy:.3f} ± {energy_uncertainty:.3f} eV")
   print(f"Ensemble size: {len(energy_ensemble)}")

**********************
 Rotational Averaging
**********************

For systems with rotational symmetry, you can average predictions over
rotations:

.. code:: python

   from pet_mad.calculator import PETMADCalculator
   from ase.build import molecule

   # Use Lebedev grid for rotational averaging
   calculator = PETMADCalculator(
       version="latest", device="cpu", rotational_average_order=14  # Lebedev grid order
   )

   # This is particularly useful for molecules
   methane = molecule("CH4")
   methane.calc = calculator

   energy = methane.get_potential_energy()
   forces = methane.get_forces()

********************
 Batched Evaluation
********************

For evaluating many structures efficiently:

.. code:: python

   import torch
   from pet_mad.calculator import PETMADCalculator
   from ase.build import bulk

   device = "cuda" if torch.cuda.is_available() else "cpu"
   calculator = PETMADCalculator(version="latest", device=device)

   # Create a dataset of structures
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   dataset = [atoms] * 100

   # Split into batches
   batch_size = 10
   batches = [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]

   all_energies = []
   all_forces = []

   for batch in batches:
       results = calculator.compute_energy(batch, compute_forces_and_stresses=True)
       all_energies.extend(results["energy"])
       all_forces.extend(results["forces"])

   print(f"Evaluated {len(all_energies)} structures")

********************************
 Density of States Calculations
********************************

.. code:: python

   from pet_mad.calculator import PETMADDOSCalculator
   from ase.build import bulk
   import matplotlib.pyplot as plt

   dos_calculator = PETMADDOSCalculator(version="latest", device="cpu")

   # Silicon crystal
   si_crystal = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")

   # Calculate DOS
   energies, dos = dos_calculator.calculate_dos(si_crystal)

   # Calculate electronic properties
   bandgap = dos_calculator.calculate_bandgap(si_crystal)
   fermi_level = dos_calculator.calculate_efermi(si_crystal)

   print(f"Bandgap: {bandgap:.3f} eV")
   print(f"Fermi level: {fermi_level:.3f} eV")

   # Plot DOS
   plt.figure(figsize=(8, 6))
   plt.plot(energies, dos)
   plt.axvline(fermi_level, color="red", linestyle="--", label="Fermi level")
   plt.xlabel("Energy (eV)")
   plt.ylabel("DOS")
   plt.legend()
   plt.show()

***************************
 Per-atom DOS Calculations
***************************

.. code:: python

   from pet_mad.calculator import PETMADDOSCalculator
   from ase.build import bulk

   dos_calculator = PETMADDOSCalculator(version="latest", device="cpu")
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")

   # Calculate DOS for each atom
   energies, dos_per_atom = dos_calculator.calculate_dos(atoms, per_atom=True)

   print(f"DOS shape: {dos_per_atom.shape}")  # (n_atoms, n_energies)

   # Calculate properties for multiple structures
   atoms_list = [bulk("Si"), bulk("C", crystalstructure="diamond")]
   energies, dos_list = dos_calculator.calculate_dos(atoms_list)

   bandgaps = dos_calculator.calculate_bandgap(atoms_list)
   fermi_levels = dos_calculator.calculate_efermi(atoms_list)

***************************************
 Dataset Visualization and Exploration
***************************************

.. code:: python

   import ase.io
   from pet_mad.explore import PETMADFeaturizer

   # Load structures (example with trajectory file)
   frames = ase.io.read("trajectory.xyz", ":")

   # Create featurizer for visualization
   featurizer = PETMADFeaturizer(version="latest")

   # Extract features for analysis
   features = featurizer(frames, None)
   print(f"Feature shape: {features.shape}")

   # For interactive visualization with chemiscope (in Jupyter)
   # import chemiscope
   # chemiscope.explore(frames, featurize=featurizer)

***************************************
 Combining with Dispersion Corrections
***************************************

.. code:: python

   import torch
   from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
   from pet_mad.calculator import PETMADCalculator
   from ase.calculators.mixing import SumCalculator
   from ase.build import bulk

   device = "cuda" if torch.cuda.is_available() else "cpu"

   # PET-MAD calculator
   pet_mad_calc = PETMADCalculator(version="latest", device=device)

   # D3 dispersion correction
   d3_calc = TorchDFTD3Calculator(device=device, xc="pbesol", damping="bj")

   # Combine calculators
   combined_calc = SumCalculator([pet_mad_calc, d3_calc])

   # Use combined calculator
   atoms = bulk("graphite")  # System where dispersion is important
   atoms.calc = combined_calc

   energy = atoms.get_potential_energy()
   forces = atoms.get_forces()

***********************************
 Error Handling and Best Practices
***********************************

.. code:: python

   from pet_mad.calculator import PETMADCalculator
   from ase.build import Atoms
   import numpy as np

   calculator = PETMADCalculator(version="latest", device="cpu")

   try:
       # PET-MAD supports elements up to Z=86 (except Astatine)
       atoms = Atoms("H2O", positions=[[0, 0, 0], [0.96, 0, 0], [0.24, 0.93, 0]])
       atoms.calc = calculator

       energy = atoms.get_potential_energy()

   except Exception as e:
       print(f"Calculation failed: {e}")

   # Check supported elements
   supported_elements = list(range(1, 87))  # H to Rn, except At (85)
   supported_elements.remove(85)  # Remove Astatine

   print(f"Supported atomic numbers: {supported_elements}")

*************************************
 Memory Management for Large Systems
*************************************

.. code:: python

   import torch
   from pet_mad.calculator import PETMADCalculator

   # For large systems, consider using mixed precision
   calculator = PETMADCalculator(
       version="latest", device="cuda", dtype=torch.float32  # Use float32 to save memory
   )

   # Clear GPU cache if needed
   if torch.cuda.is_available():
       torch.cuda.empty_cache()
