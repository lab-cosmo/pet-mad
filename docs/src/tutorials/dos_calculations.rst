################################
 Density of States Calculations
################################

This tutorial covers how to use PET-MAD-DOS for calculating electronic
density of states, Fermi levels, and bandgaps.

**************
 Introduction
**************

PET-MAD-DOS is a universal model for predicting the electronic density
of states (DOS) of materials. It uses a slightly modified PET
architecture trained on the same MAD dataset as PET-MAD. The model can
predict:

-  Electronic density of states
-  Fermi levels
-  Bandgaps
-  Per-atom contributions to DOS

*************
 Basic Usage
*************

Setting up the Calculator
=========================

.. code:: python

   from pet_mad.calculator import PETMADDOSCalculator
   from ase.build import bulk
   import torch

   # Choose device
   device = "cuda" if torch.cuda.is_available() else "cpu"

   # Initialize DOS calculator
   dos_calculator = PETMADDOSCalculator(version="latest", device=device)

Simple DOS Calculation
======================

.. code:: python

   # Create a silicon crystal
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")

   # Calculate DOS
   energies, dos = dos_calculator.calculate_dos(atoms)

   print(f"Energy range: {energies.min():.2f} to {energies.max():.2f} eV")
   print(f"DOS shape: {dos.shape}")
   print(f"Total DOS integral: {np.trapz(dos, energies):.2f}")

Electronic Properties
=====================

.. code:: python

   # Calculate bandgap and Fermi level
   bandgap = dos_calculator.calculate_bandgap(atoms)
   fermi_level = dos_calculator.calculate_efermi(atoms)

   print(f"Bandgap: {bandgap:.3f} eV")
   print(f"Fermi level: {fermi_level:.3f} eV")

***************
 Visualization
***************

Basic DOS Plot
==============

.. code:: python

   import matplotlib.pyplot as plt
   import numpy as np

   # Calculate DOS for silicon
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   energies, dos = dos_calculator.calculate_dos(atoms)
   fermi_level = dos_calculator.calculate_efermi(atoms)

   # Create plot
   plt.figure(figsize=(10, 6))
   plt.plot(energies, dos, "b-", linewidth=2, label="DOS")
   plt.axvline(fermi_level, color="red", linestyle="--", linewidth=2, label="Fermi level")
   plt.axhline(0, color="gray", linestyle="-", alpha=0.3)

   plt.xlabel("Energy (eV)")
   plt.ylabel("DOS (states/eV)")
   plt.title("Silicon Density of States")
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Comparing Materials
===================

.. code:: python

   # Compare DOS of different materials
   materials = {
       "Silicon": bulk("Si", cubic=True, a=5.43, crystalstructure="diamond"),
       "Carbon": bulk("C", cubic=True, a=3.55, crystalstructure="diamond"),
       "Germanium": bulk("Ge", cubic=True, a=5.66, crystalstructure="diamond"),
   }

   plt.figure(figsize=(12, 8))

   for i, (name, atoms) in enumerate(materials.items()):
       energies, dos = dos_calculator.calculate_dos(atoms)
       fermi_level = dos_calculator.calculate_efermi(atoms)
       bandgap = dos_calculator.calculate_bandgap(atoms)

       # Normalize DOS for comparison
       dos_normalized = dos / np.max(dos)

       plt.subplot(2, 2, i + 1)
       plt.plot(energies, dos_normalized, "b-", linewidth=2)
       plt.axvline(fermi_level, color="red", linestyle="--", linewidth=2)
       plt.xlabel("Energy (eV)")
       plt.ylabel("Normalized DOS")
       plt.title(f"{name}\nBandgap: {bandgap:.2f} eV, E_F: {fermi_level:.2f} eV")
       plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

*******************
 Advanced Features
*******************

Per-Atom DOS
============

.. code:: python

   # Calculate DOS for each atom
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   atoms = atoms.repeat((2, 2, 2))  # 64-atom supercell

   energies, dos_per_atom = dos_calculator.calculate_dos(atoms, per_atom=True)

   print(f"Per-atom DOS shape: {dos_per_atom.shape}")  # (n_atoms, n_energies)

   # Plot DOS for first few atoms
   plt.figure(figsize=(10, 6))
   for i in range(min(4, len(atoms))):
       plt.plot(energies, dos_per_atom[i], label=f"Atom {i+1}", alpha=0.7)

   plt.xlabel("Energy (eV)")
   plt.ylabel("DOS (states/eV)")
   plt.title("Per-Atom DOS")
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Batch Processing
================

.. code:: python

   # Process multiple structures at once
   structures = [
       bulk("Si", cubic=True, a=5.43, crystalstructure="diamond"),
       bulk("C", cubic=True, a=3.55, crystalstructure="diamond"),
       bulk("Ge", cubic=True, a=5.66, crystalstructure="diamond"),
   ]

   # Calculate DOS for all structures
   energies, dos_list = dos_calculator.calculate_dos(structures)

   # Calculate properties for all structures
   bandgaps = dos_calculator.calculate_bandgap(structures)
   fermi_levels = dos_calculator.calculate_efermi(structures)

   # Print results
   materials = ["Silicon", "Carbon", "Germanium"]
   for i, name in enumerate(materials):
       print(f"{name}: Bandgap = {bandgaps[i]:.3f} eV, E_F = {fermi_levels[i]:.3f} eV")

Reusing DOS Data
================

.. code:: python

   # Calculate DOS once and reuse for properties
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   energies, dos = dos_calculator.calculate_dos(atoms)

   # Reuse DOS data for property calculations (more efficient)
   bandgap = dos_calculator.calculate_bandgap(atoms, dos=dos)
   fermi_level = dos_calculator.calculate_efermi(atoms, dos=dos)

   print(f"Bandgap: {bandgap:.3f} eV")
   print(f"Fermi level: {fermi_level:.3f} eV")

****************************
 Material Analysis Examples
****************************

Semiconductor Analysis
======================

.. code:: python

   import numpy as np


   def analyze_semiconductor(atoms, name="Material"):
       """Comprehensive semiconductor analysis"""

       # Calculate DOS and properties
       energies, dos = dos_calculator.calculate_dos(atoms)
       bandgap = dos_calculator.calculate_bandgap(atoms, dos=dos)
       fermi_level = dos_calculator.calculate_efermi(atoms, dos=dos)

       # Find valence band maximum and conduction band minimum
       fermi_idx = np.argmin(np.abs(energies - fermi_level))

       # Valence band (below Fermi level)
       vb_energies = energies[energies < fermi_level]
       vb_dos = dos[energies < fermi_level]
       vbm = vb_energies[np.argmax(vb_dos)] if len(vb_energies) > 0 else None

       # Conduction band (above Fermi level)
       cb_energies = energies[energies > fermi_level + bandgap]
       cb_dos = dos[energies > fermi_level + bandgap]
       cbm = cb_energies[np.argmax(cb_dos)] if len(cb_energies) > 0 else None

       print(f"\n{name} Analysis:")
       print(f"  Bandgap: {bandgap:.3f} eV")
       print(f"  Fermi level: {fermi_level:.3f} eV")
       if vbm is not None:
           print(f"  VBM: {vbm:.3f} eV")
       if cbm is not None:
           print(f"  CBM: {cbm:.3f} eV")

       # Classification
       if bandgap < 0.1:
           classification = "Metal"
       elif bandgap < 3.0:
           classification = "Semiconductor"
       else:
           classification = "Insulator"
       print(f"  Classification: {classification}")

       return {
           "energies": energies,
           "dos": dos,
           "bandgap": bandgap,
           "fermi_level": fermi_level,
           "classification": classification,
       }


   # Analyze different materials
   materials = {
       "Silicon": bulk("Si", cubic=True, a=5.43, crystalstructure="diamond"),
       "Diamond": bulk("C", cubic=True, a=3.55, crystalstructure="diamond"),
       "Germanium": bulk("Ge", cubic=True, a=5.66, crystalstructure="diamond"),
   }

   results = {}
   for name, atoms in materials.items():
       results[name] = analyze_semiconductor(atoms, name)

Surface and Interface Studies
=============================

.. code:: python

   from ase.build import surface

   # Create silicon surface
   slab = surface("Si", (1, 0, 0), 4, vacuum=10.0)

   # Calculate DOS for surface
   energies, dos_surface = dos_calculator.calculate_dos(slab)

   # Compare with bulk
   bulk_si = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   energies_bulk, dos_bulk = dos_calculator.calculate_dos(bulk_si)

   # Normalize by number of atoms for comparison
   dos_surface_norm = dos_surface / len(slab)
   dos_bulk_norm = dos_bulk / len(bulk_si)

   plt.figure(figsize=(10, 6))
   plt.plot(energies_bulk, dos_bulk_norm, "b-", linewidth=2, label="Bulk Si")
   plt.plot(energies, dos_surface_norm, "r-", linewidth=2, label="Si(100) Surface")

   plt.xlabel("Energy (eV)")
   plt.ylabel("DOS per atom (states/eV)")
   plt.title("Bulk vs Surface DOS")
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Alloy Analysis
==============

.. code:: python

   from ase import Atoms
   import numpy as np


   def create_random_alloy(element1, element2, size, concentration):
       """Create random binary alloy"""
       # Create base structure
       base = bulk(element1, cubic=True, crystalstructure="fcc")
       alloy = base.repeat(size)

       # Randomly substitute atoms
       n_substitute = int(len(alloy) * concentration)
       indices = np.random.choice(len(alloy), n_substitute, replace=False)

       symbols = alloy.get_chemical_symbols()
       for i in indices:
           symbols[i] = element2
       alloy.set_chemical_symbols(symbols)

       return alloy


   # Create Cu-Ni alloys with different concentrations
   concentrations = [0.0, 0.25, 0.5, 0.75, 1.0]

   plt.figure(figsize=(12, 8))

   for i, conc in enumerate(concentrations):
       if conc == 0.0:
           alloy = bulk("Cu", cubic=True, crystalstructure="fcc")
           label = "Pure Cu"
       elif conc == 1.0:
           alloy = bulk("Ni", cubic=True, crystalstructure="fcc")
           label = "Pure Ni"
       else:
           alloy = create_random_alloy("Cu", "Ni", (2, 2, 2), conc)
           label = f"Cu-{conc*100:.0f}%Ni"

       energies, dos = dos_calculator.calculate_dos(alloy)
       fermi_level = dos_calculator.calculate_efermi(alloy)

       # Normalize DOS
       dos_norm = dos / np.max(dos)

       plt.subplot(2, 3, i + 1)
       plt.plot(energies, dos_norm, "b-", linewidth=2)
       plt.axvline(fermi_level, color="red", linestyle="--", linewidth=1)
       plt.xlabel("Energy (eV)")
       plt.ylabel("Normalized DOS")
       plt.title(label)
       plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Temperature Effects (Approximate)
=================================

.. code:: python

   def fermi_dirac(energies, fermi_level, temperature):
       """Fermi-Dirac distribution"""
       kT = temperature * 8.617e-5  # Convert K to eV
       return 1.0 / (1.0 + np.exp((energies - fermi_level) / kT))


   def occupied_dos(energies, dos, fermi_level, temperature=0):
       """Calculate occupied DOS at given temperature"""
       if temperature == 0:
           return dos * (energies <= fermi_level)
       else:
           fd = fermi_dirac(energies, fermi_level, temperature)
           return dos * fd


   # Calculate for silicon at different temperatures
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   energies, dos = dos_calculator.calculate_dos(atoms)
   fermi_level = dos_calculator.calculate_efermi(atoms)

   temperatures = [0, 300, 600, 1000]  # Kelvin

   plt.figure(figsize=(12, 8))

   for i, temp in enumerate(temperatures):
       occupied = occupied_dos(energies, dos, fermi_level, temp)

       plt.subplot(2, 2, i + 1)
       plt.plot(energies, dos, "b-", linewidth=2, alpha=0.5, label="Total DOS")
       plt.fill_between(energies, occupied, alpha=0.7, label="Occupied")
       plt.axvline(fermi_level, color="red", linestyle="--", linewidth=1)

       plt.xlabel("Energy (eV)")
       plt.ylabel("DOS (states/eV)")
       plt.title(f"T = {temp} K")
       plt.legend()
       plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

**************************
 Performance Optimization
**************************

Batch Processing
================

.. code:: python

   # For many structures, use batch processing
   structures = [create_structure(i) for i in range(100)]

   # Process in batches
   batch_size = 10
   all_bandgaps = []
   all_fermi_levels = []

   for i in range(0, len(structures), batch_size):
       batch = structures[i : i + batch_size]

       # Calculate properties for batch
       bandgaps = dos_calculator.calculate_bandgap(batch)
       fermi_levels = dos_calculator.calculate_efermi(batch)

       all_bandgaps.extend(bandgaps)
       all_fermi_levels.extend(fermi_levels)

Memory Management
=================

.. code:: python

   import torch

   # For large systems, use float32 to save memory
   dos_calculator = PETMADDOSCalculator(
       version="latest", device="cuda", dtype=torch.float32
   )

   # Clear GPU cache when needed
   torch.cuda.empty_cache()

******************************
 Integration with Other Tools
******************************

With Pymatgen
=============

.. code:: python

   from pymatgen.core import Structure
   from pymatgen.io.ase import AseAtomsAdaptor

   # Convert pymatgen structure to ASE
   adaptor = AseAtomsAdaptor()

   # Example with pymatgen structure
   # structure = Structure.from_file("POSCAR")
   # atoms = adaptor.get_atoms(structure)
   # energies, dos = dos_calculator.calculate_dos(atoms)

With Materials Project
======================

.. code:: python

   # Example workflow for Materials Project data
   # from pymatgen.ext.matproj import MPRester

   # def analyze_mp_structure(mp_id):
   #     with MPRester("your_api_key") as mpr:
   #         structure = mpr.get_structure_by_material_id(mp_id)
   #         atoms = adaptor.get_atoms(structure)
   #
   #         energies, dos = dos_calculator.calculate_dos(atoms)
   #         bandgap = dos_calculator.calculate_bandgap(atoms, dos=dos)
   #
   #         return bandgap

**************************
 Export and Data Handling
**************************

Saving DOS Data
===============

.. code:: python

   import numpy as np

   # Calculate DOS
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   energies, dos = dos_calculator.calculate_dos(atoms)

   # Save to file
   np.savetxt(
       "silicon_dos.dat",
       np.column_stack([energies, dos]),
       header="Energy(eV) DOS(states/eV)",
       fmt="%.6f",
   )

   # Save properties
   properties = {
       "bandgap": dos_calculator.calculate_bandgap(atoms, dos=dos),
       "fermi_level": dos_calculator.calculate_efermi(atoms, dos=dos),
   }

   import json

   with open("silicon_properties.json", "w") as f:
       json.dump(properties, f, indent=2)

Loading and Comparing
=====================

.. code:: python

   # Load experimental or reference data for comparison
   # exp_data = np.loadtxt("experimental_dos.dat")
   # exp_energies, exp_dos = exp_data[:, 0], exp_data[:, 1]

   # Compare with PET-MAD-DOS prediction
   # plt.plot(exp_energies, exp_dos, 'r-', label='Experimental')
   # plt.plot(energies, dos, 'b--', label='PET-MAD-DOS')
   # plt.legend()

*****************
 Troubleshooting
*****************

Common Issues
=============

#. **Memory errors**: Use smaller batch sizes or float32 precision
#. **Unexpected results**: Check if the structure is reasonable and
   contains supported elements
#. **Performance issues**: Use GPU acceleration and batch processing

Validation
==========

.. code:: python

   # Basic validation checks
   def validate_dos(energies, dos):
       """Basic DOS validation"""

       # Check for NaN or infinite values
       if np.any(np.isnan(dos)) or np.any(np.isinf(dos)):
           print("Warning: DOS contains NaN or infinite values")

       # Check if DOS is non-negative
       if np.any(dos < 0):
           print("Warning: DOS contains negative values")

       # Check energy range
       print(f"Energy range: {energies.min():.2f} to {energies.max():.2f} eV")
       print(f"DOS range: {dos.min():.6f} to {dos.max():.6f} states/eV")

       return True


   # Validate results
   energies, dos = dos_calculator.calculate_dos(atoms)
   validate_dos(energies, dos)
