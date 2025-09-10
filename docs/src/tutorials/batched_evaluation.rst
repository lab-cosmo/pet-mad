#############################
 Batched Evaluation Tutorial
#############################

This tutorial covers efficient batched evaluation of PET-MAD for
high-throughput calculations.

**************
 Introduction
**************

While the standard ASE calculator methods are ideal for single-point
calculations and simulations, they are inefficient when evaluating the
model on many pre-determined structures. For high-throughput screening,
database evaluation, or benchmarking, PET-MAD provides optimized batched
evaluation methods that can be orders of magnitude faster.

Key advantages of batched evaluation:

-  **Performance**: 5-10x faster than sequential single-structure
   calculations
-  **Memory efficiency**: Better GPU memory utilization
-  **Scalability**: Handles thousands of structures efficiently
-  **Flexibility**: Works with mixed chemical compositions and structure
   sizes

**************************
 Basic Batched Evaluation
**************************

Simple Batch Processing
=======================

.. code:: python

   import torch
   from pet_mad.calculator import PETMADCalculator
   from ase.build import bulk
   import numpy as np

   # Choose device
   device = "cuda" if torch.cuda.is_available() else "cpu"
   calculator = PETMADCalculator(version="latest", device=device)

   # Create a dataset of structures
   structures = []
   for i in range(100):
       # Create silicon structures with different lattice parameters
       a = 5.43 + np.random.normal(0, 0.1)
       atoms = bulk("Si", cubic=True, a=a, crystalstructure="diamond")
       structures.append(atoms)

   print(f"Created {len(structures)} structures")

   # Batched evaluation
   batch_size = 10
   all_energies = []
   all_forces = []

   for i in range(0, len(structures), batch_size):
       batch = structures[i : i + batch_size]

       # Compute energies and forces for the batch
       results = calculator.compute_energy(batch, compute_forces_and_stresses=True)

       all_energies.extend(results["energy"])
       all_forces.extend(results["forces"])

       print(f"Processed batch {i//batch_size + 1}/{(len(structures)-1)//batch_size + 1}")

   print(f"Computed {len(all_energies)} energies")
   print(f"Energy range: {min(all_energies):.3f} to {max(all_energies):.3f} eV")

Performance Comparison
======================

.. code:: python

   import time


   def time_sequential_evaluation(structures, calculator):
       """Time sequential single-structure evaluation"""
       start_time = time.time()

       energies = []
       for atoms in structures:
           atoms.calc = calculator
           energy = atoms.get_potential_energy()
           energies.append(energy)

       elapsed = time.time() - start_time
       return energies, elapsed


   def time_batched_evaluation(structures, calculator, batch_size=10):
       """Time batched evaluation"""
       start_time = time.time()

       all_energies = []
       for i in range(0, len(structures), batch_size):
           batch = structures[i : i + batch_size]
           results = calculator.compute_energy(batch)
           all_energies.extend(results["energy"])

       elapsed = time.time() - start_time
       return all_energies, elapsed


   # Compare performance with a smaller dataset
   test_structures = structures[:50]

   # Sequential evaluation
   seq_energies, seq_time = time_sequential_evaluation(test_structures, calculator)

   # Batched evaluation
   batch_energies, batch_time = time_batched_evaluation(
       test_structures, calculator, batch_size=10
   )

   print(f"Performance Comparison:")
   print(f"  Sequential: {seq_time:.2f} seconds")
   print(f"  Batched:    {batch_time:.2f} seconds")
   print(f"  Speedup:    {seq_time/batch_time:.1f}x")
   print(
       f"  Energy difference (max): {max(abs(e1-e2) for e1, e2 in zip(seq_energies, batch_energies)):.6f} eV"
   )

******************************
 Advanced Batching Strategies
******************************

Optimal Batch Size Selection
============================

.. code:: python

   def find_optimal_batch_size(structures, calculator, max_batch_size=50):
       """Find optimal batch size for given hardware and structures"""

       test_structures = structures[:100]  # Use subset for testing
       batch_sizes = [1, 5, 10, 20, 30, 40, 50]
       times = []

       print("Testing batch sizes...")

       for batch_size in batch_sizes:
           if batch_size > len(test_structures):
               break

           try:
               start_time = time.time()

               for i in range(0, len(test_structures), batch_size):
                   batch = test_structures[i : i + batch_size]
                   results = calculator.compute_energy(batch)

               elapsed = time.time() - start_time
               times.append(elapsed)

               print(f"  Batch size {batch_size}: {elapsed:.2f} seconds")

           except RuntimeError as e:
               if "out of memory" in str(e).lower():
                   print(f"  Batch size {batch_size}: Out of memory")
                   break
               else:
                   raise e

       # Find optimal batch size
       if times:
           optimal_idx = np.argmin(times)
           optimal_batch_size = batch_sizes[optimal_idx]
           print(f"\nOptimal batch size: {optimal_batch_size}")
           return optimal_batch_size
       else:
           return 1


   # Find optimal batch size
   optimal_batch_size = find_optimal_batch_size(structures, calculator)

Memory-Aware Batching
=====================

.. code:: python

   def adaptive_batching(structures, calculator, max_memory_gb=4):
       """Adaptive batching based on available memory"""

       def estimate_memory_usage(batch_size, avg_atoms_per_structure=50):
           """Rough estimate of memory usage in GB"""
           # This is a simplified estimate
           memory_per_atom = 0.001  # GB per atom (rough estimate)
           return batch_size * avg_atoms_per_structure * memory_per_atom

       # Start with a reasonable batch size
       current_batch_size = 10
       all_energies = []

       i = 0
       while i < len(structures):
           # Adjust batch size based on remaining structures
           remaining = len(structures) - i
           batch_size = min(current_batch_size, remaining)

           batch = structures[i : i + batch_size]

           try:
               results = calculator.compute_energy(batch)
               all_energies.extend(results["energy"])

               # If successful, try to increase batch size
               if current_batch_size < 50:
                   current_batch_size += 5

               i += batch_size

           except RuntimeError as e:
               if "out of memory" in str(e).lower():
                   # Reduce batch size and clear GPU cache
                   current_batch_size = max(1, current_batch_size // 2)

                   if torch.cuda.is_available():
                       torch.cuda.empty_cache()

                   print(
                       f"Reduced batch size to {current_batch_size} due to memory constraints"
                   )

                   # Retry with smaller batch
                   continue
               else:
                   raise e

       return all_energies


   # Use adaptive batching
   adaptive_energies = adaptive_batching(structures[:100], calculator)

Mixed Structure Types
=====================

.. code:: python

   def batch_mixed_structures():
       """Handle batches with different structure types and sizes"""

       # Create diverse structures
       mixed_structures = []

       # Bulk crystals
       for element in ["Si", "C", "Ge"]:
           atoms = bulk(element, cubic=True, crystalstructure="diamond")
           mixed_structures.append(atoms)

       # Molecules
       from ase.build import molecule

       for mol in ["H2O", "CO2", "CH4", "NH3"]:
           atoms = molecule(mol)
           mixed_structures.append(atoms)

       # Supercells
       for size in [(2, 2, 2), (3, 2, 1)]:
           atoms = bulk("Si", cubic=True, crystalstructure="diamond")
           atoms = atoms.repeat(size)
           mixed_structures.append(atoms)

       # Batch process mixed structures
       batch_size = 5
       all_results = []

       for i in range(0, len(mixed_structures), batch_size):
           batch = mixed_structures[i : i + batch_size]

           results = calculator.compute_energy(batch, compute_forces_and_stresses=True)

           # Store results with metadata
           for j, atoms in enumerate(batch):
               result = {
                   "formula": atoms.get_chemical_formula(),
                   "n_atoms": len(atoms),
                   "energy": results["energy"][j],
                   "energy_per_atom": results["energy"][j] / len(atoms),
                   "forces": results["forces"][j],
                   "max_force": np.max(np.linalg.norm(results["forces"][j], axis=1)),
               }
               all_results.append(result)

       return all_results


   mixed_results = batch_mixed_structures()

   # Display results
   print("Mixed Structure Results:")
   for result in mixed_results:
       print(
           f"  {result['formula']}: {result['energy_per_atom']:.3f} eV/atom, "
           f"max_force: {result['max_force']:.3f} eV/Å"
       )

******************************
 High-Throughput Applications
******************************

Database Evaluation
===================

.. code:: python

   def evaluate_structure_database(database_file, calculator, output_file):
       """Evaluate structures from a database file"""

       # Load structures (example with XYZ file)
       try:
           import ase.io

           structures = ase.io.read(database_file, ":")
           print(f"Loaded {len(structures)} structures from {database_file}")
       except:
           print("Could not load database file. Using example structures.")
           # Create example database
           structures = []
           for i in range(200):
               a = 5.43 + np.random.normal(0, 0.05)
               atoms = bulk("Si", cubic=True, a=a, crystalstructure="diamond")
               structures.append(atoms)

       # Determine optimal batch size
       optimal_batch = find_optimal_batch_size(structures[:50], calculator)

       # Process database
       results = []
       total_time = time.time()

       for i in range(0, len(structures), optimal_batch):
           batch = structures[i : i + optimal_batch]

           batch_start = time.time()
           batch_results = calculator.compute_energy(
               batch, compute_forces_and_stresses=True
           )
           batch_time = time.time() - batch_start

           # Store results
           for j, atoms in enumerate(batch):
               result = {
                   "index": i + j,
                   "formula": atoms.get_chemical_formula(),
                   "energy": batch_results["energy"][j],
                   "forces": batch_results["forces"][j].tolist(),
                   "stress": (
                       batch_results["stress"][j].tolist()
                       if "stress" in batch_results
                       else None
                   ),
               }
               results.append(result)

           # Progress update
           progress = (i + len(batch)) / len(structures) * 100
           rate = len(batch) / batch_time
           print(f"Progress: {progress:.1f}% ({rate:.1f} structures/second)")

       total_time = time.time() - total_time

       # Save results
       import json

       with open(output_file, "w") as f:
           json.dump(results, f, indent=2)

       print(f"Completed evaluation in {total_time:.1f} seconds")
       print(f"Average rate: {len(structures)/total_time:.1f} structures/second")
       print(f"Results saved to {output_file}")

       return results


   # Example usage
   # results = evaluate_structure_database("structures.xyz", calculator, "results.json")

Parameter Screening
===================

.. code:: python

   def screen_lattice_parameters():
       """Screen across lattice parameters"""

       elements = ["Si", "C", "Ge"]
       lattice_params = np.linspace(3.0, 6.5, 50)

       screening_results = []

       for element in elements:
           print(f"Screening {element}...")

           # Create structures for this element
           element_structures = []
           for a in lattice_params:
               try:
                   atoms = bulk(element, cubic=True, a=a, crystalstructure="diamond")
                   element_structures.append(atoms)
               except:
                   # Skip if structure creation fails
                   continue

           # Batch evaluate
           batch_size = 20
           element_energies = []

           for i in range(0, len(element_structures), batch_size):
               batch = element_structures[i : i + batch_size]
               results = calculator.compute_energy(batch)
               element_energies.extend(results["energy"])

           # Store results
           for a, energy in zip(lattice_params[: len(element_energies)], element_energies):
               screening_results.append(
                   {
                       "element": element,
                       "lattice_param": a,
                       "energy": energy,
                       "energy_per_atom": energy / 8,  # Diamond structure has 8 atoms
                   }
               )

       return screening_results


   screening_data = screen_lattice_parameters()

   # Analyze results
   import matplotlib.pyplot as plt

   fig, axes = plt.subplots(1, 2, figsize=(12, 5))

   elements = ["Si", "C", "Ge"]
   colors = ["blue", "black", "green"]

   for element, color in zip(elements, colors):
       element_data = [d for d in screening_data if d["element"] == element]
       if element_data:
           lattice_params = [d["lattice_param"] for d in element_data]
           energies = [d["energy"] for d in element_data]
           energies_per_atom = [d["energy_per_atom"] for d in element_data]

           axes[0].plot(
               lattice_params, energies, "o-", color=color, label=element, alpha=0.7
           )
           axes[1].plot(
               lattice_params,
               energies_per_atom,
               "o-",
               color=color,
               label=element,
               alpha=0.7,
           )

   axes[0].set_xlabel("Lattice Parameter (Å)")
   axes[0].set_ylabel("Total Energy (eV)")
   axes[0].set_title("Total Energy vs Lattice Parameter")
   axes[0].legend()
   axes[0].grid(True, alpha=0.3)

   axes[1].set_xlabel("Lattice Parameter (Å)")
   axes[1].set_ylabel("Energy per Atom (eV)")
   axes[1].set_title("Energy per Atom vs Lattice Parameter")
   axes[1].legend()
   axes[1].grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

********************************
 Specialized Evaluation Methods
********************************

Forces and Stresses
===================

.. code:: python

   # Evaluate forces and stresses efficiently
   def batch_forces_and_stresses(structures, calculator):
       """Efficient computation of forces and stresses"""

       batch_size = 15  # Smaller batch size for force calculations
       all_results = []

       for i in range(0, len(structures), batch_size):
           batch = structures[i : i + batch_size]

           # Compute all properties at once
           results = calculator.compute_energy(batch, compute_forces_and_stresses=True)

           # Analyze force characteristics
           for j, atoms in enumerate(batch):
               forces = results["forces"][j]
               stress = results["stress"][j] if "stress" in results else None

               force_magnitudes = np.linalg.norm(forces, axis=1)

               result = {
                   "formula": atoms.get_chemical_formula(),
                   "energy": results["energy"][j],
                   "max_force": np.max(force_magnitudes),
                   "rms_force": np.sqrt(np.mean(force_magnitudes**2)),
                   "stress_trace": np.trace(stress) if stress is not None else None,
               }
               all_results.append(result)

       return all_results


   # Example usage
   force_results = batch_forces_and_stresses(structures[:30], calculator)

   print("Force Analysis Results:")
   for result in force_results[:5]:  # Show first 5
       print(
           f"  {result['formula']}: max_force = {result['max_force']:.3f} eV/Å, "
           f"rms_force = {result['rms_force']:.3f} eV/Å"
       )

Uncertainty Quantification in Batches
=====================================

.. code:: python

   # Batch uncertainty quantification (requires v1.0.2)
   def batch_with_uncertainty():
       """Batch evaluation with uncertainty quantification"""

       uq_calculator = PETMADCalculator(
           version="v1.0.2",
           device=device,
           calculate_uncertainty=True,
           calculate_ensemble=True,
       )

       # Smaller batch size due to additional computation
       batch_size = 8
       test_structures = structures[:40]

       results_with_uncertainty = []

       for i in range(0, len(test_structures), batch_size):
           batch = test_structures[i : i + batch_size]

           # Process each structure in batch individually for uncertainty
           for atoms in batch:
               atoms.calc = uq_calculator
               energy = atoms.get_potential_energy()
               uncertainty = atoms.calc.get_energy_uncertainty()
               ensemble = atoms.calc.get_energy_ensemble()

               result = {
                   "formula": atoms.get_chemical_formula(),
                   "energy": energy,
                   "uncertainty": uncertainty,
                   "ensemble_std": np.std(ensemble),
                   "relative_uncertainty": uncertainty / abs(energy) * 100,
               }
               results_with_uncertainty.append(result)

       return results_with_uncertainty


   # Example usage (requires v1.0.2)
   try:
       uncertainty_results = batch_with_uncertainty()

       print("Uncertainty Analysis:")
       for result in uncertainty_results[:5]:
           print(
               f"  {result['formula']}: {result['energy']:.3f} ± {result['uncertainty']:.3f} eV "
               f"({result['relative_uncertainty']:.2f}%)"
           )
   except:
       print("Uncertainty quantification not available (requires v1.0.2)")

****************************
 Data Management and Export
****************************

Efficient Data Storage
======================

.. code:: python

   import h5py
   import json


   def save_batch_results_hdf5(results, filename):
       """Save batch results in HDF5 format for efficiency"""

       with h5py.File(filename, "w") as f:
           # Create datasets
           n_structures = len(results)

           # Basic properties
           energies = f.create_dataset("energies", (n_structures,), dtype="f8")
           formulas = f.create_dataset(
               "formulas", (n_structures,), dtype=h5py.string_dtype()
           )

           # Variable length data (forces)
           forces_group = f.create_group("forces")

           for i, result in enumerate(results):
               energies[i] = result["energy"]
               formulas[i] = result["formula"]

               if "forces" in result:
                   forces_group.create_dataset(f"structure_{i}", data=result["forces"])

       print(f"Saved {n_structures} results to {filename}")


   def load_batch_results_hdf5(filename):
       """Load batch results from HDF5 format"""

       results = []

       with h5py.File(filename, "r") as f:
           energies = f["energies"][:]
           formulas = f["formulas"][:]

           for i in range(len(energies)):
               result = {
                   "energy": float(energies[i]),
                   "formula": formulas[i].decode("utf-8"),
               }

               # Load forces if available
               if f"structure_{i}" in f["forces"]:
                   result["forces"] = f["forces"][f"structure_{i}"][:]

               results.append(result)

       return results


   # Example usage
   # save_batch_results_hdf5(mixed_results, "batch_results.h5")
   # loaded_results = load_batch_results_hdf5("batch_results.h5")

Results Analysis and Visualization
==================================

.. code:: python

   def analyze_batch_results(results):
       """Comprehensive analysis of batch results"""

       # Extract data
       energies = [r["energy"] for r in results]
       formulas = [r["formula"] for r in results]
       energies_per_atom = [
           r["energy_per_atom"] for r in results if "energy_per_atom" in r
       ]

       # Basic statistics
       print("Batch Results Analysis:")
       print(f"  Total structures: {len(results)}")
       print(f"  Energy range: {min(energies):.3f} to {max(energies):.3f} eV")
       print(f"  Mean energy: {np.mean(energies):.3f} eV")
       print(f"  Energy std: {np.std(energies):.3f} eV")

       if energies_per_atom:
           print(
               f"  Energy per atom range: {min(energies_per_atom):.3f} to {max(energies_per_atom):.3f} eV/atom"
           )

       # Composition analysis
       unique_formulas = list(set(formulas))
       print(f"  Unique compositions: {len(unique_formulas)}")

       # Visualization
       plt.figure(figsize=(15, 5))

       plt.subplot(1, 3, 1)
       plt.hist(energies, bins=20, alpha=0.7)
       plt.xlabel("Energy (eV)")
       plt.ylabel("Count")
       plt.title("Energy Distribution")
       plt.grid(True, alpha=0.3)

       if energies_per_atom:
           plt.subplot(1, 3, 2)
           plt.hist(energies_per_atom, bins=20, alpha=0.7)
           plt.xlabel("Energy per Atom (eV)")
           plt.ylabel("Count")
           plt.title("Energy per Atom Distribution")
           plt.grid(True, alpha=0.3)

       plt.subplot(1, 3, 3)
       formula_counts = {f: formulas.count(f) for f in unique_formulas}
       plt.bar(range(len(unique_formulas)), list(formula_counts.values()))
       plt.xticks(range(len(unique_formulas)), unique_formulas, rotation=45)
       plt.xlabel("Formula")
       plt.ylabel("Count")
       plt.title("Composition Distribution")
       plt.grid(True, alpha=0.3)

       plt.tight_layout()
       plt.show()

       return {
           "n_structures": len(results),
           "energy_stats": {
               "mean": np.mean(energies),
               "std": np.std(energies),
               "min": min(energies),
               "max": max(energies),
           },
           "compositions": unique_formulas,
       }


   # Analyze results
   if "mixed_results" in locals():
       analysis = analyze_batch_results(mixed_results)

****************
 Best Practices
****************

Performance Optimization
========================

#. **Batch Size**: Start with 10-20 structures and adjust based on
   memory and performance
#. **GPU Memory**: Monitor GPU memory usage and adjust batch size
   accordingly
#. **Mixed Precision**: Use ``dtype=torch.float32`` for
   memory-constrained systems
#. **Progress Tracking**: Use progress bars for long-running evaluations

Error Handling
==============

.. code:: python

   def robust_batch_evaluation(structures, calculator, batch_size=10):
       """Robust batch evaluation with error handling"""

       successful_results = []
       failed_indices = []

       for i in range(0, len(structures), batch_size):
           batch = structures[i : i + batch_size]
           batch_indices = list(range(i, min(i + batch_size, len(structures))))

           try:
               results = calculator.compute_energy(batch)

               # Store successful results
               for j, idx in enumerate(batch_indices):
                   result = {
                       "index": idx,
                       "formula": batch[j].get_chemical_formula(),
                       "energy": results["energy"][j],
                   }
                   successful_results.append(result)

           except Exception as e:
               print(f"Batch {i//batch_size + 1} failed: {str(e)}")

               # Try individual structures in failed batch
               for j, (atoms, idx) in enumerate(zip(batch, batch_indices)):
                   try:
                       atoms.calc = calculator
                       energy = atoms.get_potential_energy()

                       result = {
                           "index": idx,
                           "formula": atoms.get_chemical_formula(),
                           "energy": energy,
                       }
                       successful_results.append(result)

                   except Exception as e2:
                       print(f"  Structure {idx} failed: {str(e2)}")
                       failed_indices.append(idx)

       print(f"Successfully processed {len(successful_results)} structures")
       print(f"Failed structures: {len(failed_indices)}")

       return successful_results, failed_indices


   # Example usage
   # robust_results, failed_idx = robust_batch_evaluation(structures[:50], calculator)

Quality Control
===============

.. code:: python

   def validate_batch_results(results, structures):
       """Validate batch evaluation results"""

       print("Validating batch results...")

       # Check for missing results
       if len(results) != len(structures):
           print(f"Warning: Expected {len(structures)} results, got {len(results)}")

       # Check for unreasonable energies
       energies = [r["energy"] if isinstance(r, dict) else r for r in results]

       # Basic energy checks
       if any(np.isnan(e) or np.isinf(e) for e in energies):
           print("Warning: NaN or infinite energies detected")

       # Check energy range (structure-dependent)
       mean_energy = np.mean(energies)
       std_energy = np.std(energies)

       outliers = [
           i for i, e in enumerate(energies) if abs(e - mean_energy) > 3 * std_energy
       ]

       if outliers:
           print(f"Warning: {len(outliers)} potential outliers detected")
           for idx in outliers[:5]:  # Show first 5
               formula = (
                   structures[idx].get_chemical_formula()
                   if idx < len(structures)
                   else "Unknown"
               )
               print(f"  Structure {idx} ({formula}): {energies[idx]:.3f} eV")

       print("Validation completed")


   # Validate results
   if "all_energies" in locals():
       validate_batch_results(all_energies, structures)
