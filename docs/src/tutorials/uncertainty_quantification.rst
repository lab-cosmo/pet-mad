Uncertainty Quantification
===========================

This tutorial covers how to use PET-MAD's uncertainty quantification features to assess the reliability of predictions.

Introduction
------------

PET-MAD provides uncertainty estimation capabilities that help assess the reliability of energy predictions. This is particularly important when:

- Exploring new chemical spaces far from training data
- Making decisions based on model predictions
- Propagating uncertainties to derived properties
- Identifying when additional training data might be needed

The uncertainty quantification in PET-MAD uses two main approaches:

1. **Energy Uncertainty**: Direct uncertainty estimation of energy predictions
2. **Ensemble Methods**: Shallow ensemble of the last layers for energy predictions

Basic Usage
-----------

Enabling Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pet_mad.calculator import PETMADCalculator
   from ase.build import bulk, molecule
   import numpy as np

   # Enable uncertainty quantification (requires v1.0.2)
   calculator = PETMADCalculator(
       version="v1.0.2",
       device="cpu",
       calculate_uncertainty=True,
       calculate_ensemble=True
   )

Basic Uncertainty Calculation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create a simple system
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   atoms.calc = calculator

   # Calculate energy (this also computes uncertainties)
   energy = atoms.get_potential_energy()

   # Get uncertainty estimates
   energy_uncertainty = atoms.calc.get_energy_uncertainty()
   energy_ensemble = atoms.calc.get_energy_ensemble()

   print(f"Energy: {energy:.3f} ± {energy_uncertainty:.3f} eV")
   print(f"Ensemble size: {len(energy_ensemble)}")
   print(f"Ensemble std: {np.std(energy_ensemble):.3f} eV")

Understanding the Output
------------------------

Energy Uncertainty
~~~~~~~~~~~~~~~~~~

The energy uncertainty represents the model's confidence in its prediction:

.. code-block:: python

   # Low uncertainty example (training-like data)
   si_bulk = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   si_bulk.calc = calculator

   energy = si_bulk.get_potential_energy()
   uncertainty = si_bulk.calc.get_energy_uncertainty()

   print(f"Silicon bulk: {energy:.3f} ± {uncertainty:.3f} eV")
   print(f"Relative uncertainty: {uncertainty/abs(energy)*100:.2f}%")

Ensemble Predictions
~~~~~~~~~~~~~~~~~~~~

The ensemble provides multiple predictions from slightly different model variations:

.. code-block:: python

   energy_ensemble = atoms.calc.get_energy_ensemble()

   print(f"Ensemble statistics:")
   print(f"  Mean: {np.mean(energy_ensemble):.3f} eV")
   print(f"  Std:  {np.std(energy_ensemble):.3f} eV")
   print(f"  Min:  {np.min(energy_ensemble):.3f} eV")
   print(f"  Max:  {np.max(energy_ensemble):.3f} eV")

Practical Applications
----------------------

Screening Chemical Space
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   from ase.build import molecule

   molecules = ['H2O', 'CO2', 'CH4', 'NH3', 'H2S']
   energies = []
   uncertainties = []

   for mol_name in molecules:
       mol = molecule(mol_name)
       mol.calc = calculator

       energy = mol.get_potential_energy()
       uncertainty = mol.calc.get_energy_uncertainty()

       energies.append(energy)
       uncertainties.append(uncertainty)

       print(f"{mol_name}: {energy:.3f} ± {uncertainty:.3f} eV")

   # Visualize uncertainty vs energy
   plt.figure(figsize=(10, 6))
   plt.errorbar(range(len(molecules)), energies, yerr=uncertainties,
                fmt='o', capsize=5, capthick=2)
   plt.xticks(range(len(molecules)), molecules)
   plt.ylabel('Energy (eV)')
   plt.title('Molecular Energies with Uncertainties')
   plt.grid(True, alpha=0.3)
   plt.show()

Geometry Optimization with Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ase.optimize import BFGS
   from ase.build import molecule
   import numpy as np

   # Create distorted water molecule
   water = molecule("H2O")
   positions = water.get_positions()
   positions += np.random.normal(0, 0.2, positions.shape)  # Add larger distortion
   water.set_positions(positions)
   water.calc = calculator

   # Track uncertainty during optimization
   uncertainties = []
   energies = []

   def track_uncertainty():
       energy = water.get_potential_energy()
       uncertainty = water.calc.get_energy_uncertainty()
       energies.append(energy)
       uncertainties.append(uncertainty)
       print(f"Step {len(energies)}: E = {energy:.3f} ± {uncertainty:.3f} eV")

   # Initial uncertainty
   track_uncertainty()

   # Optimize with tracking
   optimizer = BFGS(water)

   for i in range(10):  # Limited steps for demonstration
       optimizer.run(1)
       track_uncertainty()

       if optimizer.converged():
           break

   # Plot optimization trajectory
   plt.figure(figsize=(12, 4))

   plt.subplot(1, 2, 1)
   plt.plot(energies, 'b-o')
   plt.xlabel('Optimization Step')
   plt.ylabel('Energy (eV)')
   plt.title('Energy Convergence')
   plt.grid(True, alpha=0.3)

   plt.subplot(1, 2, 2)
   plt.plot(uncertainties, 'r-o')
   plt.xlabel('Optimization Step')
   plt.ylabel('Uncertainty (eV)')
   plt.title('Uncertainty Evolution')
   plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Uncertainty in Molecular Dynamics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
   from ase.md.verlet import VelocityVerlet
   from ase import units

   # Small system for demonstration
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   atoms = atoms.repeat((2, 2, 2))  # 64 atoms
   atoms.calc = calculator

   # Set up MD
   MaxwellBoltzmannDistribution(atoms, temperature_K=300)
   md = VelocityVerlet(atoms, timestep=1.0*units.fs)

   # Track uncertainties during MD
   md_uncertainties = []
   md_energies = []
   times = []

   for i in range(50):  # Short MD run
       md.run(1)

       energy = atoms.get_potential_energy()
       uncertainty = atoms.calc.get_energy_uncertainty()

       md_energies.append(energy)
       md_uncertainties.append(uncertainty)
       times.append(i * 1.0)  # fs

       if i % 10 == 0:
           print(f"Time {i} fs: E = {energy:.3f} ± {uncertainty:.3f} eV")

   # Analyze uncertainty evolution
   plt.figure(figsize=(12, 4))

   plt.subplot(1, 2, 1)
   plt.plot(times, md_energies, 'b-')
   plt.xlabel('Time (fs)')
   plt.ylabel('Energy (eV)')
   plt.title('MD Energy Trajectory')
   plt.grid(True, alpha=0.3)

   plt.subplot(1, 2, 2)
   plt.plot(times, md_uncertainties, 'r-')
   plt.xlabel('Time (fs)')
   plt.ylabel('Uncertainty (eV)')
   plt.title('Uncertainty During MD')
   plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Advanced Analysis
-----------------

Uncertainty vs Distance from Training Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create structures at different lattice parameters
   lattice_params = np.linspace(5.0, 6.0, 11)  # Around Si equilibrium
   si_energies = []
   si_uncertainties = []

   for a in lattice_params:
       atoms = bulk("Si", cubic=True, a=a, crystalstructure="diamond")
       atoms.calc = calculator

       energy = atoms.get_potential_energy()
       uncertainty = atoms.calc.get_energy_uncertainty()

       si_energies.append(energy)
       si_uncertainties.append(uncertainty)

   plt.figure(figsize=(12, 4))

   plt.subplot(1, 2, 1)
   plt.plot(lattice_params, si_energies, 'b-o')
   plt.xlabel('Lattice Parameter (Å)')
   plt.ylabel('Energy (eV)')
   plt.title('Energy vs Lattice Parameter')
   plt.grid(True, alpha=0.3)

   plt.subplot(1, 2, 2)
   plt.plot(lattice_params, si_uncertainties, 'r-o')
   plt.xlabel('Lattice Parameter (Å)')
   plt.ylabel('Uncertainty (eV)')
   plt.title('Uncertainty vs Lattice Parameter')
   plt.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.show()

Ensemble Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detailed ensemble analysis
   atoms = molecule("CH4")
   atoms.calc = calculator

   energy = atoms.get_potential_energy()
   ensemble = atoms.calc.get_energy_ensemble()

   # Statistical analysis
   mean_energy = np.mean(ensemble)
   std_energy = np.std(ensemble)
   min_energy = np.min(ensemble)
   max_energy = np.max(ensemble)

   print(f"Ensemble Analysis:")
   print(f"  Main prediction: {energy:.3f} eV")
   print(f"  Ensemble mean:   {mean_energy:.3f} eV")
   print(f"  Ensemble std:    {std_energy:.3f} eV")
   print(f"  Range: [{min_energy:.3f}, {max_energy:.3f}] eV")
   print(f"  Spread: {max_energy - min_energy:.3f} eV")

   # Histogram of ensemble predictions
   plt.figure(figsize=(10, 6))
   plt.hist(ensemble, bins=20, alpha=0.7, density=True)
   plt.axvline(energy, color='red', linestyle='--', linewidth=2, label='Main prediction')
   plt.axvline(mean_energy, color='blue', linestyle='--', linewidth=2, label='Ensemble mean')
   plt.xlabel('Energy (eV)')
   plt.ylabel('Probability Density')
   plt.title('Distribution of Ensemble Predictions')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Uncertainty Propagation
-----------------------

Error Propagation in Derived Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate formation energy with uncertainty
   def formation_energy_with_uncertainty(compound_atoms, reference_atoms_dict):
       """Calculate formation energy with uncertainty propagation"""

       # Calculate compound energy
       compound_atoms.calc = calculator
       compound_energy = compound_atoms.get_potential_energy()
       compound_uncertainty = compound_atoms.calc.get_energy_uncertainty()

       # Calculate reference energies
       reference_energies = {}
       reference_uncertainties = {}

       for element, atoms in reference_atoms_dict.items():
           atoms.calc = calculator
           ref_energy = atoms.get_potential_energy()
           ref_uncertainty = atoms.calc.get_energy_uncertainty()

           reference_energies[element] = ref_energy
           reference_uncertainties[element] = ref_uncertainty

       # Calculate formation energy (simplified for binary compound)
       # This would need to be adapted for the specific compound
       formation_energy = compound_energy  # Simplified calculation

       # Uncertainty propagation (simplified)
       total_uncertainty = np.sqrt(
           compound_uncertainty**2 +
           sum(u**2 for u in reference_uncertainties.values())
       )

       return formation_energy, total_uncertainty

   # Example usage (simplified)
   h2o = molecule("H2O")
   references = {
       'H': molecule("H2"),
       'O': molecule("O2")
   }

   # This is a simplified example - real formation energy calculation
   # would require proper stoichiometry and reference state handling

Confidence Intervals
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from scipy import stats

   def calculate_confidence_interval(ensemble, confidence=0.95):
       """Calculate confidence interval from ensemble predictions"""
       alpha = 1 - confidence
       lower_percentile = (alpha/2) * 100
       upper_percentile = (1 - alpha/2) * 100

       lower_bound = np.percentile(ensemble, lower_percentile)
       upper_bound = np.percentile(ensemble, upper_percentile)

       return lower_bound, upper_bound

   # Example with molecule
   atoms = molecule("NH3")
   atoms.calc = calculator

   energy = atoms.get_potential_energy()
   ensemble = atoms.calc.get_energy_ensemble()

   # Calculate confidence intervals
   ci_68 = calculate_confidence_interval(ensemble, 0.68)  # 1σ
   ci_95 = calculate_confidence_interval(ensemble, 0.95)  # 2σ

   print(f"NH3 Energy Analysis:")
   print(f"  Point estimate: {energy:.3f} eV")
   print(f"  68% CI: [{ci_68[0]:.3f}, {ci_68[1]:.3f}] eV")
   print(f"  95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}] eV")

Quality Assessment
------------------

Uncertainty Calibration
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def assess_prediction_quality(structures, reference_energies=None):
       """Assess the quality of uncertainty estimates"""

       energies = []
       uncertainties = []

       for atoms in structures:
           atoms.calc = calculator
           energy = atoms.get_potential_energy()
           uncertainty = atoms.calc.get_energy_uncertainty()

           energies.append(energy)
           uncertainties.append(uncertainty)

       energies = np.array(energies)
       uncertainties = np.array(uncertainties)

       # Basic statistics
       mean_uncertainty = np.mean(uncertainties)
       std_uncertainty = np.std(uncertainties)

       print(f"Uncertainty Statistics:")
       print(f"  Mean uncertainty: {mean_uncertainty:.3f} eV")
       print(f"  Std of uncertainties: {std_uncertainty:.3f} eV")
       print(f"  Min uncertainty: {np.min(uncertainties):.3f} eV")
       print(f"  Max uncertainty: {np.max(uncertainties):.3f} eV")

       # If reference values are available, calculate calibration
       if reference_energies is not None:
           reference_energies = np.array(reference_energies)
           errors = np.abs(energies - reference_energies)

           # Check if uncertainties correlate with errors
           correlation = np.corrcoef(uncertainties, errors)[0, 1]
           print(f"  Uncertainty-error correlation: {correlation:.3f}")

           return energies, uncertainties, errors

       return energies, uncertainties

   # Example assessment
   test_structures = [
       bulk("Si", cubic=True, a=5.43, crystalstructure="diamond"),
       bulk("C", cubic=True, a=3.55, crystalstructure="diamond"),
       molecule("H2O"),
       molecule("CO2")
   ]

   energies, uncertainties = assess_prediction_quality(test_structures)

Outlier Detection
~~~~~~~~~~~~~~~~~

.. code-block:: python

   def detect_high_uncertainty_structures(structures, threshold_percentile=95):
       """Identify structures with unusually high uncertainty"""

       results = []

       for i, atoms in enumerate(structures):
           atoms.calc = calculator
           energy = atoms.get_potential_energy()
           uncertainty = atoms.calc.get_energy_uncertainty()

           results.append({
               'index': i,
               'formula': atoms.get_chemical_formula(),
               'energy': energy,
               'uncertainty': uncertainty,
               'relative_uncertainty': uncertainty / abs(energy) * 100
           })

       # Find high uncertainty structures
       uncertainties = [r['uncertainty'] for r in results]
       threshold = np.percentile(uncertainties, threshold_percentile)

       high_uncertainty = [r for r in results if r['uncertainty'] > threshold]

       print(f"High uncertainty structures (>{threshold_percentile}th percentile):")
       for result in high_uncertainty:
           print(f"  {result['formula']}: {result['energy']:.3f} ± {result['uncertainty']:.3f} eV "
                 f"({result['relative_uncertainty']:.2f}%)")

       return results, high_uncertainty

Best Practices
--------------

When to Use Uncertainty Quantification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **New chemical spaces**: When exploring materials not well represented in training data
2. **Critical decisions**: When model predictions inform important scientific or engineering decisions
3. **Method validation**: When comparing with experimental data or other computational methods
4. **Active learning**: When deciding which structures need additional training data

Interpreting Uncertainties
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def interpret_uncertainty(uncertainty, energy):
       """Provide interpretation guidelines for uncertainties"""

       relative_uncertainty = uncertainty / abs(energy) * 100

       if relative_uncertainty < 1.0:
           confidence = "High"
           recommendation = "Prediction is likely reliable"
       elif relative_uncertainty < 5.0:
           confidence = "Medium"
           recommendation = "Prediction is reasonably reliable, consider validation"
       else:
           confidence = "Low"
           recommendation = "Prediction should be validated with other methods"

       print(f"Uncertainty Analysis:")
       print(f"  Absolute uncertainty: {uncertainty:.3f} eV")
       print(f"  Relative uncertainty: {relative_uncertainty:.2f}%")
       print(f"  Confidence level: {confidence}")
       print(f"  Recommendation: {recommendation}")

       return confidence, recommendation

   # Example usage
   atoms = molecule("H2O")
   atoms.calc = calculator

   energy = atoms.get_potential_energy()
   uncertainty = atoms.calc.get_energy_uncertainty()

   interpret_uncertainty(uncertainty, energy)

Limitations and Considerations
------------------------------

Important Notes
~~~~~~~~~~~~~~~

1. **Model-specific**: Uncertainties are specific to the PET-MAD model and training data
2. **Not calibrated probabilities**: Uncertainties provide relative confidence, not absolute probabilities
3. **Computational cost**: Uncertainty quantification adds computational overhead
4. **Version dependency**: Only available for specific model versions (v1.0.2)

.. warning::
   Uncertainty estimates should be used as guidance for assessing prediction reliability, not as absolute error bounds. Always validate critical predictions with additional methods when possible.

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time

   # Compare performance with and without uncertainty quantification
   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")

   # Without uncertainty
   calc_fast = PETMADCalculator(version="v1.0.2", device="cpu")
   atoms.calc = calc_fast

   start_time = time.time()
   energy_fast = atoms.get_potential_energy()
   time_fast = time.time() - start_time

   # With uncertainty
   calc_uq = PETMADCalculator(
       version="v1.0.2",
       device="cpu",
       calculate_uncertainty=True,
       calculate_ensemble=True
   )
   atoms.calc = calc_uq

   start_time = time.time()
   energy_uq = atoms.get_potential_energy()
   uncertainty = atoms.calc.get_energy_uncertainty()
   time_uq = time.time() - start_time

   print(f"Performance comparison:")
   print(f"  Without UQ: {time_fast:.3f} s")
   print(f"  With UQ: {time_uq:.3f} s")
   print(f"  Overhead: {time_uq/time_fast:.1f}x")
   print(f"  Energy difference: {abs(energy_fast - energy_uq):.6f} eV")
