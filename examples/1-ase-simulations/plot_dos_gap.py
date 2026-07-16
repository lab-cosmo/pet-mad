"""
Density of States (DOS) and Bandgap calculations with UPET
================================================================

Single-point evaluation of a bulk silicon cell with PET-MAD-DOS. The two silicon atoms
are displaced off their equilibrium positions and the DOS and bandgap is computed using
the :py:class:`~upet.ase.dos.PETMADDOSCalculator`. Additionally, we showcase the
performance of the denoising algorithm. The DOS is plotted with and without denoising,
and the predicted bandgap is printed for both cases. The denoised DOS is less
oscillatory and non-negative compared to the raw DOS.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.build import bulk

from upet.ase.dos import PETMADDOSCalculator


atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
calculator = PETMADDOSCalculator(version="latest", device="cpu")
with torch.no_grad():
    results = calculator.calculate(atoms)

print(f"The keys in results is: {results.keys()}")
print(f"Predicted bandgap: {results['bandgap']:.4f} eV")
print("True Bandgap: 0.4667 eV (Under the same DFT paramters)")

energy_grid = np.arange(len(results["dos_raw"])) * calculator.energy_interval

plt.plot(energy_grid, results["dos_raw"], label="Raw DOS")
plt.plot(energy_grid, results["dos_denoised"], label="Denoised DOS")
plt.xlabel("Energy (eV)")
plt.ylabel("Density of States (States/eV)")
plt.title("Density of States for Bulk Silicon")
plt.legend()
plt.show()
