"""
Geometry optimization (LBFGS)
=============================

Two-stage geometry optimization of a slightly perturbed silicon unit cell.
First, positions are relaxed at fixed cell with
:py:class:`ase.optimize.LBFGS`. Then the cell itself is relaxed jointly
with the atomic positions by wrapping the ``Atoms`` in
:py:class:`ase.filters.FrechetCellFilter`.

The script records the maximum force and total energy at every step and
plots them so that the convergence of the two stages is visible at a
glance.
"""

import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk
from ase.filters import FrechetCellFilter
from ase.optimize import LBFGS

from upet.calculator import UPETCalculator


atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")

# perturb positions and cell so the optimizer has something to do
atoms.rattle(0.1, seed=0)  # ASE's built-in random displacement method
atoms.set_cell(atoms.cell * 1.05, scale_atoms=True)

calculator = UPETCalculator(model="pet-mad-xs", version="1.5.0", device="cpu")
atoms.calc = calculator

history = {"stage": [], "energy": [], "fmax": []}  # type: ignore


def record(stage_name):
    def _cb():
        results = calculator.results
        history["stage"].append(stage_name)
        history["energy"].append(float(results["energy"]))
        history["fmax"].append(float(np.linalg.norm(results["forces"], axis=1).max()))

    return _cb


# stage 1: positions only
opt_pos = LBFGS(atoms)
opt_pos.attach(record("positions"), interval=1)
opt_pos.run(fmax=0.05, steps=30)

# stage 2: joint position + cell relaxation
filtered = FrechetCellFilter(atoms)
opt_cell = LBFGS(filtered)
opt_cell.attach(record("cell"), interval=1)
opt_cell.run(fmax=0.05, steps=30)

steps = np.arange(len(history["energy"]))
stages = np.array(history["stage"])
boundary = (
    int(np.searchsorted(stages == "cell", True))
    if (stages == "cell").any()
    else len(stages)
)

fig, (ax_e, ax_f) = plt.subplots(1, 2, figsize=(9, 3.5))
ax_e.plot(steps, history["energy"], "o-")
ax_e.axvline(boundary - 0.5, color="k", ls="--", lw=0.8)
ax_e.set_xlabel("optimization step")
ax_e.set_ylabel("total energy [eV]")
ax_e.set_title("Energy vs. step")

ax_f.semilogy(steps, history["fmax"], "o-")
ax_f.axvline(boundary - 0.5, color="k", ls="--", lw=0.8)
ax_f.set_xlabel("optimization step")
ax_f.set_ylabel("max |force| [eV/Å]")
ax_f.set_title("Force convergence")

fig.tight_layout()
plt.show()
