from pet_mad.calculator import PETMADCalculator
from ase.build import bulk
import pytest
import numpy as np

GRID_ORDERS = [2, 3, 5, 7, 9]


def test_rot_averaging():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.rattle(0.05)
    calc = PETMADCalculator()
    atoms.calc = calc
    target_energy = atoms.get_potential_energy()
    target_forces = atoms.get_forces()

    for order in GRID_ORDERS:
        if order == 2:
            with pytest.raises(
                ValueError, match="Lebedev-Laikov grid order 2 is not available."
            ):
                atoms.calc = PETMADCalculator(rotational_average_order=order)
        else:
            atoms.calc = PETMADCalculator(rotational_average_order=order)
            averaged_energy = atoms.get_potential_energy()
            averaged_forces = atoms.get_forces()
            assert "energy_rot_std" in atoms.calc.results
            assert "forces_rot_std" in atoms.calc.results
            assert np.allclose(averaged_energy, target_energy, atol=1e-3)
            assert np.allclose(averaged_forces, target_forces, atol=5e-1)
