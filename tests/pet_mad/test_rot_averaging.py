from pet_mad.calculator import PETMADCalculator
from ase.build import bulk
import pytest

GRID_ORDERS = [2, 3, 5, 7, 9]


def test_rot_averaging():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.rattle(0.05)
    calc = PETMADCalculator()
    atoms.calc = calc

    for order in GRID_ORDERS:
        if order == 2:
            with pytest.raises(
                ValueError, match="Lebedev-Laikov grid order 2 is not available."
            ):
                atoms.calc = PETMADCalculator(rotational_average_order=order)
        else:
            leb_calc = PETMADCalculator(rotational_average_order=order)
            atoms.calc = leb_calc
            atoms.get_potential_energy()
            atoms.get_forces()
            atoms.get_stress()

            assert "energy_rot_std" in atoms.calc.results
            assert "forces_rot_std" in atoms.calc.results
            assert "stress_rot_std" in atoms.calc.results
