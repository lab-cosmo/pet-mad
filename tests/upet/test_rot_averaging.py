import numpy as np
import pytest
from ase.build import bulk

from upet._version import UPET_AVAILABLE_MODELS
from upet.calculator import UPETCalculator


GRID_ORDERS = [3, 5]
BATCH_SIZES = [1, 2, 4, 8]


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_calc_rot_averaging(model_name):
    if "-xl" in model_name or "-l" in model_name:
        pytest.skip("Skipping XL models and L models due to large size.")
    atoms = bulk("C", cubic=True, a=3.57, crystalstructure="diamond")
    atoms.rattle(0.05)
    atoms.calc = UPETCalculator(model=model_name)

    target_energy = atoms.get_potential_energy()
    target_forces = atoms.get_forces()
    target_stress = atoms.get_stress()

    atoms.calc = UPETCalculator(
        model=model_name,
        rotational_average_order=3,
        rotational_average_batch_size=8,
    )
    averaged_energy = atoms.get_potential_energy()
    averaged_forces = atoms.get_forces()
    averaged_stress = atoms.get_stress()
    assert "energy_rot_std" in atoms.calc.results
    assert "forces_rot_std" in atoms.calc.results
    assert "stress_rot_std" in atoms.calc.results
    np.testing.assert_allclose(averaged_energy, target_energy, atol=1e-2, rtol=1e-2)
    np.testing.assert_allclose(averaged_forces, target_forces, atol=1e-1, rtol=1e-1)
    np.testing.assert_allclose(averaged_stress, target_stress, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_calc_rot_averaging_non_conservative(model_name):
    if "-xl" in model_name or "-l" in model_name:
        pytest.skip("Skipping XL models and L models due to large size.")
    if model_name in ["pet-spice-s", "pet-spice-l"]:
        msg = "Non-conservative forces and stresses are not available"
        with pytest.raises(NotImplementedError, match=msg):
            _ = UPETCalculator(model=model_name, non_conservative=True)
    else:
        atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
        atoms.rattle(0.05)
        atoms.calc = UPETCalculator(model=model_name, non_conservative=True)

        target_energy = atoms.get_potential_energy()
        target_forces = atoms.get_forces()
        target_stress = atoms.get_stress()

        atoms.calc = UPETCalculator(
            model=model_name,
            non_conservative=True,
            rotational_average_order=3,
            rotational_average_batch_size=8,
        )
        averaged_energy = atoms.get_potential_energy()
        averaged_forces = atoms.get_forces()
        averaged_stress = atoms.get_stress()
        assert "energy_rot_std" in atoms.calc.results
        assert "forces_rot_std" in atoms.calc.results
        assert "stress_rot_std" in atoms.calc.results
        np.testing.assert_allclose(averaged_energy, target_energy, atol=5e-2)
        np.testing.assert_allclose(averaged_forces, target_forces, atol=1e-1)
        np.testing.assert_allclose(averaged_stress, target_stress, atol=5e-2)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batched_calc_rot_averaging(batch_size):
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.rattle(0.05)
    calc = UPETCalculator(model="pet-mad-s", rotational_average_order=3)
    batched_calc = UPETCalculator(
        model="pet-mad-s",
        rotational_average_order=3,
        rotational_average_batch_size=batch_size,
    )
    atoms.calc = calc
    target_energy = atoms.get_potential_energy()
    target_forces = atoms.get_forces()
    target_stress = atoms.get_stress()

    atoms.calc = batched_calc
    batched_energy = atoms.get_potential_energy()
    batched_forces = atoms.get_forces()
    batched_stress = atoms.get_stress()

    np.testing.assert_allclose(batched_energy, target_energy, atol=1e-6)
    np.testing.assert_allclose(batched_forces, target_forces, atol=1e-6)
    np.testing.assert_allclose(batched_stress, target_stress, atol=1e-6)
