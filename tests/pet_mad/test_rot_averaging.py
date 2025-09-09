from pet_mad.calculator import PETMADCalculator
from pet_mad.utils import rotate_atoms, get_so3_rotations, compute_rotational_average
from ase.build import bulk, molecule
import pytest
import numpy as np

GRID_ORDERS = [3, 5, 7]
NUM_ROTATIONS = [4, 8, 12]
BATCH_SIZES = [1, 2, 4, 8]


@pytest.mark.parametrize("order", GRID_ORDERS)
@pytest.mark.parametrize("num_rotations", NUM_ROTATIONS)
def test_get_rotations(order, num_rotations):
    rotations = get_so3_rotations(order, num_rotations)
    random_vector = np.random.rand(3)
    random_vector = random_vector / np.linalg.norm(random_vector)
    rotated_vectors = [rotation @ random_vector for rotation in rotations]
    np.testing.assert_allclose(np.linalg.norm(rotated_vectors, axis=1), 1.0)
    np.testing.assert_allclose(
        np.linalg.norm(np.sum(rotated_vectors, axis=0)), 0.0, atol=1e-10, rtol=1e-10
    )


def test_rotate_atoms():
    atoms = molecule("H2O")
    rotations = get_so3_rotations(3, 4)
    rotated_atoms = rotate_atoms(atoms, rotations)
    assert len(rotated_atoms) == len(rotations)
    for rotation, item in zip(rotations, rotated_atoms):
        np.testing.assert_allclose(
            item.get_positions(), atoms.get_positions() @ rotation.T
        )
        if atoms.cell is not None:
            np.testing.assert_allclose(item.get_cell(), atoms.cell.array @ rotation.T)


def test_compute_rotational_average():
    rotations = get_so3_rotations(3, 4)
    base_forces = np.random.rand(8, 3)  # n_atoms, 3
    base_stress = np.random.rand(3, 3)  # 3, 3
    results = {
        "energy": [1.0, 2.0, 3.0],
        "forces": [base_forces @ rotation.T for rotation in rotations],
        "stress": [rotation @ base_stress @ rotation.T for rotation in rotations],
    }
    rotations = get_so3_rotations(3, 4)
    averaged_results = compute_rotational_average(results, rotations)
    assert "energy_rot_std" in averaged_results
    assert "forces_rot_std" in averaged_results
    assert "stress_rot_std" in averaged_results
    np.testing.assert_allclose(averaged_results["energy"], 2.0)
    np.testing.assert_allclose(averaged_results["forces"], base_forces)
    np.testing.assert_allclose(averaged_results["stress"], base_stress)
    np.testing.assert_allclose(
        averaged_results["energy_rot_std"], np.std(results["energy"])
    )
    np.testing.assert_allclose(
        averaged_results["forces_rot_std"], 0.0, atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(
        averaged_results["stress_rot_std"], 0.0, atol=1e-10, rtol=1e-10
    )


@pytest.mark.parametrize("order", GRID_ORDERS)
def test_calc_rot_averaging(order):
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.rattle(0.05)
    calc = PETMADCalculator()
    atoms.calc = calc

    target_energy = atoms.get_potential_energy()
    target_forces = atoms.get_forces()
    target_stress = atoms.get_stress()

    atoms.calc = PETMADCalculator(rotational_average_order=order)
    averaged_energy = atoms.get_potential_energy()
    averaged_forces = atoms.get_forces()
    averaged_stress = atoms.get_stress()
    assert "energy_rot_std" in atoms.calc.results
    assert "forces_rot_std" in atoms.calc.results
    assert "stress_rot_std" in atoms.calc.results
    np.testing.assert_allclose(averaged_energy, target_energy, atol=1e-2)
    np.testing.assert_allclose(averaged_forces, target_forces, atol=1e-2)
    np.testing.assert_allclose(averaged_stress, target_stress, atol=1e-2)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
def test_batched_calc_rot_averaging(batch_size):
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.rattle(0.05)
    calc = PETMADCalculator(rotational_average_order=3)
    batched_calc = PETMADCalculator(
        rotational_average_order=3, rotational_average_batch_size=batch_size
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


def test_raises_bad_grid_order_error():
    with pytest.raises(
        ValueError, match="Lebedev-Laikov grid order 2 is not available."
    ):
        PETMADCalculator(rotational_average_order=2)
