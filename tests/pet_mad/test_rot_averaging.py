from pet_mad.calculator import PETMADCalculator
from pet_mad.utils import (
    rotate_atoms,
    compute_rotational_average,
    accumulate_rotational_moments,
)
import numpy as np
from ase import Atoms
from ase.build import bulk
import pytest

GRID_ORDERS = [2, 3, 5, 7, 9]

MODEL_PATH = "/Users/paolo/Software/pet-mad/pet-mad-dev.ckpt"


def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rotx(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def test_rot_averaging():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.rattle(0.05)
    calc = PETMADCalculator(checkpoint_path=MODEL_PATH)
    atoms.calc = calc

    for order in GRID_ORDERS:
        if order == 2:
            with pytest.raises(
                ValueError, match="Lebedev-Laikov grid order 2 is not available."
            ):
                atoms.calc = PETMADCalculator(
                    checkpoint_path=MODEL_PATH, rotational_average_order=order
                )
        else:
            leb_calc = PETMADCalculator(
                checkpoint_path=MODEL_PATH, rotational_average_order=order
            )
            atoms.calc = leb_calc
            atoms.get_potential_energy()
            atoms.get_forces()
            atoms.get_stress()

            assert "energy_rot_std" in atoms.calc.results
            assert "forces_rot_std" in atoms.calc.results
            assert "stress_rot_std" in atoms.calc.results


def test_rotate_atoms_positions_and_cell():
    """rotate_atoms must rotate positions and cell as r' = r @ R^T (row-vectors)."""
    # simple triatomic + cubic cell
    a0 = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.2, 0.3, 0.0]],
        cell=np.eye(3),
        pbc=True,
    )
    R = rotz(np.pi / 2)[None, ...]  # (1,3,3)
    out = rotate_atoms(a0, R)
    a1 = out[0]

    # expected
    pos_ref = a0.get_positions() @ R[0].T
    cell_ref = a0.cell.array @ R[0].T
    np.testing.assert_allclose(a1.get_positions(), pos_ref, atol=1e-14)
    np.testing.assert_allclose(a1.cell.array, cell_ref, atol=1e-14)


def test_accumulate_and_finalize_energy_mean_std():
    """Streaming mean/std for scalar energies equals weighted formulas."""
    rng = np.random.default_rng(0)
    B = 10
    energies = rng.normal(0.0, 1.0, size=B)
    w = rng.random(B) + 0.1
    R = np.stack([rotz(0.1 * i) for i in range(B)], axis=0)

    m1, m2 = {}, {}
    res_b = {"energy": energies}
    accumulate_rotational_moments(m1, m2, res_b, R, w)
    out = compute_rotational_average(m1, m2, float(w.sum()), suffix_std="_rot_std")

    E_mean_ref = (energies * w).sum() / w.sum()
    E_var_ref = (energies**2 * w).sum() / w.sum() - E_mean_ref**2
    E_std_ref = np.sqrt(max(E_var_ref, 0.0))

    assert out["energy"] == pytest.approx(E_mean_ref, rel=1e-12, abs=0.0)
    assert out["energy_rot_std"] == pytest.approx(E_std_ref, rel=1e-12, abs=0.0)


def test_forces_rotation_and_zero_std_when_equivariant():
    """
    If model outputs are perfect rotations of a fixed lab force, mean recovers lab force
    and std=0.
    """
    B, N = 8, 5
    R = np.stack([rotx(0.3 * i) for i in range(B)], axis=0)
    w = np.ones(B)

    F_lab_true = np.ones((N, 3))  # fixed lab-frame force
    # model returns forces in rotated frame: F_rot = F_lab @ R^T
    F_rot = np.matmul(F_lab_true[None, :, :], np.transpose(R, (0, 2, 1)))  # (B,N,3)

    m1, m2 = {}, {}
    res_b = {"forces": F_rot}
    accumulate_rotational_moments(m1, m2, res_b, R, w)
    out = compute_rotational_average(m1, m2, float(w.sum()), suffix_std="_rot_std")

    np.testing.assert_allclose(out["forces"], F_lab_true, atol=1e-14)
    np.testing.assert_allclose(out["forces_rot_std"], 0.0, atol=1e-14)


def test_stress_rotation_and_zero_std_when_equivariant():
    """
    If model outputs are perfect rotations of a fixed lab stress, mean recovers lab
    stress and std=0.
    """
    B = 9
    R = np.stack([rotz(0.2 * i) @ rotx(0.1 * i) for i in range(B)], axis=0)
    w = np.linspace(0.5, 1.5, B)

    S_lab_true = np.diag([2.0, 3.0, 5.0])
    # model returns stress in rotated frame: S_rot = R^T S_lab R
    S_rot = np.matmul(np.transpose(R, (0, 2, 1)), np.matmul(S_lab_true[None, :, :], R))

    m1, m2 = {}, {}
    res_b = {"stress": S_rot}
    accumulate_rotational_moments(m1, m2, res_b, R, w)
    out = compute_rotational_average(m1, m2, float(w.sum()), suffix_std="_rot_std")

    np.testing.assert_allclose(out["stress"], S_lab_true, atol=1e-14)
    np.testing.assert_allclose(out["stress_rot_std"], np.zeros((3, 3)), atol=1e-7)


def test_streaming_equals_naive_vector_tensor():
    """Streaming accumulation equals naive stack-and-reduce for arbitrary data."""
    rng = np.random.default_rng(1)
    B, N = 11, 7
    R = np.stack(
        [rotx(rng.uniform(0, np.pi)) @ rotz(rng.uniform(0, np.pi)) for _ in range(B)],
        axis=0,
    )
    w = rng.random(B) + 0.2

    # arbitrary "rotated-frame" data
    F_rot = rng.normal(size=(B, N, 3))
    S_rot = rng.normal(size=(B, 3, 3))

    # naive rotate-back then weighted mean/std
    F_lab = np.matmul(F_rot, R)  # (B,N,3)
    S_lab = np.matmul(R, np.matmul(S_rot, np.transpose(R, (0, 2, 1))))

    W = w.sum()
    F_mean_ref = (F_lab * w[:, None, None]).sum(0) / W
    F_std_ref = np.sqrt(((F_lab**2) * w[:, None, None]).sum(0) / W - F_mean_ref**2)

    S_mean_ref = (S_lab * w[:, None, None]).sum(0) / W
    S_std_ref = np.sqrt(((S_lab**2) * w[:, None, None]).sum(0) / W - S_mean_ref**2)

    m1, m2 = {}, {}
    res_b = {"forces": F_rot, "stress": S_rot}
    accumulate_rotational_moments(m1, m2, res_b, R, w)
    out = compute_rotational_average(m1, m2, float(W), suffix_std="_rot_std")

    np.testing.assert_allclose(out["forces"], F_mean_ref, atol=1e-12)
    np.testing.assert_allclose(out["forces_rot_std"], F_std_ref, atol=1e-12)
    np.testing.assert_allclose(out["stress"], S_mean_ref, atol=1e-12)
    np.testing.assert_allclose(out["stress_rot_std"], S_std_ref, atol=1e-7)


@pytest.fixture
def calc_with_rotavg(monkeypatch):
    """
    Construct PETMADCalculator with rotational averaging enabled and
    monkeypatch its compute_energy to return controlled per-rotation data.

    The per-rotation rotation matrix R is attached on Atoms.info["R"]
    by monkeypatching rotate_atoms; we still call your rotate_atoms internally.
    """
    # Keep original rotate_atoms
    from pet_mad.calculator import rotate_atoms as rotate_atoms_original

    def rotate_atoms_with_info(atoms, rotations):
        lst = rotate_atoms_original(atoms, rotations)
        for a, R in zip(lst, rotations):
            a.info["R"] = R
        return lst

    monkeypatch.setattr(
        "pet_mad.calculator.rotate_atoms", rotate_atoms_with_info, raising=True
    )

    # Create a calculator
    calc = PETMADCalculator(
        checkpoint_path=MODEL_PATH,
        rotational_average_order=3,
        initial_batch_size=16,
        min_batch_size=1,
    )

    # Controlled lab-frame "truth"
    E_true = 2.5
    F_lab_true = np.array([[1.0, -2.0, 0.5], [0.0, 0.0, 0.0]], dtype=float)
    S_lab_true = np.diag([1.0, 4.0, 9.0]).astype(float)

    def fake_compute_energy(atoms_list, compute_forces_and_stresses=True):
        energies = [E_true] * len(atoms_list)
        out = {"energy": np.asarray(energies, dtype=float)}
        if compute_forces_and_stresses:
            forces = [(F_lab_true @ a.info["R"].T) for a in atoms_list]
            stresses = [(a.info["R"].T @ S_lab_true @ a.info["R"]) for a in atoms_list]
            out["forces"] = np.asarray(forces, dtype=float)
            out["stress"] = np.asarray(stresses, dtype=float)
        return out

    monkeypatch.setattr(calc, "compute_energy", fake_compute_energy, raising=True)
    return calc, E_true, F_lab_true, S_lab_true


def test_calculator_rotational_average_mean_and_std(calc_with_rotavg):
    calc, E_true, F_lab_true, S_lab_true = calc_with_rotavg

    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = calc

    # Trigger ASE call path with averaging
    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    S = atoms.get_stress(voigt=False)

    assert E == pytest.approx(E_true, abs=1e-12)
    np.testing.assert_allclose(F, F_lab_true, atol=1e-12)
    np.testing.assert_allclose(S, S_lab_true, atol=1e-12)

    # std are available in results with suffix "_rot_std" and must be ~0
    assert calc.results["energy_rot_std"] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        calc.results["forces_rot_std"],
        np.zeros_like(calc.results["forces_rot_std"]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        calc.results["stress_rot_std"],
        np.zeros_like(calc.results["stress_rot_std"]),
        atol=1e-6,
    )


def test_calculator_adaptive_backoff(monkeypatch, calc_with_rotavg):
    """
    Force OOM-like behavior by raising an exception when batch_size > threshold.
    Verify that the calculator halves the batch and succeeds.
    """
    calc, E_true, F_lab_true, S_lab_true = calc_with_rotavg
    threshold = 4

    call_counts = {"calls": 0}

    def compute_energy_failing_for_large_batches(
        atoms_list, compute_forces_and_stresses=True
    ):
        call_counts["calls"] += 1
        if len(atoms_list) > threshold:
            raise RuntimeError(
                "CUDA out of memory"
            )  # your _compute_rotational_average checks this text
        # fall back to the successful path
        energies = [E_true] * len(atoms_list)
        out = {"energy": np.asarray(energies)}

        if compute_forces_and_stresses:
            forces = [(F_lab_true @ a.info["R"].T) for a in atoms_list]
            stresses = [(a.info["R"].T @ S_lab_true @ a.info["R"]) for a in atoms_list]
            out["forces"] = np.asarray(forces)
            out["stress"] = np.asarray(stresses)
        return out

    monkeypatch.setattr(
        calc, "compute_energy", compute_energy_failing_for_large_batches, raising=True
    )

    atoms = Atoms("He", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = calc

    # Should complete successfully after backoff
    E = atoms.get_potential_energy()
    F = atoms.get_forces()
    S = atoms.get_stress(voigt=False)

    assert E == pytest.approx(E_true, abs=1e-12)
    np.testing.assert_allclose(F, F_lab_true, atol=1e-12)
    np.testing.assert_allclose(S, S_lab_true, atol=1e-12)
    assert call_counts["calls"] >= 2  # at least one failure + retries


def test_unknown_key_is_rejected_in_accumulator():
    """
    Check the accumulator raises KeyError for non-per-rotation keys.
    """
    B = 3
    R = np.stack([np.eye(3) for _ in range(B)], axis=0)
    w = np.ones(B)
    m1, m2 = {}, {}
    res_b = {"foo": np.array([1.0])}  # not per-rotation; your code raises
    with pytest.raises(KeyError):
        accumulate_rotational_moments(m1, m2, res_b, R, w)
