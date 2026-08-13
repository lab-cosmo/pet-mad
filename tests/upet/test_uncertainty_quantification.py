import re

import numpy as np
import pytest
from ase.build import bulk

from upet._models import get_versions_for_model
from upet._version import UPET_AVAILABLE_MODELS
from upet.calculator import UPETCalculator


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_uncertainty_quantification(model_name):
    if "-xl" in model_name or "-l" in model_name:
        pytest.skip("Skipping XL models and L models due to large size.")
    model, size = model_name.rsplit("-", 1)
    all_model_versions = get_versions_for_model(model, size)

    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")

    for version in all_model_versions:
        calc = UPETCalculator(
            model=model_name,
            version=version,
        )
        if not calc.supports_uncertainty:
            message = (
                "Energy uncertainty and ensemble are not available for the "
                "selected model. The documentation lists the models providing "
                "uncertainty estimates."
            )
            with pytest.raises(NotImplementedError, match=re.escape(message)):
                calc.get_energy_uncertainty(atoms)
        else:
            energy_uncertainty = calc.get_energy_uncertainty(atoms)
            energy_ensemble = calc.get_energy_ensemble(atoms)

            atoms.calc = calc
            energy = atoms.get_potential_energy()

            assert np.allclose(np.mean(energy_ensemble), energy, atol=1e-6)
            assert np.allclose(energy_uncertainty, np.std(energy_ensemble), atol=3e-1)

            # getting uncertainty and ensemble without an `atoms` parameter
            energy_uncertainty_2 = calc.get_energy_uncertainty()
            energy_ensemble_2 = calc.get_energy_ensemble()
            assert np.allclose(energy_uncertainty, energy_uncertainty_2, atol=1e-6)
            assert np.allclose(energy_ensemble, energy_ensemble_2, atol=1e-6)


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_forces_stress_uncertainty_quantification(model_name):
    if "-xl" in model_name or "-l" in model_name:
        pytest.skip("Skipping XL models and L models due to large size.")
    model, size = model_name.rsplit("-", 1)
    all_model_versions = get_versions_for_model(model, size)

    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")

    for version in all_model_versions:
        calc = UPETCalculator(
            model=model_name,
            version=version,
        )
        if not calc.supports_uncertainty:
            message = (
                "Forces/stress uncertainty and ensemble are not available for "
                "the selected model. The documentation lists the models "
                "providing uncertainty estimates."
            )
            with pytest.raises(NotImplementedError, match=f"^{re.escape(message)}$"):
                calc.get_forces_uncertainty(atoms)
            continue

        forces_ensemble, stress_ensemble = calc.get_forces_and_stress_ensemble(atoms)
        forces_uncertainty = calc.get_forces_uncertainty(atoms)
        stress_uncertainty = calc.get_stress_uncertainty(atoms)

        n_atoms = len(atoms)
        assert forces_ensemble.shape[:2] == (n_atoms, 3)
        assert forces_uncertainty.shape == (n_atoms, 3)
        assert stress_ensemble.shape[0] == 6
        assert stress_uncertainty.shape == (6,)

        atoms.calc = calc
        assert np.allclose(
            np.mean(forces_ensemble, axis=2), atoms.get_forces(), atol=1e-4
        )
        assert np.allclose(
            np.mean(stress_ensemble, axis=1), atoms.get_stress(), atol=1e-4
        )

        # every member is translationally invariant, since the mean over atoms is
        # subtracted from each of them
        assert np.allclose(forces_ensemble.sum(axis=0), 0.0, atol=1e-8)


def test_forces_and_stress_ensemble():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    calc = UPETCalculator(model="pet-mad-s", version="1.5.0")

    forces_ensemble = calc.get_forces_ensemble(atoms)
    stress_ensemble = calc.get_stress_ensemble(atoms)
    forces_combined, stress_combined = calc.get_forces_and_stress_ensemble(atoms)

    assert np.allclose(forces_ensemble, forces_combined, atol=1e-4)
    assert np.allclose(stress_ensemble, stress_combined, atol=1e-4)

    # Voigt order is (xx, yy, zz, yz, xz, xy), symmetrized
    _, stress_3x3 = calc.get_forces_and_stress_ensemble(atoms, voigt=False)
    assert np.allclose(stress_3x3[0, 0], stress_combined[0])
    assert np.allclose((stress_3x3[1, 2] + stress_3x3[2, 1]) / 2, stress_combined[3])
    assert np.allclose((stress_3x3[0, 2] + stress_3x3[2, 0]) / 2, stress_combined[4])
    assert np.allclose((stress_3x3[0, 1] + stress_3x3[1, 0]) / 2, stress_combined[5])

    nc_calc = UPETCalculator(model="pet-mad-s", version="1.5.0", non_conservative=True)
    message = (
        "get_forces_and_stress_ensemble is not available when the calculator was "
        "initialized with non_conservative=True."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        nc_calc.get_forces_and_stress_ensemble(atoms)


def test_forces_ensemble_method_errors():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")

    calc = UPETCalculator(model="pet-mad-s", version="1.5.0")
    # no shipped model carries a direct forces ensemble; LLPR checkpoints trained
    # with a non-conservative forces head do
    message = (
        "Direct forces ensemble (mtt::aux::non_conservative_forces_ensemble) is "
        "not available for the selected model."
    )
    with pytest.raises(NotImplementedError, match=f"^{re.escape(message)}$"):
        calc.get_forces_ensemble(atoms, method="direct")

    nc_calc = UPETCalculator(model="pet-mad-s", version="1.5.0", non_conservative=True)
    message = (
        "method='conservative' is not available when the calculator was "
        "initialized with non_conservative=True."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        nc_calc.get_forces_ensemble(atoms, method="conservative")


def test_uncertainty_with_rotational_averaging():
    # uncertainty outputs are requested from the base model, so they are available
    # but not themselves rotationally averaged
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    calc = UPETCalculator(
        model="pet-mad-s", version="1.5.0", rotational_average_order=3
    )
    plain_calc = UPETCalculator(model="pet-mad-s", version="1.5.0")

    assert np.allclose(
        calc.get_energy_uncertainty(atoms),
        plain_calc.get_energy_uncertainty(atoms),
        atol=1e-6,
    )
    assert np.allclose(
        calc.get_forces_uncertainty(atoms),
        plain_calc.get_forces_uncertainty(atoms),
        atol=1e-6,
    )


def test_error_model_not_evaluated():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    calc = UPETCalculator(
        model="pet-mad-s",
        version="1.0.2",
    )
    atoms.calc = calc

    message = "No `atoms` provided and no previously calculated atoms found."
    with pytest.raises(ValueError, match=message):
        calc.get_energy_uncertainty()
    with pytest.raises(ValueError, match=message):
        calc.get_energy_ensemble()
    with pytest.raises(ValueError, match=message):
        calc.get_forces_ensemble()
    with pytest.raises(ValueError, match=message):
        calc.get_forces_uncertainty()
    with pytest.raises(ValueError, match=message):
        calc.get_stress_ensemble()
    with pytest.raises(ValueError, match=message):
        calc.get_stress_uncertainty()
