import re

import numpy as np
import pytest
from ase.build import bulk

from upet._models import get_versions_for_model
from upet._version import UPET_AVAILABLE_MODELS
from upet.ase import UPETCalculator


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
                "Energy uncertainty is not available for the "
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
def test_gradient_ensemble_uncertainty_quantification(model_name):
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
                "Energy ensemble is required for calculating the gradient ensemble "
                "uncertainty (forces, stress), but is not available for the selected "
                "model. The documentation lists the models providing uncertainty "
                "estimates."
            )
            with pytest.raises(NotImplementedError, match=re.escape(message)):
                calc.get_forces_uncertainty(atoms)
        else:
            forces_uncertainty = calc.get_forces_uncertainty(atoms)
            forces_ensemble = calc.get_forces_ensemble(atoms)
            stress_uncertainty = calc.get_stress_uncertainty(atoms)
            stress_ensemble = calc.get_stress_ensemble(atoms)

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


@pytest.mark.skip(
    reason="We have no shipped models with non-conservative UQ available."
)
def test_direct_forces_stress_uncertainty_quantification():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    calc = UPETCalculator(checkpoint_path="model.ckpt", variants={"energy": "r2scan"})
    nc_calc = UPETCalculator(
        checkpoint_path="model.ckpt",
        variants={"energy": "r2scan"},
        non_conservative=True,
    )

    atoms.calc = calc
    forces = atoms.get_forces()
    stress = atoms.get_stress()
    atoms.calc = nc_calc
    nc_forces = atoms.get_forces()
    nc_stress = atoms.get_stress()

    forces_uncertainty = calc.get_forces_uncertainty(atoms)
    forces_ensemble = calc.get_forces_ensemble(atoms)
    nc_forces_uncertainty = nc_calc.get_forces_uncertainty(atoms)
    nc_forces_ensemble = nc_calc.get_forces_ensemble(atoms)

    stress_uncertainty = calc.get_stress_uncertainty(atoms)
    stress_ensemble = calc.get_stress_ensemble(atoms)
    nc_stress_uncertainty = nc_calc.get_stress_uncertainty(atoms)
    nc_stress_ensemble = nc_calc.get_stress_ensemble(atoms)

    n_atoms = len(atoms)
    assert forces_ensemble.shape[:2] == (n_atoms, 3)
    assert forces_uncertainty.shape == (n_atoms, 3)
    assert stress_ensemble.shape[0] == 6
    assert stress_uncertainty.shape == (6,)
    assert nc_forces_ensemble.shape[:2] == (n_atoms, 3)
    assert nc_forces_uncertainty.shape == (n_atoms, 3)
    assert nc_stress_ensemble.shape[0] == 6
    assert nc_stress_uncertainty.shape == (6,)

    assert np.allclose(forces_uncertainty, np.std(forces_ensemble, axis=2), atol=1e-6)
    assert np.allclose(stress_uncertainty, np.std(stress_ensemble, axis=1), atol=1e-6)
    assert np.allclose(
        nc_forces_uncertainty, np.std(nc_forces_ensemble, axis=2), atol=1e-6
    )
    assert np.allclose(
        nc_stress_uncertainty, np.std(nc_stress_ensemble, axis=1), atol=1e-6
    )

    assert np.allclose(np.mean(forces_ensemble, axis=2), forces, atol=1e-4)
    assert np.allclose(np.mean(nc_forces_ensemble, axis=2), nc_forces, atol=1e-4)
    assert np.allclose(np.mean(stress_ensemble, axis=1), stress, atol=1e-4)
    assert np.allclose(np.mean(nc_stress_ensemble, axis=1), nc_stress, atol=1e-4)


def test_direct_forces_stress_uncertainty_quantification_raises_errors():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")

    calc = UPETCalculator(model="pet-mad-s", version="1.5.0", non_conservative=True)
    # no shipped model carries a direct forces ensemble; LLPR checkpoints trained
    # with a non-conservative forces head do
    message = (
        "Non-conservative {quantity} uncertainty/ensemble is not available for the "
        "selected model. Consider switching-off `non-conservative` mode. "
        "The documentation lists the models providing uncertainty estimates "
        "for non-conservative outputs."
    )
    with pytest.raises(
        NotImplementedError,
        match=re.escape(message.format(quantity="forces")),
    ):
        calc.get_forces_uncertainty(atoms)
    with pytest.raises(
        NotImplementedError,
        match=re.escape(message.format(quantity="forces")),
    ):
        calc.get_forces_ensemble(atoms)
    with pytest.raises(
        NotImplementedError,
        match=re.escape(message.format(quantity="stress")),
    ):
        calc.get_stress_uncertainty(atoms)
    with pytest.raises(
        NotImplementedError,
        match=re.escape(message.format(quantity="stress")),
    ):
        calc.get_stress_ensemble(atoms)


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
