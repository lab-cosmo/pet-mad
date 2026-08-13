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
