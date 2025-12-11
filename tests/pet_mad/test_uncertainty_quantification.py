import re

import numpy as np
import pytest
from ase.build import bulk
from packaging.version import Version

from pet_mad._version import PET_MAD_UQ_AVAILABILITY_VERSION
from pet_mad.calculator import PETMADCalculator


VERSIONS = ["1.0.2", "1.0.1"]


@pytest.mark.parametrize("version", VERSIONS)
def test_uncertainty_quantification(version):
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    if Version(version) < Version(PET_MAD_UQ_AVAILABILITY_VERSION):
        msg = (
            f"Energy uncertainty and ensemble are not available for version {version}."
        )
        with pytest.raises(NotImplementedError, match=msg):
            calc = PETMADCalculator(
                version=version, calculate_uncertainty=True, calculate_ensemble=True
            )
    else:
        calc = PETMADCalculator(version=version)

        energy_uncertainty = calc.get_energy_uncertainty(atoms)
        energy_ensemble = calc.get_energy_ensemble(atoms)

        atoms.calc = calc
        energy = atoms.get_potential_energy()

        assert np.allclose(np.mean(energy_ensemble), energy, atol=1e-6)
        assert np.allclose(energy_uncertainty, np.std(energy_ensemble), atol=1e-1)

        # getting uncertainty and ensemble without an `atoms` parameter
        energy_uncertainty_2 = calc.get_energy_uncertainty()
        energy_ensemble_2 = calc.get_energy_ensemble()
        assert np.allclose(energy_uncertainty, energy_uncertainty_2, atol=1e-6)
        assert np.allclose(energy_ensemble, energy_ensemble_2, atol=1e-6)


def test_uq_deprecation_warning():
    message = (
        "`calculate_uncertainty` is deprecated, you can directly call "
        "`calculator.get_energy_uncertainty(atoms)`"
    )
    with pytest.warns(match=re.escape(message)):
        _ = PETMADCalculator(version="latest", calculate_uncertainty=True)

    message = (
        "`calculate_ensemble` is deprecated, you can directly call "
        "`calculator.get_energy_ensemble(atoms)`"
    )
    with pytest.warns(match=re.escape(message)):
        _ = PETMADCalculator(version="latest", calculate_ensemble=True)


def test_error_model_not_evaluated():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    calc = PETMADCalculator(version="latest")
    atoms.calc = calc

    message = "No `atoms` provided and no previously calculated atoms found."
    with pytest.raises(ValueError, match=message):
        calc.get_energy_uncertainty()
    with pytest.raises(ValueError, match=message):
        calc.get_energy_ensemble()
