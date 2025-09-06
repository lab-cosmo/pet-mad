from pet_mad.calculator import PETMADCalculator
from ase.build import bulk
import numpy as np
from pet_mad._version import PET_MAD_UQ_AVAILABILITY_VERSION
from packaging.version import Version
import pytest

VERSIONS = ("1.0.2", "1.0.1")


@pytest.mark.parametrize(
    "version",
    VERSIONS,
)
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
        calc = PETMADCalculator(
            version=version, calculate_uncertainty=True, calculate_ensemble=True
        )
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        energy_uncertainty = atoms.calc.get_energy_uncertainty()
        energy_ensemble = atoms.calc.get_energy_ensemble()

        assert np.allclose(np.mean(energy_ensemble), energy, atol=1e-1)
        assert np.allclose(energy_uncertainty, np.std(energy_ensemble), atol=1e-1)


def test_error_no_uq_requested():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    calc = PETMADCalculator(
        version="latest", calculate_uncertainty=False, calculate_ensemble=False
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    with pytest.raises(ValueError, match="Energy uncertainty is not available"):
        calc.get_energy_uncertainty()
    with pytest.raises(ValueError, match="Energy ensemble is not available"):
        calc.get_energy_ensemble()


def test_error_model_not_evaluated():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    calc = PETMADCalculator(
        version="latest", calculate_uncertainty=True, calculate_ensemble=True
    )
    atoms.calc = calc
    with pytest.raises(ValueError, match="Energy uncertainty is not available"):
        calc.get_energy_uncertainty()
    with pytest.raises(ValueError, match="Energy ensemble is not available"):
        calc.get_energy_ensemble()
