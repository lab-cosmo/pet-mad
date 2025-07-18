from pet_mad.calculator import PETMADCalculator
from ase.build import bulk
import numpy as np
import pytest
from packaging.version import Version

from pet_mad._version import UQ_AVAILABILITY_VERSION

@pytest.mark.parametrize("version", ["1.1.0"])
def test_uncertainty_quantification(version):
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    if Version(version) < Version(UQ_AVAILABILITY_VERSION):
        with pytest.raises(NotImplementedError) as e:
            calc = PETMADCalculator(version=version, calculate_uncertainty=True, calculate_ensemble=True)
        assert f"Energy uncertainty and ensemble are not available for version {version}. " in str(e.value)
    else:
        calc = PETMADCalculator(version=version, calculate_uncertainty=True, calculate_ensemble=True)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        energy_uncertainty = atoms.calc.get_energy_uncertainty()
        energy_ensemble = atoms.calc.get_energy_ensemble()
        
        assert np.allclose(np.mean(energy_ensemble), energy, atol=1e-1)
        assert np.allclose(energy_uncertainty, np.std(energy_ensemble), atol=1e-1)