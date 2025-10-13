import numpy as np
import pytest
from ase.build import bulk
from packaging.version import Version

from pet_mad._version import PET_MAD_NC_AVAILABILITY_VERSION
from pet_mad.calculator import PETMADCalculator


VERSIONS = ("1.1.0", "1.0.2", "1.0.1")


@pytest.mark.parametrize(
    "version",
    VERSIONS,
)
def test_non_conservative(version):
    if Version(version) < Version(PET_MAD_NC_AVAILABILITY_VERSION):
        msg = f"Non-conservative forces and stresses are not available for version {version}."  # noqa: E501
        with pytest.raises(NotImplementedError, match=msg):
            calc = PETMADCalculator(version=version, non_conservative=True)
    else:
        atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
        calc = PETMADCalculator(version=version, non_conservative=False)
        calc_nc = PETMADCalculator(version=version, non_conservative=True)

        atoms.calc = calc
        forces = atoms.get_forces()
        stresses = atoms.get_stress()

        atoms.calc = calc_nc
        forces_nc = atoms.get_forces()
        stresses_nc = atoms.get_stress()

        assert np.allclose(forces, forces_nc, atol=1e-1)
        assert np.allclose(stresses, stresses_nc, atol=1e-1)
