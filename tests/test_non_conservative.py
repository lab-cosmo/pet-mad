from pet_mad.calculator import PETMADCalculator
from ase.build import bulk
import numpy as np
from pet_mad._version import NC_AVAILABILITY_VERSION
    
def test_non_conservative():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    calc = PETMADCalculator(version=NC_AVAILABILITY_VERSION, non_conservative=False)
    calc_nc = PETMADCalculator(version=NC_AVAILABILITY_VERSION, non_conservative=True)

    atoms.calc = calc
    forces = atoms.get_forces()
    stresses = atoms.get_stress()

    atoms.calc = calc_nc
    forces_nc = atoms.get_forces()
    stresses_nc = atoms.get_stress()

    assert np.allclose(forces, forces_nc, atol=1e-1)
    assert np.allclose(stresses, stresses_nc, atol=1e-1)