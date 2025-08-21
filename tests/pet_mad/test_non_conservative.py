from pet_mad.calculator import PETMADCalculator
from ase.build import bulk
import numpy as np
import pytest

def test_non_conservative():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")

    msg1 = (
        "trying to upgrade an old model checkpoint with unknown version, this "
        "might fail and require manual modifications"
    )
    msg2 = (
        "PET assumes that Cartesian tensors of rank 2 are stress-like, meaning that "
        "they are symmetric and intensive. If this is not the case, please use a "
        "different model."
    )

    with pytest.warns(UserWarning, match=f"({msg1}|{msg2})"):
        calc = PETMADCalculator(version="latest", non_conservative=False)
        calc_nc = PETMADCalculator(version="latest", non_conservative=True)

    atoms.calc = calc
    forces = atoms.get_forces()
    stresses = atoms.get_stress()

    atoms.calc = calc_nc
    forces_nc = atoms.get_forces()
    stresses_nc = atoms.get_stress()

    assert np.allclose(forces, forces_nc, atol=1e-1)
    assert np.allclose(stresses, stresses_nc, atol=1e-1)