from pet_mad.calculator import PETMADCalculator
from ase.build import bulk
from ase.md.langevin import Langevin
import ase.units
import pytest

def get_warnings():
    
    msg1 = (
        "trying to upgrade an old model checkpoint with unknown version, this "
        "might fail and require manual modifications"
    )
    msg2 = (
        "PET assumes that Cartesian tensors of rank 2 are stress-like, meaning that "
        "they are symmetric and intensive. If this is not the case, please use a "
        "different model."
    )
    return msg1, msg2

def test_basic_usage():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    msg1, msg2 = get_warnings()
    with pytest.warns(UserWarning, match=f"({msg1}|{msg2})"):
        atoms.calc = PETMADCalculator(version="latest")

    dyn = Langevin(atoms, 0.5 * ase.units.fs, temperature_K=310, friction=5e-3)
    dyn.run(10)
