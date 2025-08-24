from pet_mad.calculator import PETMADCalculator
from ase.build import bulk
from ase.md.langevin import Langevin
import ase.units

def test_basic_usage():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.calc = PETMADCalculator(version="latest")
    dyn = Langevin(atoms, 0.5 * ase.units.fs, temperature_K=310, friction=5e-3)
    dyn.run(10)
