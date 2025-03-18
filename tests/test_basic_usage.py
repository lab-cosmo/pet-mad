from pet_mad.calculator import PETMADCalculator
from ase.build import bulk


def test_basic_usage():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.calc = PETMADCalculator(version="latest")
    _ = atoms.get_potential_energy()
    _ = atoms.get_forces()
