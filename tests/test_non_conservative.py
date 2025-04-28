from pet_mad.calculator import PETMADCalculator
from ase.build import bulk
import pytest


def test_non_conservative():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.calc = PETMADCalculator(version="latest", non_conservative=True)
    _ = atoms.get_potential_energy()
    _ = atoms.get_forces()