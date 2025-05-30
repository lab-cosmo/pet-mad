from pet_mad.calculator import PETMADCalculator
from ase.build import bulk
import pytest


def test_basic_usage():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.calc = PETMADCalculator(version="latest")
    _ = atoms.get_potential_energy()
    _ = atoms.get_forces()


@pytest.mark.parametrize(
    "version",
    [
        "latest",
        "1.1.0",
        "1.0.1",
    ],
)
def test_version(version):
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.calc = PETMADCalculator(version=version)
    _ = atoms.get_potential_energy()
    _ = atoms.get_forces()


@pytest.mark.parametrize(
    "version",
    [
        "1.0.0",
    ],
)
def test_version_deprecated(version):
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.calc = PETMADCalculator(version=version)
    _ = atoms.get_potential_energy()
    _ = atoms.get_forces()
