import os

import pytest
from ase.build import bulk

from pet_mad._models import get_pet_mad, save_pet_mad
from pet_mad.calculator import PETMADCalculator


VERSIONS = ("1.1.0", "1.0.2", "1.0.1")


@pytest.mark.parametrize(
    "version",
    VERSIONS,
)
def test_get_pet_mad(version):
    model = get_pet_mad(version=version)
    assert model.metadata().name == f"PET-MAD v{version}"


@pytest.mark.parametrize(
    "version",
    VERSIONS,
)
def test_save_pet_mad(version, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    save_pet_mad(version=version, output=f"pet-mad-{version}.pt")
    assert os.path.exists(f"pet-mad-{version}.pt")


def test_basic_usage():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.calc = PETMADCalculator(version="latest")
    _ = atoms.get_potential_energy()
    _ = atoms.get_forces()


@pytest.mark.parametrize(
    "version",
    VERSIONS,
)
def test_version(version):
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.calc = PETMADCalculator(version=version)
    _ = atoms.get_potential_energy()
    _ = atoms.get_forces()
