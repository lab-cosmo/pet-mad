from pet_mad.calculator import PETMADCalculator
from pet_mad._models import get_pet_mad, save_pet_mad
from ase.build import bulk
import os
import pytest

VERSIONS = ("1.1.0", "1.0.1")

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

@pytest.mark.parametrize(
    "version",
    VERSIONS,
)
def test_get_pet_mad(version):
    msg1, msg2 = get_warnings()
    with pytest.warns(UserWarning, match=f"({msg1}|{msg2})"):
        model = get_pet_mad(version=version)
    assert model.metadata().name == f"PET-MAD v{version}"


@pytest.mark.parametrize(
    "version",
    VERSIONS,
)
def test_save_pet_mad(version, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    msg1, msg2 = get_warnings()
    with pytest.warns(UserWarning, match=f"({msg1}|{msg2})"):
        save_pet_mad(version=version, output=f"pet-mad-{version}.pt")
    assert os.path.exists(f"pet-mad-{version}.pt")

def test_basic_usage():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    msg1, msg2 = get_warnings()
    with pytest.warns(UserWarning, match=f"({msg1}|{msg2})"):
        atoms.calc = PETMADCalculator(version="latest")
    _ = atoms.get_potential_energy()
    _ = atoms.get_forces()


@pytest.mark.parametrize(
    "version",
    VERSIONS,
)
def test_version(version):
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    msg1, msg2 = get_warnings()
    with pytest.warns(UserWarning, match=f"({msg1}|{msg2})"):
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
    msg1, msg2 = get_warnings()
    with pytest.warns(UserWarning, match=f"({msg1}|{msg2})"):
        atoms.calc = PETMADCalculator(version=version)
    _ = atoms.get_potential_energy()
    _ = atoms.get_forces()
