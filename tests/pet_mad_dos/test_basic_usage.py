from pet_mad.calculator import PETMADDOSCalculator
from pet_mad._models import get_pet_mad_dos
from ase.build import bulk
import pytest
import torch

VERSIONS = ["1.0"]

def get_atoms():
    atoms_1 = bulk("C", cubic=True, a=3.55, crystalstructure="diamond")
    atoms_2 = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    return [atoms_1, atoms_2]

@pytest.mark.parametrize(
    "version",
    VERSIONS,
)
def test_get_pet_mad_dos(version):
    model = get_pet_mad_dos(version=version)
    assert model.metadata().name == f"PET-MAD-DOS v{version}"


def test_dos_calculation():
    calc = PETMADDOSCalculator()
    atoms = get_atoms()
    _ = calc.calculate_dos(atoms)

def test_efermi_calculation():
    calc = PETMADDOSCalculator()
    atoms = get_atoms()
    efermi = calc.calculate_efermi(atoms)
    target_efermi = torch.tensor([-10.7456,  -9.3956])

    torch.testing.assert_close(efermi, target_efermi, atol=1e-3, rtol=1e-3)


def test_bandgap_calculation():
    calc = PETMADDOSCalculator()
    atoms = get_atoms()
    bandgap = calc.calculate_bandgap(atoms)
    target_bandgap = torch.tensor([4.1198, 0.9741])

    torch.testing.assert_close(bandgap, target_bandgap, atol=1e-3, rtol=1e-3)
