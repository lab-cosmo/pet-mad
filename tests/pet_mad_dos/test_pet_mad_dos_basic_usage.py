import pytest
import torch
from ase.build import bulk

from upet._models import get_pet_mad_dos
from upet.calculator import PETMADDOSCalculator


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


@pytest.mark.parametrize(
    "per_atom",
    [True, False],
)
def test_dos_calculation(per_atom):
    calc = PETMADDOSCalculator()
    atoms = get_atoms()
    _, dos = calc._calculate_dos(atoms, per_atom=per_atom)
    if per_atom:
        assert dos.shape[0] == sum([len(item) for item in atoms])
    else:
        assert dos.shape[0] == len(atoms)


def test_dos_calculation_single_item():
    calc = PETMADDOSCalculator()
    atoms = get_atoms()[0]
    _, dos = calc._calculate_dos(atoms, per_atom=False)
    assert dos.shape[0] == 1


def test_efermi_calculation(with_dos):
    calc = PETMADDOSCalculator()
    atoms = get_atoms()
    target_efermi = torch.tensor([-10.7456, -9.3956])
    _, dos = calc._calculate_dos(atoms, per_atom=False)
    efermi = calc._calculate_efermi(atoms, dos=dos)

    torch.testing.assert_close(efermi, target_efermi, atol=1e-3, rtol=1e-3)


def test_efermi_model_prediction():
    calc = PETMADDOSCalculator()
    atoms = get_atoms()
    target_efermi = torch.tensor([-7.3819, -8.7343])
    efermi = calc._calculate_efermi(atoms, model=True)

    torch.testing.assert_close(efermi, target_efermi, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize(
    "with_dos",
    [True, False],
)
def test_efermi_calculation_finite_temperature(with_dos):
    calc = PETMADDOSCalculator()
    atoms = get_atoms()
    target_efermi = torch.tensor(-10.7198)
    if with_dos:
        _, dos = calc._calculate_dos(atoms, per_atom=False)
        efermi = calc._calculate_efermi(atoms, dos=dos)

    torch.testing.assert_close(efermi, target_efermi, atol=1e-3, rtol=1e-3)


def test_bandgap_calculation():
    calc = PETMADDOSCalculator()
    atoms = get_atoms()
    bandgap = calc._calculate_bandgap(atoms)
    target_bandgap = torch.tensor([4.1007, 0.5021])

    torch.testing.assert_close(bandgap, target_bandgap, atol=1e-3, rtol=1e-3)

def test_error_wrong_dos_shape():
    calc = PETMADDOSCalculator()
    atoms = get_atoms()
    _, dos = calc._calculate_dos(atoms, per_atom=False)
    dos = dos[:-1]
    with pytest.raises(ValueError, match="The provided DOS is inconsistent"):
        calc._calculate_efermi(atoms, dos=dos)
