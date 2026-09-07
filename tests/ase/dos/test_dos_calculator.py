import pytest
import torch
from ase.build import bulk

from upet.ase.dos import PETMADDOSCalculator, get_pet_mad_dos


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
    results = calc.calculate(
        atoms, properties=["dos_raw", "dos_raw_per_atom", "dos_denoised"]
    )
    dos = results["dos_raw"]
    dos_per_atom = results["dos_raw_per_atom"]
    dos_denoised = results["dos_denoised"]
    assert dos_per_atom.shape[0] == sum([len(item) for item in atoms])
    assert dos.shape[0] == len(atoms)
    assert dos_denoised.shape[0] == len(atoms)


def test_dos_calculation_single_item():
    calc = PETMADDOSCalculator()
    atoms = get_atoms()[0]
    results = calc.calculate(atoms, properties=["dos_raw", "dos_denoised"])
    dos = results["dos_raw"]
    denoised_dos = results["dos_denoised"]
    assert dos.shape[0] == 4806
    assert denoised_dos.shape[0] == 4806


def test_efermi_model_prediction():
    calc = PETMADDOSCalculator()
    atoms = get_atoms()
    target_efermi = torch.tensor([152.2637, 150.9113])
    efermi = calc.calculate(atoms, properties=["fermi_level"])["fermi_level"]
    torch.testing.assert_close(efermi, target_efermi, atol=1e-3, rtol=1e-3)


def test_bandgap_calculation():
    calc = PETMADDOSCalculator()
    atoms = get_atoms()
    bandgap = calc.calculate(atoms, properties=["bandgap"])["bandgap"]
    target_bandgap = torch.tensor([4.1007, 0.5021])

    torch.testing.assert_close(bandgap, target_bandgap, atol=1e-3, rtol=1e-3)
