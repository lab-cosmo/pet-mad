import pytest
from ase.build import bulk, molecule

from upet._models import get_versions_for_model
from upet._version import UPET_AVAILABLE_MODELS, UPET_NO_NC_SUPPORT_MODELS
from upet.ase import UPETCalculator


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_non_conservative(model_name):
    if "-xl" in model_name or "-l" in model_name:
        pytest.skip("Skipping XL models and L models due to large size.")
    atoms = (
        bulk("C", cubic=True, a=5.43, crystalstructure="diamond")
        if "spice" not in model_name
        else molecule("H2O")
    )

    model, size = model_name.rsplit("-", 1)
    all_model_versions = get_versions_for_model(model, size)

    for version in all_model_versions:
        if f"{model_name}-v{version}" in UPET_NO_NC_SUPPORT_MODELS:
            with pytest.raises(
                NotImplementedError,
                match="Non-conservative forces and stresses are not available",
            ):
                calc = UPETCalculator(
                    model=model_name, version=version, non_conservative=True
                )
        else:
            calc = UPETCalculator(
                model=model_name, version=version, non_conservative=True
            )
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            virial = atoms.get_stress()
            assert isinstance(energy, float)
            assert forces.shape == (len(atoms), 3)
            assert virial.shape == (6,)
