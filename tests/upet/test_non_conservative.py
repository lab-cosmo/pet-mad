import re

import pytest
from ase.build import bulk, molecule

from upet._models import get_versions_for_model
from upet._version import UPET_AVAILABLE_MODELS, UPET_NO_NC_SUPPORT_MODELS
from upet.calculator import UPETCalculator


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_non_conservative(model_name):
    non_conservative = "forces"

    if "-xl" in model_name or "-l" in model_name:
        pytest.skip("Skipping XL models and L models due to large size.")
    atoms = (
        molecule("H2O")
        if any(name in model_name for name in ("spice", "mols"))
        else bulk("C", cubic=True, a=5.43, crystalstructure="diamond")
    )

    model, size = model_name.rsplit("-", 1)
    all_model_versions = get_versions_for_model(model, size)

    for version in all_model_versions:
        if f"{model_name}-v{version}" in UPET_NO_NC_SUPPORT_MODELS:
            message = (
                f"`non-conservative={non_conservative}` option is not available "
                f"for the model {model_name}, v{version}, and a target variant "
                f"`energy`. Please choose another `non-conservative` option, "
                "use another target variant, switch to a conservative regime "
                "or choose another model."
            )
            with pytest.raises(NotImplementedError, match=re.escape(message)):
                calc = UPETCalculator(
                    model=model_name, version=version, non_conservative=non_conservative
                )
        else:
            calc = UPETCalculator(
                model=model_name, version=version, non_conservative=non_conservative
            )
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            virial = atoms.get_stress()
            assert isinstance(energy, float)
            assert forces.shape == (len(atoms), 3)
            assert virial.shape == (6,)
