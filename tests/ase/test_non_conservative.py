import re

import pytest
from _utils import non_conservative_error_message, supports_non_conservative
from ase.build import bulk, molecule

from upet._models import get_versions_for_model
from upet._version import UPET_AVAILABLE_MODELS
from upet.ase import UPETCalculator


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
        if not supports_non_conservative(model_name, version):
            message = non_conservative_error_message(
                model_name, version, non_conservative
            )
            with pytest.raises(NotImplementedError, match=re.escape(message)):
                UPETCalculator(
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
