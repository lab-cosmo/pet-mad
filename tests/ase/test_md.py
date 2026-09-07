import ase.units
import pytest
from ase.build import bulk, molecule
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet

from upet._models import get_versions_for_model
from upet._version import UPET_AVAILABLE_MODELS
from upet.ase import UPETCalculator


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
@pytest.mark.filterwarnings(  # see https://github.com/metatensor/metatomic/issues/139
    "ignore:.*invalid value encountered in scalar add.*:RuntimeWarning"
)
def test_md(model_name):
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
        calc = UPETCalculator(model=model_name, version=version)
        atoms.calc = calc
        MaxwellBoltzmannDistribution(atoms, temperature_K=300)
        dyn = VelocityVerlet(atoms, 0.5 * ase.units.fs)
        dyn.run(10)
