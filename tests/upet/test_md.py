import ase.units
import pytest
from ase.build import bulk, molecule
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from huggingface_hub import HfApi
from packaging.version import Version

from pet_mad._version import UPET_AVAILABLE_MODELS
from pet_mad.calculator import UPETCalculator


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_md(model_name):
    atoms = (
        bulk("C", cubic=True, a=5.43, crystalstructure="diamond")
        if "spice" not in model_name
        else molecule("H2O")
    )

    hf_api = HfApi()
    repo_files = hf_api.list_repo_files("lab-cosmo/upet")
    files_in_models_folder = [f[7:] for f in repo_files if f.startswith("models/")]
    model, size = model_name.rsplit("-", 1)
    all_model_files = [
        f
        for f in files_in_models_folder
        if f.startswith(f"{model}-{size}-") and f.endswith(".ckpt")
    ]
    all_model_versions = [
        Version(f.split(f"{model}-{size}-")[1].split(".ckpt")[0])
        for f in all_model_files
    ]
    all_model_versions = sorted(set(all_model_versions))

    for version in all_model_versions:
        calc = UPETCalculator(model=model_name, version=version)
        atoms.calc = calc
        MaxwellBoltzmannDistribution(atoms, temperature_K=300)
        dyn = VelocityVerlet(atoms, 0.5 * ase.units.fs)
        dyn.run(10)
