import ase.units
import pytest
from ase.build import bulk, molecule
from ase.md.langevin import Langevin
from huggingface_hub import HfApi

from pet_mad.calculator import PETMLIPCalculator


UPET_MODELS = ["pet-mad", "pet-omad", "pet-oam", "pet-omat", "pet-omatpes", "pet-spice"]


@pytest.mark.parametrize("model", UPET_MODELS)
def test_basic_usage(model):
    atoms = (
        bulk("C", cubic=True, a=5.43, crystalstructure="diamond")
        if model != "pet-spice"
        else molecule("H2O")
    )

    hf_api = HfApi()
    repo_files = hf_api.list_repo_files("lab-cosmo/upet")
    files_in_models_folder = [f[7:] for f in repo_files if f.startswith("models/")]
    all_model_files = [
        f
        for f in files_in_models_folder
        if f.startswith(f"{model}-") and f.endswith(".ckpt")
    ]

    for model_file in all_model_files:
        split = model_file.split("-")
        size, version = (
            split[2],
            split[3][1:].replace(".ckpt", ""),
        )  # remove the "v" prefix and ".ckpt" suffix
        calc = PETMLIPCalculator(model=model, size=size, version=version)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        virial = atoms.get_stress()
        assert isinstance(energy, float)
        assert forces.shape == (len(atoms), 3)
        assert virial.shape == (6,)


def test_md():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    atoms.calc = PETMLIPCalculator(model="pet-mad", size="s", version="1.0.2")
    dyn = Langevin(atoms, 0.5 * ase.units.fs, temperature_K=310, friction=5e-3)
    dyn.run(10)
