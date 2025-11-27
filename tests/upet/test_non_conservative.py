import pytest
from ase.build import bulk, molecule
from huggingface_hub import HfApi
from packaging.version import Version

from upet._version import UPET_AVAILABLE_MODELS, UPET_NO_NC_SUPPORT_MODELS
from upet.calculator import UPETCalculator


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_non_conservative(model_name):
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
