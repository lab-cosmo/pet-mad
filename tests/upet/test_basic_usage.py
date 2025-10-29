import pytest
from ase.build import bulk, molecule
from huggingface_hub import HfApi
from packaging.version import Version

from upet._models import get_upet, upet_get_size_to_load, upet_get_version_to_load
from upet._version import UPET_AVAILABLE_MODELS
from upet.calculator import UPETCalculator


@pytest.mark.parametrize("size", ["s", "m", "l", "xl", "xs"])
def test_upet_get_size_to_load(size):
    model = "pet-omat"
    if size in ["l", "m", "s", "xs"]:
        returned_size = upet_get_size_to_load(model, requested_size=size)
        assert returned_size == size
    else:
        with pytest.raises(
            ValueError, match=f"Requested size {size} not available for model {model}"
        ):
            upet_get_size_to_load(model, requested_size=size)


@pytest.mark.parametrize("version", ["0.0.0", "0.1.0", "1.0.0"])
def test_upet_get_version_to_load(version):
    model = "pet-omat"
    size = "l"
    if version in ["0.1.0", "1.0.0"]:
        returned_version = upet_get_version_to_load(
            model, size, requested_version=version
        )
        assert str(returned_version) == version
    else:
        with pytest.raises(
            ValueError,
            match=(
                f"Requested version {version} not available "
                f"for model {model} size {size}."
            ),
        ):
            upet_get_version_to_load(model, size, requested_version=version)


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_get_upet(model_name):
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
        model, size = model_name.rsplit("-", 1)
        get_upet(model=model, size=size, version=version)


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_basic_usage(model_name):
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
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        virial = atoms.get_stress()
        assert isinstance(energy, float)
        assert forces.shape == (len(atoms), 3)
        assert virial.shape == (6,)
