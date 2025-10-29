import numpy as np
import pytest
from ase.build import bulk, molecule
from huggingface_hub import HfApi
from packaging.version import Version

from upet._version import UPET_AVAILABLE_MODELS, UPET_UQ_SUPPORTED_MODELS
from upet.calculator import UPETCalculator


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_uncertainty_quantification(model_name):
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
        if f"{model_name}-v{version}" in UPET_UQ_SUPPORTED_MODELS:
            calc = UPETCalculator(
                model=model_name,
                version=version,
                calculate_uncertainty=True,
                calculate_ensemble=True,
            )
            atoms.calc = calc
            energy = atoms.get_potential_energy()
            energy_uncertainty = atoms.calc.get_energy_uncertainty()
            energy_ensemble = atoms.calc.get_energy_ensemble()

            assert np.allclose(np.mean(energy_ensemble), energy, atol=1e-1)
            assert np.allclose(energy_uncertainty, np.std(energy_ensemble), atol=1e-1)
        else:
            with pytest.raises(
                NotImplementedError,
                match="Energy uncertainty and ensemble are not available",
            ):
                calc = UPETCalculator(
                    model=model_name,
                    version=version,
                    calculate_uncertainty=True,
                    calculate_ensemble=True,
                )
                atoms.calc = calc
                atoms.get_potential_energy()


def test_error_no_uq_requested():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    calc = UPETCalculator(
        model="pet-mad-s",
        version="1.0.2",
        calculate_uncertainty=False,
        calculate_ensemble=False,
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    with pytest.raises(ValueError, match="Energy uncertainty is not available"):
        calc.get_energy_uncertainty()
    with pytest.raises(ValueError, match="Energy ensemble is not available"):
        calc.get_energy_ensemble()


def test_error_model_not_evaluated():
    atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
    calc = UPETCalculator(
        model="pet-mad-s",
        version="1.0.2",
        calculate_uncertainty=True,
        calculate_ensemble=True,
    )
    atoms.calc = calc
    with pytest.raises(ValueError, match="Energy uncertainty is not available"):
        calc.get_energy_uncertainty()
    with pytest.raises(ValueError, match="Energy ensemble is not available"):
        calc.get_energy_ensemble()
