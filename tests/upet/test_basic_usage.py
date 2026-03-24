import pytest
from ase.build import bulk, molecule

from upet._models import (
    get_upet,
    get_versions_for_model,
    list_available_models,
    upet_resolve_model,
)
from upet._version import UPET_AVAILABLE_MODELS
from upet.calculator import UPETCalculator


@pytest.mark.parametrize("size", ["s", "m", "l", "xl", "xs", "xxs"])
def test_upet_resolve_model_size(size):
    model = "pet-omat"
    if size in ["l", "m", "s", "xs", "xl"]:
        returned_size, _ = upet_resolve_model(model, requested_size=size)
        assert returned_size == size
    else:
        with pytest.raises(
            ValueError, match=f"Requested size {size} not available for model {model}"
        ):
            upet_resolve_model(model, requested_size=size)


@pytest.mark.parametrize("version", ["0.0.0", "0.1.0", "0.2.0", "1.0.0"])
def test_upet_resolve_model_version(version):
    model = "pet-omat"
    size = "l"
    if version in ["0.1.0", "0.2.0", "1.0.0"]:
        _, returned_version = upet_resolve_model(
            model, requested_size=size, requested_version=version
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
            upet_resolve_model(model, requested_size=size, requested_version=version)


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_get_upet(model_name):
    if "-xl" in model_name or "-l" in model_name:
        pytest.skip("Skipping XL models and L models due to large size.")
    model, size = model_name.rsplit("-", 1)
    all_model_versions = get_versions_for_model(model, size)

    for version in all_model_versions:
        get_upet(model=model, size=size, version=version)


@pytest.mark.parametrize("model", ["pet-mad"])
@pytest.mark.parametrize("size", ["xs", "s", "m"])
def test_list_available_models(model: str, size: str):
    available_models = list_available_models(model=model, size=size)
    for m in available_models:
        assert m.startswith(f"{model}-{size}")


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_basic_usage(model_name):
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
        calc = UPETCalculator(model=model_name, version=version)
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        virial = atoms.get_stress()
        assert isinstance(energy, float)
        assert forces.shape == (len(atoms), 3)
        assert virial.shape == (6,)
