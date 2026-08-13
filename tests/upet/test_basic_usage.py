import os

import pytest
from ase.build import bulk, molecule

from upet._models import (
    get_available_models,
    get_sizes_for_model,
    get_upet,
    get_versions_for_model,
    list_upet,
    save_upet,
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


def test_available_models_are_published():
    """Check that every model offered here can be downloaded.

    Models published on the hub but missing from the list are not an error:
    a model can be uploaded before it is offered here.
    """
    published = {
        f"{model}-{size}"
        for model in get_available_models()
        for size in get_sizes_for_model(model)
    }
    missing = sorted(set(UPET_AVAILABLE_MODELS) - published)

    assert missing == [], f"not published on the hub: {missing}"


def test_list_models():
    result = list_upet(print_summary=False)
    assert len(result) > 0
    assert all(
        "model" in entry and "size" in entry and "version" in entry for entry in result
    )
    assert any(entry["model"] == "pet-mad" for entry in result)


def test_list_sizes_for_model():
    result = list_upet(model="pet-mad", print_summary=False)
    assert len(result) > 0
    assert all(entry["model"] == "pet-mad" for entry in result)


def test_list_versions_for_model_and_size():
    result = list_upet(model="pet-mad", size="s", print_summary=False)
    assert len(result) > 0
    assert all(entry["model"] == "pet-mad" and entry["size"] == "s" for entry in result)


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_basic_usage(model_name):
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
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        virial = atoms.get_stress()
        assert isinstance(energy, float)
        assert forces.shape == (len(atoms), 3)
        assert virial.shape == (6,)


def test_save_upet(tmp_path):
    output_path = str(tmp_path / "pet-mad-xs.pt")
    save_upet(model="pet-mad", size="xs", output=output_path)
    assert os.path.isfile(output_path)
    assert os.path.getsize(output_path) > 0
