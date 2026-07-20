import copy

import pytest
import torch
from ase.build import molecule
from metatomic.torch import ModelOutput, System
from metatrain.pet import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from upet._models import _load_model_with_custom_heads
from upet._weighted_sum import (
    ARCHITECTURE_NAME,
    WeightedSumModel,
    collect_specs,
    create_weighted_sum_checkpoint,
    extract_wrapped_checkpoint,
    load_specs_yaml,
    parse_inline_head,
)


def _minimal_hypers():
    hypers = copy.deepcopy(get_default_hypers("pet")["model"])
    for key in [
        "d_pet",
        "d_head",
        "d_node",
        "d_feedforward",
        "num_heads",
        "num_attention_layers",
        "num_gnn_layers",
    ]:
        hypers[key] = 1
    return hypers


def _dataset_info(units=("eV", "eV")):
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "energy/a": get_energy_target_info(
                "energy/a", {"quantity": "energy", "unit": units[0]}
            ),
            "energy/b": get_energy_target_info(
                "energy/b", {"quantity": "energy", "unit": units[1]}
            ),
        },
    )


@pytest.fixture(scope="module")
def base_model():
    model = PET(_minimal_hypers(), _dataset_info())
    model.eval()
    return model


@pytest.fixture(scope="module")
def water_system(base_model):
    atoms = molecule("H2O")
    system = System(
        types=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32),
        positions=torch.tensor(atoms.get_positions(), dtype=torch.float32),
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    return get_system_with_neighbor_lists(system, base_model.requested_neighbor_lists())


def test_load_specs_yaml(tmp_path):
    path = tmp_path / "heads.yaml"
    path.write_text("heads:\n  energy/mix:\n    energy/a: 0.3\n    energy/b: 0.7\n")
    assert load_specs_yaml(str(path)) == {
        "energy/mix": {"energy/a": 0.3, "energy/b": 0.7}
    }


def test_load_specs_yaml_legacy_single_head(tmp_path):
    path = tmp_path / "heads.yaml"
    path.write_text("name: energy/mix\nspec:\n  energy/a: 0.3\n  energy/b: 0.7\n")
    assert load_specs_yaml(str(path)) == {
        "energy/mix": {"energy/a": 0.3, "energy/b": 0.7}
    }


def test_load_specs_yaml_missing_heads_section(tmp_path):
    path = tmp_path / "heads.yaml"
    path.write_text("not_heads:\n  foo: 1\n")
    with pytest.raises(ValueError, match="has no 'heads' section"):
        load_specs_yaml(str(path))


def test_parse_inline_head():
    name, spec = parse_inline_head("energy/mix = energy/a:0.3, energy/b:0.7")
    assert name == "energy/mix"
    assert spec == {"energy/a": 0.3, "energy/b": 0.7}


def test_parse_inline_head_malformed():
    with pytest.raises(ValueError, match="neither a head name"):
        parse_inline_head("energy/mix")


def test_collect_specs_from_yaml(tmp_path):
    path = tmp_path / "heads.yaml"
    path.write_text("heads:\n  energy/mix:\n    energy/a: 0.3\n    energy/b: 0.7\n")
    assert collect_specs(str(path), None) == {
        "energy/mix": {"energy/a": 0.3, "energy/b": 0.7}
    }


def test_collect_specs_inline_only():
    specs = collect_specs(None, ["energy/mix = energy/a:0.5, energy/b:0.5"])
    assert specs == {"energy/mix": {"energy/a": 0.5, "energy/b": 0.5}}


def test_collect_specs_nothing_to_do():
    with pytest.raises(ValueError, match="nothing to do"):
        collect_specs(None, None)


def test_weighted_sum_rejects_empty_specs(base_model):
    with pytest.raises(ValueError, match="no weighted-sum heads requested"):
        WeightedSumModel(base_model, {})


def test_weighted_sum_rejects_head_with_no_sources(base_model):
    with pytest.raises(ValueError, match="'x' has no sources"):
        WeightedSumModel(base_model, {"x": {}})


def test_weighted_sum_rejects_existing_output_name(base_model):
    with pytest.raises(ValueError, match="already exists as a model output"):
        WeightedSumModel(base_model, {"energy/a": {"energy/b": 1.0}})


def test_weighted_sum_rejects_unknown_source(base_model):
    with pytest.raises(ValueError, match="unknown source target 'energy/nope'"):
        WeightedSumModel(base_model, {"energy/mix": {"energy/nope": 1.0}})


def test_weighted_sum_rejects_mismatched_units():
    model = PET(_minimal_hypers(), _dataset_info(units=("eV", "kcal/mol")))
    with pytest.raises(ValueError, match="incompatible with"):
        WeightedSumModel(model, {"energy/mix": {"energy/a": 0.3, "energy/b": 0.7}})


def test_weighted_sum_forward_matches_manual_combination(base_model, water_system):
    wrapped = WeightedSumModel(
        base_model, {"energy/mix": {"energy/a": 0.3, "energy/b": 0.7}}
    )
    assert "energy/mix" in wrapped.supported_outputs()

    with torch.no_grad():
        separate = base_model(
            [water_system],
            {
                "energy/a": ModelOutput(sample_kind="system"),
                "energy/b": ModelOutput(sample_kind="system"),
            },
        )
        combined = wrapped(
            [water_system], {"energy/mix": ModelOutput(sample_kind="system")}
        )

    a = separate["energy/a"].block().values.squeeze()
    b = separate["energy/b"].block().values.squeeze()
    mix = combined["energy/mix"].block().values.squeeze()
    assert torch.allclose(mix, 0.3 * a + 0.7 * b, atol=1e-6)
    assert set(combined.keys()) == {"energy/mix"}


def test_weighted_sum_checkpoint_round_trip(base_model, water_system):
    specs = {"energy/mix": {"energy/a": 0.3, "energy/b": 0.7}}
    wrapped = WeightedSumModel(base_model, specs)

    reloaded = WeightedSumModel.load_checkpoint(wrapped.get_checkpoint())
    assert reloaded.new_names == wrapped.new_names
    assert reloaded.sources == wrapped.sources
    assert reloaded.coefficients == wrapped.coefficients

    with torch.no_grad():
        original = wrapped(
            [water_system], {"energy/mix": ModelOutput(sample_kind="system")}
        )
        after_reload = reloaded(
            [water_system], {"energy/mix": ModelOutput(sample_kind="system")}
        )
    assert torch.allclose(
        original["energy/mix"].block().values, after_reload["energy/mix"].block().values
    )


def test_weighted_sum_export_produces_atomistic_model(base_model):
    wrapped = WeightedSumModel(
        base_model, {"energy/mix": {"energy/a": 0.3, "energy/b": 0.7}}
    )
    exported = wrapped.export()
    assert "energy/mix" in exported.capabilities().outputs


def test_create_and_extract_weighted_sum_checkpoint(base_model, tmp_path):
    base_ckpt = tmp_path / "base.ckpt"
    torch.save(base_model.get_checkpoint(), base_ckpt)

    wsum_ckpt = tmp_path / "wsum.ckpt"
    create_weighted_sum_checkpoint(
        str(base_ckpt),
        {"energy/mix": {"energy/a": 0.3, "energy/b": 0.7}},
        str(wsum_ckpt),
    )

    raw = torch.load(wsum_ckpt, map_location="cpu", weights_only=False)
    assert raw["architecture_name"] == ARCHITECTURE_NAME
    assert raw["weighted_sum_heads"] == {
        "energy/mix": {"energy/a": 0.3, "energy/b": 0.7}
    }

    loaded = _load_model_with_custom_heads(str(wsum_ckpt))
    assert isinstance(loaded, WeightedSumModel)
    assert "energy/mix" in loaded.supported_outputs()

    extracted_ckpt = tmp_path / "extracted.ckpt"
    extract_wrapped_checkpoint(str(wsum_ckpt), str(extracted_ckpt))
    extracted = torch.load(extracted_ckpt, map_location="cpu", weights_only=False)
    assert extracted["architecture_name"] == "pet"


def test_extract_wrapped_checkpoint_rejects_plain_checkpoint(base_model, tmp_path):
    base_ckpt = tmp_path / "base.ckpt"
    torch.save(base_model.get_checkpoint(), base_ckpt)
    with pytest.raises(ValueError, match="not a weighted-sum checkpoint"):
        extract_wrapped_checkpoint(str(base_ckpt), str(tmp_path / "out.ckpt"))


def test_load_model_with_custom_heads_passes_through_plain_checkpoint(
    base_model, tmp_path
):
    base_ckpt = tmp_path / "base.ckpt"
    torch.save(base_model.get_checkpoint(), base_ckpt)
    loaded = _load_model_with_custom_heads(str(base_ckpt))
    assert isinstance(loaded, PET)
