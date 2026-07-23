import copy
import sys

import pytest
import torch
from ase.build import molecule
from metatensor.torch import Labels
from metatomic.torch import ModelEvaluationOptions, ModelOutput, System
from metatrain.pet import PET
from metatrain.utils.architectures import get_default_hypers
from metatrain.utils.data import DatasetInfo
from metatrain.utils.data.target_info import (
    get_energy_target_info,
    get_generic_target_info,
)
from metatrain.utils.neighbor_lists import get_system_with_neighbor_lists

from upet._models import _load_model_with_custom_heads
from upet._weighted_sum import (
    ARCHITECTURE_NAME,
    WeightedSumHead,
    WeightedSumModel,
    _main,
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


def _water_system(model, positions=None):
    atoms = molecule("H2O")
    if positions is None:
        positions = torch.tensor(atoms.get_positions(), dtype=torch.float32)
    system = System(
        types=torch.tensor(atoms.get_atomic_numbers(), dtype=torch.int32),
        positions=positions,
        cell=torch.zeros(3, 3),
        pbc=torch.tensor([False, False, False]),
    )
    return get_system_with_neighbor_lists(system, model.requested_neighbor_lists())


@pytest.fixture(scope="module")
def water_system(base_model):
    return _water_system(base_model)


def _nc_force_dataset_info():
    target = {
        "quantity": "force",
        "unit": "eV/A",
        "sample_kind": "atom",
        "type": {"cartesian": {"rank": 1}},
        "num_subtargets": 1,
    }
    return DatasetInfo(
        length_unit="Angstrom",
        atomic_types=[1, 6, 7, 8],
        targets={
            "non_conservative_force/a": get_generic_target_info(
                "non_conservative_force/a", target
            ),
            "non_conservative_force/b": get_generic_target_info(
                "non_conservative_force/b", target
            ),
        },
    )


@pytest.fixture(scope="module")
def nc_force_model():
    model = PET(_minimal_hypers(), _nc_force_dataset_info())
    model.eval()
    return model


@pytest.fixture(scope="module")
def nc_force_water_system(nc_force_model):
    return _water_system(nc_force_model)


def test_load_specs_yaml(tmp_path):
    path = tmp_path / "heads.yaml"
    path.write_text(
        "heads:\n"
        "  energy/mix:\n"
        "    sources:\n"
        "      energy/a: 0.3\n"
        "      energy/b: 0.7\n"
    )
    assert load_specs_yaml(str(path)) == {
        "energy/mix": WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})
    }


def test_load_specs_yaml_with_normalize_coefficients(tmp_path):
    path = tmp_path / "heads.yaml"
    path.write_text(
        "heads:\n"
        "  energy/mix:\n"
        "    sources:\n"
        "      energy/a: 1\n"
        "      energy/b: 3\n"
        "    normalize_coefficients: true\n"
    )
    assert load_specs_yaml(str(path)) == {
        "energy/mix": WeightedSumHead(
            sources={"energy/a": 1, "energy/b": 3}, normalize_coefficients=True
        )
    }


def test_load_specs_yaml_legacy_single_head(tmp_path):
    path = tmp_path / "heads.yaml"
    path.write_text(
        "name: energy/mix\nspec:\n  sources:\n    energy/a: 0.3\n    energy/b: 0.7\n"
    )
    assert load_specs_yaml(str(path)) == {
        "energy/mix": WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})
    }


def test_load_specs_yaml_missing_heads_section(tmp_path):
    path = tmp_path / "heads.yaml"
    path.write_text("not_heads:\n  foo: 1\n")
    with pytest.raises(ValueError, match="has no 'heads' section"):
        load_specs_yaml(str(path))


def test_parse_inline_head():
    name, spec = parse_inline_head("energy/mix = energy/a:0.3, energy/b:0.7")
    assert name == "energy/mix"
    assert spec == WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})


def test_parse_inline_head_malformed():
    with pytest.raises(ValueError, match="neither a head name"):
        parse_inline_head("energy/mix")


def test_collect_specs_from_yaml(tmp_path):
    path = tmp_path / "heads.yaml"
    path.write_text(
        "heads:\n"
        "  energy/mix:\n"
        "    sources:\n"
        "      energy/a: 0.3\n"
        "      energy/b: 0.7\n"
    )
    assert collect_specs(str(path), None) == {
        "energy/mix": WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})
    }


def test_collect_specs_inline_only():
    specs = collect_specs(None, ["energy/mix = energy/a:0.5, energy/b:0.5"])
    assert specs == {
        "energy/mix": WeightedSumHead(sources={"energy/a": 0.5, "energy/b": 0.5})
    }


def test_collect_specs_nothing_to_do():
    with pytest.raises(ValueError, match="nothing to do"):
        collect_specs(None, None)


def test_weighted_sum_head_requires_sources():
    # `sources` has no default, so a typo'd or forgotten field is a TypeError at
    # construction time -- the whole point of moving off an untyped Dict[str, Any].
    with pytest.raises(TypeError):
        WeightedSumHead()


def test_weighted_sum_head_rejects_unknown_field():
    with pytest.raises(TypeError):
        WeightedSumHead(sources={"energy/a": 0.5, "energy/b": 0.5}, typo_key=True)


def test_weighted_sum_rejects_empty_specs(base_model):
    with pytest.raises(ValueError, match="no weighted-sum heads requested"):
        WeightedSumModel(base_model, {})


def test_weighted_sum_rejects_head_with_no_sources(base_model):
    with pytest.raises(ValueError, match="'x' has no sources"):
        WeightedSumModel(base_model, {"x": WeightedSumHead(sources={})})


def test_weighted_sum_rejects_existing_output_name(base_model):
    with pytest.raises(ValueError, match="already exists as a model output"):
        WeightedSumModel(
            base_model, {"energy/a": WeightedSumHead(sources={"energy/b": 1.0})}
        )


def test_weighted_sum_rejects_unknown_source(base_model):
    with pytest.raises(ValueError, match="unknown source target 'energy/nope'"):
        WeightedSumModel(
            base_model, {"energy/mix": WeightedSumHead(sources={"energy/nope": 1.0})}
        )


def test_weighted_sum_rejects_mismatched_units():
    model = PET(_minimal_hypers(), _dataset_info(units=("eV", "kcal/mol")))
    with pytest.raises(ValueError, match="incompatible with"):
        WeightedSumModel(
            model,
            {"energy/mix": WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})},
        )


def test_weighted_sum_allows_non_summing_coefficients_by_default(base_model):
    wrapped = WeightedSumModel(
        base_model,
        {"energy/diff": WeightedSumHead(sources={"energy/a": 1.0, "energy/b": -1.0})},
    )
    assert wrapped.coefficients == [[1.0, -1.0]]


def test_weighted_sum_normalize_coefficients_rescales_to_sum_one(base_model):
    wrapped = WeightedSumModel(
        base_model,
        {
            "energy/mix": WeightedSumHead(
                sources={"energy/a": 1, "energy/b": 3}, normalize_coefficients=True
            )
        },
    )
    assert wrapped.coefficients == [[0.25, 0.75]]


def test_weighted_sum_normalize_coefficients_rejects_zero_sum(base_model):
    with pytest.raises(ValueError, match="coefficients sum to 0.0, cannot normalize"):
        WeightedSumModel(
            base_model,
            {
                "energy/mix": WeightedSumHead(
                    sources={"energy/a": 1.0, "energy/b": -1.0},
                    normalize_coefficients=True,
                )
            },
        )


def test_weighted_sum_default_description_is_auto_generated(base_model):
    wrapped = WeightedSumModel(
        base_model,
        {"energy/mix": WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})},
    )
    assert wrapped.descriptions == ["0.3 * energy/a + 0.7 * energy/b"]
    assert (
        wrapped.supported_outputs()["energy/mix"].description
        == "0.3 * energy/a + 0.7 * energy/b"
    )


def test_weighted_sum_custom_description_overrides_default(base_model):
    wrapped = WeightedSumModel(
        base_model,
        {
            "energy/mix": WeightedSumHead(
                sources={"energy/a": 0.3, "energy/b": 0.7},
                description="30/70 a/b mix",
            )
        },
    )
    assert wrapped.descriptions == ["30/70 a/b mix"]
    assert wrapped.supported_outputs()["energy/mix"].description == "30/70 a/b mix"


def test_weighted_sum_forward_matches_manual_combination(base_model, water_system):
    wrapped = WeightedSumModel(
        base_model,
        {"energy/mix": WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})},
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


def test_weighted_sum_forward_with_difference_head(base_model, water_system):
    wrapped = WeightedSumModel(
        base_model,
        {"energy/diff": WeightedSumHead(sources={"energy/a": 1.0, "energy/b": -1.0})},
    )

    with torch.no_grad():
        separate = base_model(
            [water_system],
            {
                "energy/a": ModelOutput(sample_kind="system"),
                "energy/b": ModelOutput(sample_kind="system"),
            },
        )
        combined = wrapped(
            [water_system], {"energy/diff": ModelOutput(sample_kind="system")}
        )

    a = separate["energy/a"].block().values.squeeze()
    b = separate["energy/b"].block().values.squeeze()
    diff = combined["energy/diff"].block().values.squeeze()
    assert torch.allclose(diff, a - b, atol=1e-6)


def test_weighted_sum_checkpoint_round_trip(base_model, water_system):
    specs = {
        "energy/mix": WeightedSumHead(
            sources={"energy/a": 1, "energy/b": 3}, normalize_coefficients=True
        ),
        "energy/diff": WeightedSumHead(sources={"energy/a": 1.0, "energy/b": -1.0}),
    }
    wrapped = WeightedSumModel(base_model, specs)
    assert wrapped.coefficients == [[0.25, 0.75], [1.0, -1.0]]

    reloaded = WeightedSumModel.load_checkpoint(wrapped.get_checkpoint())
    assert reloaded.new_names == wrapped.new_names
    assert reloaded.sources == wrapped.sources
    assert reloaded.coefficients == wrapped.coefficients
    assert reloaded.descriptions == wrapped.descriptions

    outputs = {
        "energy/mix": ModelOutput(sample_kind="system"),
        "energy/diff": ModelOutput(sample_kind="system"),
    }
    with torch.no_grad():
        original = wrapped([water_system], outputs)
        after_reload = reloaded([water_system], outputs)
    for name in outputs:
        assert torch.allclose(
            original[name].block().values, after_reload[name].block().values
        )


def test_weighted_sum_export_produces_atomistic_model(base_model):
    wrapped = WeightedSumModel(
        base_model,
        {"energy/mix": WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})},
    )
    exported = wrapped.export()
    assert "energy/mix" in exported.capabilities().outputs


def test_create_and_extract_weighted_sum_checkpoint(base_model, tmp_path):
    base_ckpt = tmp_path / "base.ckpt"
    torch.save(base_model.get_checkpoint(), base_ckpt)

    wsum_ckpt = tmp_path / "wsum.ckpt"
    create_weighted_sum_checkpoint(
        str(base_ckpt),
        {"energy/mix": WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})},
        str(wsum_ckpt),
    )

    raw = torch.load(wsum_ckpt, map_location="cpu", weights_only=False)
    assert raw["architecture_name"] == ARCHITECTURE_NAME
    assert raw["weighted_sum_heads"] == {
        "energy/mix": {
            "sources": {"energy/a": 0.3, "energy/b": 0.7},
            "description": "0.3 * energy/a + 0.7 * energy/b",
        }
    }

    loaded = _load_model_with_custom_heads(str(wsum_ckpt))
    assert isinstance(loaded, WeightedSumModel)
    assert "energy/mix" in loaded.supported_outputs()

    extracted_ckpt = tmp_path / "extracted.ckpt"
    extract_wrapped_checkpoint(str(wsum_ckpt), str(extracted_ckpt))
    extracted = torch.load(extracted_ckpt, map_location="cpu", weights_only=False)
    assert extracted["architecture_name"] == "pet"


def test_main_cli_writes_checkpoint_and_prints_summary(
    base_model, tmp_path, monkeypatch, capsys
):
    # Exercises _main end to end (argv -> collect_specs ->
    # create_weighted_sum_checkpoint -> summary print), which every other test
    # bypasses by calling collect_specs / create_weighted_sum_checkpoint directly.
    base_ckpt = tmp_path / "base.ckpt"
    torch.save(base_model.get_checkpoint(), base_ckpt)

    config_path = tmp_path / "heads.yaml"
    config_path.write_text(
        "heads:\n"
        "  energy/mix:\n"
        "    sources:\n"
        "      energy/a: 0.3\n"
        "      energy/b: 0.7\n"
    )
    output_ckpt = tmp_path / "wsum.ckpt"

    monkeypatch.setattr(
        sys,
        "argv",
        ["upet-weighted-sum", str(base_ckpt), str(config_path), str(output_ckpt)],
    )
    _main()

    captured = capsys.readouterr()
    assert f"wrote {output_ckpt} with 1 weighted-sum head(s):" in captured.out
    assert "energy/mix = 0.3 * energy/a + 0.7 * energy/b" in captured.out

    raw = torch.load(output_ckpt, map_location="cpu", weights_only=False)
    assert raw["architecture_name"] == ARCHITECTURE_NAME
    assert raw["weighted_sum_heads"]["energy/mix"]["sources"] == {
        "energy/a": 0.3,
        "energy/b": 0.7,
    }


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


def test_weighted_sum_forward_preserves_gradients_for_conservative_energy(base_model):
    # The whole point of combining conservative-energy heads (as opposed to
    # combining already-predicted values) is that forces/stress for the combined
    # head come for free from a single backward pass -- see the module docstring
    # and docs/src/fine-tuning.rst. That only holds if `forward` never detaches
    # the sources' values, so exercise it end to end with `requires_grad` positions
    # instead of `torch.no_grad()`, which every other forward test uses.
    wrapped = WeightedSumModel(
        base_model,
        {"energy/mix": WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})},
    )

    positions_separate = torch.tensor(
        molecule("H2O").get_positions(), dtype=torch.float32, requires_grad=True
    )
    separate = base_model(
        [_water_system(base_model, positions_separate)],
        {
            "energy/a": ModelOutput(sample_kind="system"),
            "energy/b": ModelOutput(sample_kind="system"),
        },
    )
    (grad_a,) = torch.autograd.grad(
        separate["energy/a"].block().values.sum(),
        positions_separate,
        retain_graph=True,
    )
    (grad_b,) = torch.autograd.grad(
        separate["energy/b"].block().values.sum(), positions_separate
    )

    positions_mix = torch.tensor(
        molecule("H2O").get_positions(), dtype=torch.float32, requires_grad=True
    )
    combined = wrapped(
        [_water_system(base_model, positions_mix)],
        {"energy/mix": ModelOutput(sample_kind="system")},
    )
    (grad_mix,) = torch.autograd.grad(
        combined["energy/mix"].block().values.sum(), positions_mix
    )

    assert torch.allclose(grad_mix, 0.3 * grad_a + 0.7 * grad_b, atol=1e-5)


def test_weighted_sum_forward_per_atom_with_selected_atoms(base_model, water_system):
    # LAMMPS, under domain decomposition, requests per-atom outputs together
    # with `selected_atoms` -- see the comment in `WeightedSumModel.forward`.
    # Every other forward test uses `sample_kind="system"` and no `selected_atoms`.
    wrapped = WeightedSumModel(
        base_model,
        {"energy/mix": WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})},
    )
    selected_atoms = Labels(
        names=["system", "atom"],
        values=torch.tensor([[0, 0], [0, 2]], dtype=torch.int32),
    )

    with torch.no_grad():
        separate = base_model(
            [water_system],
            {
                "energy/a": ModelOutput(sample_kind="atom"),
                "energy/b": ModelOutput(sample_kind="atom"),
            },
            selected_atoms=selected_atoms,
        )
        combined = wrapped(
            [water_system],
            {"energy/mix": ModelOutput(sample_kind="atom")},
            selected_atoms=selected_atoms,
        )

    a = separate["energy/a"].block().values
    b = separate["energy/b"].block().values
    mix = combined["energy/mix"].block().values
    assert mix.shape[0] == 2  # only the 2 selected atoms, not all 3
    assert torch.allclose(mix, 0.3 * a + 0.7 * b, atol=1e-6)
    assert (
        combined["energy/mix"].block().samples == separate["energy/a"].block().samples
    )


def test_weighted_sum_combines_non_conservative_force_sources(
    nc_force_model, nc_force_water_system
):
    # Only energy-like (conservative) sources are covered elsewhere; the module
    # docstring also documents non_conservative_force/stress-like sources, which
    # take the "plain linear combination of predicted values" path instead of the
    # autograd path, so exercise that separately.
    wrapped = WeightedSumModel(
        nc_force_model,
        {
            "non_conservative_force/mix": WeightedSumHead(
                sources={
                    "non_conservative_force/a": 0.3,
                    "non_conservative_force/b": 0.7,
                }
            )
        },
    )

    with torch.no_grad():
        separate = nc_force_model(
            [nc_force_water_system],
            {
                "non_conservative_force/a": ModelOutput(sample_kind="atom"),
                "non_conservative_force/b": ModelOutput(sample_kind="atom"),
            },
        )
        combined = wrapped(
            [nc_force_water_system],
            {"non_conservative_force/mix": ModelOutput(sample_kind="atom")},
        )

    a = separate["non_conservative_force/a"].block().values
    b = separate["non_conservative_force/b"].block().values
    mix = combined["non_conservative_force/mix"].block().values
    assert torch.allclose(mix, 0.3 * a + 0.7 * b, atol=1e-6)


def test_weighted_sum_rejects_mismatched_components_or_properties():
    # Same quantity/unit (so that check passes), but different `num_subtargets`,
    # which differs in `properties` -- the mismatched-unit test already covers
    # the quantity/unit branch, but nothing exercised this second check.
    target = {
        "quantity": "energy",
        "unit": "eV",
        "sample_kind": "system",
        "type": "scalar",
    }
    model = PET(
        _minimal_hypers(),
        DatasetInfo(
            length_unit="Angstrom",
            atomic_types=[1, 6, 7, 8],
            targets={
                "generic/a": get_generic_target_info(
                    "generic/a", {**target, "num_subtargets": 1}
                ),
                "generic/b": get_generic_target_info(
                    "generic/b", {**target, "num_subtargets": 2}
                ),
            },
        ),
    )
    with pytest.raises(ValueError, match="different components or properties"):
        WeightedSumModel(
            model,
            {
                "generic/mix": WeightedSumHead(
                    sources={"generic/a": 0.5, "generic/b": 0.5}
                )
            },
        )


def test_weighted_sum_upgrade_checkpoint_rejects_version_mismatch(base_model):
    wrapped = WeightedSumModel(
        base_model,
        {"energy/mix": WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})},
    )
    checkpoint = wrapped.get_checkpoint()
    checkpoint["model_ckpt_version"] = 999
    with pytest.raises(RuntimeError, match="version 999"):
        WeightedSumModel.upgrade_checkpoint(checkpoint)


def test_weighted_sum_export_rejects_unsupported_dtype():
    # Use a fresh model instance rather than the shared `base_model` fixture:
    # `.to(dtype)` mutates the model in place, which would leak into every other
    # test in this module if it were done on the shared fixture.
    model = PET(_minimal_hypers(), _dataset_info())
    model.eval()
    model.to(torch.float16)
    wrapped = WeightedSumModel(
        model,
        {"energy/mix": WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})},
    )
    with pytest.raises(ValueError, match="unsupported dtype"):
        wrapped.export()


def test_weighted_sum_export_matches_wrapped_module_output(base_model, water_system):
    # test_weighted_sum_export_produces_atomistic_model only checks the exported
    # capabilities; nothing runs the exported AtomisticModel itself to confirm it
    # numerically reproduces what the pre-export module computes.
    wrapped = WeightedSumModel(
        base_model,
        {"energy/mix": WeightedSumHead(sources={"energy/a": 0.3, "energy/b": 0.7})},
    )
    with torch.no_grad():
        before = wrapped(
            [water_system], {"energy/mix": ModelOutput(sample_kind="system")}
        )

    exported = wrapped.export()
    options = ModelEvaluationOptions(
        length_unit="Angstrom",
        outputs={"energy/mix": ModelOutput(sample_kind="system")},
    )
    with torch.no_grad():
        after = exported([water_system], options, check_consistency=True)

    assert torch.allclose(
        before["energy/mix"].block().values,
        after["energy/mix"].block().values,
        atol=1e-6,
    )
