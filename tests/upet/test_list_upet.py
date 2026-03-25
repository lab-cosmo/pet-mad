from upet._models import list_upet


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
