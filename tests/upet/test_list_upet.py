from upet._models import list_upet


def test_list_models():
    result = list_upet(print_summary=False)
    assert "models" in result
    assert len(result["models"]) > 0
    assert "pet-mad" in result["models"]


def test_list_sizes_for_model():
    result = list_upet(model="pet-mad", print_summary=False)
    assert result["model"] == "pet-mad"
    assert "sizes" in result
    assert len(result["sizes"]) > 0


def test_list_versions_for_model_and_size():
    result = list_upet(model="pet-mad", size="s", print_summary=False)
    assert result["model"] == "pet-mad"
    assert result["size"] == "s"
    assert len(result["versions"]) > 0
