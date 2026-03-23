from upet._models import list_upet


def test_list_models(capsys):
    result = list_upet()
    assert "models" in result
    assert len(result["models"]) > 0
    assert "pet-mad" in result["models"]

    captured = capsys.readouterr()
    assert "pet-mad" in captured.out


def test_list_models_no_print(capsys):
    result = list_upet(print_summary=False)
    assert "models" in result
    captured = capsys.readouterr()
    assert captured.out == ""


def test_list_sizes_for_model(capsys):
    result = list_upet(model="pet-mad")
    assert result["model"] == "pet-mad"
    assert "sizes" in result
    assert len(result["sizes"]) > 0
    for size, versions in result["sizes"].items():
        assert len(versions) > 0

    captured = capsys.readouterr()
    assert "pet-mad" in captured.out


def test_list_versions_for_model_and_size(capsys):
    result = list_upet(model="pet-mad", size="s")
    assert result["model"] == "pet-mad"
    assert result["size"] == "s"
    assert "versions" in result
    assert len(result["versions"]) > 0

    captured = capsys.readouterr()
    assert "pet-mad" in captured.out
