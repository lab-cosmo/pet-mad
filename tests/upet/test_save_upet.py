import os

from upet._models import save_upet


def test_save_upet(tmp_path):
    output_path = str(tmp_path / "pet-mad-xs.pt")
    save_upet(model="pet-mad", size="xs", output=output_path)
    assert os.path.isfile(output_path)
    assert os.path.getsize(output_path) > 0


def test_save_upet_default_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_upet(model="pet-mad", size="xs")
    # default name is "{model}-{size}-v{version}.pt"
    saved_files = [f for f in os.listdir(tmp_path) if f.endswith(".pt")]
    assert len(saved_files) == 1
    assert saved_files[0].startswith("pet-mad-xs-v")
