import os

from upet._models import save_upet


def test_save_upet(tmp_path):
    output_path = str(tmp_path / "pet-mad-xs.pt")
    save_upet(model="pet-mad", size="xs", output=output_path)
    assert os.path.isfile(output_path)
    assert os.path.getsize(output_path) > 0
