from packaging.version import Version

from upet._models import parse_checkpoint_filename


def test_parse_standard_checkpoint_name():
    model, size, version = parse_checkpoint_filename("pet-mad-s-v1.2.3.ckpt")
    assert model == "pet-mad"
    assert size == "s"
    assert version == Version("1.2.3")


def test_parse_non_standard_checkpoint_name():
    model, size, version = parse_checkpoint_filename("custom_name.ckpt")
    assert model is None
    assert size is None
    assert version is None
