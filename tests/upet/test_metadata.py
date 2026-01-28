import pytest

from upet._metadata import get_upet_metadata
from upet._models import get_versions_for_model
from upet._version import UPET_AVAILABLE_MODELS


@pytest.mark.parametrize("model_name", UPET_AVAILABLE_MODELS)
def test_get_upet_metadata(model_name):
    if "-xl" in model_name or "-l" in model_name:
        pytest.skip("Skipping XL models and L models due to large size.")
    model, size = model_name.rsplit("-", 1)
    all_model_versions = get_versions_for_model(model, size)

    for version in all_model_versions:
        model, size = model_name.rsplit("-", 1)
        metadata = get_upet_metadata(model, size, version)

        if "mad" in model.lower():
            authors = [
                "Arslan Mazitov (arslan.mazitov@epfl.ch)",
                "Filippo Bigi",
                "Matthias Kellner",
                "Paolo Pegolo",
                "Davide Tisi",
                "Guillaume Fraux",
                "Sergey Pozdnyakov",
                "Philip Loche",
                "Michele Ceriotti (michele.ceriotti@epfl.ch)",
            ]
        else:
            authors = [
                "Filippo Bigi (filippo.bigi@epfl.ch)",
                "Arslan Mazitov (arslan.mazitov@epfl.ch)",
                "Paolo Pegolo",
                "Michele Ceriotti (michele.ceriotti@epfl.ch)",
            ]

        assert metadata.name == f"{model.upper()}-{size.upper()} v{version}"
        assert metadata.description == (
            r"A universal interatomic potential for advanced materials modeling "
            r"based on a Point-Edge Transformer (PET) architecture, and trained on "
            r"the {} dataset. Model size: {}".format(model.split("-")[1].upper(), size)
        )

        assert metadata.authors == authors
        assert metadata.references == {
            "architecture": ["https://arxiv.org/abs/2305.19302v3"],
            "model": [
                "https://doi.org/10.1038/s41467-025-65662-7",
                "https://arxiv.org/abs/2601.16195",
            ],
        }
