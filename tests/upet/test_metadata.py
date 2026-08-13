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

        description = (
            r"A universal interatomic potential for advanced materials modeling "
            r"based on a Point-Edge Transformer (PET) architecture, and trained on "
            r"the {} dataset. Model size: {}. Model version: {}.".format(
                model.split("-")[1].upper(), size, version
            )
        )
        references = {
            "architecture": ["https://arxiv.org/abs/2305.19302v3"],
            "model": [
                "https://doi.org/10.1038/s41467-025-65662-7",
                "https://arxiv.org/abs/2601.16195",
            ],
        }

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
        elif "mols" in model.lower():
            # PET-MOLS is not a universal potential, and has its own paper
            description = (
                r"A machine-learning interatomic potential to study organic "
                r"molecular crystals, trained on periodic PBE0+MBD reference data, "
                r"covering 12 elements and a broad range of organic motifs "
                r"subsampled from the Cambridge Structural Database. "
                r"Model size: {}. Model version: {}.".format(size, version)
            )
            authors = [
                "Matthias Kellner (matthias.kellner@epfl.ch)",
                "Ruben Rodriguez-Madrid",
                "Jacob B. Holmes",
                "Victor Paul Principe",
                "Seio Inoue",
                "Lyndon Emsley",
                "Michele Ceriotti (michele.ceriotti@epfl.ch)",
            ]
            references = {"model": ["https://arxiv.org/abs/2603.06236"]}
        elif "omol" in model.lower():
            description = description.replace("the OMOL dataset", "the OMol25 dataset")
            authors = [
                "Filippo Bigi (filippo.bigi@epfl.ch)",
                "Paolo Pegolo (paolo.pegolo@epfl.ch)",
                "Arslan Mazitov (arslan.mazitov@epfl.ch)",
                "Jonathan Schmidt",
                "Michele Ceriotti (michele.ceriotti@epfl.ch)",
            ]
            references = {
                "architecture": ["https://doi.org/10.1088/2632-2153/ae6417"],
                "model": [
                    "https://doi.org/10.1088/2632-2153/ae6417",
                    "https://arxiv.org/abs/2505.08762",
                ],
            }
        else:
            authors = [
                "Filippo Bigi (filippo.bigi@epfl.ch)",
                "Arslan Mazitov (arslan.mazitov@epfl.ch)",
                "Paolo Pegolo",
                "Michele Ceriotti (michele.ceriotti@epfl.ch)",
            ]

        assert metadata.name == f"{model.upper()}-{size.upper()} v{version}"
        assert metadata.description == description
        assert metadata.authors == authors
        assert metadata.references == references
