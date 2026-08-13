"""Metadata for the checkpoints that do not carry their own."""

from typing import Optional

from metatomic.torch import ModelMetadata


def get_upet_metadata(
    model: Optional[str] = None,
    size: Optional[str] = None,
    version: Optional[str] = None,
) -> ModelMetadata:
    """Describe a model that was published without any metadata of its own.

    Newer checkpoints store their name, description, authors and references,
    which is where the ones of a new model belong; this only covers the models
    published before that.
    """
    description = (
        r"A universal interatomic potential for advanced materials modeling "
        r"based on a Point-Edge Transformer (PET) architecture, and trained on "
        r"the {} dataset. Model size: {}. Model version: {}."
    )
    references = {
        "architecture": ["https://arxiv.org/abs/2305.19302v3"],
        "model": [
            "https://doi.org/10.1038/s41467-025-65662-7",
            "https://arxiv.org/abs/2601.16195",
        ],
    }

    if model and size and version:
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
        metadata = ModelMetadata(
            name=f"{model.upper()}-{size.upper()} v{version}",
            description=description.format(model.split("-")[1].upper(), size, version),
            authors=authors,
            references=references,
        )
    else:
        metadata = ModelMetadata(
            name="Custom UPET",
            description=description.format("custom", "unknown", "unknown"),
            authors=[],
            references=references,
        )

    return metadata


def get_pet_mad_dos_metadata(version: str):
    return ModelMetadata(
        name=f"PET-MAD-DOS v{version}",
        description="A universal machine learning model for the electronic density of states",  # noqa: E501
        authors=[
            "Wei Bin How (weibin.how@epfl.ch)",
            "Pol Febrer",
            "Sanggyu Chong",
            "Arslan Mazitov",
            "Filippo Bigi",
            "Matthias Kellner",
            "Sergey Pozdnyakov",
            "Michele Ceriotti (michele.ceriotti@epfl.ch)",
        ],
        references={
            "architecture": ["https://arxiv.org/abs/2508.09000"],
            "model": [],
        },
    )
