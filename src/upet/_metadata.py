from typing import Optional

from metatomic.torch import ModelMetadata


# Official spelling of the training dataset, when it differs from the one that
# can be derived from the model name.
DATASET_NAMES = {"pet-omol": "OMol25"}

# Same order as the model paper (doi:10.1088/2632-2153/ae6417).
OMOL_AUTHORS = [
    "Filippo Bigi (filippo.bigi@epfl.ch)",
    "Paolo Pegolo (paolo.pegolo@epfl.ch)",
    "Arslan Mazitov (arslan.mazitov@epfl.ch)",
    "Jonathan Schmidt",
    "Michele Ceriotti (michele.ceriotti@epfl.ch)",
]

# ``dataset`` is not an accepted key of ModelMetadata.references, so the OMol25
# dataset paper is listed together with the model one.
OMOL_REFERENCES = {
    # the original PET preprint is added by PET's own default metadata
    "architecture": ["https://doi.org/10.1088/2632-2153/ae6417"],
    "model": [
        "https://doi.org/10.1088/2632-2153/ae6417",
        "https://arxiv.org/abs/2505.08762",
    ],
}


# PET-MOLS is not a universal potential, so it does not use the generic
# description below.
MOLS_DESCRIPTION = (
    r"A machine-learning interatomic potential to study organic molecular "
    r"crystals, trained on periodic PBE0+MBD reference data, covering 12 "
    r"elements and a broad range of organic motifs subsampled from the "
    r"Cambridge Structural Database. Model size: {}. Model version: {}."
)

# Same order as the model paper (arXiv:2603.06236).
MOLS_AUTHORS = [
    "Matthias Kellner (matthias.kellner@epfl.ch)",
    "Ruben Rodriguez-Madrid",
    "Jacob B. Holmes",
    "Victor Paul Principe",
    "Seio Inoue",
    "Lyndon Emsley",
    "Michele Ceriotti (michele.ceriotti@epfl.ch)",
]

MOLS_REFERENCES = {"model": ["https://arxiv.org/abs/2603.06236"]}


def get_upet_metadata(
    model: Optional[str] = None,
    size: Optional[str] = None,
    version: Optional[str] = None,
) -> ModelMetadata:
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
        dataset = DATASET_NAMES.get(model, model.split("-")[1].upper())
        description_text = description.format(dataset, size, version)
        if "mols" in model.lower():
            authors = MOLS_AUTHORS
            references = MOLS_REFERENCES
            description_text = MOLS_DESCRIPTION.format(size, version)
        elif "omol" in model.lower():
            authors = OMOL_AUTHORS
            references = OMOL_REFERENCES
        elif "mad" in model.lower():
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
            description=description_text,
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
