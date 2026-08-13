from upet._metadata import get_upet_metadata


PET_REFERENCES = {
    "architecture": ["https://arxiv.org/abs/2305.19302v3"],
    "model": [
        "https://doi.org/10.1038/s41467-025-65662-7",
        "https://arxiv.org/abs/2601.16195",
    ],
}


def test_metadata_default():
    metadata = get_upet_metadata("pet-omat", "s", "1.0.0")

    assert metadata.name == "PET-OMAT-S v1.0.0"
    assert metadata.description == (
        "A universal interatomic potential for advanced materials modeling based "
        "on a Point-Edge Transformer (PET) architecture, and trained on the OMAT "
        "dataset. Model size: s. Model version: 1.0.0."
    )
    assert metadata.authors == [
        "Filippo Bigi (filippo.bigi@epfl.ch)",
        "Arslan Mazitov (arslan.mazitov@epfl.ch)",
        "Paolo Pegolo",
        "Michele Ceriotti (michele.ceriotti@epfl.ch)",
    ]
    assert metadata.references == PET_REFERENCES


def test_metadata_mad():
    metadata = get_upet_metadata("pet-mad", "xs", "1.5.0")

    assert metadata.name == "PET-MAD-XS v1.5.0"
    assert metadata.authors == [
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
    assert metadata.references == PET_REFERENCES


def test_metadata_omol():
    """Check the dataset name, which is not the one derived from the model name."""
    metadata = get_upet_metadata("pet-omol", "s", "1.0.0")

    assert metadata.name == "PET-OMOL-S v1.0.0"
    assert metadata.description == (
        "A universal interatomic potential for advanced materials modeling based "
        "on a Point-Edge Transformer (PET) architecture, and trained on the "
        "OMol25 dataset. Model size: s. Model version: 1.0.0."
    )
    assert metadata.authors == [
        "Filippo Bigi (filippo.bigi@epfl.ch)",
        "Paolo Pegolo (paolo.pegolo@epfl.ch)",
        "Arslan Mazitov (arslan.mazitov@epfl.ch)",
        "Jonathan Schmidt",
        "Michele Ceriotti (michele.ceriotti@epfl.ch)",
    ]
    assert metadata.references == {
        "architecture": ["https://doi.org/10.1088/2632-2153/ae6417"],
        "model": [
            "https://doi.org/10.1088/2632-2153/ae6417",
            "https://arxiv.org/abs/2505.08762",
        ],
    }


def test_metadata_mols():
    """Check the description of the only model that is not a universal potential."""
    metadata = get_upet_metadata("pet-mols", "s", "1.1.0")

    assert metadata.name == "PET-MOLS-S v1.1.0"
    assert metadata.description == (
        "A machine-learning interatomic potential to study organic molecular "
        "crystals, trained on periodic PBE0+MBD reference data, covering 12 "
        "elements and a broad range of organic motifs subsampled from the "
        "Cambridge Structural Database. Model size: s. Model version: 1.1.0."
    )
    assert metadata.authors == [
        "Matthias Kellner (matthias.kellner@epfl.ch)",
        "Ruben Rodriguez-Madrid",
        "Jacob B. Holmes",
        "Victor Paul Principe",
        "Seio Inoue",
        "Lyndon Emsley",
        "Michele Ceriotti (michele.ceriotti@epfl.ch)",
    ]
    assert metadata.references == {"model": ["https://arxiv.org/abs/2603.06236"]}
