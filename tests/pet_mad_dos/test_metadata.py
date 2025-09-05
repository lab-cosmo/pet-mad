from pet_mad.utils import get_pet_mad_dos_metadata


def test_get_metadata():
    metadata = get_pet_mad_dos_metadata("1.0")
    assert metadata.name == "PET-MAD-DOS v1.0"
    assert (
        metadata.description
        == "A universal machine learning model for the electronic density of states"
    )
    assert metadata.authors == [
        "Wei Bin How (weibin.how@epfl.ch)",
        "Pol Febrer",
        "Sanggyu Chong",
        "Arslan Mazitov",
        "Filippo Bigi",
        "Matthias Kellner",
        "Sergey Pozdnyakov",
        "Michele Ceriotti (michele.ceriotti@epfl.ch)",
    ]
    assert metadata.references == {
        "architecture": ["https://arxiv.org/abs/2508.09000"],
        "model": [],
    }
