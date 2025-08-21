from pet_mad.utils import get_pet_mad_metadata

def test_get_metadata():
    metadata = get_pet_mad_metadata("1.0.2")
    assert metadata.name == "PET-MAD v1.0.2"
    assert metadata.description == "A universal interatomic potential for advanced materials modeling"
    assert metadata.authors == ["Arslan Mazitov (arslan.mazitov@epfl.ch)", "Filippo Bigi", "Matthias Kellner", "Paolo Pegolo", "Davide Tisi", "Guillaume Fraux", "Sergey Pozdnyakov", "Philip Loche", "Michele Ceriotti (michele.ceriotti@epfl.ch)"]
    assert metadata.references == {"architecture": ["https://arxiv.org/abs/2305.19302v3"], "model": ["http://arxiv.org/abs/2503.14118"]}