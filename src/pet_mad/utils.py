from metatomic.torch import ModelMetadata

def get_metadata(version: str):
    return ModelMetadata(
        name=f"PET-MAD v{version}",
        description="A universal interatomic potential for advanced materials modeling",
        authors=[
            "Arslan Mazitov (arslan.mazitov@epfl.ch)",
            "Filippo Bigi",
            "Matthias Kellner",
            "Paolo Pegolo",
            "Davide Tisi",
            "Guillaume Fraux",
            "Sergey Pozdnyakov",
            "Philip Loche",
            "Michele Ceriotti (michele.ceriotti@epfl.ch)",
        ],
        references={
            "architecture": ["https://arxiv.org/abs/2305.19302v3"],
            "model": ["http://arxiv.org/abs/2503.14118"],
        }, 
    )


