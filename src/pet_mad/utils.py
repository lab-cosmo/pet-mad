from metatomic.torch import ModelMetadata
from ase import Atoms
from typing import List, Union, Optional
import torch
from pathlib import Path
from urllib.parse import unquote
from huggingface_hub import hf_hub_download
import re


hf_pattern = re.compile(
    r"(?P<endpoint>https://[^/]+)/"
    r"(?P<repo_id>[^/]+/[^/]+)/"
    r"resolve/"
    r"(?P<revision>[^/]+)/"
    r"(?P<filename>.+)"
)

NUM_ELECTRONS_PER_ELEMENT = {
    "Al": 3.0,
    "As": 5.0,
    "Ba": 10.0,
    "Be": 4.0,
    "Bi": 15.0,
    "B": 3.0,
    "Br": 7.0,
    "Ca": 10.0,
    "Cd": 12.0,
    "Cl": 7.0,
    "Co": 17.0,
    "C": 4.0,
    "Cr": 14.0,
    "Cs": 9.0,
    "Fe": 16.0,
    "F": 7.0,
    "Ga": 13.0,
    "Ge": 14.0,
    "H": 1.0,
    "In": 13.0,
    "I": 7.0,
    "Ir": 15.0,
    "K": 9.0,
    "Li": 3.0,
    "Mg": 2.0,
    "Mn": 15.0,
    "Na": 9.0,
    "Nb": 13.0,
    "Ni": 18.0,
    "N": 5.0,
    "O": 6.0,
    "Os": 16.0,
    "Pb": 14.0,
    "Po": 16.0,
    "P": 5.0,
    "Pt": 16.0,
    "Re": 15.0,
    "Rn": 18.0,
    "Sb": 15.0,
    "Se": 6.0,
    "Si": 4.0,
    "Sn": 14.0,
    "S": 6.0,
    "Sr": 10.0,
    "Ta": 13.0,
    "Te": 6.0,
    "Ti": 12.0,
    "Tl": 13.0,
    "V": 13.0,
    "W": 14.0,
    "Y": 11.0,
    "Zn": 20.0,
    "Zr": 12.0,
    "Ag": 19.0,
    "Ar": 8.0,
    "Au": 19.0,
    "Ce": 12.0,
    "Dy": 20.0,
    "Er": 22.0,
    "Eu": 17.0,
    "Gd": 18.0,
    "He": 2.0,
    "Hf": 12.0,
    "Hg": 20.0,
    "Ho": 21.0,
    "Kr": 8.0,
    "La": 11.0,
    "Lu": 25.0,
    "Mo": 14.0,
    "Nd": 14.0,
    "Ne": 8.0,
    "Pd": 18.0,
    "Pm": 15.0,
    "Pr": 13.0,
    "Rb": 9.0,
    "Rh": 17.0,
    "Ru": 16.0,
    "Sc": 11.0,
    "Sm": 16.0,
    "Tb": 19.0,
    "Tc": 15.0,
    "Tm": 23.0,
    "Xe": 18.0,
    "Yb": 24.0,
    "Cu": 11.0,
}


def get_pet_mad_metadata(version: str):
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


def get_pet_mad_dos_metadata(version: str):
    return ModelMetadata(
        name=f"PET-MAD-DOS v{version}",
        description="A universal machine learning model for the electronic density of states",
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


def get_num_electrons(atoms: Union[Atoms, List[Atoms]]) -> torch.Tensor:
    num_electrons = []
    if isinstance(atoms, Atoms):
        atoms = [atoms]
    for item in atoms:
        num_electrons.append(
            int(sum([NUM_ELECTRONS_PER_ELEMENT[symbol] for symbol in item.symbols]))
        )
    num_electrons = torch.tensor(num_electrons)
    return num_electrons


def hf_hub_download_url(
    url: str,
    hf_token: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> str:
    """Wrapper around `hf_hub_download` allowing passing the URL directly.

    Function is in inverse of `hf_hub_url`
    """

    match = hf_pattern.match(url)

    if not match:
        raise ValueError(f"URL '{url}' has an invalid format for the Hugging Face Hub.")

    endpoint = match.group("endpoint")
    repo_id = match.group("repo_id")
    revision = unquote(match.group("revision"))
    filename = unquote(match.group("filename"))

    # Extract subfolder if applicable
    parts = filename.split("/", 1)
    if len(parts) == 2:
        subfolder, filename = parts
    else:
        subfolder = None
    print("Subfolder: ", subfolder)
    print("Filename: ", filename)
    print("Revision: ", revision)
    print("Endpoint: ", endpoint)
    print("Repo ID: ", repo_id)
    print("HF Token: ", hf_token)
    print("Cache Dir: ", cache_dir)
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        cache_dir=cache_dir,
        revision=revision,
        token=hf_token,
        endpoint=endpoint,
    )
