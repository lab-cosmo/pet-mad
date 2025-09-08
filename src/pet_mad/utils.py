from metatomic.torch import ModelMetadata
from ase import Atoms
from ase.units import kB
from typing import List, Union, Optional, Dict
import torch
from pathlib import Path
from urllib.parse import unquote
from huggingface_hub import hf_hub_download
import numpy as np
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

AVAILABLE_LEBEDEV_GRID_ORDERS = [
    3,
    5,
    7,
    9,
    11,
    13,
    15,
    17,
    19,
    21,
    23,
    25,
    27,
    29,
    31,
    35,
    41,
    47,
    53,
    59,
    65,
    71,
    77,
    83,
    89,
    95,
    101,
    107,
    113,
    119,
    125,
    131,
]


def rotate_atoms(
    atoms: Atoms,
    rotations: np.ndarray,
) -> List[Atoms]:
    """
    Create rotated copies of `atoms` using rotation matrices (active convention).
    Row-vector layout: r' = r @ R^T.
    """
    pos = atoms.get_positions()
    has_cell = (atoms.cell is not None) and (atoms.cell.rank > 0)
    cell = atoms.cell.array if has_cell else None

    rotated_atoms: List[Atoms] = []
    for Ri in rotations:
        ai = atoms.copy()
        ai.positions = np.ascontiguousarray(pos @ Ri.T)
        if has_cell:
            ai.cell = np.ascontiguousarray(cell @ Ri.T)
        rotated_atoms.append(ai)
    return rotated_atoms


def compute_rotational_average(
    m1: Dict[str, Union[np.ndarray, float]],
    m2: Dict[str, Union[np.ndarray, float]],
    total_weight: float,
    suffix_std: str = "_rot_std",
) -> Dict[str, np.ndarray | float]:
    """
    Convert accumulated weighted first/second moments to mean and std.
    Uses population (probability-weighted) variance: Var = E[x^2] - (E[x])^2.
    """
    out: Dict[str, np.ndarray | float] = {}
    W = float(total_weight)

    for key, s1 in m1.items():
        s2 = m2.get(key, None)
        if s2 is None:
            # No variance info; just pass through
            out[key] = s1 if isinstance(s1, float) else np.ascontiguousarray(s1)
            continue

        if isinstance(s1, (float, np.floating)):
            mean = float(s1 / W)
            var = float(s2 / W) - mean * mean
            std = float(np.sqrt(max(var, 0.0)))
            out[key] = mean
            out[key + suffix_std] = std
        else:
            mean = np.ascontiguousarray(s1 / W)
            var = np.ascontiguousarray(s2 / W) - mean * mean
            # guard tiny negative due to roundoff
            var[var < 0] = 0.0
            std = np.sqrt(var, dtype=mean.dtype, where=np.ones_like(var, dtype=bool))
            out[key] = mean
            out[key + suffix_std] = std

    return out


def accumulate_rotational_moments(
    m1: Dict[str, Union[np.ndarray, float]],
    m2: Dict[str, Union[np.ndarray, float]],
    results_b: Dict[str, np.ndarray],
    rotations_b: np.ndarray,  # (b,3,3)
    weights_b: np.ndarray,  # (b,)
) -> None:
    """
    Stream (online) accumulation of weighted first and second moments for a batch.

    Conventions:
      - Scalars (shape (b,)): accumulate sum(w*x) and sum(w*x^2).
      - "*forces" (b,N,3): rotate back to lab (F_lab = F_rot @ R), then accumulate
        sum(w*F_lab) and sum(w*F_lab^2) elementwise.
      - "*stress" (b,3,3): S_lab = R S_rot R^T, then accumulate sum(w*S_lab) and sum(w*S_lab^2).
      - Fallback: if leading dim is b, treat remaining dims elementwise as above.
    """
    w = weights_b
    b = w.shape[0]
    Rt = np.transpose(rotations_b, (0, 2, 1))  # (b,3,3)

    for key, val in results_b.items():
        arr = np.asarray(val)

        # Scalar (b,)
        if arr.ndim == 1 and arr.shape[0] == b:
            wx = (w * arr).sum()
            wx2 = (w * (arr**2)).sum()
            m1[key] = m1.get(key, 0.0) + float(wx)
            m2[key] = m2.get(key, 0.0) + float(wx2)
            continue

        # Forces (b,N,3)
        if key.endswith("forces") and arr.ndim == 3 and arr.shape[-1] == 3:
            # Rotate back and weight
            # sum_i w_i * (F_i @ R_i), elementwise second moment via (F_lab**2)
            F_lab = np.matmul(arr, rotations_b)  # (b,N,3)
            wF = w[:, None, None] * F_lab  # (b,N,3)
            wx = wF.sum(axis=0)  # (N,3)
            wx2 = (w[:, None, None] * (F_lab**2)).sum(axis=0)  # (N,3)

            m1[key] = wx if key not in m1 else (m1[key] + wx)
            m2[key] = wx2 if key not in m2 else (m2[key] + wx2)
            continue

        # Stress: (b,3,3)
        if key.endswith("stress") and arr.ndim == 3 and arr.shape[-1] == 3:
            RS = np.matmul(rotations_b, arr)  # (b,3,3)
            S_lab = np.matmul(RS, Rt)  # (b,3,3)
            wS = S_lab * w[:, None, None]  # (b,3,3)
            wx = wS.sum(axis=0)  # (3,3)
            wx2 = (w[:, None, None] * (S_lab**2)).sum(axis=0)  # (3,3)

            m1[key] = wx if key not in m1 else (m1[key] + wx)
            m2[key] = wx2 if key not in m2 else (m2[key] + wx2)
            continue

        # Otherwise: leave untouched (not per-rotation)
        # (You could raise here if strictness is desired.)
        if key not in m1 or key not in m2:
            raise KeyError(f"{key} not in results")


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


def fermi_dirac_distribution(
    energies: torch.Tensor, mu: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    """
    Fermi-Dirac distribution function.

    :param energies: Energy grid.
    :param mu: Fermi level.
    :param T: Temperature.
    :return: Fermi-Dirac distribution function.
    """
    x = -(energies - mu) / (kB * T)  # Note the negative sign
    return torch.sigmoid(x)


def get_num_electrons(atoms: Union[Atoms, List[Atoms]]) -> torch.Tensor:
    """
    Get the number of electrons for a given ase.Atoms object, or a list of ase.Atoms
    objects.

    :param atoms: ASE atoms object or a list of ASE atoms objects
    :return: Number of electrons for each ase.Atoms object stored in a torch.Tensor
    format.
    """
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
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        subfolder=subfolder,
        cache_dir=cache_dir,
        revision=revision,
        token=hf_token,
        endpoint=endpoint,
    )
