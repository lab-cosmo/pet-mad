import logging
from typing import List, Optional

import ase
import ase.calculators
import ase.calculators.calculator
from metatensor.torch.atomistic import ModelMetadata
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator
from metatrain.utils.io import load_model
import importlib


METADATA = ModelMetadata(
    name="PET-MAD",
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
VERSIONS = ("latest", "1.0.1", "1.0.0")
BASE_URL = (
    "https://huggingface.co/lab-cosmo/pet-mad/resolve/{}/models/pet-mad-latest.ckpt"
)


class PETMADCalculator(ase.calculators.calculator.Calculator):
    """
    PET-MAD ASE Calculator

    :param version: PET-MAD version to use. Supported versions are "latest",
    "v1.0.1", "1.0.0". Defaults to "latest".
    :param checkpoint_path: path to a checkpoint file to load the model from. If
        provided, the `version` parameter is ignored.
    :param additional_outputs: Dictionary of additional outputs to be computed by
        the model. These outputs will always be computed whenever the
        `calculate` function is called (e.g. by `ase.Atoms.get_potential_energy`,
        `ase.optimize.optimize.Dynamics.run`, *etc.*) and stored in the
        `additional_outputs` attribute. If you want more control over when
        and how to compute specific outputs, you should use `run_model` instead.
    :param extensions_directory: if the model uses extensions, we will try to load
        them from this directory
    :param check_consistency: should we check the model for consistency when
        running, defaults to False.
    :param device: torch device to use for the calculation. If `None`, we will try
        the options in the model's `supported_device` in order.
    :param non_conservative: if ``True``, the model will be asked to
        compute non-conservative forces and stresses. This can afford a speed-up,
        potentially at the expense of physical correctness (especially in molecular
        dynamics simulations).
    """

    implemented_properties = ase.calculators.calculator.all_properties

    def __init__(
        self,
        version: str = "latest",
        checkpoint_path: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        if version not in VERSIONS:
            raise ValueError(
                f"Version {version} is not supported. Supported versions are {VERSIONS}"
            )
        if version in ("1.0.0"):
            if not importlib.util.find_spec("pet_neighbors_convert"):
                raise ImportError(
                    f"PET-MAD v{version} is now deprecated. Please consider using the "
                    "`latest` version. If you still want to use it, please install the "
                    "pet-mad package with optional dependencies: "
                    "pip install pet-mad[deprecated]"
                )

        if checkpoint_path is not None:
            logging.info(f"Loading PET-MAD model from checkpoint: {checkpoint_path}")
            path = checkpoint_path
        else:
            logging.info(f"Downloading PET-MAD model version: {version}")
            path = BASE_URL.format(f"v{version}" if version != "latest" else "main")
            print(f"Downloading PET-MAD model version: {path}")
        model = load_model(path).export(METADATA)
        self.model = model
        self._calculator = MetatensorCalculator(
            model, *args, non_conservative=False, **kwargs
        )

    def calculate(
        self,
        atoms: ase.Atoms,
        properties: List[str],
        system_changes: List[str],
    ) -> None:
        self._calculator.calculate(atoms, properties, system_changes)
        self.results = self._calculator.results

    def compute_energy(
        self,
        atoms: ase.Atoms,
        compute_forces_and_stresses: bool = False,
    ) -> dict:
        return self._calculator.compute_energy(
            atoms, compute_forces_and_stresses=compute_forces_and_stresses
        )
