import importlib.util
import logging
import os
import warnings
from platformdirs import user_cache_dir
from typing import Optional

from metatensor.torch.atomistic import ModelMetadata
from metatensor.torch.atomistic.ase_calculator import MetatensorCalculator
from metatrain.utils.io import load_model as load_metatrain_model

warnings.filterwarnings(
    "ignore",
    message=("PET assumes that Cartesian tensors"),
)

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
VERSIONS = ("latest", "1.1.0", "1.0.1", "1.0.0")
BASE_URL = (
    "https://huggingface.co/lab-cosmo/pet-mad/resolve/{}/models/pet-mad-latest.ckpt"
)


class PETMADCalculator(MetatensorCalculator):
    """
    PET-MAD ASE Calculator
    """

    def __init__(
        self,
        version: str = "latest",
        checkpoint_path: Optional[str] = None,
        *,
        check_consistency=False,
        device=None,
    ):
        """
        :param version: PET-MAD version to use. Supported versions are "latest",
            "v1.0.1", "1.0.0". Defaults to "latest".
        :param checkpoint_path: path to a checkpoint file to load the model from. If
            provided, the `version` parameter is ignored.
        :param check_consistency: should we check the model for consistency when
            running, defaults to False.
        :param device: torch device to use for the calculation. If `None`, we will try
            the options in the model's `supported_device` in order.
        """

        if version not in VERSIONS:
            raise ValueError(
                f"Version {version} is not supported. Supported versions are {VERSIONS}"
            )

        extensions_directory = None
        if version == "1.0.0":
            if not importlib.util.find_spec("pet_neighbors_convert"):
                raise ImportError(
                    f"PET-MAD v{version} is now deprecated. Please consider using the "
                    "`latest` version. If you still want to use it, please install the "
                    "pet-mad package with optional dependencies: "
                    "pip install pet-mad[deprecated]"
                )
                extensions_directory = "extensions"

        if checkpoint_path is not None:
            logging.info(f"Loading PET-MAD model from checkpoint: {checkpoint_path}")
            path = checkpoint_path
        else:
            logging.info(f"Downloading PET-MAD model version: {version}")
            path = BASE_URL.format(
                f"v{version}" if version not in ("latest", "1.1.0") else "main"
            )
        model = load_metatrain_model(path).export(METADATA)

        cache_dir = user_cache_dir("pet-mad", "metatensor")
        os.makedirs(cache_dir, exist_ok=True)

        pt_path = cache_dir + f"/pet-mad-{version}.pt"
        extensions_directory = (
            (cache_dir + "/" + extensions_directory)
            if extensions_directory is not None
            else None
        )

        logging.info(f"Exporting checkpoint to TorchScript at {pt_path}")
        model.save(pt_path, collect_extensions=extensions_directory)

        super().__init__(
            pt_path,
            extensions_directory=extensions_directory,
            check_consistency=check_consistency,
            device=device,
            non_conservative=False,
            additional_outputs={},
        )
