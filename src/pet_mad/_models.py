import importlib.util
import warnings
import logging

from metatomic.torch import ModelMetadata, AtomisticModel
from metatrain.utils.io import load_model as load_metatrain_model

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


def get_pet_mad(*, version="latest", checkpoint_path=None) -> AtomisticModel:
    """Get a metatomic ``AtomisticModel`` for PET-MAD.

    :param version: PET-MAD version to use. Supported versions are "latest", "v1.0.1",
        "1.0.0". Defaults to "latest".
    :param checkpoint_path: path to a checkpoint file to load the model from. If
        provided, the `version` parameter is ignored.
    """
    if version not in VERSIONS:
        raise ValueError(
            f"Version {version} is not supported. Supported versions are {VERSIONS}"
        )

    if version == "1.0.0":
        if not importlib.util.find_spec("pet_neighbors_convert"):
            raise ImportError(
                f"PET-MAD v{version} is now deprecated. Please consider using the "
                "`latest` version. If you still want to use it, please install the "
                "pet-mad package with optional dependencies: "
                "pip install pet-mad[deprecated]"
            )

        import pet_neighbors_convert  # noqa: F401

    if checkpoint_path is not None:
        logging.info(f"Loading PET-MAD model from checkpoint: {checkpoint_path}")
        path = checkpoint_path
    else:
        logging.info(f"Downloading PET-MAD model version: {version}")
        path = BASE_URL.format(
            f"v{version}" if version not in ("latest", "1.1.0") else "main"
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message="PET assumes that Cartesian tensors of rank 2 are stress-like",
        )
        model = load_metatrain_model(path)

    return model.export(METADATA)


def save_pet_mad(*, version="latest", checkpoint_path=None, output=None):
    """
    Save the PET-MAD model to a TorchScript file (``pet-mad-xxx.pt``). These files can
    be used with LAMMPS and other tools to run simulations without Python.

    :param version: PET-MAD version to use. Supported versions are "latest", "v1.0.1",
        "1.0.0". Defaults to "latest".
    :param checkpoint_path: path to a checkpoint file to load the model from. If
        provided, the `version` parameter is ignored.
    :param output: path to use for the output model, defaults to
        ``pet-mad-{version}.pt`` when using a version, or the checkpoint path when using
        a checkpoint.
    """
    extensions_directory = None
    if version == "1.0.0":
        logging.info("putting TorchScript extensions in `extensions/`")
        extensions_directory = "extensions"

    model = get_pet_mad(version, checkpoint_path)

    if output is None:
        if checkpoint_path is None:
            output = f"pet-mad-{version}.pt"
        else:
            raise

    model.save(output, collect_extensions=extensions_directory)
    logging.info(f"saved pet-mad model to {output}")
