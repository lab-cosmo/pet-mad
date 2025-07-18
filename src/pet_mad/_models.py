import importlib.util
import logging
import warnings
from typing import Optional

from metatomic.torch import AtomisticModel
from metatrain.utils.io import load_model as load_metatrain_model
from .utils import get_metadata

from packaging.version import Version


LATEST_VERSION = "1.1.0"
AVAILABLE_VERSIONS = ("1.1.0", "1.0.1", "1.0.0")

BASE_URL = "https://huggingface.co/lab-cosmo/pet-mad/resolve/{tag}/models/pet-mad-{version}.ckpt"


def get_pet_mad(
    *, version: str = LATEST_VERSION, checkpoint_path: Optional[str] = None
) -> AtomisticModel:
    """Get a metatomic ``AtomisticModel`` for PET-MAD.

    :param version: PET-MAD version to use. Supported versions are
        "1.1.0", "1.0.1", "1.0.0". Defaults to latest available version.
    :param checkpoint_path: path to a checkpoint file to load the model from. If
        provided, the `version` parameter is ignored.
    """
    if not isinstance(version, Version):
        version = Version(version)

    if version not in [Version(v) for v in AVAILABLE_VERSIONS]:
        raise ValueError(
            f"Version {version} is not supported. Supported versions are {AVAILABLE_VERSIONS}"
        )

    if version == Version("1.0.0"):
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
        path = BASE_URL.format(tag=f"v{version}", version=f"v{version}")

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message="PET assumes that Cartesian tensors of rank 2 are stress-like",
        )
        model = load_metatrain_model(path)

    metadata = get_metadata(version)
    return model.export(metadata)


def save_pet_mad(*, version: str = LATEST_VERSION, checkpoint_path=None, output=None):
    """
    Save the PET-MAD model to a TorchScript file (``pet-mad-xxx.pt``). These files can
    be used with LAMMPS and other tools to run simulations without Python.

    :param version: PET-MAD version to use. Supported versions are "1.1.0",
        "1.0.1", "1.0.0". Defaults to the latest version.
    :param checkpoint_path: path to a checkpoint file to load the model from. If
        provided, the `version` parameter is ignored.
    :param output: path to use for the output model, defaults to
        ``pet-mad-{version}.pt`` when using a version, or the checkpoint path when using
        a checkpoint.
    """
    extensions_directory = None
    if version == Version("1.0.0"):
        logging.info("Putting TorchScript extensions in `extensions/`")
        extensions_directory = "extensions"

    model = get_pet_mad(version=version, checkpoint_path=checkpoint_path)

    if output is None:
        if checkpoint_path is None:
            output = f"pet-mad-{version}.pt"
        else:
            raise

    model.save(output, collect_extensions=extensions_directory)
    logging.info(f"Saved PET-MAD model to {output}")
