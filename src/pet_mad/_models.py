import importlib.util
import logging
import warnings
from typing import Optional

from metatomic.torch import AtomisticModel
from metatrain.utils.io import load_model as load_metatrain_model
from metatrain.utils.io import _hf_hub_download_url
from .utils import get_pet_mad_metadata, get_pet_mad_dos_metadata
from urllib.parse import urlparse
from urllib.request import urlretrieve
from .modules import BandgapModel
import torch

from packaging.version import Version

from ._version import (
    PET_MAD_LATEST_STABLE_VERSION,
    PET_MAD_AVAILABLE_VERSIONS,
    PET_MAD_DOS_LATEST_STABLE_VERSION,
    PET_MAD_DOS_AVAILABLE_VERSIONS,
)

BASE_URL = "https://huggingface.co/lab-cosmo/pet-mad/resolve/{tag}/models/pet-mad-{version}.ckpt"


def get_pet_mad(
    *, version: str = "latest", checkpoint_path: Optional[str] = None
) -> AtomisticModel:
    """Get a metatomic ``AtomisticModel`` for PET-MAD.

    :param version: PET-MAD version to use. Defaults to the latest stable version.
    :param checkpoint_path: path to a checkpoint file to load the model from. If
        provided, the `version` parameter is ignored.
    """
    if version == "latest":
        version = Version(PET_MAD_LATEST_STABLE_VERSION)
    if not isinstance(version, Version):
        version = Version(version)

    if version not in [Version(v) for v in PET_MAD_AVAILABLE_VERSIONS]:
        raise ValueError(
            f"Version {version} is not supported. Supported versions are {PET_MAD_AVAILABLE_VERSIONS}"
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

    metadata = get_pet_mad_metadata(version)
    return model.export(metadata)


def save_pet_mad(*, version: str = "latest", checkpoint_path=None, output=None):
    """
    Save the PET-MAD model to a TorchScript file (``pet-mad-xxx.pt``). These files can
    be used with LAMMPS and other tools to run simulations without Python.

    :param version: PET-MAD version to use. Defaults to the latest stable version.
    :param checkpoint_path: path to a checkpoint file to load the model from. If
        provided, the `version` parameter is ignored.
    :param output: path to use for the output model, defaults to
        ``pet-mad-{version}.pt`` when using a version, or the checkpoint path when using
        a checkpoint.
    """
    if version == "latest":
        version = Version(PET_MAD_LATEST_STABLE_VERSION)
    if not isinstance(version, Version):
        version = Version(version)

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


BASE_URL_PET_MAD_DOS = "https://huggingface.co/lab-cosmo/pet-mad-dos/resolve/{tag}/models/pet-mad-dos-{version}.pt"
BASE_URL_BANDGAP_MODEL = (
    "https://huggingface.co/lab-cosmo/pet-mad-dos/resolve/{tag}/models/bandgap-model.pt"
)


def get_pet_mad_dos(
    *, version: str = "latest", model_path: Optional[str] = None
) -> AtomisticModel:
    """Get a metatomic ``AtomisticModel`` for PET-MAD-DOS.

    :param version: PET-MAD-DOS version to use. Defaults to latest available version.
    :param model_path: path to a Torch-Scripted metatomic ``AtomisticModel``. If
        provided, the `version` parameter is ignored.
    """
    if version == "latest":
        version = Version(PET_MAD_DOS_LATEST_STABLE_VERSION)
    if not isinstance(version, Version):
        version = Version(version)

    if version not in [Version(v) for v in PET_MAD_DOS_AVAILABLE_VERSIONS]:
        raise ValueError(
            f"Version {version} is not supported. Supported versions are {PET_MAD_DOS_AVAILABLE_VERSIONS}"
        )

    if model_path is not None:
        logging.info(f"Loading PET-MAD-DOS model from checkpoint: {model_path}")
        path = model_path
    else:
        logging.info(f"Downloading PET-MAD-DOS model version: {version}")
        path = BASE_URL_PET_MAD_DOS.format(tag=f"v{version}", version=f"v{version}")

    model = load_metatrain_model(path)
    metadata = get_pet_mad_dos_metadata(version)
    model._metadata = metadata
    return model


def _get_bandgap_model(version: str = "latest", model_path: Optional[str] = None):
    """
    Get a bandgap model for PET-MAD-DOS
    """
    if version == "latest":
        version = Version(PET_MAD_DOS_LATEST_STABLE_VERSION)
    if not isinstance(version, Version):
        version = Version(version)

    if version not in [Version(v) for v in PET_MAD_DOS_AVAILABLE_VERSIONS]:
        raise ValueError(
            f"Version {version} is not supported. Supported versions are {PET_MAD_DOS_AVAILABLE_VERSIONS}"
        )

    if model_path is not None:
        logging.info(
            f"Loading the PET-MAD-DOS bandgap model from checkpoint: {model_path}"
        )
        path = model_path
    else:
        logging.info(f"Downloading bandgap model version: {version}")
        path = BASE_URL_BANDGAP_MODEL.format(tag=f"v{version}")
        path = str(path)
        url = urlparse(path)

        if url.scheme:
            if url.netloc == "huggingface.co":
                path = _hf_hub_download_url(url=url.geturl(), hf_token=None)
            else:
                # Avoid caching generic URLs due to lack of a model hash for proper cache
                # invalidation
                path, _ = urlretrieve(url=url.geturl())

    model = BandgapModel()
    model.load_state_dict(torch.load(path, weights_only=False, map_location="cpu"))
    return model
