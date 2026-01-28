import logging
import os
import re
import warnings
from typing import Optional, Tuple
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch
from huggingface_hub import HfApi, hf_hub_download
from metatomic.torch import AtomisticModel
from metatrain.utils.io import load_model as load_metatrain_model
from packaging.version import Version

from ._metadata import get_pet_mad_dos_metadata, get_upet_metadata
from ._version import (
    PET_MAD_DOS_AVAILABLE_VERSIONS,
    PET_MAD_DOS_LATEST_STABLE_VERSION,
)
from .modules import BandgapModel
from .utils import hf_hub_download_url


CHECKPOINT_NAME_PATTERN = re.compile(
    r"^(?P<model>pet-[\w]+)-(?P<size>xs|s|m|l|xl)-v?(?P<version>\d+\.\d+\.\d+)\.ckpt$"
)


def parse_checkpoint_filename(path: str) -> Tuple[str, str, Version]:
    """
    Try to parse model, size, and version from a checkpoint filename.

    Returns (model, size, version) if filename matches standard pattern,
    (None, None, None) otherwise.

    Examples:
        "pet-mad-s-v1.0.2.ckpt" -> ("pet-mad", "s", Version("1.0.2"))
        "model.ckpt" -> (None, None, None)
    """
    filename = os.path.basename(path)
    match = CHECKPOINT_NAME_PATTERN.match(filename)
    if match:
        return (
            match.group("model"),
            match.group("size"),
            Version(match.group("version")),
        )
    return (None, None, None)


def upet_get_size_to_load(model: str, requested_size: Optional[str] = None) -> str:
    """
    Get the size of a UPET model.

    :param model: name of the model.
    :param requested_size: a requested size of the model.
    :return: If the model has multiple sizes available, the
        sizes will be chosen based on the following priority: s > m > xs > l > xl,
        depending on availability.
    """
    # We need to inspect the models in https://huggingface.co/lab-cosmo/upet/tree/main/models
    # and get the available sizes for each model.
    hf_api = HfApi()
    repo_files = hf_api.list_repo_files("lab-cosmo/upet")
    files_in_models_folder = [f[7:] for f in repo_files if f.startswith("models/")]
    all_model_files = [
        f
        for f in files_in_models_folder
        if f.startswith(f"{model}-") and f.endswith(".ckpt")
    ]
    all_model_sizes = [f.split(f"{model}-")[1].split("-")[0] for f in all_model_files]
    all_model_sizes = sorted(set(all_model_sizes))

    if requested_size is not None:
        if requested_size in all_model_sizes:
            return requested_size
        else:
            raise ValueError(
                f"Requested size {requested_size} not available for model {model}. "
                f"Available sizes are: {all_model_sizes}"
            )

    if "s" in all_model_sizes:
        return "s"
    elif "m" in all_model_sizes:
        return "m"
    elif "xs" in all_model_sizes:
        return "xs"
    elif "l" in all_model_sizes:
        return "l"
    elif "xl" in all_model_sizes:
        return "xl"
    else:
        raise ValueError(f"No sizes found for model {model}")


def upet_get_version_to_load(
    model: str, size: str, requested_version: Optional[Version] = None
) -> Version:
    """
    Get the version of a UPET model.

    :param model: name of the model.
    :param size: size of the model.
    :param requested_version: a requested version of the model.
    :return: the version to load.
    """
    if requested_version == "latest":
        requested_version = None

    hf_api = HfApi()
    repo_files = hf_api.list_repo_files("lab-cosmo/upet")
    files_in_models_folder = [f[7:] for f in repo_files if f.startswith("models/")]
    all_model_files = [
        f
        for f in files_in_models_folder
        if f.startswith(f"{model}-{size}-") and f.endswith(".ckpt")
    ]
    all_model_versions = [
        Version(f.split(f"{model}-{size}-")[1].split(".ckpt")[0])
        for f in all_model_files
    ]
    all_model_versions = sorted(set(all_model_versions))

    if requested_version is not None:
        if not isinstance(requested_version, Version):
            requested_version = Version(requested_version)
        if requested_version in all_model_versions:
            return requested_version
        else:
            raise ValueError(
                f"Requested version {requested_version} not available for model "
                f"{model} size {size}. Available versions are: "
                f"{list(str(v) for v in all_model_versions)}"
            )

    return max(all_model_versions)


def get_upet(
    *,
    model: Optional[str] = None,
    size: Optional[str] = None,
    version: str = "latest",
    checkpoint_path: Optional[str] = None,
) -> AtomisticModel:
    """Get a metatomic ``AtomisticModel`` for a UPET MLIP.

    :param model: name of the UPET model. Required when not using checkpoint_path,
        or when checkpoint_path has non-standard naming.
    :param size: size of the UPET model. Required when not using checkpoint_path,
        or when checkpoint_path has non-standard naming.
    :param version: version of the UPET model.
    :param checkpoint_path: path to a checkpoint file to load the model from.
        If the filename follows standard naming (e.g., "pet-mad-s-v1.0.2.ckpt"),
        model/size/version are extracted automatically.
    """
    if checkpoint_path is not None:
        # Try to parse info from checkpoint filename
        if not (model and size and version):
            model, size, version = parse_checkpoint_filename(checkpoint_path)
        logging.info(f"Loading model from checkpoint: {checkpoint_path}")
        path = checkpoint_path
    else:
        # Remote loading requires model and size
        if model is None or size is None:
            raise ValueError(
                "'model' and 'size' are required when not using checkpoint_path"
            )

        if version == "latest":
            version = upet_get_version_to_load(model, size, requested_version=version)
        if not isinstance(version, Version):
            version = Version(version)

        model_string = f"{model}-{size}-v{version}.ckpt"
        logging.info(f"Loading pre-trained model: {model_string}")
        path = hf_hub_download(
            repo_id="lab-cosmo/upet",
            filename=model_string,
            subfolder="models",
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message="PET assumes that Cartesian tensors of rank 2 are stress-like",
        )
        loaded_model = load_metatrain_model(path)

    # Generate metadata based on available info
    metadata = get_upet_metadata(model=model, size=size, version=str(version))
    return loaded_model.export(metadata)


def save_upet(
    *,
    model: Optional[str] = None,
    size: Optional[str] = None,
    version: str = "latest",
    checkpoint_path: Optional[str] = None,
    output: Optional[str] = None,
):
    """
    Save the UPET model to a TorchScript file. These files can be used with
    LAMMPS and other tools to run simulations without Python.

    :param model: name of the UPET model.
    :param size: size of the UPET model.
    :param version: UPET version to use. Defaults to the latest stable version.
    :param checkpoint_path: path to a checkpoint file to load the model from.
    :param output: path for the output model. Defaults to "{model}-{size}-v{version}.pt"
        or "model.pt" for non-standard checkpoint names.
    """
    loaded_model = get_upet(
        model=model, size=size, version=version, checkpoint_path=checkpoint_path
    )

    if output is None:
        if checkpoint_path is not None:
            parsed = parse_checkpoint_filename(checkpoint_path)
            if parsed is not None:
                model, size, version = parsed
                output = f"{model}-{size}-v{version}.pt"
            else:
                output = "model.pt"
        elif model is not None and size is not None:
            output = f"{model}-{size}-v{version}.pt"
        else:
            output = "model.pt"

    loaded_model.save(output)
    logging.info(f"Saved UPET model to {output}")


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
            f"Version {version} is not supported. Supported versions are "
            f"{PET_MAD_DOS_AVAILABLE_VERSIONS}"
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
            f"Version {version} is not supported. Supported versions are "
            f"{PET_MAD_DOS_AVAILABLE_VERSIONS}"
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
                path = hf_hub_download_url(url=url.geturl(), hf_token=None)
            else:
                # Avoid caching generic URLs due to lack of a model hash for proper
                # cache invalidation
                path, _ = urlretrieve(url=url.geturl())

    model = BandgapModel()
    model.load_state_dict(torch.load(path, weights_only=False, map_location="cpu"))
    return model
