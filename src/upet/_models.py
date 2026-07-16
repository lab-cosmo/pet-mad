import logging
import os
import re
import warnings
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import torch
from huggingface_hub import HfApi, hf_hub_download
from metatomic.torch import AtomisticModel
from metatrain.utils.io import load_model as load_metatrain_model
from packaging.version import Version

from ._metadata import get_upet_metadata
from ._version import DEPRECATED_MODELS


CHECKPOINT_NAME_PATTERN = re.compile(
    r"^(?P<model>pet-[\w]+)-(?P<size>xs|s|m|l|xl)-v?(?P<version>\d+\.\d+\.\d+)\.ckpt$"
)


@lru_cache(maxsize=1)
def _get_upet_repo_files() -> List[str]:
    """Cached listing of files in the lab-cosmo/upet repository models folder."""
    hf_api = HfApi()
    repo_files = hf_api.list_repo_files("lab-cosmo/upet")
    return [f[7:] for f in repo_files if f.startswith("models/")]


def get_available_models() -> List[str]:
    """Get all available base model names from the HuggingFace repository.

    :return: Sorted list of base model names (e.g., ["pet-mad", "pet-omat", ...])
    """
    files = _get_upet_repo_files()
    models = set()
    for f in files:
        match = CHECKPOINT_NAME_PATTERN.match(f)
        if match:
            models.add(match.group("model"))
    return sorted(models)


def get_sizes_for_model(model: str) -> List[str]:
    """Get all available sizes for a given model from the cached repo files.

    :param model: Base model name (e.g., "pet-mad", "pet-omat")
    :return: Sorted list of available sizes
    """
    files = _get_upet_repo_files()
    prefix = f"{model}-"
    model_files = [f for f in files if f.startswith(prefix) and f.endswith(".ckpt")]
    sizes = [f.split(prefix)[1].split("-")[0] for f in model_files]
    return sorted(set(sizes))


def get_versions_for_model(model: str, size: str) -> List[Version]:
    """Get all available versions for a given model/size from the cached repo files.

    :param model: Base model name (e.g., "pet-mad", "pet-omat")
    :param size: Model size (e.g., "s", "m", "l")
    :return: Sorted list of available versions
    """
    files = _get_upet_repo_files()
    prefix = f"{model}-{size}-"
    model_files = [f for f in files if f.startswith(prefix) and f.endswith(".ckpt")]
    versions = [Version(f.split(prefix)[1].split(".ckpt")[0]) for f in model_files]
    return sorted(set(versions))


def parse_checkpoint_filename(
    path: str,
) -> Union[Tuple[str, str, Version], Tuple[None, None, None]]:
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


def upet_resolve_model(
    model: str,
    requested_size: Optional[str] = None,
    requested_version: Optional[str] = None,
) -> Tuple[str, Version]:
    """
    Resolve size and version for a UPET model in a single operation.

    :param model: Base model name (e.g., "pet-mad", "pet-omat")
    :param requested_size: Specific size to use, or None for default
    :param requested_version: Specific version or "latest"/None for newest
    :return: Tuple of (size, version)
    """
    all_model_sizes = get_sizes_for_model(model)

    # Resolve size
    if requested_size is not None:
        if requested_size in all_model_sizes:
            size = requested_size
        else:
            raise ValueError(
                f"Requested size {requested_size} not available for model {model}. "
                f"Available sizes are: {all_model_sizes}"
            )
    elif "s" in all_model_sizes:
        size = "s"
    elif "m" in all_model_sizes:
        size = "m"
    elif "xs" in all_model_sizes:
        size = "xs"
    elif "l" in all_model_sizes:
        size = "l"
    elif "xl" in all_model_sizes:
        size = "xl"
    else:
        raise ValueError(f"No sizes found for model {model}")

    all_model_versions = get_versions_for_model(model, size)

    # Resolve version
    if requested_version is None or requested_version == "latest":
        version = max(all_model_versions)
    else:
        if not isinstance(requested_version, Version):
            requested_version = Version(requested_version)
        if requested_version in all_model_versions:
            version = requested_version
        else:
            raise ValueError(
                f"Requested version {requested_version} not available for model "
                f"{model} size {size}. Available versions are: "
                f"{list(str(v) for v in all_model_versions)}"
            )

    return size, version


def _resolve_and_download_checkpoint(
    model: str,
    size: Optional[str] = None,
    version: Optional[Union[str, Version]] = "latest",
) -> Tuple[str, Version, str]:
    """
    Resolve size/version for a UPET model and download its checkpoint from
    the ``lab-cosmo/upet`` HuggingFace repository, caching it locally.

    Shared by :func:`_get_upet_exported_atomistic_model` and
    :class:`upet.nvalchemi.UPETWrapper`'s ``from_checkpoint``.

    :param model: Base model name (e.g., "pet-mad", "pet-omat")
    :param size: Specific size to use, or None for the default
    :param version: Specific version, or "latest"/None for the newest
    :return: Tuple of (resolved size, resolved version, local checkpoint path)
    """
    # Resolve size and version via upet_resolve_model, which handles
    # defaults (prefers 's'), validates against available checkpoints,
    # and correctly resolves "latest" to an actual version number.
    # Previously, passing version="latest" would reach hf_hub_download
    # as None and produce a broken "pet-mad-s-vNone" filename.
    requested_version = (
        None if (version is None or version == "latest") else str(version)
    )
    size, version = upet_resolve_model(
        model=model,
        requested_size=size,
        requested_version=requested_version,
    )

    model_name = f"{model}-{size}-v{version}"
    if model_name in DEPRECATED_MODELS:
        warn_msg = (
            f"Model {model_name} is deprecated and may not be supported in "
            "future versions. Please switch to a newer model for better "
            "performance and support."
        )
        warnings.warn(warn_msg, category=DeprecationWarning, stacklevel=2)

    model_string = f"{model_name}.ckpt"
    logging.info(f"Loading pre-trained model: {model_string}")
    path = hf_hub_download(
        repo_id="lab-cosmo/upet",
        filename=model_string,
        subfolder="models",
    )
    return size, version, path


def _get_upet_exported_atomistic_model(
    model: Optional[str] = None,
    size: Optional[str] = None,
    version: Optional[Union[str, Version]] = "latest",
    checkpoint_path: Optional[str] = None,
) -> AtomisticModel:
    """
    Internal helper to load a UPET AtomisticModel without caching or TorchScript.

    This function is separate from get_upet() to allow for caching and post-processing
    in the public API function.
    """
    if checkpoint_path is not None:
        # Try to parse info from checkpoint filename
        model, size, version = parse_checkpoint_filename(checkpoint_path)
        logging.info(f"Loading model from checkpoint: {checkpoint_path}")
        path = checkpoint_path
    else:
        if model is None:
            raise ValueError("'model' is required when not using checkpoint_path")

        size, version, path = _resolve_and_download_checkpoint(model, size, version)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            message="PET assumes that Cartesian tensors of rank 2 are stress-like",
        )
        loaded_model = load_metatrain_model(path)

    metadata = get_upet_metadata(model=model, size=size, version=str(version))
    exported_model = loaded_model.export(metadata)
    return exported_model


def get_upet(
    *,
    model: Optional[str] = None,
    size: Optional[str] = None,
    version: Optional[Union[str, Version]] = "latest",
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
        model/size/version are extracted automatically, while the `model` and
        `version` parameters are ignored.
    """
    exported_model = _get_upet_exported_atomistic_model(
        model=model, size=size, version=version, checkpoint_path=checkpoint_path
    )

    # TorchScript the model
    for parameter in exported_model.parameters():
        parameter.requires_grad = False
    exported_model = exported_model.eval()
    exported_model = torch.jit.script(exported_model)
    return exported_model


def save_upet(
    *,
    model: Optional[str] = None,
    size: Optional[str] = None,
    version: Optional[str] = "latest",
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
    exported_model = _get_upet_exported_atomistic_model(
        model=model, size=size, version=version, checkpoint_path=checkpoint_path
    )

    if output is None:
        if checkpoint_path is not None:
            model, size, version = parse_checkpoint_filename(checkpoint_path)
            if model and size and version:
                output = f"{model}-{size}-v{version}.pt"
            else:
                output = "model.pt"
        elif model and size:
            output = f"{model}-{size}-v{version}.pt"
        else:
            output = "model.pt"

    exported_model.to("cpu").save(output)
    logging.info(f"Saved UPET model to {output}")


def list_upet(
    *,
    model: Optional[str] = None,
    size: Optional[str] = None,
    print_summary: bool = True,
) -> List[dict]:
    """List available UPET models, sizes, and versions.

    When called without arguments, returns all available model/size/version
    combinations. When ``model`` is given, filters to that model. When both
    ``model`` and ``size`` are given, filters to that specific combination.

    :param model: Base model name (e.g., "pet-mad", "pet-omat"). If ``None``,
        lists all available models.
    :param size: Model size (e.g., "s", "m", "l"). If ``None`` and ``model`` is
        given, lists all sizes for that model.
    :param print_summary: Whether to print a human-readable summary to stdout.
        Defaults to ``True``.
    :return: A list of dictionaries, each with keys ``"model"``, ``"size"``,
        and ``"version"``.
    """
    if model is None:
        models = get_available_models()
    else:
        models = [model]

    result = []
    for m in models:
        if size is None:
            sizes = get_sizes_for_model(m)
        else:
            sizes = [size]
        for s in sizes:
            for v in get_versions_for_model(m, s):
                result.append({"model": m, "size": s, "version": str(v)})

    if print_summary:
        if not result:
            print("No UPET models found.")
        else:
            print("Available UPET models:")
            for entry in result:
                print(f"  - {entry['model']}-{entry['size']} v{entry['version']}")

    return result
