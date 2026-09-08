import logging
from typing import Optional
from urllib.parse import urlparse
from urllib.request import urlretrieve

import torch
import torch.nn as nn
from metatomic.torch import AtomisticModel
from metatrain.utils.io import load_model as load_metatrain_model
from packaging.version import Version

from ..._metadata import get_pet_mad_dos_metadata
from ..._version import (
    PET_MAD_DOS_AVAILABLE_VERSIONS,
    PET_MAD_DOS_LATEST_STABLE_VERSION,
)
from ...utils import hf_hub_download_url


class CNNModel(nn.Module):
    """
    A minimalistic CNN model for predicting the bandgap or Fermi level from the
    electronic density of states.
    """

    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=64,
            kernel_size=32,
            stride=1,
            dilation=1,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=64, kernel_size=16, dilation=1, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=64, out_channels=64, kernel_size=8, dilation=1, padding=1
        )
        self.conv4 = nn.Conv1d(
            in_channels=64, out_channels=64, kernel_size=8, dilation=1, padding=1
        )
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.fc1 = nn.Linear(1024, 1024)  # Adjust size after pooling
        self.fc2 = nn.Linear(1024, 1)
        self.silu = nn.SiLU()

    def forward(self, x, last_layer=False):
        x = self.silu(self.conv1(x))  # Conv1 + ReLU
        x = self.pool(x)  # MaxPooling1D
        x = self.silu(self.conv2(x))  # Conv2 + ReLU
        x = self.pool(x)  # MaxPooling1D
        x = self.silu(self.conv3(x))  # Conv2 + ReLU
        x = self.pool(x)  # MaxPooling1D
        x = self.silu(self.conv4(x))  # Conv2 + ReLU
        x = self.pool(x)  # MaxPooling1D
        x = x.view(x.size(0), -1)  # Flatten
        x = self.silu(self.fc1(x))  # Fully Connected Layer 1
        output = self.fc2(x)  # Fully Connected Layer 2 (Output)
        if last_layer:
            return x
        else:
            return output


BASE_URL_PET_MAD_DOS = "https://huggingface.co/lab-cosmo/pet-mad-dos/resolve/{tag}/models/pet-mad-dos-{version}.ckpt"
BASE_URL_BANDGAP_MODEL = (
    "https://huggingface.co/lab-cosmo/pet-mad-dos/resolve/{tag}/models/bandgap-model.pt"
)
BASE_URL_FERMI_MODEL = (
    "https://huggingface.co/lab-cosmo/pet-mad-dos/resolve/{tag}/models/fermi-model.pt"
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
        print(f"Loading PET-MAD-DOS model from checkpoint: {model_path}")
        path = model_path
    else:
        print(f"Downloading PET-MAD-DOS model version: {version}")
        path = BASE_URL_PET_MAD_DOS.format(tag="main", version=f"v{version}")

    model = load_metatrain_model(path)
    metadata = get_pet_mad_dos_metadata(version)
    exported_model = model.export(metadata)
    return exported_model


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
        path = BASE_URL_BANDGAP_MODEL.format(tag="main")
        path = str(path)
        url = urlparse(path)

        if url.scheme:
            if url.netloc == "huggingface.co":
                path = hf_hub_download_url(url=url.geturl(), hf_token=None)
            else:
                # Avoid caching generic URLs due to lack of a model hash for proper
                # cache invalidation
                path, _ = urlretrieve(url=url.geturl())

    model = CNNModel()
    model.load_state_dict(torch.load(path, weights_only=False, map_location="cpu"))
    return model


def _get_fermi_model(version: str = "latest", model_path: Optional[str] = None):
    """
    Get a Fermi level model for PET-MAD-DOS
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
            f"Loading the PET-MAD-DOS Fermi level model from checkpoint: {model_path}"
        )
        path = model_path
    else:
        logging.info(f"Downloading Fermi level model version: {version}")
        # Set to main for now until the next version gets published
        path = BASE_URL_FERMI_MODEL.format(tag="main")
        path = str(path)
        url = urlparse(path)

        if url.scheme:
            if url.netloc == "huggingface.co":
                path = hf_hub_download_url(url=url.geturl(), hf_token=None)
            else:
                # Avoid caching generic URLs due to lack of a model hash for proper
                # cache invalidation
                path, _ = urlretrieve(url=url.geturl())

    model = CNNModel()
    model.load_state_dict(torch.load(path, weights_only=False, map_location="cpu"))
    return model
