import logging
import os
from typing import Optional

from metatomic.torch.ase_calculator import MetatomicCalculator
from metatomic.torch import ModelOutput
from platformdirs import user_cache_dir

from packaging.version import Version

from ._models import get_pet_mad
from ._version import LATEST_VERSION


class PETMADCalculator(MetatomicCalculator):
    """
    PET-MAD ASE Calculator
    """

    def __init__(
        self,
        version: str = "latest",
        checkpoint_path: Optional[str] = None,
        *,
        check_consistency: bool = False,
        device: Optional[str] = None,
        non_conservative: bool = False,
        do_gradients_with_energy: bool = True,
    ):
        """
        :param version: PET-MAD version to use. Supported versions are
            "1.1.0", "1.0.1", "1.0.0". Defaults to latest available version.
        :param checkpoint_path: path to a checkpoint file to load the model from. If
            provided, the `version` parameter is ignored.
        :param check_consistency: should we check the model for consistency when
            running, defaults to False.
        :param device: torch device to use for the calculation. If `None`, we will try
            the options in the model's `supported_device` in order.
        :param non_conservative: whether to use the non-conservative regime of forces
            and stresses prediction. Defaults to False. Only available for PET-MAD
            version 1.1.0 or higher.
        :param do_gradients_with_energy: whether to compute gradients with respect to
            the energy. Defaults to True.

        """

        if version == "latest":
            version = Version(LATEST_VERSION)
        if not isinstance(version, Version):
            version = Version(version)

        model = get_pet_mad(version=version, checkpoint_path=checkpoint_path)

        cache_dir = user_cache_dir("pet-mad", "metatensor")
        os.makedirs(cache_dir, exist_ok=True)

        extensions_directory = None
        if version == Version("1.0.0"):
            extensions_directory = "extensions"

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
            additional_outputs={},
            extensions_directory=extensions_directory,
            check_consistency=check_consistency,
            device=device,
            non_conservative=non_conservative,
            do_gradients_with_energy=do_gradients_with_energy,
        )