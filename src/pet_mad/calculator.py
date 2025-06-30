import logging
import os
from typing import Optional

from metatomic.torch.ase_calculator import MetatomicCalculator
from platformdirs import user_cache_dir

from ._models import get_pet_mad


class PETMADCalculator(MetatomicCalculator):
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
        :param version: PET-MAD version to use. Supported versions are "latest", "1.1.0",
            "1.0.1", "1.0.0". Defaults to "latest".
        :param checkpoint_path: path to a checkpoint file to load the model from. If
            provided, the `version` parameter is ignored.
        :param check_consistency: should we check the model for consistency when
            running, defaults to False.
        :param device: torch device to use for the calculation. If `None`, we will try
            the options in the model's `supported_device` in order.
        """

        model = get_pet_mad(version=version, checkpoint_path=checkpoint_path)

        cache_dir = user_cache_dir("pet-mad", "metatensor")
        os.makedirs(cache_dir, exist_ok=True)

        extensions_directory = None
        if version == "1.0.0":
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
            extensions_directory=extensions_directory,
            check_consistency=check_consistency,
            device=device,
            non_conservative=False,
            additional_outputs={},
        )
