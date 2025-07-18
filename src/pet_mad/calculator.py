import logging
import os
from typing import Optional

from metatomic.torch.ase_calculator import MetatomicCalculator
from metatomic.torch import ModelOutput
from platformdirs import user_cache_dir

from packaging.version import Version

from ._models import get_pet_mad

LATEST_VERSION = Version("1.1.0")
UQ_AVAILABILITY_VERSION = Version("1.2.0rc1")


class PETMADCalculator(MetatomicCalculator):
    """
    PET-MAD ASE Calculator
    """

    def __init__(
        self,
        version: str = "latest",
        checkpoint_path: Optional[str] = None,
        calculate_uncertainty: bool = False,
        calculate_ensemble: bool = False,
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
        :param calculate_uncertainty: whether to calculate energy uncertainty.
            Defaults to False. Only available for PET-MAD version 1.2.0 or higher.
        :param calculate_ensemble: whether to calculate energy ensemble.
            Defaults to False. Only available for PET-MAD version 1.2.0 or higher.
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
            version = LATEST_VERSION
        if not isinstance(version, Version):
            version = Version(version)

        additional_outputs = {}
        if calculate_uncertainty or calculate_ensemble:
            if version < UQ_AVAILABILITY_VERSION:
                raise NotImplementedError(
                    f"Energy uncertainty and ensemble are not available for version {version}. "
                    f"Please use PET-MAD version {UQ_AVAILABILITY_VERSION} or higher, "
                    f"or disable the calculation of energy uncertainty and energy ensemble."
                )
            else:
                if calculate_uncertainty:
                    additional_outputs["energy_uncertainty"] = ModelOutput(
                        quantity="energy", unit="eV", per_atom=False
                    )
                if calculate_ensemble:
                    additional_outputs["energy_ensemble"] = ModelOutput(
                        quantity="energy", unit="eV", per_atom=False
                    )

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
            additional_outputs=additional_outputs,
            extensions_directory=extensions_directory,
            check_consistency=check_consistency,
            device=device,
            non_conservative=non_conservative,
            do_gradients_with_energy=do_gradients_with_energy,
        )

    def _get_uq_output(self, output_name: str):
        if output_name not in self.additional_outputs:
            quantity = output_name.split("_")[1]
            raise ValueError(
                f"Energy {quantity} is not available. Please make sure that you have initialized the "
                f"calculator with `calculate_{quantity}=True` and performed evaluation. "
                f"This option is only available for PET-MAD version {UQ_AVAILABILITY_VERSION} or higher."
            )
        return (
            self.additional_outputs[output_name]
            .block()
            .values.detach()
            .numpy()
            .squeeze()
        )

    def get_energy_uncertainty(self):
        return self._get_uq_output("energy_uncertainty")

    def get_energy_ensemble(self):
        return self._get_uq_output("energy_ensemble")
