import logging
import os
from typing import Optional, Tuple, List, Union

from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator
from platformdirs import user_cache_dir
import torch
import numpy as np
from ase import Atoms

from packaging.version import Version

from ._models import get_pet_mad, get_pet_mad_dos, _get_bandgap_model
from .utils import get_num_electrons
from ._version import (
    PET_MAD_LATEST_STABLE_VERSION,
    PET_MAD_UQ_AVAILABILITY_VERSION,
    PET_MAD_NC_AVAILABILITY_VERSION,
    PET_MAD_DOS_LATEST_STABLE_VERSION,
)


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
    ):
        """
        :param version: PET-MAD version to use. Defaults to the latest stable version.
        :param checkpoint_path: path to a checkpoint file to load the model from. If
            provided, the `version` parameter is ignored.
        :param calculate_uncertainty: whether to calculate energy uncertainty.
            Defaults to False. Only available for PET-MAD version 1.0.2.
        :param calculate_ensemble: whether to calculate energy ensemble.
            Defaults to False. Only available for PET-MAD version 1.0.2.
        :param check_consistency: should we check the model for consistency when
            running, defaults to False.
        :param device: torch device to use for the calculation. If `None`, we will try
            the options in the model's `supported_device` in order.
        :param non_conservative: whether to use the non-conservative regime of forces
            and stresses prediction. Defaults to False. Only available for PET-MAD
            version 1.1.0 or higher.

        """

        if version == "latest":
            version = Version(PET_MAD_LATEST_STABLE_VERSION)
        if not isinstance(version, Version):
            version = Version(version)

        if non_conservative and version < Version(PET_MAD_NC_AVAILABILITY_VERSION):
            raise NotImplementedError(
                f"Non-conservative forces and stresses are not available for version {version}. "
                f"Please use PET-MAD version {PET_MAD_NC_AVAILABILITY_VERSION} or higher."
            )

        additional_outputs = {}
        if calculate_uncertainty or calculate_ensemble:
            if version < Version(PET_MAD_UQ_AVAILABILITY_VERSION):
                raise NotImplementedError(
                    f"Energy uncertainty and ensemble are not available for version {version}. "
                    f"Please use PET-MAD version {PET_MAD_UQ_AVAILABILITY_VERSION} or higher, "
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
        )

    def _get_uq_output(self, output_name: str):
        if output_name not in self.additional_outputs:
            quantity = output_name.split("_")[1]
            raise ValueError(
                f"Energy {quantity} is not available. Please make sure that you have initialized the "
                f"calculator with `calculate_{quantity}=True` and performed evaluation. "
                f"This option is only available for PET-MAD version {PET_MAD_UQ_AVAILABILITY_VERSION} or higher."
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


ENERGY_LOWER_BOUND = -159.6456
ENERGY_UPPER_BOUND = 79.1528 + 1.5
ENERGY_INTERVAL = 0.05


class PETMADDOSCalculator(MetatomicCalculator):
    """
    PET-MAD DOS Calculator
    """

    def __init__(
        self,
        version: str = "latest",
        model_path: Optional[str] = None,
        bandgap_model_path: Optional[str] = None,
        *,
        check_consistency: bool = False,
        device: Optional[str] = None,
    ):
        if version == "latest":
            version = Version(PET_MAD_DOS_LATEST_STABLE_VERSION)
        if not isinstance(version, Version):
            version = Version(version)

        model = get_pet_mad_dos(version=version, model_path=model_path)
        bandgap_model = _get_bandgap_model(
            version=version, model_path=bandgap_model_path
        )

        super().__init__(
            model,
            additional_outputs={},
            check_consistency=check_consistency,
            device=device,
        )
        self._bandgap_model = bandgap_model

        n_points = np.ceil((ENERGY_UPPER_BOUND - ENERGY_LOWER_BOUND) / ENERGY_INTERVAL)
        self.energy_grid = torch.arange(n_points) * ENERGY_INTERVAL + ENERGY_LOWER_BOUND

    def calculate_bandgap(self, atoms: Union[Atoms, List[Atoms]]) -> torch.Tensor:
        """
        Calculate the bandgap for a given ase.Atoms object,
        or a list of ase.Atoms objects.

        :param atoms: ASE atoms object or a list of ASE atoms objects
        :return: bandgap values for each atoms object.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        _, dos = self.calculate_dos(atoms, per_atom=False)
        num_atoms = torch.tensor([len(item) for item in atoms], device=dos.device)
        dos = dos / num_atoms.unsqueeze(1)
        bandgap = self._bandgap_model(
            dos.unsqueeze(1)
        ).detach()  # Need to make the inputs [n_predictions, 1, 4806]
        bandgap = torch.nn.functional.relu(bandgap).squeeze()
        return bandgap

    def calculate_dos(
        self, atoms: Union[Atoms, List[Atoms]], per_atom: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the density of states for a given ase.Atoms object,
        or a list of ase.Atoms objects.

        :param atoms: ASE atoms object or a list of ASE atoms objects
        :param per_atom: Whether to return the density of states per atom.
        :return: Energy grid and corresponding DOS values in torch.Tensor format.
        """
        results = self.run_model(
            atoms, outputs={"mtt::dos": ModelOutput(per_atom=per_atom)}
        )
        dos = results["mtt::dos"].block().values
        return self.energy_grid, dos

    def calculate_efermi(self, atoms: Union[Atoms, List[Atoms]]) -> float:
        """
        Get the Fermi energy for a given ase.Atoms object,
        or a list of ase.Atoms objects, based on a predicted
        density of states.

        :param atoms: ASE atoms object or a list of ASE atoms objects
        :return: Fermi energy.
        """
        _, dos = self.calculate_dos(atoms, per_atom=False)
        cdos = torch.cumulative_trapezoid(dos, dx=ENERGY_INTERVAL)
        num_electrons = get_num_electrons(atoms)
        num_electrons.to(dos.device)
        efermi_indices = torch.argmax(
            (cdos > num_electrons.unsqueeze(1)).float(), dim=1
        )
        efermi = self.energy_grid[efermi_indices]
        return efermi
