import logging
import os
from typing import Optional, Tuple, List

from metatomic.torch.ase_calculator import MetatomicCalculator
from platformdirs import user_cache_dir
from metatomic.torch import ModelOutput
import torch
import numpy as np
from ase import Atoms

from packaging.version import Version

from ._models import get_pet_mad, get_pet_mad_dos, _get_bandgap_model
from ._version import LATEST_VERSION, LATEST_PET_MAD_DOS_VERSION
from .utils import get_num_electrons


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
        )


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
            version = Version(LATEST_PET_MAD_DOS_VERSION)
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

    def calculate_bandgap(self, atoms: Atoms | List[Atoms]) -> torch.Tensor:
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
        self, atoms: Atoms | List[Atoms], per_atom: bool = False
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

    def calculate_efermi(self, atoms: Atoms | List[Atoms]) -> float:
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
