import logging
import os
from typing import List, Optional, Tuple, Union

import ase.calculators.calculator
import numpy as np
import torch
from ase import Atoms
from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator, SymmetrizedCalculator
from packaging.version import Version
from platformdirs import user_cache_dir

from ._models import (
    _get_bandgap_model,
    get_pet_mad_dos,
    get_upet,
    upet_get_size_to_load,
    upet_get_version_to_load,
)
from ._version import (
    PET_MAD_DOS_LATEST_STABLE_VERSION,
    UPET_AVAILABLE_MODELS,
)
from .utils import (
    fermi_dirac_distribution,
    get_num_electrons,
)


STR_TO_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
}
DTYPE_TO_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
}


class UPETCalculator(ase.calculators.calculator.Calculator):
    """
    ASE Calculator for universal MLIPs based on the PET architecture.
    """

    def __init__(
        self,
        model: str,
        version: str = "latest",
        dtype: Optional[torch.dtype] = None,
        checkpoint_path: Optional[str] = None,
        calculate_uncertainty: bool = False,
        calculate_ensemble: bool = False,
        rotational_average_order: Optional[int] = None,
        rotational_average_batch_size: Optional[int] = None,
        *,
        device: Optional[str] = None,
        non_conservative: bool = False,
        check_consistency: bool = False,
    ):
        """
        :param model: PET-MLIP model to use. Can be one of the following:
            - "pet-mad-s": PET-MAD model (size "s', materials and molecules, PBEsol)
            - "pet-omad-l": PET-OMAD model (size "l", materials and molecules, PBEsol,
                slower and more accurate)
            - "pet-omat-xs": PET-OMat model (size "xs", materials, PBE)
            - "pet-omat-s": PET-OMat model (size "s", materials, PBE)
            - "pet-omat-m": PET-OMat model (size "m", materials, PBE)
            - "pet-omat-l": PET-OMat model (size "l", materials, PBE)
            - "pet-omat-xl": PET-OMat model (size "xl", materials, PBE)
            - "pet-oam-l": PET-OAM model (size "l", materials,
                Materials-Project-consistent PBE)
            - "pet-oam-xl": PET-OAM model (size "xl", materials,
                Materials-Project-consistent PBE)
            - "pet-omatpes-l": PET-OMATPES model (size "l", materials, r2SCAN)
            - "pet-spice-s": PET-SPICE model (size "s", molecules, ωB97M-D3)
            - "pet-spice-l": PET-SPICE model (size "l", molecules, ωB97M-D3)
        :param version: version of the model to use. Defaults to the latest stable
            version.
        :param dtype: dtype to use for the calculations. If `None`, we will use the
            default dtype.
        :param checkpoint_path: checkpoint path to a checkpoint file to load the model
            from. Mainly designed for loading fine-tuned models.
        :param calculate_uncertainty: whether to calculate energy uncertainty.
            Defaults to False. Only available for PET-MAD version 1.0.2.
        :param calculate_ensemble: whether to calculate energy ensemble.
            Defaults to False. Only available for PET-MAD version 1.0.2.
        :param rotational_average_order: order of the Lebedev-Laikov grid used for
            averaging the prediction over rotations.
        :param rotational_average_num_additional_rotations: the number of additional
            rotations sampled from 0 to 2pi angle applied on top of the each
            Lebedev-Laikov rotation vector when performing rotational averaging.
            Defaults to 1, which means that by default only the Lebedev-Laikov grid
            is used for rotational averaging.
        :param rotational_average_batch_size: batch size to use for the rotational
            averaging. If `None`, all rotations will be computed at once.
        :param device: torch device to use for the calculation. If `None`, we will try
            the options in the model's `supported_device` in order.
        :param non_conservative: whether to use the non-conservative regime of forces
            and stresses prediction. Defaults to False. Available for all models,
            except:
                - PET-MAD models with version < 1.1.0
                - PET-SPICE models
        :param check_consistency: whether internal consistency checks should be
            performed. Mainly for developers, defaults to False.
        """
        super().__init__()

        if model.lower() not in UPET_AVAILABLE_MODELS:
            raise ValueError(
                f"Model {model} is not available. Please select one of the following: "
                f"{UPET_AVAILABLE_MODELS}"
            )
        model, size = model.rsplit("-", 1)
        size = upet_get_size_to_load(model, requested_size=size)
        version = upet_get_version_to_load(model, size, requested_version=version)

        if not isinstance(version, Version):
            version = Version(version)

        loaded_model = get_upet(
            model=model,
            size=size,  # type: ignore
            version=version,
            checkpoint_path=checkpoint_path,
        )

        model_outputs = loaded_model.capabilities().outputs
        if non_conservative:
            # Check for "non_conservative_{forces/stress} availability"
            if (
                "non_conservative_forces" not in model_outputs
                or "non_conservative_stress" not in model_outputs
            ):
                raise NotImplementedError(
                    "Non-conservative forces and stresses are not available for the "
                    f"model {model.lower()}. Please check the documentation of this "
                    "class for more information."
                )
        if calculate_uncertainty or calculate_ensemble:
            if (
                "energy_uncertainty" not in model_outputs
                or "energy_ensemble" not in model_outputs
            ):
                raise NotImplementedError(
                    "Energy uncertainty and ensemble are not available for the "
                    f"model {model.lower()}. Please check the documentation of this "
                    "class for more information."
                )
            self._uq_is_available = True
        else:
            self._uq_is_available = False

        if dtype is not None:
            if isinstance(dtype, str):
                assert dtype in STR_TO_DTYPE, f"Invalid dtype: {dtype}"
                dtype = STR_TO_DTYPE[dtype]
            loaded_model._capabilities.dtype = DTYPE_TO_STR[dtype]
            loaded_model = loaded_model.to(dtype=dtype, device=device)

        cache_dir = user_cache_dir("upet", "metatensor")
        os.makedirs(cache_dir, exist_ok=True)

        pt_path = cache_dir + f"/{model}-{size}-v{version}.pt"
        logging.info(f"Exporting checkpoint to TorchScript at {pt_path}")
        loaded_model.save(pt_path, collect_extensions=None)

        self.calculator = MetatomicCalculator(
            pt_path,
            extensions_directory=None,
            check_consistency=check_consistency,
            device=device,
            non_conservative=non_conservative,
        )
        self.implemented_properties = self.calculator.implemented_properties

        if rotational_average_order is not None:
            self.calculator = SymmetrizedCalculator(
                self.calculator,
                l_max=rotational_average_order,
                batch_size=rotational_average_batch_size,
                store_rotational_std=True,
            )

    def calculate(
        self, atoms: Atoms, properties: List[str], system_changes: List[str]
    ) -> None:
        """
        Compute some ``properties`` with this calculator, and return them in the format
        expected by ASE.

        This is not intended to be called directly by users, but to be an implementation
        detail of ``atoms.get_energy()`` and related functions. See
        :py:meth:`ase.calculators.calculator.Calculator.calculate` for more information.

        If the `rotational_average_order` parameter is set during initialization, the
        prediction will be averaged over unique rotations in the Lebedev-Laikov grid of
        a chosen order.

        If the `rotational_average_batch_size` parameter is set during initialization,
        averaging will be performed in batches of the given size to avoid out of memory
        errors.
        """

        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )

        self.calculator.calculate(atoms, properties, system_changes)
        self.results = self.calculator.results

    def get_energy_uncertainty(
        self, atoms: Optional[Atoms] = None, per_atom: bool = False
    ) -> np.ndarray:
        """
        Get the energy uncertainty for a given :py:class:`ase.Atoms` object.

        :param atoms: ASE atoms object. If ``None``, the last calculated atoms will be
            used.
        :param per_atom: Whether to return the energy uncertainty per atom.
        :return: Energy uncertainty in numpy.ndarray format.
        """
        if atoms is None:
            if self.atoms is None:
                raise ValueError(
                    "No `atoms` provided and no previously calculated atoms found."
                )
            else:
                atoms = self.atoms

        outputs = self.calculator.run_model(
            atoms,
            outputs={
                # TODO: handle variants if we have a a model with them
                "energy_uncertainty": ModelOutput(
                    quantity="energy", unit="eV", per_atom=per_atom
                )
            },
        )

        return outputs["energy_uncertainty"].block().values.detach().cpu().numpy()

    def get_energy_ensemble(
        self, atoms: Optional[Atoms] = None, per_atom: bool = False
    ) -> np.ndarray:
        """
        Get the ensemble of energies for a given :py:class:`ase.Atoms` object.

        :param atoms: ASE atoms object. If ``None``, the last calculated atoms will be
            used.
        :param per_atom: Whether to return the energies per atom.
        :return: Energy uncertainty in numpy.ndarray format.
        """

        if atoms is None:
            if self.atoms is None:
                raise ValueError(
                    "No `atoms` provided and no previously calculated atoms found."
                )
            else:
                atoms = self.atoms

        outputs = self.calculator.run_model(
            atoms,
            outputs={
                # TODO: handle variants if we have a a model with them
                "energy_ensemble": ModelOutput(
                    quantity="energy", unit="eV", per_atom=per_atom
                )
            },
        )

        return outputs["energy_ensemble"].block().values.detach().cpu().numpy()


ENERGY_LOWER_BOUND = -159.6456  # Lower bound of the energy grid for DOS
ENERGY_UPPER_BOUND = 79.1528 + 1.5  # Upper bound of the energy grid for DOS
ENERGY_INTERVAL = 0.05  # Interval of the energy grid for DOS

# If we want to calculate the Fermi level at a given temperature, we need to search
# it around the Fermi level at 0 K. To do this, we first set a certain energy window
# with a certain number of grid points to calculate the integrated DOS. Next, we
# interpolate the integrated DOS to a finer grid and find the Fermi level that
# gives the correct number of electrons.
ENERGY_WINDOW = 0.5
ENERGY_GRID_NUM_POINTS_COARSE = 1000
ENERGY_GRID_NUM_POINTS_FINE = 10000


class PETMADDOSCalculator:
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
        """
        :param version: PET-MAD-DOS version to use. Defaults to the latest stable
            version.
        :param model_path: path to a Torch-Scripted model file to load the model from.
            If provided, the `version` parameter is ignored.
        :param bandgap_model_path: path to a PyTorch checkpoint file with the bandgap
            model. If provided, the `version` parameter is ignored.
        :param check_consistency: should we check the model for consistency when
            running, defaults to False.
        :param device: torch device to use for the calculation. If `None`, we will try
            the options in the model's `supported_device` in order.

        """
        if version == "latest":
            version = Version(PET_MAD_DOS_LATEST_STABLE_VERSION)
        if not isinstance(version, Version):
            version = Version(version)

        model = get_pet_mad_dos(version=version, model_path=model_path)
        bandgap_model = _get_bandgap_model(
            version=version, model_path=bandgap_model_path
        )

        self.calculator = MetatomicCalculator(
            model,
            additional_outputs={},
            check_consistency=check_consistency,
            device=device,
        )
        self._bandgap_model = bandgap_model

        n_points = np.ceil((ENERGY_UPPER_BOUND - ENERGY_LOWER_BOUND) / ENERGY_INTERVAL)
        self._energy_grid = (
            torch.arange(n_points) * ENERGY_INTERVAL + ENERGY_LOWER_BOUND
        )

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
        results = self.calculator.run_model(
            atoms, outputs={"mtt::dos": ModelOutput(per_atom=per_atom)}
        )
        dos = results["mtt::dos"].block().values
        return self._energy_grid.clone(), dos

    def calculate_bandgap(
        self, atoms: Union[Atoms, List[Atoms]], dos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate the bandgap for a given ase.Atoms object,or a list of ase.Atoms
        objects. By default, the density of states is first calculated using the
        `calculate_dos` method, and the the bandgap is derived from the DOS by a
        BandgapModel. Alternatively, the density of states can be provided as an
        input parameter to avoid re-calculating the DOS.

        :param atoms: ASE atoms object or a list of ASE atoms objects
        :param dos: Density of states for the given atoms. If not provided, the
            density of states is calculated using the `calculate_dos` method.
        :return: bandgap values for each ase.Atoms object object stored in a
            torch.Tensor format.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        if dos is None:
            _, dos = self.calculate_dos(atoms, per_atom=False)
        if dos.shape[0] != len(atoms):
            raise ValueError(
                f"The provided DOS is inconsistent with the provided `atoms` "
                f"parameter: {len(atoms)} != {dos.shape[0]}. Please either set "
                "`dos = None` or provide a consistent DOS, computed with "
                "`per_atom = False`."
            )
        num_atoms = torch.tensor([len(item) for item in atoms], device=dos.device)
        dos = dos / num_atoms.unsqueeze(1)
        bandgap = self._bandgap_model(
            dos.unsqueeze(1)
        ).detach()  # Need to make the inputs [n_predictions, 1, 4806]
        bandgap = torch.nn.functional.relu(bandgap).squeeze()
        return bandgap

    def calculate_efermi(
        self,
        atoms: Union[Atoms, List[Atoms]],
        dos: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """
        Get the Fermi energy for a given ase.Atoms object, or a list of ase.Atoms
        objects, based on a predicted density of states at a given temperature.
        By default, the density of states is first calculated using the `calculate_dos`
        method, and the Fermi level is calculated at T=0 K. Alternatively, the density
        of states can be provided as an input parameter to avoid re-calculating the DOS.

        :param atoms: ASE atoms object or a list of ASE atoms objects
        :param dos: Density of states for the given atoms. If not provided, the
            density of states is calculated using the `calculate_dos` method.
        :param temperature: Temperature (K). Defaults to 0 K.
        :return: Fermi energy for each ase.Atoms object stored in a torch.Tensor
        format.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        if dos is None:
            _, dos = self.calculate_dos(atoms, per_atom=False)
        if dos.shape[0] != len(atoms):
            raise ValueError(
                f"The provided DOS is inconsistent with the provided `atoms` "
                f"parameter: {len(atoms)} != {dos.shape[0]}. Please either set "
                "`dos = None` or provide a consistent DOS, computed with "
                "`per_atom = False`."
            )
        cdos = torch.cumulative_trapezoid(dos, dx=ENERGY_INTERVAL)
        num_electrons = get_num_electrons(atoms)
        num_electrons.to(dos.device)
        efermi_indices = torch.argmax(
            (cdos > num_electrons.unsqueeze(1)).float(), dim=1
        )
        efermi = self._energy_grid[efermi_indices]
        if temperature > 0.0:
            efermi_grid_trial = torch.linspace(
                efermi.min() - ENERGY_WINDOW,
                efermi.max() + ENERGY_WINDOW,
                ENERGY_GRID_NUM_POINTS_COARSE,
            )
            occupancies = fermi_dirac_distribution(
                self._energy_grid.unsqueeze(0),
                efermi_grid_trial.unsqueeze(1),
                temperature,
            )
            idos = torch.trapezoid(dos.unsqueeze(1) * occupancies, self._energy_grid)
            idos_interp = torch.nn.functional.interpolate(
                idos.unsqueeze(0),
                size=ENERGY_GRID_NUM_POINTS_FINE,
                mode="linear",
                align_corners=True,
            )[0]
            efermi_grid_interp = torch.nn.functional.interpolate(
                efermi_grid_trial.unsqueeze(0).unsqueeze(0),
                size=ENERGY_GRID_NUM_POINTS_FINE,
                mode="linear",
                align_corners=True,
            )[0][0]
            # Soft approximation of argmax using temperature scaling
            residue = idos_interp - num_electrons.unsqueeze(1)
            # Use softmax with a sharp temperature to approximate argmax
            tau = 0.0001  # Small temperature for sharp approximation
            weights = torch.softmax(-torch.abs(residue) / tau, dim=1)
            efermi = torch.sum(weights * efermi_grid_interp.unsqueeze(0), dim=1)
        return efermi
