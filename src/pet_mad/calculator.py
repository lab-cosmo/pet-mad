import logging
import os
from typing import Optional, Tuple, List, Union, Dict, Any

from metatomic.torch import ModelOutput
from metatomic.torch.ase_calculator import MetatomicCalculator
from platformdirs import user_cache_dir
import torch
import numpy as np
from ase import Atoms

from packaging.version import Version

from ._models import get_pet_mad, get_pet_mad_dos, _get_bandgap_model
from .utils import (
    get_num_electrons,
    fermi_dirac_distribution,
    rotate_atoms,
    compute_rotational_average,
    AVAILABLE_LEBEDEV_GRID_ORDERS,
    get_so3_rotations,
)
from ._version import (
    PET_MAD_LATEST_STABLE_VERSION,
    PET_MAD_UQ_AVAILABILITY_VERSION,
    PET_MAD_NC_AVAILABILITY_VERSION,
    PET_MAD_DOS_LATEST_STABLE_VERSION,
)

STR_TO_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
}
DTYPE_TO_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
}


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
        rotational_average_order: Optional[int] = None,
        rotational_average_num_primitive_rotations: int = 4,
        rotational_average_batch_size: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
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
        :param rotational_average_order: order of the Lebedev-Laikov grid used for averaging
            the prediction over rotations.
        :param rotational_average_num_primitive_rotations: number of primitive rotations used
            for sampling the unit sphere around each Lebedev-Laikov rotation vector.
        :param rotational_average_batch_size: batch size to use for the rotational averaging.
            If `None`, all rotations will be computed at once.
        :param dtype: dtype to use for the calculations. If `None`, we will use the
            default dtype.
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

        if dtype is not None:
            if isinstance(dtype, str):
                assert dtype in STR_TO_DTYPE, f"Invalid dtype: {dtype}"
                dtype = STR_TO_DTYPE[dtype]
            model._capabilities.dtype = DTYPE_TO_STR[dtype]
            model = model.to(dtype=dtype, device=device)

        self._rotations: List[np.ndarray] = []
        if rotational_average_order is not None:
            assert rotational_average_num_primitive_rotations > 0, (
                "Number of primitive rotations must be greater than 0."
            )
            if rotational_average_order not in AVAILABLE_LEBEDEV_GRID_ORDERS:
                raise ValueError(
                    f"Lebedev-Laikov grid order {rotational_average_order} is not available. "
                    f"Please use one of the following orders: {AVAILABLE_LEBEDEV_GRID_ORDERS}."
                )

            self._rotations = get_so3_rotations(
                rotational_average_order,
                rotational_average_num_primitive_rotations,
            )
        self._rotational_average_batch_size = rotational_average_batch_size

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

    def calculate(
        self, atoms: Atoms, properties: List[str], system_changes: List[str]
    ) -> None:
        """
        Compute some ``properties`` with this calculator, and return them in the format
        expected by ASE.

        This is not intended to be called directly by users, but to be an implementation
        detail of ``atoms.get_energy()`` and related functions. See
        :py:meth:`ase.calculators.calculator.Calculator.calculate` for more information.

        If the `rotational_average_order` parameter is set during initialization, the prediction
        will be averaged over unique rotations in the Lebedev-Laikov grid of a chosen order.
        """
        if len(self._rotations) == 0:
            super().calculate(atoms, properties, system_changes)
        else:
            rotated_atoms_list = rotate_atoms(atoms, self._rotations)
            compute_forces_and_stresses = (
                True
                if any(
                    p in properties
                    for p in [
                        "forces",
                        "stress",
                        "non_conservative_forces",
                        "non_conservative_stress",
                    ]
                )
                else False
            )
            batch_size = (
                self._rotational_average_batch_size
                if self._rotational_average_batch_size is not None
                else len(rotated_atoms_list)
            )
            batches = [
                rotated_atoms_list[i : i + batch_size]
                for i in range(0, len(rotated_atoms_list), batch_size)
            ]
            results: Dict[str, Any] = {}
            for batch in batches:
                batch_results = self.compute_energy(batch, compute_forces_and_stresses)
                for key, value in batch_results.items():
                    results.setdefault(key, [])
                    results[key].extend([value] if isinstance(value, float) else value)
            results = compute_rotational_average(results, self._rotations)
            self.results.update(results)

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
        """
        :param version: PET-MAD-DOS version to use. Defaults to the latest stable version.
        :param model_path: path to a Torch-Scripted model file to load the model from. If
            provided, the `version` parameter is ignored.
        :param bandgap_model_path: path to a PyTorch checkpoint file with the bandgap model.
            If provided, the `version` parameter is ignored.
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

        super().__init__(
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
        results = self.run_model(
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
