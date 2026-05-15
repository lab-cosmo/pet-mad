import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import ase.calculators.calculator
import numpy as np
import torch
import torch.nn.functional as F
from ase import Atoms
from metatomic.torch import ModelOutput
from metatomic_ase import MetatomicCalculator, SymmetrizedCalculator
from packaging.version import Version
from scipy.ndimage import gaussian_filter1d

from ._models import (
    _get_bandgap_model,
    _get_fermi_model,
    get_pet_mad_dos,
    get_upet,
    parse_checkpoint_filename,
    upet_resolve_model,
)
from ._version import (
    PET_MAD_DOS_LATEST_STABLE_VERSION,
    UPET_AVAILABLE_MODELS,
    UPET_UQ_SUPPORTED_MODELS,
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
        model: Optional[str] = None,
        version: Optional[str] = "latest",
        dtype: Optional[torch.dtype] = None,
        checkpoint_path: Optional[str] = None,
        variants: Optional[Dict[str, Optional[str]]] = None,
        rotational_average_order: Optional[int] = None,
        rotational_average_batch_size: Optional[int] = None,
        *,
        device: Optional[str] = None,
        non_conservative: bool = False,
        check_consistency: bool = False,
    ):
        """
        :param model: PET-MLIP model to use. Required when not using checkpoint_path.
            Can be one of the following:

            - "pet-mad-xs": PET-MAD-1.5 model (size "xs", materials and molecules,
              r2SCAN)
            - "pet-mad-s": PET-MAD-1.5 model (size "s", materials and molecules, r2SCAN)
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
            version. Deprecated model versions:

            - "pet-mad-s-v1.0.2": PET-MAD-1 model (size "s", materials and molecules,
              PBEsol)
            - "pet-omad-xs-v1.0.0": PET-OMAD model (size "xs", materials and molecules,
              PBEsol)
            - "pet-omad-s-v1.0.0": PET-OMAD model (size "s", materials and molecules,
              PBEsol)
            - "pet-omad-l-v0.1.0": PET-OMAD model (size "l", materials and molecules,
              PBEsol)
        :param dtype: dtype to use for the calculations. If `None`, we will use the
            default dtype.
        :param checkpoint_path: path to a checkpoint file to load the model from.
            If the filename follows standard naming (e.g., "pet-mad-s-v1.0.2.ckpt"),
            model/size/version are extracted automatically, and the `model`, `size`, and
            `version` parameters are ignored.
        :param variants: dictionary specifying which variant to use for each output.
            This option allows to choose the evaluation head when multiple variants
            are available for a given output. For example, if both ``energy/pbe`` and
            ``energy/r2scan`` variants are available for ``energy`` target, one can
            select which one to use by setting the ``variants`` parameter to
            ``{"energy": "r2scan"}``. If ``energy`` is set to a variant also the
            uncertainty and non-conservative outputs will be taken from this variant.
            If not provided, the default variant for each output will be used
            (for example: ``energy`` with no variant specification).
        :param rotational_average_order: order of the Lebedev-Laikov grid used for
            averaging the prediction over rotations.
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

        # Branch 1: Loading from a local checkpoint
        if checkpoint_path is not None:
            model_name, size, version = parse_checkpoint_filename(checkpoint_path)
        # Branch 2: Loading from HuggingFace
        else:
            if model is None:
                raise ValueError(
                    "'model' parameter is required when not using checkpoint_path"
                )

            if model.lower() not in UPET_AVAILABLE_MODELS:
                raise ValueError(
                    f"Model {model} is not available. Please select one of the "
                    f"following: {UPET_AVAILABLE_MODELS}"
                )

            model_name, size = model.rsplit("-", 1)
            size, version = upet_resolve_model(
                model_name,
                requested_size=size,
                requested_version=version if version != "latest" else None,
            )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)

            loaded_model = get_upet(
                model=model_name,
                size=size,
                version=version,
                checkpoint_path=checkpoint_path,
            )

        model_outputs = loaded_model.capabilities().outputs
        if non_conservative:
            selected_variant = None if variants is None else variants.get("energy")
            variant_postfix = f"/{selected_variant}" if selected_variant else ""
            nc_forces_key = "non_conservative_forces" + variant_postfix
            nc_stress_key = "non_conservative_stress" + variant_postfix
            if nc_forces_key not in model_outputs or nc_stress_key not in model_outputs:
                raise NotImplementedError(
                    "Non-conservative forces and stresses are not available for the "
                    f"model {model}, v{version}. Please run without "
                    "non_conservative=True, or choose another model."
                )

        if dtype is not None:
            if isinstance(dtype, str):
                assert dtype in STR_TO_DTYPE, f"Invalid dtype: {dtype}"
                dtype = STR_TO_DTYPE[dtype]
            loaded_model._capabilities.dtype = DTYPE_TO_STR[dtype]
            loaded_model = loaded_model.to(dtype=dtype, device=device)

        self.calculator = MetatomicCalculator(
            loaded_model,
            extensions_directory=None,
            check_consistency=check_consistency,
            device=device,
            variants=variants,
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

    def _run_uq(
        self,
        atoms: Optional[Atoms] = None,
        per_atom: bool = False,
        key: str = "energy_uncertainty",
    ) -> np.ndarray:
        if not self.calculator._calculate_uncertainty:
            raise NotImplementedError(
                "Energy uncertainty and ensemble are not available for the selected "
                "model. For uncertainty estimates, please use one of the following "
                f"models: {UPET_UQ_SUPPORTED_MODELS}"
            )

        if atoms is None:
            if self.atoms is None:
                raise ValueError(
                    "No `atoms` provided and no previously calculated atoms found."
                )
            else:
                atoms = self.atoms

        outputs = self.calculator.run_model(
            atoms,
            outputs={key: ModelOutput(quantity="energy", unit="eV", per_atom=per_atom)},
        )

        return outputs[key].block().values.detach().cpu().numpy()

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
        key = self.calculator._energy_uq_key
        return self._run_uq(atoms=atoms, per_atom=per_atom, key=key)

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
        key = self.calculator._energy_uq_key.replace("_uncertainty", "_ensemble")
        return self._run_uq(atoms=atoms, per_atom=per_atom, key=key)


# For PET-MAD-DOS predictions
ENERGY_LOWER_BOUND = -159.6456  # Lower bound of the energy grid for DOS
ENERGY_UPPER_BOUND = 80.6528  # Upper bound of the energy grid for DOS
ENERGY_INTERVAL = 0.05  # Interval of the energy grid for DOS

# For PET-MAD-DOS Targets Computation
TARGET_ENERGY_LOWER_BOUND = -149.6456  # Lower bound of the energy grid for DOS
TARGET_ENERGY_UPPER_BOUND = 80.6528  # Upper bound of the energy grid for DOS
TARGET_ENERGY_INTERVAL = 0.05  # Interval of the energy grid for DOS

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
        fermi_model_path: Optional[str] = None,
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
        fermi_model = _get_fermi_model(version=version, model_path=fermi_model_path)

        self.calculator = MetatomicCalculator(
            model,
            additional_outputs={},
            check_consistency=check_consistency,
            device=device,
        )
        self._bandgap_model = bandgap_model
        self._fermi_model = fermi_model
        self.UQ_model = None  # UQ model is heavy so it will not be loaded by default.
        self.sigmoid = torch.nn.Sigmoid()

        n_points = np.ceil((ENERGY_UPPER_BOUND - ENERGY_LOWER_BOUND) / ENERGY_INTERVAL)
        self._energy_grid = (
            torch.arange(n_points) * ENERGY_INTERVAL + ENERGY_LOWER_BOUND
        )
        target_n_points = np.ceil(
            (TARGET_ENERGY_UPPER_BOUND - TARGET_ENERGY_LOWER_BOUND)
            / TARGET_ENERGY_INTERVAL
        )
        self._target_energy_grid = (
            torch.arange(target_n_points) * TARGET_ENERGY_INTERVAL
            + TARGET_ENERGY_LOWER_BOUND
        )

    def calculate_dos(
        self,
        atoms: Union[Atoms, List[Atoms]],
        per_atom: bool = False,
        denoise: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the density of states for a given ase.Atoms object,
        or a list of ase.Atoms objects.

        :param atoms: ASE atoms object or a list of ASE atoms objects
        :param per_atom: Whether to return the density of states per atom.
        :param denoise: Whether to apply denoising to the calculated DOS.
        :return: Energy grid and corresponding DOS values in torch.Tensor format.
        """
        results = self.calculator.run_model(
            atoms, outputs={"mtt::dos": ModelOutput(per_atom=per_atom)}
        )
        dos = results["mtt::dos"].block().values
        if denoise:
            if per_atom:
                raise NotImplementedError(
                    "Denoising is not implemented for per-atom DOS."
                    " Please set `per_atom=False` to use denoising."
                )
            _, dos = self.denoise_predictions(atoms, dos, self._energy_grid.clone())
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
        model=False,
    ) -> torch.Tensor:
        """
        Get the Fermi energy for a given ase.Atoms object, or a list of ase.Atoms
        objects, based on a predicted density of states at a given temperature.
        By default, the density of states is first calculated using the `calculate_dos`
        method, and the Fermi level is calculated at T=0 K. Alternatively, the density
        of states can be provided as an input parameter to avoid re-calculating the DOS.
        There are two methods to calculate the Fermi level: (1) the default method,
        which is based on charge neutrality and cumulative DOS, and (2) a model-based
        method that uses a dedicated CNN model trained to predict the Fermi level
        directly from the DOS. The model-based method can be activated by setting
        `model=True`. The model-based method is less susceptible to model noise
        especially for gapped systems.

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
        if model:
            logging.info(
                "Calculating Fermi level with the model-based method. This method is"
                " more robust to noise in the DOS, especially for gapped systems"
            )
            num_atoms = torch.tensor([len(item) for item in atoms], device=dos.device)
            dos = dos / num_atoms.unsqueeze(1)
            efermi = self._fermi_model(dos.unsqueeze(1)).detach()
            efermi = efermi.squeeze()

        else:
            logging.info(
                "Calculating Fermi level with the default method "
                "based on charge neutrality and cumulative DOS."
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
                idos = torch.trapezoid(
                    dos.unsqueeze(1) * occupancies, self._energy_grid
                )
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

    def denoise_predictions(
        self,
        atoms: Union[Atoms, List[Atoms]],
        dos: torch.Tensor,
        energies: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Denoise the predicted DOS by enforcing physical consistency between the DOS
        and the Fermi level predicted by the model. The denoising procedure is detailed
        in the PET-MAD-DOS paper. The procedure is summarized as:

        1) Predict the Fermi level from the original DOS
        2) A 1-D Gaussian filter with a standard deviation of 0.3eV is applied to the
              original DOS to smooth out spurious peaks and noise.
        3) The filtered DOS is passed through a modified sigmoid function such that
            the inflection point is at 0.1 an the slope is 100
        4) The output is used as a multiplier on the DOS output to obtain a
            thresholded DOS
        5) The thresholded DOS is scaled such that the physical Fermi level of the DOS
            lie on the same point as the predicted by the model in the first step.

        :param atoms: ASE atoms object or a list of ASE atoms objects
        :param dos: Density of states for the given atoms
        :param energies: Energy grid corresponding to the DOS. If not provided,
            the default energy grid of PET-MAD-DOS will be used.
        :return: Energy grid and corresponding denoised DOS values for each ase.Atoms
            object stored in torch.Tensor format.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        fermi = self.calculate_efermi(atoms, dos=dos, model=True)
        n_electrons = get_num_electrons(atoms).to(dos.device)
        num_atoms = torch.tensor([len(item) for item in atoms], device=dos.device)
        dos = dos / num_atoms.unsqueeze(1)
        n_electrons = n_electrons / num_atoms
        if energies is None:
            energies = self._energy_grid.clone().to(dos.device)
        dos_filtered = gaussian_filter1d(dos.cpu().numpy(), sigma=0.3 / ENERGY_INTERVAL)
        dos_filtered = torch.from_numpy(dos_filtered).to(dos.device)
        sigmoid_input = 100 * (dos_filtered - 0.1)
        multiplier = self.sigmoid(sigmoid_input)
        dos_thresholded = dos * multiplier
        cdos_thresholded = torch.cumulative_trapezoid(
            dos_thresholded, x=energies, dim=1
        )
        fermi_indexes = torch.searchsorted(energies, fermi)
        if len(fermi_indexes.shape) == 0:
            fermi_indexes = fermi_indexes.unsqueeze(0)
        current_electrons = cdos_thresholded.gather(1, fermi_indexes.unsqueeze(1))
        scaling_factor = n_electrons.flatten() / current_electrons.flatten()
        dos_denoised = dos_thresholded * scaling_factor.unsqueeze(1)
        dos_rescaled = dos_denoised * num_atoms.unsqueeze(1)
        return energies, dos_rescaled

    def align_dos(
        self, predicted_DOS: torch.Tensor, true_DOS: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Align the predicted DOS to the true DOS by finding the optimal energy shift that
        minimizes the mean squared error between the predicted and true DOS in a given
        energy window defined by the mask. It is assumed that predictions have a larger
        energy grid than the true_DOS. The output is the true_DOS expanded to the same
        energy grid as the predicted_DOS in a way that aligns the two spectras.

        :param predicted_DOS: Predicted density of states for a given atoms.
        :param true_DOS: True density of states for the same atoms.
        :param mask: Integer boolean tensor (1/0) indicating the energy window to
            consider for the alignment.
        :return: Aligned predicted and true DOS.
        """

        device = predicted_DOS.device
        true_DOS = true_DOS.to(device)
        mask = mask.to(device)
        sum_sq_smaller = torch.sum((true_DOS**2) * mask, dim=1, keepdim=True)
        batch_size = predicted_DOS.shape[0]
        bigger_reshaped = predicted_DOS.unsqueeze(0)
        kernel = (true_DOS * mask).unsqueeze(1)
        cross_corr = F.conv1d(bigger_reshaped, kernel, groups=batch_size)
        cross_corr = cross_corr.squeeze(0)
        bigger_sq_reshaped = (predicted_DOS**2).unsqueeze(0)
        mask_kernel = mask.unsqueeze(1)
        sum_sq_bigger = F.conv1d(bigger_sq_reshaped, mask_kernel, groups=batch_size)
        sum_sq_bigger = sum_sq_bigger.squeeze(0)
        losses = sum_sq_bigger - 2 * cross_corr + sum_sq_smaller
        losses = torch.clamp(losses, min=0.0)
        front_tail = torch.cumsum(predicted_DOS**2, dim=1)
        shape_difference = predicted_DOS.shape[1] - true_DOS.shape[1]
        additional_error = torch.hstack(
            [
                torch.zeros(len(predicted_DOS), device=predicted_DOS.device).reshape(
                    -1, 1
                ),
                front_tail[:, :shape_difference],
            ]
        )
        total_losses = losses + additional_error
        final_loss, shift = torch.min(total_losses, dim=1)
        aligned_true_DOS = []
        aligned_true_masks = []
        for index, s in enumerate(shift):
            front_pad = torch.zeros(s, device=predicted_DOS.device)
            back_pad = torch.zeros(
                predicted_DOS.shape[1] - true_DOS.shape[1] - s,
                device=predicted_DOS.device,
            )
            true_DOS_padded = torch.hstack([front_pad, true_DOS[index], back_pad]).int()
            true_Mask_padded = torch.hstack([front_pad+1, mask[index], back_pad]).int()
            aligned_true_DOS.append(true_DOS_padded)
            aligned_true_masks.append(true_Mask_padded)

        return (
            predicted_DOS,
            torch.vstack(aligned_true_DOS),
            torch.vstack(aligned_true_masks),
        )

    def compute_DOS_and_mask_from_eigenvalues(
        self,
        eigenvalues: torch.Tensor,
        kweights: Optional[torch.Tensor] = None,
        energy_grid: Optional[torch.Tensor] = None,
        sigma: float = 0.3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the density of states and the corresponding mask from a given set of
        eigenvalues and k-point weights. The DOS is computed by broadening each
        eigenvalue with a Gaussian function and summing over all eigenvalues. The mask
        is a boolean tensor indicating which energy grid points are reliable enough to
        compute the loss on.

        :param eigenvalues: Tensor of shape (n_kpoints, n_bands) containing the
            eigenvalues.
        :param kweights: Tensor of shape (n_kpoints,) containing the weights of each
            k-point.
        :param energy_grid: Tensor containing the energy grid on which to compute the
            DOS. Defaults to the energy grid defined in the PET-MAD-DOS model.
        :param sigma: Standard deviation for Gaussian broadening in eV.
            Defaults to 0.3 eV.
        :return: DOS and mask
        """

        if kweights is None:
            kweights = (
                torch.ones(eigenvalues.shape[0], device=eigenvalues.device)
                / eigenvalues.shape[0]
            )
        if energy_grid is None:
            energy_grid = self._energy_grid.clone()
            device = energy_grid.device
        confident_upper_energy_bound = torch.min(eigenvalues[:, -1]) - 3 * sigma
        eigenvalues = eigenvalues.to(device)
        kweights = kweights.to(device)
        n_bands = eigenvalues.shape[1]
        eigenvalues = eigenvalues.flatten()
        kweights = kweights.squeeze().repeat(n_bands)
        delta_E = (energy_grid - eigenvalues[:, None]) / sigma
        gaussian_weights = torch.exp(-0.5 * delta_E**2)
        normalization = 1 / np.sqrt(2 * np.pi * sigma**2)
        dos = torch.sum(kweights[:, None] * gaussian_weights, dim=0) * normalization
        mask = (energy_grid <= confident_upper_energy_bound).to(torch.int)

        return dos, mask

    def pad_dos_and_mask_for_training(
        self,
        dos: torch.Tensor,
        mask: torch.Tensor,
        target_length: int = 4806,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad the DOS and mask tensors with zeros to a target length. The padding is done
        on the left side of the tensors.

        :param dos: Tensor containing the density of states values.
        :param mask: Tensor containing the mask values.
        :param target_length: The target length to pad the tensors to. Defaults to 4806.
        :return: Padded DOS and mask tensors.
        """

        current_length = dos.shape[0] if dos.ndim == 1 else dos.shape[1]
        if current_length >= target_length:
            logging.info(
                "No padding needed for DOS and mask. Current length: ",
                current_length,
                " Target length: ",
                target_length,
            )
            return dos, mask
        padding_length = target_length - current_length
        logging.info(
            "Padding DOS and mask with zeros to the left. Padding length: ",
            padding_length,
            " Please use this value for "
            "n_extra_targets parameter for the loss function "
            "in the training hyperparameters YAML file.",
        )
        dos_padded = F.pad(dos, (padding_length, 0), mode="constant", value=0)
        mask_padded = F.pad(mask, (padding_length, 0), mode="constant", value=0)
        return dos_padded, mask_padded
