import warnings
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import ase.calculators.calculator
import numpy as np
import torch
from ase import Atoms
from metatomic.torch import ModelOutput
from metatomic_ase import (
    MetatomicCalculator,
    SymmetrizedCalculator,
)
from packaging.version import Version

from ._models import (
    _get_bandgap_model,
    _get_fermi_model,
    get_pet_mad_dos,
    get_upet,
    parse_checkpoint_filename,
    upet_resolve_model,
)
from ._uncertainty import (
    UQ_ERROR_MSG,
    UQ_GRAD_ERROR_MSG,
    UQ_NC_ERROR_MSG,
    run_direct_uq,
    run_gradient_ensemble_uq,
    stress_ensemble_to_voigt,
)
from ._version import (
    PET_MAD_DOS_LATEST_STABLE_VERSION,
    UPET_AVAILABLE_MODELS,
)
from .utils import (
    dos_from_eigenvalues,
    get_num_electrons,
    pad_dos,
    torch_gaussian_filter1d,
)


BASE_QUANTITIES = ("energy", "non_conservative_forces", "non_conservative_stress")

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
        non_conservative: Union[bool, Literal["forces", "stress"]] = False,
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
            - "pet-omol-s": PET-OMol model (size "s", molecules, ωB97M-V)
            - "pet-omol-m": PET-OMol model (size "m", molecules, ωB97M-V)
            - "pet-omol-l": PET-OMol model (size "l", molecules, ωB97M-V)
            - "pet-mols-s": PET-MOLS model (size "s", organic molecular crystals,
              PBE0+MBD)
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
            and / or stresses prediction. Available options are:

            - False: use the conservative regime (default)
            - True: use the non-conservative regime for both forces and stresses
            - "forces": use the non-conservative regime for forces only
            - "stress": use the non-conservative regime for stresses only

            Defaults to False. Available for all models, except:

            - PET-MAD models with version < 1.1.0
            - PET-SPICE models
            - PET-MOLS models

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
        self._model_outputs = model_outputs
        selected_variant = None if variants is None else variants.get("energy")
        variant_prefix = "mtt::aux::" if selected_variant else ""
        variant_postfix = f"/{selected_variant}" if selected_variant else ""

        quantity_keys = {}
        for quantity in BASE_QUANTITIES:
            quantity_key = f"{quantity}{variant_postfix}"
            # metatrain names the uncertainty and ensemble outputs of any target
            # other than a plain "energy" with an "mtt::aux::" prefix
            prefix = "mtt::aux::" if quantity != "energy" else variant_prefix
            uncertainty_key = f"{prefix}{quantity}{variant_postfix}_uncertainty"
            ensemble_key = f"{prefix}{quantity}{variant_postfix}_ensemble"
            quantity_keys[quantity] = {
                "quantity": quantity_key,
                "uncertainty": uncertainty_key,
                "ensemble": ensemble_key,
            }

        self._quantity_keys = quantity_keys

        if non_conservative:
            requested_nc_quantities = (
                ("forces", "stress")
                if non_conservative is True
                else (non_conservative,)
            )
            for nc_quantity in requested_nc_quantities:
                nc_quantity_key = quantity_keys[f"non_conservative_{nc_quantity}"][
                    "quantity"
                ]
                if nc_quantity_key not in model_outputs:
                    raise NotImplementedError(
                        f"`non-conservative={non_conservative}` option is not "
                        f"available for the model {model} v{version}, and a target "
                        f"variant `{selected_variant or 'energy'}`. Please choose "
                        f"another `non-conservative` option, use another target "
                        "variant, switch to a conservative regime or choose "
                        "another model."
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

    @property
    def _base_calculator(self) -> MetatomicCalculator:
        """The underlying calculator, unwrapped from rotational averaging."""
        calc = self.calculator
        if isinstance(calc, SymmetrizedCalculator):
            return calc.base_calculator
        return calc

    @property
    def _non_conservative_forces(self) -> bool:
        """Whether ``get_forces`` returns the model's direct forces."""
        return self._base_calculator.parameters["non_conservative"] in (True, "forces")

    @property
    def supports_uncertainty(self) -> bool:
        """Whether the calculator supports uncertainty quantification."""
        return self._base_calculator._calculate_uncertainty

    def _resolve_atoms(self, atoms: Optional[Atoms]) -> Atoms:
        """Fall back to the last calculated atoms when none are given."""
        if atoms is not None:
            return atoms
        if self.atoms is None:
            raise ValueError(
                "No `atoms` provided and no previously calculated atoms found."
            )
        return self.atoms

    def get_energy_uncertainty(
        self, atoms: Optional[Atoms] = None, per_atom: bool = False
    ) -> np.ndarray:
        """
        Get the energy uncertainty for a given :py:class:`ase.Atoms` object.

        :param atoms: ASE atoms object. If ``None``, the last calculated atoms will be
            used.
        :param per_atom: Whether to return the energy uncertainty per atom.
        :return: Energy uncertainty in numpy.ndarray format.

        The uncertainty is not rotationally averaged, even when the calculator
        is: it is requested from the base model directly.
        """
        key = self._quantity_keys["energy"]["uncertainty"]
        if key not in self._model_outputs:
            raise NotImplementedError(UQ_ERROR_MSG.format(key="Energy uncertainty"))
        return run_direct_uq(
            calculator=self._base_calculator,
            atoms=self._resolve_atoms(atoms),
            key=key,
            per_atom=per_atom,
        )

    def get_energy_ensemble(
        self, atoms: Optional[Atoms] = None, per_atom: bool = False
    ) -> np.ndarray:
        """
        Get the ensemble of energies for a given :py:class:`ase.Atoms` object.

        :param atoms: ASE atoms object. If ``None``, the last calculated atoms will be
            used.
        :param per_atom: Whether to return the energies per atom.
        :return: Energy ensemble in numpy.ndarray format.

        The ensemble is not rotationally averaged, even when the calculator is:
        it is requested from the base model directly.
        """
        key = self._quantity_keys["energy"]["ensemble"]
        if key not in self._model_outputs:
            raise NotImplementedError(UQ_ERROR_MSG.format(key="Energy ensemble"))
        return run_direct_uq(
            calculator=self._base_calculator,
            atoms=self._resolve_atoms(atoms),
            key=key,
            per_atom=per_atom,
        )

    def _resolve_forces_regime(self, non_conservative: Optional[bool]) -> bool:
        """Which force regime an uncertainty request refers to, defaulting to the
        one the calculator was built with."""
        if non_conservative is None:
            return self._non_conservative_forces
        if non_conservative not in (True, False):
            raise TypeError(
                f"`non_conservative` must be a bool or None, got {non_conservative!r}."
            )
        if self._non_conservative_forces and not non_conservative:
            raise ValueError(
                "`non_conservative=False` is not available for a calculator built "
                "with non-conservative forces: the conservative ensemble would not "
                "match the forces this calculator returns. Build the calculator in "
                "the conservative regime instead."
            )
        return non_conservative

    def get_forces_uncertainty(
        self,
        atoms: Optional[Atoms] = None,
        non_conservative: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Get the forces uncertainty for a given :py:class:`ase.Atoms` object.

        :param atoms: ASE atoms object. If ``None``, the last calculated atoms will be
            used.
        :param non_conservative: which force uncertainty to use, see
            :py:meth:`get_forces_ensemble`. Defaults to the regime the calculator
            was built with. This is the spread of the corresponding ensemble, so it
            refers to the same forces the calculator itself returns — except for a
            model carrying a direct uncertainty but no direct ensemble, where the
            model's own (unprojected, hence larger) uncertainty is returned instead.
        :return: Forces uncertainty as numpy.ndarray with shape [n_atoms, 3],
            in eV/Angstrom.
        """
        non_conservative = self._resolve_forces_regime(non_conservative)
        keys = self._quantity_keys["non_conservative_forces"]
        if non_conservative and keys["ensemble"] not in self._model_outputs:
            # without an ensemble to take the spread of, the head's own uncertainty
            # output is the only option, unprojected net force and all
            key = keys["uncertainty"]
            if key not in self._model_outputs:
                raise NotImplementedError(
                    UQ_NC_ERROR_MSG.format(key="Non-conservative forces uncertainty")
                )
            return run_direct_uq(
                calculator=self._base_calculator,
                atoms=self._resolve_atoms(atoms),
                key=key,
                per_atom=True,
            )
        return self.get_forces_ensemble(atoms, non_conservative).std(axis=-1)

    def get_forces_ensemble(
        self,
        atoms: Optional[Atoms] = None,
        non_conservative: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Get the ensemble of forces for a given :py:class:`ase.Atoms` object.

        :param atoms: ASE atoms object. If ``None``, the last calculated atoms will be
            used.
        :param non_conservative: how to build the ensemble.

            - ``False``: gradients of the energy ensemble, so that every member is
              conservative by construction.
            - ``True``: ensemble of the model's own non-conservative force head.
              Unless the calculator itself is in the non-conservative regime, the
              ensemble is re-centered on the conservative forces, so that only its
              spread comes from the direct head. The other way around is an error:
              a non-conservative calculator cannot serve a conservative ensemble.

            Defaults to the regime the calculator was built with.
        :return: Forces ensemble as numpy.ndarray with shape [n_atoms, 3, n_ensemble],
            in eV/Angstrom.
        """
        atoms = self._resolve_atoms(atoms)

        if self._resolve_forces_regime(non_conservative):
            key = self._quantity_keys["non_conservative_forces"]["ensemble"]
            if key not in self._model_outputs:
                raise NotImplementedError(
                    UQ_NC_ERROR_MSG.format(key="Non-conservative forces ensemble")
                )
            forces_ensemble = run_direct_uq(
                calculator=self._base_calculator,
                atoms=atoms,
                key=key,
                per_atom=True,
            )
            # `MetatomicCalculator` removes the net force from the direct forces, so
            # take it off every member too: it carries most of the head's spread and
            # none of it reaches the forces anyone uses
            forces_ensemble -= np.mean(forces_ensemble, axis=0, keepdims=True)
            if not self._non_conservative_forces:
                shift = self.get_forces(atoms) - forces_ensemble.mean(axis=-1)
                forces_ensemble += shift[:, :, np.newaxis]
        else:
            key = self._quantity_keys["energy"]["ensemble"]
            if key not in self._model_outputs:
                raise NotImplementedError(
                    UQ_GRAD_ERROR_MSG.format(key="Energy ensemble")
                )
            forces_ensemble = run_gradient_ensemble_uq(
                calculator=self._base_calculator,
                atoms=atoms,
                key=key,
                gradients=("positions",),
            )["positions"]
        return forces_ensemble

    def get_stress_uncertainty(
        self, atoms: Optional[Atoms] = None, voigt: bool = True
    ) -> np.ndarray:
        """
        TODO
        """
        if self._base_calculator.parameters["non_conservative"] in (True, "stress"):
            key = self._quantity_keys["non_conservative_stress"]["uncertainty"]
            if key not in self._model_outputs:
                raise NotImplementedError(
                    UQ_NC_ERROR_MSG.format(key="Non-conservative stress uncertainty")
                )
            stress_uncertainty = run_direct_uq(
                calculator=self._base_calculator,
                atoms=self._resolve_atoms(atoms),
                key=key,
                per_atom=False,
            )
            if voigt:
                stress_uncertainty = stress_ensemble_to_voigt(stress_uncertainty)
        else:
            stress_uncertainty = self.get_stress_ensemble(atoms, voigt=voigt).std(
                axis=-1
            )
        return stress_uncertainty

    def get_stress_ensemble(
        self, atoms: Optional[Atoms] = None, voigt: bool = True
    ) -> np.ndarray:
        """
        TODO
        """
        if self._base_calculator.parameters["non_conservative"] in (True, "stress"):
            key = self._quantity_keys["non_conservative_stress"]["ensemble"]
            if key not in self._model_outputs:
                raise NotImplementedError(
                    UQ_NC_ERROR_MSG.format(key="Non-conservative stress ensemble")
                )
            stress_ensemble = run_direct_uq(
                calculator=self._base_calculator,
                atoms=self._resolve_atoms(atoms),
                key=key,
                per_atom=False,
            )
        else:
            key = self._quantity_keys["energy"]["ensemble"]
            if key not in self._model_outputs:
                raise NotImplementedError(
                    UQ_GRAD_ERROR_MSG.format(key="Energy ensemble")
                )
            stress_ensemble = run_gradient_ensemble_uq(
                calculator=self._base_calculator,
                atoms=self._resolve_atoms(atoms),
                key=key,
                gradients=("strain",),
            )["strain"]

        if voigt:
            stress_ensemble = stress_ensemble_to_voigt(stress_ensemble)
        return stress_ensemble


# For PET-MAD-DOS predictions
ENERGY_INTERVAL = 0.05  # Interval of the energy grid for DOS


class PETMADDOSCalculator(ase.calculators.calculator.Calculator):
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
        super().__init__()
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
        self.sigmoid = torch.nn.Sigmoid()

        self.output_size = len(model.module.property_labels["mtt::dos"][0])
        self.sigma = torch.tensor(
            0.3
        )  # Standard deviation for Gaussian broadening in eV
        self.energy_interval = ENERGY_INTERVAL

    def calculate(
        self,
        atoms: Atoms,
        properties: Sequence[
            Literal[
                "dos_raw", "dos_denoised", "dos_raw_per_atom", "bandgap", "fermi_level"
            ]
        ] = ("dos_raw", "dos_denoised", "bandgap", "fermi_level"),
        system_changes: Sequence[str] = (),
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate the density of states, bandgap, and Fermi level for a given ase.Atoms
        object, or a list of ase.Atoms objects.

        :param atoms: ASE atoms object or a list of ASE atoms objects
        :param properties: List of what needs to be calculated.
        :param system_changes: List of what has changed since last calculation.
            Currently ignored, but required for compatibility with ASE.
        :return: Dictionary containing the calculated properties.
        """

        super().calculate(
            atoms=atoms,
            properties=properties,
            system_changes=system_changes,
        )
        # atoms = [atoms]
        results = {}
        # Check for invalid parameter combinations
        # need a per_atom = False to get the appropriate DOS for the
        # bandgap and Fermi level CNNs
        dos = self._calculate_dos(atoms, per_atom=False)
        if "dos_raw" in properties:
            results["dos_raw"] = dos.squeeze()
        if "fermi_level" in properties or "dos_denoised" in properties:
            fermi_level = self._calculate_efermi(atoms, dos)
            # Temporary fixed addition until Arslan merges huggingface
        if "fermi_level" in properties:
            results["fermi_level"] = fermi_level
        if "bandgap" in properties:
            bandgap = self._calculate_bandgap(atoms, dos)
            results["bandgap"] = bandgap
        # If denoise is True, we need to apply the denoising procedure to the
        # predicted DOS before returning it.
        if "dos_denoised" in properties:
            results["dos_denoised"] = self._denoise_predictions(
                atoms, fermi_level, dos
            ).squeeze()
        if "dos_raw_per_atom" in properties:
            results["dos_raw_per_atom"] = self._calculate_dos(atoms, per_atom=True)

        # If per_atom is True, we need to calculate the per-atom DOS separately, as the
        # bandgap and Fermi level models are trained on the total DOS, not the per-atom
        # DOS.

        self.results = results

        return results

    def _calculate_dos(
        self,
        atoms: Union[Atoms, List[Atoms]],
        per_atom: bool = False,
    ) -> torch.Tensor:
        """
        Calculate the density of states for a given ase.Atoms object,
        or a list of ase.Atoms objects.

        :param atoms: ASE atoms object or a list of ASE atoms objects
        :param per_atom: Whether to return the density of states per atom.
        :param denoise: Whether to apply denoising to the calculated DOS.
        :return: Energy grid and corresponding DOS values in torch.Tensor format.
        """
        results = self.calculator.run_model(
            atoms,
            outputs={
                "mtt::dos": ModelOutput(sample_kind="atom" if per_atom else "system")
            },
        )
        dos = results["mtt::dos"].block().values

        return dos

    def _calculate_bandgap(
        self, atoms: Union[Atoms, List[Atoms]], dos: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate the bandgap for a given ase.Atoms object,or a list of ase.Atoms
        objects. By default, the density of states is first calculated using the
        `calculate_dos` method, and the the bandgap is derived from the DOS by a
        BandgapModel. Alternatively, the density of states can be provided as an
        input parameter to avoid re-calculating the DOS.

        :param atoms: ASE atoms object or a list of ASE atoms objects
        :param dos: Density of states for the given atoms
        :return: bandgap values for each ase.Atoms object object stored in a
            torch.Tensor format.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        num_atoms = torch.tensor([len(item) for item in atoms], device=dos.device)
        dos = dos / num_atoms.unsqueeze(1)
        bandgap = self._bandgap_model(
            dos.unsqueeze(1)
        ).detach()  # Need to make the inputs [n_predictions, 1, 4806]
        bandgap = torch.nn.functional.relu(bandgap).squeeze()  # Remove negative gaps
        return bandgap

    def _calculate_efermi(
        self,
        atoms: Union[Atoms, List[Atoms]],
        dos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get the Fermi energy for a given ase.Atoms object, or a list of ase.Atoms
        objects, using a dedicated CNN model trained to predict the Fermi level
        directly from the predicted density of states.

        :param atoms: ASE atoms object or a list of ASE atoms objects
        :param dos: Density of states for the given atoms. If not provided, the
            density of states is calculated using the `calculate_dos` method.
        :return: Fermi energy for each ase.Atoms object stored in a torch.Tensor
            format.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        num_atoms = torch.tensor([len(item) for item in atoms], device=dos.device)
        dos = dos / num_atoms.unsqueeze(1)
        efermi = self._fermi_model(dos.unsqueeze(1)).detach()
        efermi = efermi.squeeze()
        return efermi

    def _denoise_predictions(
        self,
        atoms: Union[Atoms, List[Atoms]],
        fermi: torch.Tensor,
        dos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denoise the predicted DOS by enforcing physical consistency between the DOS
        and the Fermi level predicted by the model. The denoising procedure is detailed
        in the PET-MAD-DOS paper. The procedure is summarized as:

        1) A 1-D Gaussian filter with a standard deviation of 0.3eV is applied to the
              original DOS to smooth out spurious peaks and noise.
        2) The filtered DOS is passed through a modified sigmoid function such that
            the inflection point is at 0.1 an the slope is 100
        3) The output is used as a multiplier on the DOS output to obtain a
            thresholded DOS
        4) The thresholded DOS is scaled such that the physical Fermi level of the DOS
            lie on the same point as the predicted by the model in the first step.

        :param atoms: ASE atoms object or a list of ASE atoms objects
        :param fermi: Predicted Fermi levels for the given atoms
        :param dos: Density of states for the given atoms
        :param energies: Energy grid corresponding to the DOS. If not provided,
            the default energy grid of PET-MAD-DOS will be used.
        :return: Energy grid and corresponding denoised DOS values for each ase.Atoms
            object stored in torch.Tensor format.
        """
        if isinstance(atoms, Atoms):
            atoms = [atoms]
        n_electrons = get_num_electrons(atoms).to(dos.device)
        num_atoms = torch.tensor([len(item) for item in atoms], device=dos.device)
        dos = dos / num_atoms.unsqueeze(1)
        n_electrons = n_electrons / num_atoms

        dos_filtered = torch_gaussian_filter1d(dos, sigma=0.3 / ENERGY_INTERVAL)
        sigmoid_input = 100 * (dos_filtered - 0.1)
        multiplier = self.sigmoid(sigmoid_input)
        dos_thresholded = dos * multiplier
        cdos_thresholded = torch.cumulative_trapezoid(
            dos_thresholded, dx=ENERGY_INTERVAL, dim=1
        )
        # Ensure that fermi is within the energy grid
        fermi_indexes = torch.min(
            fermi // ENERGY_INTERVAL, torch.tensor(dos.shape[1] - 1)
        ).long()
        if len(fermi_indexes.shape) == 0:
            fermi_indexes = fermi_indexes.unsqueeze(0)
        current_electrons = cdos_thresholded.gather(1, fermi_indexes.unsqueeze(1))
        scaling_factor = n_electrons.flatten() / current_electrons.flatten()
        dos_denoised = dos_thresholded * scaling_factor.unsqueeze(1)
        dos_rescaled = dos_denoised * num_atoms.unsqueeze(1)
        return dos_rescaled

    def dos_from_eigenvalues(
        self,
        eigenvalues: torch.Tensor,
        kweights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calls the `dos_from_eigenvalues` function with PET-MAD-DOS default parameters.
        The function is useful to compute the DOS and mask from eigenvalues and
        k-point weights from DFT calculations in a way that is consistent with
        PET-MAD-DOS.

        :param eigenvalues: Tensor of shape (n_kpoints, n_bands) containing the
            eigenvalues.
        :param kweights: Tensor of shape (n_kpoints,) containing the weights of each
            k-point.
        :return: DOS
        """

        energy_grid_min = torch.min(eigenvalues)
        energy_grid_max = torch.max(eigenvalues)
        energy_grid = torch.arange(
            energy_grid_min - 10 * self.sigma,
            energy_grid_max + 10 * self.sigma,
            self.energy_interval,
            device=eigenvalues.device,
        )

        dos, mask = dos_from_eigenvalues(
            energy_grid,
            self.sigma,
            eigenvalues,
            kweights,
        )

        dos = self.pad_dos(dos, mask)

        return dos

    def pad_dos(
        self,
        dos: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pads the input DOS to the length required for PET-MAD-DOS training/
        finetuning. It calls the `pad_dos` utility function with PET-MAD-DOS
        default parameters. At the end, it replaces the
        regions where the DOS is not well-defined with zeros.

        :param dos: Tensor containing the density of states values.
        :param mask: Tensor containing the mask values.
        :return: Padded DOS
        """

        dos_padded, mask_padded = pad_dos(dos, mask, self.output_size)
        dos_padded[~mask_padded] = float("nan")

        return dos_padded
