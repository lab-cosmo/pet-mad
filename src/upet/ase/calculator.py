import warnings
from typing import Dict, List, Literal, Optional, Union

import ase.calculators.calculator
import numpy as np
import torch
from ase import Atoms
from metatomic_ase import MetatomicCalculator, SymmetrizedCalculator

from .._models import (
    get_upet,
    parse_checkpoint_filename,
    upet_resolve_model,
)
from .._version import (
    UPET_AVAILABLE_MODELS,
)
from ._uncertainty import (
    UQ_ERROR_MSG,
    UQ_GRAD_ERROR_MSG,
    UQ_NC_ERROR_MSG,
    run_direct_uq,
    run_gradient_ensemble_uq,
    stress_ensemble_to_voigt,
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
        Calculate the energy uncertainty for a given :py:class:`ase.Atoms` object.
        Note, that the uncertainty is not rotationally averaged, even when the
        calculator is: it is requested from the base model directly.

        :param atoms: ASE atoms object. If ``None``, the last calculated atoms will be
            used.
        :param per_atom: Whether to return the energy uncertainty per atom.
        :return: Energy uncertainty in numpy.ndarray format.
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
        Calculate the energy ensemble for a given :py:class:`ase.Atoms` object.
        Note, that the ensemble is not rotationally averaged, even when the
        calculator is: it is requested from the base model directly.

        :param atoms: ASE atoms object. If ``None``, the last calculated atoms will be
            used.
        :param per_atom: Whether to return the energies per atom.
        :return: Energy ensemble in numpy.ndarray format.
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

    def get_forces_uncertainty(
        self,
        atoms: Optional[Atoms] = None,
        non_conservative: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Calculate the forces uncertainty for a given :py:class:`ase.Atoms` object
        through a standard deviation of the forces ensemble. Can be calculated in
        two ways: conservative or non-conservative, where the default is controlled
        by the `non_conservative` parameter of the calculator. Optionnaly, the
        non-conservative forces uncertainty can be requested explicitly for faster
        evaluation through the ``non_conservative=True`` flag, even when the calculator
        itself is initialized in the conservative regime.
        Calculating the conservative forces uncertainty for a non-conservative forces
        calculator is not supported.

        :param atoms: ASE atoms object. If ``None``, the last calculated atoms will be
            used.
        :param non_conservative: whether to use the non-conservative regime of forces
            uncertainty calculation. If ``None``, the regime of the calculator is used.
        :return: Forces uncertainty as numpy.ndarray with shape [n_atoms, 3],
            in eV/Angstrom.
        """
        return self.get_forces_ensemble(atoms, non_conservative).std(axis=-1)

    def get_forces_ensemble(
        self,
        atoms: Optional[Atoms] = None,
        non_conservative: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Calculate the forces ensemble for a given :py:class:`ase.Atoms` object.

        Can be calculated in two ways: conservative or non-conservative, where
        the default is controlled by the `non_conservative` parameter of the
        calculator. Optionnaly, the non-conservative forces ensemble can be requested
        explicitly for faster evaluation through the ``non_conservative=True`` flag,
        even when the calculator itself is initialized in the conservative regime.
        If the calculator is in the conservative regime, the non-conservative forces
        ensemble is centered on the conservative forces values, while keeping the
        non-conservative ensemble spread.
        Calculating the conservative forces ensemble for a non-conservative forces
        calculator is not supported.

        :param atoms: ASE atoms object. If ``None``, the last calculated atoms will be
            used.
        :param non_conservative: whether to use the non-conservative regime of forces
            ensemble calculation. If ``None``, the regime of the calculator is used.
        :return: Forces ensemble as numpy.ndarray with shape [n_atoms, 3, n_ensemble],
            in eV/Angstrom.
        """

        # We allow for non_conservative forces ensemble in two cases:
        # 1. The calculator was built with non_conservative=True, so the forces are
        #   non-conservative and the ensemble is requested for the same regime.
        # 2. The calculator was built with non_conservative=False, but the user
        #   explicitly requests non_conservative=True for faster evaluation.
        # Otherwise, for a conservaitve forces calculation and
        # non_conservative=False, we use the gradient ensemble forces UQ.

        calc_nc_requested = self._base_calculator.parameters["non_conservative"] in (
            True,
            "forces",
        )
        non_conservative = (
            calc_nc_requested if non_conservative is None else non_conservative
        )
        atoms = self._resolve_atoms(atoms)
        if non_conservative or calc_nc_requested:
            key = self._quantity_keys["non_conservative_forces"]["ensemble"]
            if key not in self._model_outputs:
                raise NotImplementedError(UQ_NC_ERROR_MSG.format(key="forces"))
            forces_ensemble = run_direct_uq(
                calculator=self._base_calculator,
                atoms=atoms,
                key=key,
                per_atom=True,
            )
            # Centering the non-conservative forces ensemble so the net force is zero,
            # similarly to how the `MetatomicCalculator` does it for non-conservative
            # forces prediction.
            forces_ensemble -= np.mean(forces_ensemble, axis=0, keepdims=True)
            if not calc_nc_requested:
                # Re-center the non-conservative ensemble on the conservative forces
                # values, while keeping the non-conservative ensemble spread
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
        self,
        atoms: Optional[Atoms] = None,
        voigt: bool = True,
        non_conservative: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Calculate the stress uncertainty for a given :py:class:`ase.Atoms` object
        through a standard deviation of the stress ensemble.

        Can be calculated in two ways: conservative or non-conservative, where the
        default is controlled by the `non_conservative` parameter of the calculator.
        Optionnaly, the non-conservative stress uncertainty can be requested explicitly
        for faster evaluation through the ``non_conservative=True`` flag, even when the
        calculator itself is initialized in the conservative regime.
        Calculating the conservative stress uncertainty for a non-conservative stress
        calculator is not supported.

        :param atoms: ASE atoms object. If ``None``, the last calculated atoms will be
            used.
        :param non_conservative: whether to use the non-conservative regime of stress
            uncertainty calculation. If ``None``, the regime of the calculator is used.
        :return: Stress uncertainty as numpy.ndarray with shape [6,] if ``voigt=True``
            or [3, 3] if ``voigt=False``, in eV/Angstrom^3.
        """
        return self.get_stress_ensemble(
            atoms, voigt=voigt, non_conservative=non_conservative
        ).std(axis=-1)

    def get_stress_ensemble(
        self,
        atoms: Optional[Atoms] = None,
        voigt: bool = True,
        non_conservative: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Calculate the stress ensemble for a given :py:class:`ase.Atoms` object.

        Can be calculated in two ways: conservative or non-conservative, where the
        default is controlled by the `non_conservative` parameter of the calculator.
        Optionnaly, the non-conservative stress ensemble can be requested explicitly
        for faster evaluation through the ``non_conservative=True`` flag, even when the
        calculator itself is initialized in the conservative regime.
        Calculating the conservative stress uncertainty for a non-conservative stress
        calculator is not supported.

        :param atoms: ASE atoms object. If ``None``, the last calculated atoms will be
            used.
        :param non_conservative: whether to use the non-conservative regime of stress
            ensemble calculation. If ``None``, the regime of the calculator is used.
        :return: Stress uncertainty as numpy.ndarray with shape [6, n_ensemble] if
            ``voigt=True`` or [3, 3, n_ensemble] if ``voigt=False``, in eV/Angstrom^3.
        """
        calc_nc_requested = self._base_calculator.parameters["non_conservative"] in (
            True,
            "stress",
        )
        non_conservative = (
            calc_nc_requested if non_conservative is None else non_conservative
        )
        if non_conservative or calc_nc_requested:
            key = self._quantity_keys["non_conservative_stress"]["ensemble"]
            if key not in self._model_outputs:
                raise NotImplementedError(UQ_NC_ERROR_MSG.format(key="stress"))
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
