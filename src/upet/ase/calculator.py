import warnings
from typing import Dict, List, Optional

import ase.calculators.calculator
import numpy as np
import torch
from ase import Atoms
from metatomic.torch import ModelOutput
from metatomic_ase import MetatomicCalculator, SymmetrizedCalculator

from .._models import (
    get_upet,
    parse_checkpoint_filename,
    upet_resolve_model,
)
from .._version import (
    UPET_AVAILABLE_MODELS,
    UPET_UQ_SUPPORTED_MODELS,
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
            nc_forces_key = "non_conservative_force" + variant_postfix
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
