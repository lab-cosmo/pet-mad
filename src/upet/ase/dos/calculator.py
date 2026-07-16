from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import ase.calculators.calculator
import torch
from ase import Atoms
from metatomic.torch import ModelOutput
from metatomic_ase import MetatomicCalculator
from packaging.version import Version

from ..._version import (
    PET_MAD_DOS_LATEST_STABLE_VERSION,
)
from ._models import (
    _get_bandgap_model,
    _get_fermi_model,
    get_pet_mad_dos,
)
from .utils import (
    dos_from_eigenvalues,
    get_num_electrons,
    pad_dos,
    torch_gaussian_filter1d,
)


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
            atoms, outputs={"mtt::dos": ModelOutput(per_atom=per_atom)}
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
