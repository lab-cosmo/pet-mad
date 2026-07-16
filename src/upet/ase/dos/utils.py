from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from ase import Atoms
from ase.units import kB


NUM_ELECTRONS_PER_ELEMENT = {
    "Al": 3.0,
    "As": 5.0,
    "Ba": 10.0,
    "Be": 4.0,
    "Bi": 15.0,
    "B": 3.0,
    "Br": 7.0,
    "Ca": 10.0,
    "Cd": 12.0,
    "Cl": 7.0,
    "Co": 17.0,
    "C": 4.0,
    "Cr": 14.0,
    "Cs": 9.0,
    "Fe": 16.0,
    "F": 7.0,
    "Ga": 13.0,
    "Ge": 14.0,
    "H": 1.0,
    "In": 13.0,
    "I": 7.0,
    "Ir": 15.0,
    "K": 9.0,
    "Li": 3.0,
    "Mg": 2.0,
    "Mn": 15.0,
    "Na": 9.0,
    "Nb": 13.0,
    "Ni": 18.0,
    "N": 5.0,
    "O": 6.0,
    "Os": 16.0,
    "Pb": 14.0,
    "Po": 16.0,
    "P": 5.0,
    "Pt": 16.0,
    "Re": 15.0,
    "Rn": 18.0,
    "Sb": 15.0,
    "Se": 6.0,
    "Si": 4.0,
    "Sn": 14.0,
    "S": 6.0,
    "Sr": 10.0,
    "Ta": 13.0,
    "Te": 6.0,
    "Ti": 12.0,
    "Tl": 13.0,
    "V": 13.0,
    "W": 14.0,
    "Y": 11.0,
    "Zn": 20.0,
    "Zr": 12.0,
    "Ag": 19.0,
    "Ar": 8.0,
    "Au": 19.0,
    "Ce": 12.0,
    "Dy": 20.0,
    "Er": 22.0,
    "Eu": 17.0,
    "Gd": 18.0,
    "He": 2.0,
    "Hf": 12.0,
    "Hg": 20.0,
    "Ho": 21.0,
    "Kr": 8.0,
    "La": 11.0,
    "Lu": 25.0,
    "Mo": 14.0,
    "Nd": 14.0,
    "Ne": 8.0,
    "Pd": 18.0,
    "Pm": 15.0,
    "Pr": 13.0,
    "Rb": 9.0,
    "Rh": 17.0,
    "Ru": 16.0,
    "Sc": 11.0,
    "Sm": 16.0,
    "Tb": 19.0,
    "Tc": 15.0,
    "Tm": 23.0,
    "Xe": 18.0,
    "Yb": 24.0,
    "Cu": 11.0,
}


def fermi_dirac_distribution(
    energies: torch.Tensor, mu: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    """
    Fermi-Dirac distribution function.

    :param energies: Energy grid.
    :param mu: Fermi level.
    :param T: Temperature.
    :return: Fermi-Dirac distribution function.
    """
    x = -(energies - mu) / (kB * T)  # Note the negative sign
    return torch.sigmoid(x)


def get_num_electrons(atoms: Union[Atoms, List[Atoms]]) -> torch.Tensor:
    """
    Get the number of electrons for a given ase.Atoms object, or a list of ase.Atoms
    objects.

    :param atoms: ASE atoms object or a list of ASE atoms objects
    :return: Number of electrons for each ase.Atoms object stored in a torch.Tensor
    format.
    """
    num_electrons = []
    if isinstance(atoms, Atoms):
        atoms = [atoms]
    for item in atoms:
        num_electrons.append(
            int(sum([NUM_ELECTRONS_PER_ELEMENT[symbol] for symbol in item.symbols]))
        )
    num_electrons = torch.tensor(num_electrons)
    return num_electrons


def align_dos(
    predicted_DOS: torch.Tensor, true_DOS: torch.Tensor, mask: torch.Tensor
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
    # Ensure that everything is on the same device and has the same dtype
    device = predicted_DOS.device
    true_DOS = true_DOS.float().to(device)
    original_length = predicted_DOS.shape[1]
    original_predicted_DOS = predicted_DOS.clone()
    dos_pad = torch.zeros_like(predicted_DOS)
    predicted_DOS = torch.hstack([dos_pad, predicted_DOS, dos_pad])
    predicted_DOS = predicted_DOS.float()

    mask = mask.float().to(device)

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
            torch.zeros(len(predicted_DOS), device=predicted_DOS.device).reshape(-1, 1),
            front_tail[:, :shape_difference],
        ]
    )
    total_losses = losses + additional_error
    final_loss, shift = torch.min(total_losses, dim=1)
    aligned_true_DOS = []
    aligned_true_masks = []
    for index, s in enumerate(shift):
        if s - original_length < 0:
            front_cut = int(original_length - s)
            back_pad = torch.zeros(
                predicted_DOS.shape[1] - true_DOS.shape[1] - s - original_length,
                device=predicted_DOS.device,
            )
            true_DOS_padded = torch.hstack(
                [true_DOS[index, front_cut:], back_pad]
            ).int()
            true_Mask_padded = torch.hstack([mask[index], back_pad + 1]).int()
        elif s - original_length > 0:
            front_pad = torch.zeros(s - original_length, device=predicted_DOS.device)
            back_pad = torch.zeros(
                predicted_DOS.shape[1] - true_DOS.shape[1] - s - original_length,
                device=predicted_DOS.device,
            )
            true_DOS_padded = torch.hstack([front_pad, true_DOS[index], back_pad]).int()
            true_Mask_padded = torch.hstack(
                [front_pad + 1, mask[index], back_pad]
            ).int()
        aligned_true_DOS.append(true_DOS_padded)
        aligned_true_masks.append(true_Mask_padded)

    return (
        original_predicted_DOS,
        torch.vstack(aligned_true_DOS),
        torch.vstack(aligned_true_masks),
    )


def dos_from_eigenvalues(
    energy_grid: torch.Tensor,
    sigma: torch.Tensor,
    eigenvalues: torch.Tensor,
    kweights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the density of states and the corresponding mask from a given set of
    eigenvalues and k-point weights. The DOS is computed by broadening each
    eigenvalue with a Gaussian function and summing over all eigenvalues. The mask
    is a boolean tensor indicating which energy grid points are reliable enough to
    compute the loss on.

    :param energy_grid: Tensor containing the energy grid on which to compute the
        DOS. Defaults to the energy grid defined in the PET-MAD-DOS model.
    :param sigma: Standard deviation for Gaussian broadening in eV.
        Defaults to 0.3 eV.
    :param eigenvalues: Tensor of shape (n_kpoints, n_bands) containing the
        eigenvalues.
    :param kweights: Tensor of shape (n_kpoints,) containing the weights of each
        k-point. If None, it is assumed that all k-points have equal weight.
        Defaults to None.
    :return: DOS and mask
    """

    if kweights is None:
        kweights = (
            torch.ones(eigenvalues.shape[0], device=eigenvalues.device)
            / eigenvalues.shape[0]
        )
    device = energy_grid.device
    confident_upper_energy_bound = torch.min(eigenvalues[:, -1]) - 3 * sigma
    eigenvalues = eigenvalues.to(device)
    kweights = kweights.to(device)
    n_bands = eigenvalues.shape[1]
    eigenvalues = eigenvalues.flatten()
    kweights = kweights.squeeze().repeat(n_bands)
    delta_E = (energy_grid - eigenvalues[:, None]) / sigma
    gaussian_weights = torch.exp(-0.5 * delta_E**2)
    normalization = 1 / torch.sqrt(2 * torch.pi * sigma**2)
    dos = torch.sum(kweights[:, None] * gaussian_weights, dim=0) * normalization
    mask = (energy_grid <= confident_upper_energy_bound).to(torch.int)

    return dos, mask


def pad_dos(
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
        return dos, mask
    padding_length = target_length - current_length
    dos_padded = F.pad(dos, (padding_length, 0), mode="constant", value=0)
    mask_padded = F.pad(mask, (padding_length, 0), mode="constant", value=0)
    return dos_padded, mask_padded


def torch_gaussian_filter1d(
    input_tensor: torch.Tensor, sigma: float, truncate: float = 4.0
) -> torch.Tensor:
    """
    Pure PyTorch implementation of SciPy's gaussian_filter1d.
    Supports batched 1D inputs of shape (B, N).
    """
    if sigma <= 0:
        return input_tensor

    # Calculate the radius and kernel size (matches SciPy's truncation logic)
    radius = int(truncate * sigma + 0.5)
    # Create the 1D coordinate grid centered at 0
    x = torch.arange(
        -radius, radius + 1, device=input_tensor.device, dtype=input_tensor.dtype
    )
    # Compute the Gaussian kernel: exp(-x^2 / (2 * sigma^2))
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()  # Normalize to sum to 1

    # Reshape kernel for conv1d: (out_channels, in_channels/groups, kernel_width)
    kernel = kernel.view(1, 1, -1)

    # Reshape input to (B, 1, N) for conv1d processing
    original_shape = input_tensor.shape
    if len(original_shape) == 1:
        x_input = input_tensor.view(1, 1, -1)
    elif len(original_shape) == 2:
        x_input = input_tensor.unsqueeze(1)
    else:
        raise ValueError("Expected a 1D or 2D (batched) tensor.")

    # Apply reflect padding to mirror SciPy's default behavior ('reflect')
    x_padded = F.pad(x_input, (radius, radius), mode="reflect")

    output = F.conv1d(x_padded, kernel, groups=1)

    return output.squeeze(1)
