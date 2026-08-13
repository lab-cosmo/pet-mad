"""Running the uncertainty outputs of a model.

The functions here take the calculator wrapping the model, and an already
resolved :py:class:`ase.Atoms`. Whether the model provides these outputs at all
is decided by the caller, which is the only one able to say so.
"""

from typing import List, Optional, Tuple

import numpy as np
from ase import Atoms
from metatomic.torch import ModelOutput
from metatomic_ase import MetatomicCalculator


def stress_ensemble_to_voigt(stress_ensemble: np.ndarray) -> np.ndarray:
    """Convert a [3, 3, n_ensemble] stress ensemble to Voigt [6, n_ensemble].

    ASE's public ``full_3x3_to_voigt_6_stress`` averages each off-diagonal
    component with itself instead of with its transpose, so the symmetrized
    conversion is written out here.
    """
    s = stress_ensemble
    return np.stack(
        [
            s[0, 0],
            s[1, 1],
            s[2, 2],
            (s[1, 2] + s[2, 1]) / 2,
            (s[0, 2] + s[2, 0]) / 2,
            (s[0, 1] + s[1, 0]) / 2,
        ]
    )


def run_energy_uq(
    calculator: MetatomicCalculator,
    atoms: Atoms,
    key: str,
    per_atom: bool = False,
) -> np.ndarray:
    """Get the ``key`` output of the model, which is an energy."""
    outputs = calculator.run_model(
        atoms,
        outputs={
            key: ModelOutput(unit="eV", sample_kind="atom" if per_atom else "system")
        },
    )

    return outputs[key].block().values.detach().cpu().numpy()


def run_forces_stress_uq(
    calculator: MetatomicCalculator,
    atoms: Atoms,
    ensemble_key: str,
    energy_key: Optional[str],
    compute_forces: bool = True,
    compute_stress: bool = True,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute force and/or stress ensembles as gradients of the energy ensemble.

    Returns a tuple (forces_ensemble, stress_ensemble) where each is None if not
    requested. Forces ensemble has shape [n_atoms, 3, n_ensemble], stress ensemble
    has shape [3, 3, n_ensemble].
    """
    assert compute_forces or compute_stress

    explicit_gradients: List[str] = []
    if compute_forces:
        explicit_gradients.append("positions")
    if compute_stress:
        explicit_gradients.append("strain")

    outputs_request = {
        ensemble_key: ModelOutput(
            unit="eV", sample_kind="system", explicit_gradients=explicit_gradients
        )
    }
    if energy_key is not None:
        # PET models refuse an ensemble request that does not also ask for the
        # energy it is built from; the value itself is unused here
        outputs_request[energy_key] = ModelOutput(unit="eV", sample_kind="system")

    block = calculator.run_model(atoms, outputs_request)[ensemble_key].block()

    forces_ensemble = None
    stress_ensemble = None

    if compute_forces:
        # gradient shape: [n_atoms, 3, n_ensemble]
        forces_ensemble = (
            -block.gradient("positions").values.detach().cpu().double().numpy()
        )
        # remove the mean over atoms for each ensemble member to impose
        # translational invariance
        forces_ensemble = forces_ensemble - np.mean(
            forces_ensemble, axis=0, keepdims=True
        )

    if compute_stress:
        # gradient shape: [1, 3, 3, n_ensemble] (single system)
        # -> [3, 3, n_ensemble]
        stress_ensemble = (
            block.gradient("strain").values.detach().cpu().double().numpy()[0]
            / atoms.cell.volume
        )

    return forces_ensemble, stress_ensemble


def run_direct_forces_uq(
    calculator: MetatomicCalculator, atoms: Atoms, key: str
) -> np.ndarray:
    """Get the direct forces ensemble, shape [n_atoms, 3, n_ensemble]."""
    outputs = calculator.run_model(
        atoms,
        outputs={key: ModelOutput(unit="eV/Angstrom", sample_kind="atom")},
    )

    # shape: [n_atoms, 3, n_ensemble]
    return outputs[key].block().values.detach().cpu().numpy()


def run_direct_forces_uncertainty(
    calculator: MetatomicCalculator, atoms: Atoms, key: str
) -> np.ndarray:
    """Get the model's built-in direct forces uncertainty, shape [n_atoms, 3]."""
    outputs = calculator.run_model(
        atoms,
        outputs={key: ModelOutput(unit="eV/Angstrom", sample_kind="atom")},
    )

    # shape: [n_atoms, 3, 1] -> [n_atoms, 3]
    return outputs[key].block().values.detach().cpu().numpy().squeeze(-1)
