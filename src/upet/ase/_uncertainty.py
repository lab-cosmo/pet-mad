"""Running the uncertainty outputs of a model.

The functions here take the calculator wrapping the model, and an already
resolved :py:class:`ase.Atoms`. Whether the model provides these outputs at all
is decided by the caller, which is the only one able to say so.
"""

from typing import Tuple

import numpy as np
from ase import Atoms
from metatomic.torch import ModelOutput
from metatomic_ase import MetatomicCalculator


UQ_ERROR_MSG = (
    "{key} is not available for the selected model. "
    "The documentation lists the models providing uncertainty estimates."
)

UQ_GRAD_ERROR_MSG = (
    "{key} is required for calculating the gradient ensemble uncertainty "
    "(forces, stress), but is not available for the selected model. "
    "The documentation lists the models providing uncertainty estimates."
)

UQ_NC_ERROR_MSG = (
    "Non-conservative {key} uncertainty/ensemble is not available for the selected "
    "model. Consider switching-off `non-conservative` mode. "
    "The documentation lists the models providing uncertainty estimates "
    "for non-conservative outputs."
)


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


def run_direct_uq(
    calculator: MetatomicCalculator,
    atoms: Atoms,
    key: str,
    per_atom: bool = False,
) -> np.ndarray:
    if "force" in key:
        unit = "eV/A"
        per_atom = True
    elif "stress" in key:
        unit = "eV/A^3"
        per_atom = False
    else:
        unit = "eV"
    outputs = calculator.run_model(
        atoms,
        outputs={
            key: ModelOutput(unit=unit, sample_kind="atom" if per_atom else "system")
        },
    )

    return outputs[key].block().values.detach().cpu().numpy().squeeze()


def run_gradient_ensemble_uq(
    calculator: MetatomicCalculator,
    atoms: Atoms,
    key: str,
    gradients: Tuple[str, ...] = ("positions", "strain"),
):
    output = calculator.run_model(
        atoms,
        outputs={
            key: ModelOutput(
                unit="eV", sample_kind="system", explicit_gradients=list(gradients)
            )
        },
    )[key].block()

    results = {}
    for gradient in gradients:
        gradient_ensemble = (
            output.gradient(gradient).values.detach().cpu().double().numpy()
        )
        if gradient == "positions":
            gradient_ensemble *= -1.0
            gradient_ensemble -= np.mean(gradient_ensemble, axis=0, keepdims=True)
        elif gradient == "strain":
            gradient_ensemble = gradient_ensemble[0] / atoms.cell.volume
        results[gradient] = gradient_ensemble

    return results
