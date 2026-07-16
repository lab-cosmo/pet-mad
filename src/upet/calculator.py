"""Deprecated: use :mod:`upet.ase` / :mod:`upet.ase.dos` instead."""

import warnings

from .ase import UPETCalculator
from .ase.dos import PETMADDOSCalculator


warnings.warn(
    "upet.calculator is deprecated; use `from upet.ase import UPETCalculator` "
    "and `from upet.ase.dos import PETMADDOSCalculator` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["UPETCalculator", "PETMADDOSCalculator"]
