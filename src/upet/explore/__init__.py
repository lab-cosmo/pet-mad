"""Deprecated: use :mod:`upet.ase.explore` instead."""

import warnings

from ..ase.explore import PETMADFeaturizer


warnings.warn(
    "upet.explore is deprecated, use `from upet.ase.explore import "
    "PETMADFeaturizer` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PETMADFeaturizer"]
