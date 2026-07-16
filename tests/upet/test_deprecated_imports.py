"""Tests for the deprecated top-level import shims.

``upet.calculator`` and ``upet.explore`` were the pre-restructure public
paths for ``UPETCalculator``/``PETMADDOSCalculator`` and
``PETMADFeaturizer`` respectively. They must keep working (with a
``DeprecationWarning``) so external code importing from them doesn't break.
"""

import importlib
import sys

import pytest


def _fresh_import(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_calculator_shim_warns_and_reexports():
    with pytest.warns(DeprecationWarning, match="upet.calculator is deprecated"):
        shim = _fresh_import("upet.calculator")

    from upet.ase import UPETCalculator
    from upet.ase.dos import PETMADDOSCalculator

    assert shim.UPETCalculator is UPETCalculator
    assert shim.PETMADDOSCalculator is PETMADDOSCalculator


def test_explore_shim_warns_and_reexports():
    with pytest.warns(DeprecationWarning, match="upet.explore is deprecated"):
        shim = _fresh_import("upet.explore")

    from upet.ase.explore import PETMADFeaturizer

    assert shim.PETMADFeaturizer is PETMADFeaturizer
