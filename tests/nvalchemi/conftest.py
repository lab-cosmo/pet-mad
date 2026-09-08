"""Fixtures shared by the `upet.nvalchemi` test modules.

pytest imports this file before the ``test_wrapper_*`` modules run their
``pytest.importorskip("nvalchemi")`` guard, so nothing here may import
``nvalchemi`` (or ``_helpers``, which does) at module scope — the fixture
bodies import what they need instead. Only the fixtures of tests that got
past the guard ever execute.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import types
from typing import TYPE_CHECKING

import pytest
import torch


if TYPE_CHECKING:
    from nvalchemi.data import Batch

    from upet.nvalchemi import UPETWrapper


# metatrain.pet.__init__ imports metatrain.pet.trainer, which imports
# metatrain.utils.distributed.slurm, which pulls in `hostlist` — a SLURM-only
# helper most environments don't have installed. Stub it out so the import
# chain resolves and these tests can still exercise PET's pure-torch code
# paths. (upet.nvalchemi.wrapper does the same thing internally, but only
# once it is imported, so this stays self-contained regardless of import
# order.)
if "hostlist" not in sys.modules:
    try:
        import hostlist  # noqa: F401
    except ImportError:
        sys.modules["hostlist"] = types.ModuleType("hostlist")


def _use_torch_libomp() -> None:
    """Point inductor at the same ``libomp`` torch already has open (macOS).

    Inductor builds its C++ kernels with Apple clang, which needs an explicit
    ``-lomp``. When conda's ``llvm-openmp`` is installed it wins the search in
    ``torch._inductor.cpp_builder._get_openmp_args``, so the kernel links a
    *second* OpenMP runtime — conda's install name is ``@rpath/libomp.dylib``
    while the one torch bundles carries an absolute path, so dyld loads both.
    The moment a compiled kernel enters a parallel region the second runtime
    initializes and aborts the interpreter (``OMP: Error #15``), which kills
    the whole pytest session rather than a single test.

    ``OMP_PREFIX`` is checked before conda, so pointing it at the prefix that
    torch's own ``libomp`` was built against makes both link the same image.
    """
    if sys.platform != "darwin" or "OMP_PREFIX" in os.environ:
        return

    torch_libomp = pathlib.Path(torch.__file__).parent / "lib" / "libomp.dylib"
    if not torch_libomp.exists():
        return
    try:
        # `otool -D` prints the library's install name, e.g.
        # /opt/homebrew/opt/libomp/lib/libomp.dylib -> prefix /opt/homebrew/opt/libomp
        install_name = (
            subprocess.run(
                ["otool", "-D", str(torch_libomp)],
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.splitlines()[-1]
            .strip()
        )
    except (OSError, subprocess.CalledProcessError, IndexError):
        return

    prefix = pathlib.Path(install_name).parent.parent
    # Inductor needs the headers too, and rejects the prefix without them.
    if (prefix / "include" / "omp.h").exists():
        os.environ["OMP_PREFIX"] = str(prefix)


_use_torch_libomp()


@pytest.fixture
def wrapper() -> UPETWrapper:
    """UPETWrapper with zero composition and unit scaler."""
    from _helpers import ATOMIC_NUMBERS, tiny_hypers

    from upet.nvalchemi import UPETWrapper

    torch.manual_seed(0)
    composition = torch.zeros(len(ATOMIC_NUMBERS), dtype=torch.float32)
    scale = torch.tensor(1.0, dtype=torch.float32)
    return UPETWrapper(
        atomic_types=ATOMIC_NUMBERS,
        hypers=tiny_hypers(),
        composition_energy=composition,
        scale_energy=scale,
    )


@pytest.fixture
def single_batch() -> Batch:
    from _helpers import make_water
    from nvalchemi.data import Batch

    return Batch.from_data_list([make_water()])


@pytest.fixture
def multi_batch() -> Batch:
    """Two H2O molecules as a batched system (B=2, N=6)."""
    from _helpers import make_water
    from nvalchemi.data import Batch

    return Batch.from_data_list([make_water(), make_water()])


@pytest.fixture
def pbc_batch() -> Batch:
    from _helpers import make_pbc_water
    from nvalchemi.data import Batch

    return Batch.from_data_list([make_pbc_water()])
