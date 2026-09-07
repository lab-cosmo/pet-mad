"""The compiled backend matches eager.

``from_checkpoint(compile_model=True)`` wraps the three ``PETBackend``
building blocks in ``torch.compile``; these check that doing so does not
change the numbers.

On macOS inductor's generated C++ kernel would otherwise link a second
``libomp.dylib`` and OpenMP would abort the interpreter (``OMP: Error
#15``), taking the whole session with it; ``conftest._use_torch_libomp``
points inductor at the copy torch already has open.
"""

from __future__ import annotations

import os

import pytest
import torch


pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed; skipping")


from _helpers import make_pbc_water  # noqa: E402

from upet._models import _resolve_and_download_checkpoint  # noqa: E402
from upet.nvalchemi import UPETWrapper  # noqa: E402


_CHECKPOINT_MODEL = "pet-mad-xs"
_CHECKPOINT_VERSION = "1.5.0"  # 'grid' adaptive cutoff
_SOLVER_CHECKPOINT_PATH = "pet-mad-xs-v1.6.0.ckpt"  # 'solver' adaptive cutoff


def test_adaptive_cutoff_grid_compilation_raises_error():
    """Tests if compiling a 'grid' adaptive-cutoff model is rejected up front."""
    model_name, size = _CHECKPOINT_MODEL.rsplit("-", 1)
    _, _, checkpoint_path = _resolve_and_download_checkpoint(
        model_name, size, _CHECKPOINT_VERSION
    )
    with pytest.raises(
        ValueError, match="is not supported for PET models using the 'grid'"
    ):
        UPETWrapper.from_checkpoint(
            checkpoint_path=checkpoint_path,
            compile_model=True,
        )


@pytest.mark.parametrize("fullgraph", [True, False])
def test_adaptive_cutoff_solver_compilation_works(fullgraph):
    """Compiling a 'solver' adaptive-cutoff model works."""
    if not os.path.exists(_SOLVER_CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint {_SOLVER_CHECKPOINT_PATH} not found.")
    UPETWrapper.from_checkpoint(
        checkpoint_path=_SOLVER_CHECKPOINT_PATH,
        compile_model=True,
        fullgraph=fullgraph,
    )


@pytest.mark.parametrize("fullgraph", [True, False])
def test_compiled_inference_matches_eager(fullgraph):
    if not os.path.exists(_SOLVER_CHECKPOINT_PATH):
        pytest.skip(f"Checkpoint {_SOLVER_CHECKPOINT_PATH} not found.")
    eager = UPETWrapper.from_checkpoint(
        checkpoint_path=_SOLVER_CHECKPOINT_PATH,
        compile_model=False,
    )
    compiled = UPETWrapper.from_checkpoint(
        checkpoint_path=_SOLVER_CHECKPOINT_PATH,
        compile_model=True,
        fullgraph=fullgraph,
    )
    eager.model_config.active_outputs = {"energy", "forces", "stress"}
    compiled.model_config.active_outputs = {"energy", "forces", "stress"}
    water = make_pbc_water()
    e_eager = eager.forward(water)["energy"]
    e_compiled = compiled.forward(water)["energy"]
    f_eager = eager.forward(water)["forces"]
    f_compiled = compiled.forward(water)["forces"]
    s_eager = eager.forward(water)["stress"]
    s_compiled = compiled.forward(water)["stress"]
    torch.testing.assert_close(e_compiled.detach(), e_eager.detach())
    torch.testing.assert_close(f_compiled.detach(), f_eager.detach())
    torch.testing.assert_close(s_compiled.detach(), s_eager.detach())
