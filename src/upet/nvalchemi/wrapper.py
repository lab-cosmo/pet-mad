# SPDX-License-Identifier: BSD-3-Clause
"""UPET / PET model wrapper for `nvalchemi-toolkit
<https://github.com/NVIDIA/nvalchemi-toolkit>`_.

Wraps the pure-torch :class:`metatrain.pet.modules.backend.PETBackend` as a
:class:`nvalchemi.models.base.BaseModelMixin`-compatible model, so that any
UPET / PET-MAD checkpoint can be driven through nvalchemi-toolkit's batched
data pipeline, neighbor-list hooks, and MD integrators. ``PETBackend`` is the
structure-preprocessing / featurization / prediction core of the PET
architecture, operating purely on :class:`torch.Tensor` objects (no
``metatomic.torch.System`` / ``metatensor.torch.TensorMap`` at call time), so
it is ``torch.compile``-friendly.

:class:`UPETWrapper` owns a ``PETBackend`` (built from hypers + atomic types)
and adds only the nvalchemi-specific glue:

* translating a :class:`nvalchemi.data.Batch` into the concatenated plain
  tensors the backend expects (:meth:`UPETWrapper.adapt_input`);
* driving the three backend building blocks (``preprocess``,
  ``calculate_features``, ``predict``);
* gradient / affine-strain wiring for conservative forces and stress;
* the flat composition / scaler buffers decoded from the checkpoint.

Usage
-----
Fetch a named UPET model directly from HuggingFace (see
https://github.com/lab-cosmo/upet for available names, or list them
programmatically via :func:`upet.list_upet`)::

    from upet.nvalchemi import UPETWrapper
    import torch

    model = UPETWrapper.from_checkpoint(
        model="pet-mad-s", version="1.5.0", device=torch.device("cuda")
    )

Or load a local checkpoint file directly (e.g. ``pet-mad-xs-v1.5.0.ckpt``)::

    model = UPETWrapper.from_checkpoint(
        checkpoint_path="pet-mad-xs-v1.5.0.ckpt", device=torch.device("cuda")
    )

Notes
-----
* Forces and stress are derived from the energy via autograd
  (``autograd_outputs = {"forces", "stress"}``). The non-conservative PET
  heads are intentionally skipped.
* The upstream composition model and scaler — originally wrapped as
  ``metatomic.torch.AtomisticModel`` with serialized ``TensorMap`` buffers
  — are decoded once at :meth:`UPETWrapper.from_checkpoint` time into two
  flat torch buffers (``composition_energy``, ``scale_energy``) so the
  forward path has no metatensor dependency.
* Only the ``energy`` output is registered on the backend; the long-range
  module is skipped entirely.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models._utils import (
    autograd_forces_and_stresses,
    autograd_stresses,
    prepare_strain,
)
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
from torch import nn

from .._models import _resolve_and_download_checkpoint
from .utils import (
    ENERGY_OUTPUT_SHAPES,
    decode_tensor_map_values,
    ensure_hostlist_stub,
    filter_state_dict,
    ignore_nonleaf_grad_warning,
    normalize_hypers,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

ensure_hostlist_stub()

__all__ = ["UPETWrapper"]


class UPETWrapper(nn.Module, BaseModelMixin):
    """:class:`nvalchemi.models.base.BaseModelMixin` wrapper around UPET/PET.

    Builds and owns a :class:`metatrain.pet.modules.backend.PETBackend`
    (from *hypers* and *atomic_types*) and drives its three building
    blocks. Handles:

    * translating a :class:`nvalchemi.data.Batch` into the concatenated
      plain tensors consumed by ``PETBackend.preprocess``
      (:meth:`adapt_input`);
    * enabling gradients on ``positions`` when autograd outputs are active,
      and wiring the affine strain trick for stress;
    * applying the flat composition / scaler buffers decoded from the
      checkpoint at load time;
    * producing :class:`nvalchemi._typing.ModelOutputs` with ``energy``,
      ``forces``, and ``stress``.

    :param atomic_types: Atomic numbers in species-index order.
    :param hypers: PET hyper-parameters (see
        :data:`upet.nvalchemi.utils.REQUIRED_HYPERS`).
    :param composition_energy: Per-species reference energy, shape
        ``[num_species]``, indexed by species index (not atomic number).
    :param scale_energy: Scalar (0-dim) tensor used as the global energy
        scale.
    """

    def __init__(
        self,
        atomic_types: Sequence[int],
        hypers: dict[str, Any],
        composition_energy: torch.Tensor,
        scale_energy: torch.Tensor,
    ) -> None:
        from metatrain.pet.modules.backend import PETBackend

        super().__init__()

        self.atomic_types = list(atomic_types)
        self.hypers = normalize_hypers(hypers)
        # Width of the adaptive-cutoff smooth taper; not cached on
        # ``PETBackend`` itself, so the wrapper keeps its own copy to pass
        # into ``preprocess`` at call time.
        self._cutoff_width_adaptive = float(self.hypers["cutoff_width_adaptive"])

        self.backend = PETBackend(self.hypers, self.atomic_types)
        self.backend.add_output("energy", ENERGY_OUTPUT_SHAPES)

        # Set to True by `from_checkpoint(compile_model=True)`, which
        # compiles the three backend methods; controls the Dynamo config
        # patching applied around the backend calls in `forward` /
        # `compute_embeddings`.
        self._compiled = False

        # Per-species reference energy (shape [num_species]) indexed by
        # species index (backend.species_to_species_index lookup), not
        # atomic number. Non-persistent: decoded at `from_checkpoint` time
        # from the raw metatensor buffer.
        self.register_buffer(
            "composition_energy", composition_energy.clone(), persistent=False
        )
        self.register_buffer(
            "scale_energy", scale_energy.clone().reshape(()), persistent=False
        )

        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            autograd_outputs=frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset(),
            optional_inputs=frozenset({"cell", "neighbor_list_shifts"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=float(self.hypers["cutoff"]),
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )

    # ------------------------------------------------------------------
    # BaseModelMixin required properties
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Node/graph embedding shapes.

        Embeddings concatenate the per-layer node features with the
        cutoff-weighted, neighbor-summed per-layer edge features (see
        :meth:`compute_embeddings`), so the dimension is
        ``num_readout_layers * (d_node + d_pet)``.
        """
        dim = self.backend.num_readout_layers * (
            self.backend.d_node + self.backend.d_pet
        )
        return {"node_embeddings": (dim,), "graph_embeddings": (dim,)}

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def cutoff(self) -> float:
        """Interaction cutoff in Angstroms."""
        return float(self.backend.cutoff)

    @property
    def _model_dtype(self) -> torch.dtype:
        """Current dtype of the backend's parameters.

        Read live from ``parameters()`` so it stays correct after
        ``.to(dtype=...)`` calls.
        """
        try:
            return next(self.backend.parameters()).dtype
        except StopIteration:
            return torch.float32

    # ------------------------------------------------------------------
    # Backend invocation (compile-aware)
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _backend_ctx(self):
        """Context for calling the backend building blocks.

        When the backend methods have been ``torch.compile``-d
        (``self._compiled``), the Dynamo flags required to capture the
        data-dependent ``max_edges_per_node`` size must be active while the
        compiled functions trace. In eager mode this is a no-op.
        """
        if self._compiled:
            with (
                torch._dynamo.config.patch(
                    capture_scalar_outputs=True,
                    capture_dynamic_output_shape_ops=True,
                    specialize_int=True,
                ),
                ignore_nonleaf_grad_warning(),
            ):
                yield
        else:
            yield

    # ------------------------------------------------------------------
    # Input preparation
    # ------------------------------------------------------------------

    def adapt_input(
        self, data: AtomicData | Batch, **_kwargs: Any
    ) -> dict[str, torch.Tensor]:
        """Translate a :class:`nvalchemi.data.Batch` into backend input tensors.

        Produces the concatenated, plain-tensor structure representation
        that ``PETBackend.preprocess`` consumes. All the edge manipulation
        (NEF reshaping, adaptive cutoffs, reversed-neighbor indexing) then
        happens inside the backend.

        Handles ``AtomicData -> Batch`` promotion and gradient enabling on
        ``positions`` when an autograd output is active. Strain handling
        (for stress) is done by :meth:`forward` **before** calling this
        method, so that the scaled positions/cell flow through the full
        featurisation.

        :param data: Input data. ``AtomicData`` inputs are promoted to a
            single-graph ``Batch``.
        :return: Keyword arguments for ``PETBackend.preprocess``:
            ``positions``, ``centers``, ``neighbors``, ``species``,
            ``cells``, ``cell_shifts``, ``system_indices``.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        dtype = self._model_dtype

        # Cast positions to model dtype and enable gradients when needed.
        # Clone so the original batch tensor is never mutated in-place.
        positions = data.positions.to(dtype=dtype)
        if self.model_config.autograd_outputs & self.model_config.active_outputs:
            positions = positions.clone()
            positions.requires_grad_(True)
        # Store the prepared positions back on the batch so that downstream
        # autograd calls in `forward` can reach them via `data.positions`.
        data["positions"] = positions

        return self._collect_backend_inputs(data, dtype)

    def _collect_backend_inputs(
        self, data: Batch, dtype: torch.dtype
    ) -> dict[str, torch.Tensor]:
        """Gather the ``PETBackend.preprocess`` kwargs from a prepared batch.

        Reads ``data.positions`` as-is (the caller is responsible for any
        dtype cast / gradient setup) and assembles the remaining structure
        tensors.

        :param data: Batch whose ``positions`` are already prepared.
        :param dtype: Model dtype, used to cast ``cells``.
        :return: Keyword arguments for ``PETBackend.preprocess``.
        """
        positions = data.positions
        device = positions.device
        num_graphs = int(data.num_graphs)

        centers = data.neighbor_list[:, 0].long()
        neighbors = data.neighbor_list[:, 1].long()
        species = data.atomic_numbers.long()
        system_indices = data.batch_idx.long()

        # Integer PBC shifts [E, 3] — zero for non-PBC systems.
        raw_shifts = getattr(data, "neighbor_list_shifts", None)
        if raw_shifts is None:
            cell_shifts = torch.zeros(
                centers.shape[0], 3, dtype=torch.long, device=device
            )
        else:
            cell_shifts = raw_shifts.to(dtype=torch.long, device=device)

        # Cell [B, 3, 3] — identity for non-PBC systems.
        raw_cell = getattr(data, "cell", None)
        if raw_cell is None:
            cells = (
                torch.eye(3, dtype=dtype, device=device)
                .unsqueeze(0)
                .expand(num_graphs, -1, -1)
                .contiguous()
            )
        else:
            cells = raw_cell.to(dtype=dtype, device=device)

        return {
            "positions": positions,
            "centers": centers,
            "neighbors": neighbors,
            "species": species,
            "cells": cells,
            "cell_shifts": cell_shifts,
            "system_indices": system_indices,
        }

    def adapt_output(
        self,
        raw_output: dict[str, torch.Tensor | None],
        data: AtomicData | Batch,
    ) -> ModelOutputs:
        """Map raw PET outputs to the standard ``ModelOutputs`` layout.

        :param raw_output: Dict with optional ``energy``, ``forces``,
            ``stress`` tensors.
        :param data: Original input batch (forwarded to
            ``BaseModelMixin.adapt_output``).
        :return: Ordered dict keyed by the wrapper's active outputs.
        """
        mapped: dict[str, torch.Tensor] = {}
        energy = raw_output.get("energy")
        if energy is not None:
            mapped["energy"] = energy.unsqueeze(-1) if energy.ndim == 1 else energy
        if raw_output.get("forces") is not None:
            mapped["forces"] = raw_output["forces"]
        if raw_output.get("stress") is not None:
            mapped["stress"] = raw_output["stress"]
        return super().adapt_output(mapped, data)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> ModelOutputs:
        """Run the PET backend and return energy / forces / stress.

        The energy comes from ``PETBackend.preprocess`` ->
        ``PETBackend.calculate_features`` -> ``PETBackend.predict`` (the
        latter already sums the node and cutoff-weighted edge contributions
        over all readout layers). The flat scaler / composition buffers are
        then applied.

        Conservative forces are derived via :func:`torch.autograd.grad` of
        the total energy with respect to positions. Stresses use the
        affine-strain trick from ``nvalchemi.models._utils``.

        :param data: Input batch.
        :param kwargs: Forwarded to :meth:`adapt_input`.
        :return: Dict with the active output keys populated.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        active = self.model_config.active_outputs & self.model_config.outputs
        compute_forces = "forces" in active
        compute_stresses = "stress" in active

        # Set up the affine strain BEFORE adapt_input so the scaled
        # positions and cell flow through the full featurisation.
        displacement: torch.Tensor | None = None
        orig_positions: torch.Tensor | None = None
        orig_cell: torch.Tensor | None = None
        if compute_stresses and getattr(data, "cell", None) is not None:
            scaled_pos, scaled_cell, displacement = prepare_strain(
                data.positions.to(self._model_dtype),
                data.cell.to(self._model_dtype),
                data.batch_idx,
            )
            orig_positions = data.positions
            orig_cell = data.cell
            data["positions"] = scaled_pos
            data["cell"] = scaled_cell

        inputs = self.adapt_input(data, **kwargs)
        positions = data.positions  # updated in-place by adapt_input

        with self._backend_ctx():
            batch_data = self.backend.preprocess(
                inputs["positions"],
                inputs["centers"],
                inputs["neighbors"],
                inputs["species"],
                inputs["cells"],
                inputs["cell_shifts"],
                inputs["system_indices"],
                self._cutoff_width_adaptive,
            )
            node_features_list, edge_features_list = self.backend.calculate_features(
                batch_data
            )
            atomic_predictions, _, _ = self.backend.predict(
                node_features_list,
                edge_features_list,
                batch_data,
                inputs["cells"],
                inputs["system_indices"],
                ["energy"],
            )

        per_atom = atomic_predictions["energy"][0]  # [N, 1]
        species_idx = self.backend.species_to_species_index[inputs["species"]]
        # Scaler first, then composition (matches upstream PET ordering).
        per_atom = self.scale_energy * per_atom
        per_atom = per_atom + self.composition_energy[species_idx].unsqueeze(-1)

        num_graphs = int(data.num_graphs)
        energy = torch.zeros(
            num_graphs, 1, dtype=per_atom.dtype, device=per_atom.device
        )
        energy.scatter_add_(0, inputs["system_indices"].unsqueeze(-1), per_atom)

        result: dict[str, torch.Tensor] = {"energy": energy}

        need_stress = (
            compute_stresses and displacement is not None and orig_cell is not None
        )
        if compute_forces and need_stress:
            # A single backward for both forces and stress. Two separate
            # ``autograd.grad`` calls would run backward twice over the
            # same graph, which clashes with ``torch.compile``'s
            # donated-buffer optimization.
            forces, stress = autograd_forces_and_stresses(
                energy,
                positions,
                displacement,
                orig_cell,
                num_graphs,
            )
            result["forces"] = forces
            result["stress"] = stress
        elif compute_forces:
            (grad,) = torch.autograd.grad(energy.sum(), positions)
            result["forces"] = -grad
        elif need_stress:
            result["stress"] = autograd_stresses(
                energy,
                displacement,
                orig_cell,
                num_graphs,
            )

        # Restore the batch's original positions/cell if strain was
        # applied, so the caller sees no mutation from the stress trick.
        if orig_positions is not None and orig_cell is not None:
            data["positions"] = orig_positions
            data["cell"] = orig_cell

        return self.adapt_output(result, data)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Compute node and graph embeddings without autograd.

        The node embedding is the concatenation of the per-layer node
        features with the cutoff-weighted, neighbor-summed per-layer edge
        features::

            node = cat(node_features_list, dim=1)
            edge = (cat(edge_features_list, dim=2) * cutoff_factors).sum(neighbors)
            feats = cat([node, edge], dim=1)

        Writes ``node_embeddings``
        (``[N, num_readout_layers*(d_node+d_pet)]``) and
        ``graph_embeddings`` (``[B, ...]``, sum-pooled over atoms) into
        *data* and returns it. Does **not** mutate ``model_config``.

        :param data: Input data.
        :param kwargs: Forwarded to :meth:`adapt_input`.
        :return: The same batch with ``node_embeddings`` and
            ``graph_embeddings`` attached.
        """
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        with torch.no_grad():
            # Build inputs without enabling gradients on positions
            # (embeddings are autograd-free), so adapt_input's grad toggle
            # is bypassed.
            data["positions"] = data.positions.to(dtype=self._model_dtype)
            inputs = self._collect_backend_inputs(data, self._model_dtype)
            with self._backend_ctx():
                batch_data = self.backend.preprocess(
                    inputs["positions"],
                    inputs["centers"],
                    inputs["neighbors"],
                    inputs["species"],
                    inputs["cells"],
                    inputs["cell_shifts"],
                    inputs["system_indices"],
                    self._cutoff_width_adaptive,
                )
                node_features_list, edge_features_list = (
                    self.backend.calculate_features(batch_data)
                )

            node_features = torch.cat(node_features_list, dim=1)
            edge_features = torch.cat(edge_features_list, dim=2)
            edge_features = (
                edge_features * batch_data["cutoff_factors"][:, :, None]
            ).sum(dim=1)
            node_feats = torch.cat([node_features, edge_features], dim=1)

        # Write node embeddings directly to the atoms group to avoid the
        # default "system" routing used by `setattr` on unknown keys.
        atoms_group = data._atoms_group
        if atoms_group is not None:
            atoms_group["node_embeddings"] = node_feats
        else:
            data.node_embeddings = node_feats

        hidden_dim = node_feats.shape[-1]
        graph_embeddings = torch.zeros(
            data.num_graphs,
            hidden_dim,
            device=node_feats.device,
            dtype=node_feats.dtype,
        )
        graph_embeddings.scatter_add_(
            0,
            data.batch_idx.long().unsqueeze(-1).expand(-1, hidden_dim),
            node_feats,
        )
        data.graph_embeddings = graph_embeddings
        return data

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        model: str | None = None,
        version: str | None = "latest",
        dtype: torch.dtype | None = None,
        checkpoint_path: Path | str | None = None,
        device: torch.device | None = None,
        compile_model: bool = False,
        **compile_kwargs: Any,
    ) -> "UPETWrapper":
        """Load a UPET/PET checkpoint from disk or HuggingFace.

        Either *checkpoint_path* (a local file) or *model* (a name to fetch
        from HuggingFace, e.g. ``"pet-mad-s"``, optionally with *version*)
        must be given. The full list of available models and versions can
        be listed programmatically via :func:`upet.list_upet`.

        :class:`UPETWrapper` supports compiling the PET backend building
        blocks (``preprocess`` / ``calculate_features`` / ``predict``) via
        ``torch.compile``, controlled by *compile_model*. Models using the
        ``'grid'`` adaptive-cutoff method (e.g. ``pet-mad`` <= v1.5.0)
        cannot be compiled because of a break in the autograd backward; use
        a ``'solver'``-method checkpoint (``pet-mad`` >= v1.6.0) instead.

        :param model: Model name to fetch from HuggingFace, either a
            combined ``<model>-<size>`` name (e.g. ``"pet-mad-s"``) or a
            bare base name (e.g. ``"pet-mad"``). Used when
            *checkpoint_path* is ``None``; ignored otherwise.
        :param version: Model version to fetch, or ``"latest"`` / ``None``
            for the newest available. Ignored when *checkpoint_path* is
            given. Defaults to ``"latest"``.
        :param dtype: If set, cast the backend and composition/scaler
            buffers to this dtype before returning.
        :param checkpoint_path: Path to a local PET checkpoint file
            (``.ckpt`` / ``.pt``). If ``None``, *model* must be given
            instead.
        :param device: Target device. Defaults to CPU.
        :param compile_model: ``torch.compile`` the three backend building
            blocks. Sets eval mode and freezes parameters; the model is
            **inference-only** after this step.
        :param compile_kwargs: Forwarded verbatim to each ``torch.compile``
            call (e.g. ``fullgraph=True``, ``mode=...``, ``dynamic=...``).
        :return: The loaded wrapper.
        :raises ValueError: When ``compile_model`` is requested for a
            ``'grid'`` adaptive-cutoff model, or when neither
            *checkpoint_path* nor *model* is given.
        :raises FileNotFoundError: When *checkpoint_path* is neither an
            existing local file nor a parseable named model.
        """
        # Make sure metatomic's custom torch ops are registered before
        # torch.load, otherwise the ScriptObject metadata unpickling fails.
        import metatomic.torch  # noqa: F401
        from metatrain.pet import PET

        if device is None:
            device = torch.device("cpu")

        if checkpoint_path is not None and Path(checkpoint_path).is_file():
            checkpoint_path = str(checkpoint_path)
        elif model is not None:
            model_name, size = model.rsplit("-", 1)
            _, _, checkpoint_path = _resolve_and_download_checkpoint(
                model_name, size, version
            )
        else:
            raise ValueError(
                "UPETWrapper.from_checkpoint requires either `checkpoint_path` "
                "(a local file path) or `model` (e.g. 'pet-mad-s', optionally "
                "with `version`)."
            )

        raw = torch.load(str(checkpoint_path), weights_only=False, map_location=device)

        if isinstance(raw, dict) and "wrapped_model_checkpoint" in raw:
            raw = raw["wrapped_model_checkpoint"]

        # Bring an old checkpoint up to the current model version (adds the
        # `backend.` prefix and any missing hypers). Mutates `raw` in place.
        raw = PET.upgrade_checkpoint(raw)

        model_data = raw["model_data"]
        hypers = dict(model_data["model_hypers"])
        atomic_types = list(model_data["dataset_info"].atomic_types)
        # Prefer the best weights (``best_model_state_dict``); fall back to
        # the last epoch (``model_state_dict``). Exported / best-only
        # checkpoints carry only the former.
        raw_sd = raw.get("best_model_state_dict") or raw.get("model_state_dict")
        if raw_sd is None:
            raise KeyError(
                "Checkpoint has neither 'best_model_state_dict' nor "
                "'model_state_dict' keys. Checkpoint may be corrupted or not "
                "a PET checkpoint."
            )

        composition_values = decode_tensor_map_values(
            raw_sd["additive_models.0.energy_composition_buffer"]
        )  # [num_species, 1]
        composition_energy = composition_values.squeeze(-1).clone()
        scale_values = decode_tensor_map_values(
            raw_sd["scaler.energy_scaler_buffer"]
        )  # [1, 1]
        scale_energy = scale_values.reshape(()).clone()

        backend_sd = filter_state_dict(raw_sd)
        wrapper = cls(
            atomic_types=atomic_types,
            hypers=hypers,
            composition_energy=composition_energy,
            scale_energy=scale_energy,
        )
        wrapper.backend.load_state_dict(backend_sd, strict=True)

        if dtype is not None:
            wrapper.backend = wrapper.backend.to(dtype=dtype)
            wrapper.composition_energy = wrapper.composition_energy.to(dtype=dtype)
            wrapper.scale_energy = wrapper.scale_energy.to(dtype=dtype)

        wrapper = wrapper.to(device)

        if compile_model:
            wrapper.eval()
            for param in wrapper.parameters():
                param.requires_grad = False
            # The 'grid' adaptive-cutoff method (what pet-mad <= v1.5.0 was
            # trained with) cannot be safely compiled: autograd backward
            # through the compiled grid cutoff aborts at the C++ level. The
            # 'solver' method (pet-mad >= v1.6.0) is fully compatible.
            uses_grid_adaptive = (
                wrapper.backend.num_neighbors_adaptive is not None
                and str(wrapper.backend.adaptive_cutoff_method).lower() == "grid"
            )
            if uses_grid_adaptive:
                raise ValueError(
                    "compile_model=True is not supported for PET models using "
                    "the 'grid' adaptive-cutoff method (e.g. pet-mad-xs "
                    "<= v1.5.0): autograd backward through the compiled grid "
                    "cutoff aborts. Load a checkpoint trained with the "
                    "'solver' method (e.g. pet-mad-xs >= v1.6.0) to use "
                    "torch.compile, or run the grid model in eager mode "
                    "(compile_model=False)."
                )
            wrapper.backend.preprocess = torch.compile(
                wrapper.backend.preprocess, **compile_kwargs
            )
            wrapper.backend.calculate_features = torch.compile(
                wrapper.backend.calculate_features, **compile_kwargs
            )
            wrapper.backend.predict = torch.compile(
                wrapper.backend.predict, **compile_kwargs
            )
            wrapper._compiled = True
        return wrapper

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_model(self, path: Path, as_state_dict: bool = False) -> None:
        """Serialize the wrapper to disk in a pure-torch layout.

        Writes a plain dict containing the backend ``state_dict``, the
        hyper-parameters, the atomic-type list, and the composition/scaler
        buffers. The output is **not** a metatrain / metatomic checkpoint —
        it is a self-contained snapshot that can be reloaded by
        constructing ``UPETWrapper(atomic_types, hypers, ...)`` and calling
        ``load_state_dict`` on its backend.

        :param path: Output path.
        :param as_state_dict: If ``True``, save only the backend's
            ``state_dict``. Defaults to ``False`` (saves the full
            snapshot).
        """
        if as_state_dict:
            torch.save(self.backend.state_dict(), path)
        else:
            snapshot = {
                "backend_state_dict": self.backend.state_dict(),
                "hypers": self.hypers,
                "atomic_types": self.atomic_types,
                "composition_energy": self.composition_energy.detach().cpu(),
                "scale_energy": self.scale_energy.detach().cpu(),
            }
            torch.save(snapshot, path)
