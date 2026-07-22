"""
Fixed-coefficient weighted-sum heads for PET models with multiple heads sharing
one backbone (e.g. several DFT-functional variants, or several committee
members).

This module provides:

- :class:`WeightedSumModel`, a thin wrapper around a trained model (in practice,
  a :class:`metatrain.pet.PET` instance with several targets) that adds one or
  more new outputs, each a fixed linear combination of existing targets. The
  combination is formed on the wrapped model's own *physical* predictions (i.e.
  after its scaler and additive contributions have been applied), so a combined
  output equals ``sum_i c_i * X_i``.
- :func:`create_weighted_sum_checkpoint`, which loads a trained checkpoint,
  attaches the requested heads, and writes the result to a new ``.ckpt`` file.

Supported source quantities
----------------------------
A combined head's sources must all share the same quantity, unit, and block
layout (checked at construction time), but the quantity itself can be anything
the wrapped model predicts directly as a value -- in practice, for PET:

- ``energy``-like targets ("conservative" heads): forces and stress are *not*
  predicted directly, but obtained by MD engines differentiating the energy
  output w.r.t. atomic positions/strain. Combined values stay attached to the
  autograd graph built by the wrapped model's own forward pass, so forces and
  stresses for a combined energy head follow from a single backward pass on
  that one combined scalar -- equal to the fixed linear combination of the
  sources' own forces/stress.
- ``non_conservative_force``/``non_conservative_stress``-like targets: these
  are predicted directly (no autograd involved), so combining them is a plain
  linear combination of the predicted values themselves.

Either way, if a *single* ``forward`` call requests several combined heads at
once, it still costs one forward pass of the wrapped model, since sources
shared between them are only requested once (see ``forward``'s ``inner``
dict). This is a property of the model API, not a guarantee about any
particular caller: it only pays off when the caller actually batches several
named outputs into one request, e.g. via ``MetatomicCalculator``'s
``additional_outputs``. Typical MD stepping (ASE dynamics, ``pair_style
metatomic``) requests one head at a time, each in its own forward call, so
there is nothing to deduplicate there. The combination only touches
predicted *values*, not explicit gradient blocks -- a source requested with
``explicit_gradients`` would not have those gradients combined, so combined
heads should be built from targets that are evaluated without explicit
gradients (the normal case for both conservative-energy and
non-conservative-force/stress targets).

Checkpoint compatibility
-------------------------
Metatrain resolves a checkpoint's architecture by importing it as a submodule of
the ``metatrain`` package itself (see ``metatrain.utils.architectures``); there is
no plugin mechanism for out-of-tree architectures. A checkpoint written by
:func:`create_weighted_sum_checkpoint` therefore cannot be loaded by plain
``metatrain.utils.io.load_model`` or by ``mtt export`` -- both would need to
recognize ``WeightedSumModel`` as an architecture, and neither can. It nests the
*unmodified* checkpoint of the wrapped model instead (so that checkpoint stays a
standard, independently loadable ``pet`` checkpoint), and stores the
weighted-sum recipe alongside it. ``upet`` special-cases this on load (see
``_load_model_with_custom_heads`` in ``_models.py``): it recognizes the
checkpoint's ``architecture_name`` and reconstructs the model directly, without
going through metatrain's architecture registry. This is enough for ``upet``
(``get_upet`` / ``save_upet`` with ``checkpoint_path=...``) to export the combined
model to a TorchScript ``.pt`` for MD.
"""

import argparse
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)
from metatrain.pet import PET
from metatrain.utils.dtype import dtype_to_str
from metatrain.utils.io import load_model as load_metatrain_model
from metatrain.utils.io import model_from_checkpoint
from metatrain.utils.metadata import merge_metadata


#: Marker stored as ``checkpoint["architecture_name"]`` for checkpoints written by
#: :func:`create_weighted_sum_checkpoint`. Deliberately distinct from any real
#: metatrain architecture name (see module docstring).
ARCHITECTURE_NAME = "pet_weighted_sum"


class WeightedSumModel(torch.nn.Module):
    """A wrapped model plus N extra outputs, each a fixed weighted sum of existing
    targets of the wrapped model.

    The combination heads have no parameters of their own, so they play no part in
    training; they only exist at export / inference time. Because of that,
    ``WeightedSumModel`` does not implement metatrain's ``ModelInterface`` (which
    assumes hyperparameters and a ``dataset_info`` of its own): it is a plain
    ``torch.nn.Module`` that duck-types the subset of that interface upet needs
    (``get_checkpoint`` / ``load_checkpoint`` / ``export`` / ``supported_outputs``
    / ``requested_neighbor_lists`` / ``requested_inputs``).

    :param model: a loaded, export-ready model (in practice a
        :class:`metatrain.pet.PET` instance) with multiple conservative targets.
    :param specs: ``{new_head_name: {"sources": {source_target_name: coefficient,
        ...}, "normalize_coefficients": bool, "description": str}, ...}``. Each
        ``source_target_name`` must name an existing target of ``model``.
        Coefficients are used as given (so a difference or an arbitrary
        rescaling works with no extra flag); set ``"normalize_coefficients"``
        to ``True`` to instead rescale them (dividing each by their sum) so
        that they end up summing to 1, e.g. for a weighted average given as
        un-normalized weights such as ``{"a": 1, "b": 2, "c": 1}``. It
        defaults to ``False``. ``"description"`` is optional and defaults to
        an auto-generated ``"c1 * source1 + c2 * source2 + ..."`` string
        (using the coefficients actually applied, i.e. after normalization if
        requested); set it to give the head a different, custom description
        (stored on the exported ``ModelOutput``) instead.
    """

    __checkpoint_version__ = 1

    # class-level annotations: TorchScript needs these to type the attributes
    new_names: List[str]
    sources: List[List[str]]
    coefficients: List[List[float]]
    descriptions: List[str]

    _SPEC_KEYS = {"sources", "normalize_coefficients", "description"}

    def __init__(self, model: PET, specs: Dict[str, Dict[str, Any]]) -> None:
        super().__init__()
        self.model = model

        if len(specs) == 0:
            raise ValueError("no weighted-sum heads requested")

        self.new_names = list(specs.keys())
        self.sources = []
        self.coefficients = []
        self.descriptions = []
        self.outputs: Dict[str, ModelOutput] = dict(model.outputs)

        for new_name in self.new_names:
            head_spec = specs[new_name]
            unknown_keys = set(head_spec) - self._SPEC_KEYS
            if unknown_keys:
                raise ValueError(
                    f"in head '{new_name}': unknown key(s) {sorted(unknown_keys)}; "
                    f"expected 'sources' and optionally "
                    f"'normalize_coefficients', 'description'"
                )
            if "sources" not in head_spec:
                raise ValueError(f"in head '{new_name}': missing 'sources'")

            source_coefficients = head_spec["sources"]
            normalize_coefficients = bool(
                head_spec.get("normalize_coefficients", False)
            )
            sources = list(source_coefficients.keys())
            coefficients = [float(source_coefficients[s]) for s in sources]

            if len(sources) == 0:
                raise ValueError(f"'{new_name}' has no sources")
            if new_name in model.outputs:
                raise ValueError(f"'{new_name}' already exists as a model output")
            for source in sources:
                if source not in model.target_names:
                    # in particular, this rejects weighted-sum heads as sources:
                    # they are not targets of `model`, so they never appear in
                    # target_names
                    raise ValueError(
                        f"unknown source target '{source}' for head '{new_name}'; "
                        f"the model provides {model.target_names}"
                    )
            if normalize_coefficients:
                total = sum(coefficients)
                if math.isclose(total, 0.0, abs_tol=1e-12):
                    raise ValueError(
                        f"in head '{new_name}': coefficients sum to {total}, "
                        f"cannot normalize (would divide by zero); pass "
                        f"coefficients that already reflect the combination "
                        f"you want, or set normalize_coefficients: false"
                    )
                coefficients = [c / total for c in coefficients]

            # Blocks are combined positionally in `forward`, so the sources of a
            # given head must agree on layout, quantity and unit. Checking here is
            # cheap, and turns a silently meaningless sum into a loud error.
            reference = model.dataset_info.targets[sources[0]]
            for source in sources[1:]:
                info = model.dataset_info.targets[source]
                if info.quantity != reference.quantity or info.unit != reference.unit:
                    raise ValueError(
                        f"in head '{new_name}': '{source}' is "
                        f"({info.quantity}, {info.unit}), incompatible with "
                        f"({reference.quantity}, {reference.unit})"
                    )
                if info.layout.keys != reference.layout.keys:
                    raise ValueError(
                        f"in head '{new_name}': '{source}' has a different block layout"
                    )
                for ref_block, block in zip(
                    reference.layout.blocks(), info.layout.blocks(), strict=True
                ):
                    if (
                        block.components != ref_block.components
                        or block.properties != ref_block.properties
                    ):
                        raise ValueError(
                            f"in head '{new_name}': '{source}' has different "
                            f"components or properties"
                        )

            default_description = " + ".join(
                f"{c} * {s}" for s, c in zip(sources, coefficients, strict=True)
            )
            description = str(head_spec.get("description", default_description))

            self.sources.append(sources)
            self.coefficients.append(coefficients)
            self.descriptions.append(description)
            self.outputs[new_name] = ModelOutput(
                quantity=reference.quantity,
                unit=reference.unit,
                sample_kind="atom",
                description=description,
            )

    def supported_outputs(self) -> Dict[str, ModelOutput]:
        return self.outputs

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return self.model.requested_neighbor_lists()

    def requested_inputs(self) -> Dict[str, ModelOutput]:
        if hasattr(self.model, "requested_inputs"):
            return self.model.requested_inputs()
        return {}

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        # The caller (an MD engine) asks only for the weighted-sum head(s) it
        # wants, so pull in the sources they depend on. Propagating sample_kind
        # keeps per-atom vs per-structure consistent, which LAMMPS relies on:
        # under domain decomposition it requests per-atom energies together with
        # selected_atoms. Heads sharing a source request it once, so several
        # weighted-sum heads still cost a single pass.
        inner: Dict[str, ModelOutput] = {}
        for key, value in outputs.items():
            inner[key] = value

        for h in range(len(self.new_names)):
            new_name = self.new_names[h]
            if new_name in outputs:
                requested = outputs[new_name]
                for source in self.sources[h]:
                    if source not in inner:
                        inner[source] = ModelOutput(
                            quantity=requested.quantity,
                            unit=requested.unit,
                            sample_kind=requested.sample_kind,
                            explicit_gradients=requested.explicit_gradients,
                        )

        # The single forward pass. In eval mode these TensorMaps already carry the
        # scaler and the additive contributions.
        result = self.model(systems, inner, selected_atoms)

        for h in range(len(self.new_names)):
            new_name = self.new_names[h]
            if new_name not in outputs:
                continue
            sources = self.sources[h]
            coefficients = self.coefficients[h]

            first = result[sources[0]]
            blocks: List[TensorBlock] = []
            for i in range(len(first.blocks())):
                reference = first.block(i)
                # plain float * Tensor: no detach, no buffers, graph left intact
                values = reference.values * coefficients[0]
                for j in range(1, len(sources)):
                    values = (
                        values + result[sources[j]].block(i).values * coefficients[j]
                    )
                blocks.append(
                    TensorBlock(
                        values=values,
                        samples=reference.samples,
                        components=reference.components,
                        properties=reference.properties,
                    )
                )
            result[new_name] = TensorMap(keys=first.keys, blocks=blocks)

        # hand back only what was asked for, dropping the sources we pulled in
        out: Dict[str, TensorMap] = {}
        for key, value in result.items():
            if key in outputs:
                out[key] = value
        return out

    def get_checkpoint(self) -> Dict[str, Any]:
        specs = {}
        for i, new_name in enumerate(self.new_names):
            specs[new_name] = {
                # already normalized (if requested) at construction time, so
                # `normalize_coefficients` itself doesn't need to round-trip
                "sources": dict(
                    zip(self.sources[i], self.coefficients[i], strict=True)
                ),
                "description": self.descriptions[i],
            }
        return {
            "architecture_name": ARCHITECTURE_NAME,
            "model_ckpt_version": self.__checkpoint_version__,
            # the wrapped model's own checkpoint, untouched: this keeps it a
            # standard, independently loadable checkpoint for its own
            # architecture (e.g. "pet"), and lets metatrain's own version
            # upgrade machinery apply to it unmodified.
            "wrapped_model_checkpoint": self.model.get_checkpoint(),
            "weighted_sum_heads": specs,
        }

    @classmethod
    def upgrade_checkpoint(cls, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        if checkpoint["model_ckpt_version"] != cls.__checkpoint_version__:
            raise RuntimeError(
                "Unable to upgrade the weighted-sum-head checkpoint: it is using "
                f"version {checkpoint['model_ckpt_version']}, while the current "
                f"version is {cls.__checkpoint_version__}."
            )
        return checkpoint

    @classmethod
    def load_checkpoint(
        cls, checkpoint: Dict[str, Any], context: str = "export"
    ) -> "WeightedSumModel":
        checkpoint = cls.upgrade_checkpoint(checkpoint)
        # metatrain's own version-upgrade machinery applies to the nested
        # checkpoint, since it is an unmodified checkpoint for a real
        # architecture (e.g. "pet").
        wrapped_model = model_from_checkpoint(
            checkpoint["wrapped_model_checkpoint"], context=context
        )
        return cls(wrapped_model, checkpoint["weighted_sum_heads"])

    def export(self, metadata: Optional[ModelMetadata] = None) -> AtomisticModel:
        dtype = next(self.model.parameters()).dtype
        if dtype not in self.model.__supported_dtypes__:
            raise ValueError(f"unsupported dtype {dtype} for {type(self.model)}")

        # Make sure the model is all in the same dtype: mirrors PET.export().
        self.model.to(dtype)

        # The CompositionModel holds TensorMaps that cannot be registered as
        # torch buffers, so `module.to(...)` does not move them: mirrors
        # PET.export(), which does this unconditionally (correct for float32
        # models too, since the values are cast to the prediction dtype at use
        # time).
        self.model.additive_models[0].weights_to(torch.device("cpu"), torch.float64)

        interaction_ranges = [self.model.num_gnn_layers * self.model.cutoff]
        for additive_model in self.model.additive_models:
            if hasattr(additive_model, "cutoff_radius"):
                interaction_ranges.append(additive_model.cutoff_radius)
        interaction_range = max(interaction_ranges)

        capabilities = ModelCapabilities(
            outputs=self.supported_outputs(),
            atomic_types=self.model.atomic_types,
            interaction_range=interaction_range,
            length_unit=self.model.dataset_info.length_unit,
            supported_devices=self.model.__supported_devices__,
            dtype=dtype_to_str(dtype),
        )

        metadata = merge_metadata(self.model.metadata, metadata)

        return AtomisticModel(self.eval(), metadata, capabilities)


def load_specs_yaml(path: str) -> Dict[str, Dict[str, Any]]:
    """Load a ``{heads: {name: {sources: {source: coeff, ...},
    normalize_coefficients: bool, description: str}, ...}}`` YAML file, as
    produced by hand or by an upstream calibration procedure.
    ``normalize_coefficients`` and ``description`` are optional per head; see
    :class:`WeightedSumModel` for their meaning and defaults.

    Example::

        heads:
          energy/mix:
            sources:
              energy/pbe:  0.25
              energy/pbesol: 0.75
            description: "25/75 PBE/PBEsol mix"
          energy/diff:
            sources:
              energy/pbe:  1.0
              energy/pbesol: -1.0
          energy/calibrated-mix:
            sources:
              energy/pbe:  1
              energy/pbesol: 3
            normalize_coefficients: true
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    if "heads" in config:
        return config["heads"]
    if "name" in config and "spec" in config:  # single-head legacy format
        return {config["name"]: config["spec"]}
    raise ValueError(f"'{path}' has no 'heads' section")


def parse_inline_head(text: str) -> Tuple[str, Dict[str, Any]]:
    """Parse ``'energy/mix = energy/pbe:0.25, energy/pbesol:0.75'``.

    Inline heads only specify sources, used as given; use a YAML file if a
    head needs ``normalize_coefficients: true``.
    """
    if "=" not in text:
        raise ValueError(
            f"'{text}' is neither a head name from the YAML nor an inline "
            f"definition of the form 'name = source:coeff, source:coeff'"
        )
    name, _, body = text.partition("=")
    sources: Dict[str, float] = {}
    for term in body.split(","):
        if ":" not in term:
            raise ValueError(f"malformed term '{term.strip()}' in '{text}'")
        source, _, coefficient = term.rpartition(":")
        sources[source.strip()] = float(coefficient)
    return name.strip(), {"sources": sources}


def collect_specs(
    config_path: Optional[str],
    requested: Optional[List[str]],
) -> Dict[str, Dict[str, Any]]:
    """Resolve a YAML file and ``--head`` flags into
    ``{name: {"sources": {source: coeff}, ...}}``."""
    available: Dict[str, Dict[str, Any]] = {}
    if config_path is not None:
        available = load_specs_yaml(config_path)

    if not requested:
        if not available:
            raise ValueError("nothing to do: no YAML heads and no --head given")
        return available

    specs: Dict[str, Dict[str, Any]] = {}
    for item in requested:
        if item in available:
            specs[item] = available[item]
        else:
            name, spec = parse_inline_head(item)
            specs[name] = spec
    return specs


def create_weighted_sum_checkpoint(
    checkpoint_path: str,
    specs: Dict[str, Dict[str, Any]],
    output_checkpoint_path: str,
) -> None:
    """Attach fixed-coefficient weighted-sum heads to a trained checkpoint and
    write the result as a new ``.ckpt`` file.

    The wrapped model's own checkpoint is embedded unmodified, so the result
    stays fully reconstructible; it is loadable via ``upet``
    (``get_upet``/``save_upet`` with ``checkpoint_path=...``), but not via plain
    ``metatrain.utils.io.load_model`` or ``mtt export`` (see the module
    docstring).

    :param checkpoint_path: path to a trained checkpoint (e.g. a PET ``.ckpt``
        with several energy, non-conservative-force, and/or
        non-conservative-stress targets).
    :param specs: ``{new_head_name: {"sources": {source_target_name:
        coefficient, ...}, "normalize_coefficients": bool}, ...}`` (see
        :class:`WeightedSumModel`). Sources for a given head must all be
        present in the same checkpoint and share the same quantity, unit, and
        block layout (e.g. combine ``energy/...`` sources into an
        ``energy/...`` head, or ``non_conservative_stress/...`` sources into a
        ``non_conservative_stress/...`` head -- not a mix of different
        quantities).
    :param output_checkpoint_path: where to write the resulting ``.ckpt``.
    """
    model = load_metatrain_model(checkpoint_path)
    wrapped = WeightedSumModel(model, specs)
    torch.save(wrapped.get_checkpoint(), output_checkpoint_path)


def extract_wrapped_checkpoint(
    weighted_sum_checkpoint_path: str,
    output_checkpoint_path: str,
) -> None:
    """Pull the plain, unmodified checkpoint nested inside a weighted-sum
    checkpoint back out to its own ``.ckpt`` file.

    The weighted-sum heads themselves have no learnable parameters (they are a
    fixed linear combination, fixed at attach time), so there is nothing to
    fine-tune *in* a weighted-sum checkpoint; only the wrapped model can be
    fine-tuned, and only through ``mtt train``/``mtt finetune``, which -- like
    ``mtt export`` -- cannot load a weighted-sum checkpoint directly (see the
    module docstring). This is the inverse of the embedding
    :func:`create_weighted_sum_checkpoint` performs, so that only the
    self-contained weighted-sum ``.ckpt`` needs to be kept around long-term:
    extract, fine-tune, then re-attach (with the same or updated coefficients)
    to get a fresh weighted-sum checkpoint.

    :param weighted_sum_checkpoint_path: path to a ``.ckpt`` written by
        :func:`create_weighted_sum_checkpoint`.
    :param output_checkpoint_path: where to write the extracted, plain ``.ckpt``
        (e.g. a standard ``pet`` checkpoint, loadable/fine-tunable by ``mtt``).
    """
    checkpoint = torch.load(
        weighted_sum_checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("architecture_name") != ARCHITECTURE_NAME:
        raise ValueError(
            f"'{weighted_sum_checkpoint_path}' is not a weighted-sum checkpoint "
            f"(architecture_name={checkpoint.get('architecture_name')!r}, "
            f"expected {ARCHITECTURE_NAME!r})"
        )
    torch.save(checkpoint["wrapped_model_checkpoint"], output_checkpoint_path)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Attach one or more fixed-coefficient weighted-sum heads to a "
            "trained checkpoint (e.g. PET with several energy, "
            "non-conservative-force, and/or non-conservative-stress targets) "
            "and save the result as a new .ckpt, loadable by upet."
        ),
    )
    parser.add_argument("checkpoint", help="input .ckpt with the source heads")
    parser.add_argument("config", nargs="?", help="YAML file with a `heads` section")
    parser.add_argument("output", help="path of the output .ckpt")
    parser.add_argument(
        "--head",
        action="append",
        dest="heads",
        metavar="NAME | 'NAME = SRC:C, SRC:C'",
        help=(
            "head to attach: either a name from the YAML, or an inline "
            "definition. Repeatable. If omitted, every head in the YAML is "
            "attached."
        ),
    )
    args = parser.parse_args()

    specs = collect_specs(args.config, args.heads)
    create_weighted_sum_checkpoint(args.checkpoint, specs, args.output)

    print(f"wrote {args.output} with {len(specs)} weighted-sum head(s):")
    for new_name, spec in specs.items():
        description = " + ".join(f"{c} * {s}" for s, c in spec["sources"].items())
        print(f"  {new_name} = {description}")


if __name__ == "__main__":
    _main()
