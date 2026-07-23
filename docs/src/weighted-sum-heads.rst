.. _weighted-sum-heads:

Combining heads with fixed-coefficient weighted sums
======================================================

If a checkpoint has several heads sharing the same backbone (for example
several DFT-functional variants trained as ``energy/pbe``,
``energy/pbesol``, ...), UPET can attach one or more extra heads that are a
fixed linear combination of the existing ones (e.g. an ``energy/mix`` head
equal to ``0.25 * energy/pbe + 0.75 * energy/pbesol``). The combination is
computed on the wrapped model's own physical predictions, so forces and
stresses for a combined energy head follow automatically from a single
backward pass. Since the extra head does not have free parameters, no
retraining is needed.

Coefficients are used as given, so heads that don't sum to 1 (such as a
difference or correction head, or an arbitrary rescale) need nothing special.
For a weighted average given as un-normalized weights (e.g. ``{"a": 1, "b":
2, "c": 1}``), set ``normalize_coefficients: true`` on that head to rescale
them (dividing each by their sum) so they end up summing to 1.

Each linear combination head also gets a description, stored on the
exported output and shown by tools that introspect a model's capabilities.
By default it is auto-generated from the sources and coefficients (e.g.
``"0.25 * energy/pbe + 0.75 * energy/pbesol"``); explicitly set
``description`` per head to override this default and enforce a custom
description.

Command line
------------

Describe the heads to attach in a YAML file, one entry per head under
``sources:``:

.. code-block:: yaml

   # wsum_heads.yaml
   heads:
     energy/mix:
       sources:
         energy/pbe: 0.25
         energy/pbesol: 0.75
       description: "25/75 PBE/PBEsol mix"
     energy/diff:
       sources:
         energy/pbe: 1.0
         energy/pbesol: -1.0
     energy/calibrated-mix:
       sources:
         energy/pbe: 1
         energy/pbesol: 3
       normalize_coefficients: true

and attach them to a trained checkpoint:

.. code-block:: bash

   python -c "from upet._weighted_sum import _main; _main()" \
       model.ckpt wsum_heads.yaml model-wsum.ckpt

This writes every head listed in the YAML file. Use one or more ``--head``
flags to attach only a subset, either by name from the YAML or as an
inline definition. Inline heads only specify sources, used as given; use
the YAML file for a head that needs ``normalize_coefficients: true``:

.. code-block:: bash

   python -c "from upet._weighted_sum import _main; _main()" \
       model.ckpt wsum_heads.yaml model-wsum.ckpt \
       --head energy/mix \
       --head "energy/other-mix = energy/pbe:0.5, energy/pbe0:0.5"

.. note::

   Prefer this ``python -c`` form over ``python -m upet._weighted_sum``:
   invoking a submodule with ``-m`` makes Python import it twice (once as
   part of ``upet``, once as ``__main__``), which triggers a spurious
   ``RuntimeWarning`` from ``runpy``.

Python API
----------

.. code-block:: python

   from upet import create_weighted_sum_checkpoint
   from upet._weighted_sum import WeightedSumHead

   create_weighted_sum_checkpoint(
       checkpoint_path="model.ckpt",
       specs={
           "energy/mix": WeightedSumHead(
               sources={"energy/pbe": 0.25, "energy/pbesol": 0.75},
               description="25/75 PBE/PBEsol mix",
           ),
           "energy/diff": WeightedSumHead(
               sources={"energy/pbe": 1.0, "energy/pbesol": -1.0},
           ),
           "energy/calibrated-mix": WeightedSumHead(
               sources={"energy/pbe": 1, "energy/pbesol": 3},
               normalize_coefficients=True,
           ),
       },
       output_checkpoint_path="model-wsum.ckpt",
   )

The resulting ``model-wsum.ckpt`` embeds the original, unmodified
checkpoint alongside the weighted-sum recipe, so it stays usable with
:py:func:`~upet.get_upet`/:py:func:`~upet.save_upet` via
``checkpoint_path=...``:

.. code-block:: python

   import upet

   upet.save_upet(checkpoint_path="model-wsum.ckpt", output="model-wsum.pt")

Because it is not a standard metatrain architecture checkpoint, a
weighted-sum checkpoint cannot be loaded with plain
``metatrain.utils.io.load_model`` or exported with ``mtt export``/``mtt
eval``. If you need to fine-tune further, first pull the original
checkpoint back out with :py:func:`~upet.extract_wrapped_checkpoint`:

.. code-block:: python

   from upet import extract_wrapped_checkpoint

   extract_wrapped_checkpoint("model-wsum.ckpt", "model.ckpt")
