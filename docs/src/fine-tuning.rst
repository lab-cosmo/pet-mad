.. _fine-tuning:

Fine-tuning
===========

.. note::

   Detailed fine-tuning instructions are work in progress. In the meantime,
   refer to the `metatrain fine-tuning tutorial
   <https://docs.metatensor.org/metatrain/latest/generated_examples/0-beginner/02-fine-tuning.html>`_,
   which covers the full workflow end-to-end.

.. note::

   Due to the complexity of the data processing pipeline for fine-tuning PET-MAD-DOS,
   the reader is instead referred to the
   `PET-MAD-DOS fine-tuning tutorial in the atomistic cookbook
   <https://atomistic-cookbook.org/examples/pet-mad-dos/
   pet-mad-dos.html#finetuning-pet-mad-dos-on-specific-applications>`_
   for a step-by-step walkthrough of the fine-tuning process for PET-MAD-DOS models.


UPET models can be fine-tuned using the `metatrain
<https://docs.metatensor.org/metatrain/latest/>`_ library. We currently
recommend fine-tuning from our **PET-OMat** models, as they are
pre-trained on a very large dataset and come in all sizes (from XS to XL),
giving a good trade-off for most applications.

Head selection
--------------

By default, :py:class:`~upet.calculator.UPETCalculator` uses the energy
and non-conservative forces/stresses heads **provided with the
pre-trained models**. If you fine-tune a model and create a new head for
your energy target, you need to explicitly select the corresponding
variant at runtime (and similarly for non-conservative forces and
stresses).

As a running example, suppose you fine-tuned the energy head and named
it ``energy/finetune`` in the ``options.yaml`` file passed to
``mtt train``.

ASE interface
~~~~~~~~~~~~~

Load the fine-tuned checkpoint and construct the calculator with the
``variants`` parameter:

.. code-block:: python

   from upet.calculator import UPETCalculator

   # For the new energy head called "energy/finetune"
   calc = UPETCalculator(checkpoint_path="finetuned.ckpt", variants={"energy": "finetune"})

The same applies to non-conservative forces and stresses, if you created
new heads for them during fine-tuning.

metatrain interface
~~~~~~~~~~~~~~~~~~~

When evaluating with ``mtt eval``, select the new head in the
``options.yaml`` file:

.. code-block:: yaml

   systems: your-test-dataset.xyz
   targets:
     energy/finetune:
       key: "energy"
       unit: "eV"

LAMMPS interface
~~~~~~~~~~~~~~~~

Select the new head with the ``variant/energy`` parameter in the
``pair_style metatomic`` command:

.. code-block:: none

   read_data silicon.data

   pair_style metatomic model.pt variant/energy finetune
   pair_coeff * * 14

Combining heads with fixed-coefficient weighted sums
------------------------------------------------------

If a checkpoint has several heads sharing the same backbone (for example
several DFT-functional variants trained as ``energy/pbe``,
``energy/pbe0``, ...), UPET can attach one or more extra heads that are a
fixed linear combination of the existing ones (e.g. an ``energy/mix`` head
equal to ``0.25 * energy/pbe + 0.75 * energy/pbe0``). The combination is
computed on the wrapped model's own physical predictions, so forces and
stresses for a combined energy head follow automatically from a single
backward pass -- no retraining is needed.

Command line
~~~~~~~~~~~~

Describe the heads to attach in a YAML file:

.. code-block:: yaml

   # wsum_heads.yaml
   heads:
     energy/mix:
       energy/pbe: 0.25
       energy/pbe0: 0.75

and attach them to a trained checkpoint:

.. code-block:: bash

   python -c "from upet._weighted_sum import _main; _main()" \
       model.ckpt wsum_heads.yaml model-wsum.ckpt

This writes every head listed in the YAML file. Use one or more ``--head``
flags to attach only a subset, either by name from the YAML or as an
inline definition:

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
~~~~~~~~~~

.. code-block:: python

   from upet import create_weighted_sum_checkpoint

   create_weighted_sum_checkpoint(
       checkpoint_path="model.ckpt",
       specs={"energy/mix": {"energy/pbe": 0.25, "energy/pbe0": 0.75}},
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
