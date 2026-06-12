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
