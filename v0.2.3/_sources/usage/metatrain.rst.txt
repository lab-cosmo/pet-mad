.. _usage_metatrain:

Batched evaluation with metatrain
=================================

Efficient evaluation of UPET models on a dataset is available from the
command line via `metatrain
<https://docs.metatensor.org/metatrain/latest/>`_, which is installed as a
dependency of UPET.

Step 1: export the model to TorchScript
---------------------------------------

Fetch and convert a UPET checkpoint from the HuggingFace repository:

.. code-block:: bash

   mtt export https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-mad-s-v1.5.0.ckpt -o model.pt

Alternatively, fetch and save the model with the UPET Python API:

.. code-block:: python

   import upet

   # Save the latest version of PET-MAD-S to a TorchScript file
   upet.save_upet(
       model="pet-mad",
       size="s",
       version="1.5.0",
       output="model.pt",
   )

Both commands download the model and convert it to TorchScript.

Step 2: write the evaluation options
------------------------------------

Create an ``options.yaml`` file describing the dataset (in ``extxyz``
format) and the targets to predict:

.. code-block:: yaml

   systems: your-test-dataset.xyz
   targets:
     energy:
       key: "energy"
       unit: "eV"

Step 3: run ``mtt eval``
------------------------

.. code-block:: bash

   mtt eval model.pt options.yaml --batch-size=16 --output=predictions.xyz

This writes ``predictions.xyz`` with the predicted energies and forces for
each structure in the dataset. For more options, see the `metatrain
evaluation docs
<https://docs.metatensor.org/metatrain/latest/getting-started/usage.html#evaluation>`_.
