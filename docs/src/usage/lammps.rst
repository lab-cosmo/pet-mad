.. _usage_lammps:

LAMMPS
======

UPET models can drive LAMMPS molecular dynamics via ``pair_style
metatomic``, on both CPU and GPU (with KOKKOS).

1. Install LAMMPS with metatomic support
----------------------------------------

Follow the `metatomic LAMMPS install instructions
<https://docs.metatensor.org/metatomic/latest/engines/lammps.html#how-to-install-the-code>`_.
We recommend the pre-built conda binaries.

2. Export the UPET model
------------------------

Fetch and convert a UPET checkpoint from the HuggingFace repository:

.. code-block:: bash

   mtt export https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-mad-s-v1.5.0.ckpt -o model.pt

This downloads the model and converts it to a TorchScript file compatible
with LAMMPS, using ``metatomic`` and ``metatrain`` under the hood. Other
UPET models can be prepared in the same way:

.. code-block:: bash

   mtt export https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-omat-xs-v1.0.0.ckpt -o model.pt
   mtt export https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-omatpes-l-v0.1.0.ckpt -o model.pt

3. CPU version
--------------

Prepare a LAMMPS input file using ``pair_style metatomic`` and defining
the mapping from LAMMPS atom types in the data file to the element UPET
expects, using ``pair_coeff``. Below, LAMMPS atom type 1 is silicon
(atomic number 14). The :download:`silicon.in <lammps/silicon.in>` and
:download:`silicon.data <lammps/silicon.data>` files are also checked
into the repository and can be downloaded directly:

.. literalinclude:: lammps/silicon.in
   :language: none

Create the ``silicon.data`` data file:

.. literalinclude:: lammps/silicon.data
   :language: none

Run LAMMPS:

.. code-block:: bash

   lmp -in lammps.in                # serial version
   mpirun -np 1 lmp -in lammps.in   # MPI version

.. warning::

   The ``neigh_modify`` settings above are particularly important for
   running PET-MAD v1.5 models, especially for dense systems with a large
   number of neighbors. This is due to the adaptive cutoff strategy, which
   requires a large initial buffer of neighbors before truncation. If your
   system is extremely dense, you may need to increase the ``one`` and
   ``page`` parameters even further to avoid neighbor-list overflow.
   Finally, ``binsize`` is important for the KOKKOS version of the code
   to avoid a bug with an empty neighbor list.

4. KOKKOS-enabled GPU version
-----------------------------

Running LAMMPS with KOKKOS and GPU support is similar to the CPU version,
but with a few additional flags. The ``lammps.in`` and ``silicon.data``
files are unchanged:

.. code-block:: bash

   lmp -in in.lammps -k on g 1 -pk kokkos newton on neigh half -sf kk                 # serial
   mpirun -np 1 lmp -in in.lammps -k on g 1 -pk kokkos newton on neigh half -sf kk    # MPI

The ``-k on g 1 -sf kk`` flags activate the KOKKOS subroutines. ``g 1``
specifies how many GPUs the simulation is parallelized over — adjust it
if you run a large system on multiple GPUs.

Important notes
---------------

- For **CPU calculations**, use a single MPI task unless simulating large
  systems (30+ Å box size). Multi-threading can be enabled with:

  .. code-block:: bash

     export OMP_NUM_THREADS=4

- For **GPU calculations**, use **one MPI task per GPU**.
