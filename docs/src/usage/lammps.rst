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
(atomic number 14):

.. code-block:: none

   units metal
   atom_style atomic

   read_data silicon.data

   pair_style metatomic model.pt device cuda  # change device to 'cpu' to use the CPU instead
   pair_coeff * * 14

   neighbor 2.0 bin
   neigh_modify one 100000 page 1000000 binsize 5.5

   timestep 0.001

   dump myDump all xyz 10 trajectory.xyz
   dump_modify myDump element Si

   thermo_style multi
   thermo 1

   velocity all create 300 87287 mom yes rot yes

   fix 1 all nvt temp 300 300 0.10
   run_style verlet


   run 100

Create the ``silicon.data`` data file:

.. code-block:: none

   # LAMMPS data file for Silicon unit cell
   8 atoms
   1 atom types

   0.0  5.43  xlo xhi
   0.0  5.43  ylo yhi
   0.0  5.43  zlo zhi

   Masses

   1  28.084999992775295 # Si

   Atoms # atomic

   1   1   0   0   0
   2   1   1.3575   1.3575   1.3575
   3   1   0   2.715   2.715
   4   1   1.3575   4.0725   4.0725
   5   1   2.715   0   2.715
   6   1   4.0725   1.3575   4.0725
   7   1   2.715   2.715   0
   8   1   4.0725   4.0725   1.3575

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
