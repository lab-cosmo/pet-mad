.. _usage_ase:

ASE interface
=============

UPET ships an ASE-compatible calculator,
:py:class:`~upet.calculator.UPETCalculator`, that wraps any UPET model and
exposes the standard ASE ``get_potential_energy`` / ``get_forces`` /
``get_stress`` API. This page covers the full surface of ASE-driven
workflows: basic evaluation, loading from a local checkpoint,
non-conservative forces and stresses, uncertainty quantification,
rotational averaging, empirical dispersion corrections, electronic DOS
prediction and dataset visualization.

.. seealso::

   For runnable end-to-end ASE workflows — single-point evaluation,
   geometry optimization and MD in the NVE / NVT / NPT ensembles — see
   the :doc:`example gallery
   </generated_examples/1-ase-simulations/index>`.


Basic usage
-----------

To perform a single-structure evaluation, build an ASE ``Atoms`` object
and attach a :py:class:`~upet.calculator.UPETCalculator`. The model name
is formed by combining the model family and the size, e.g. ``pet-mad-s``,
``pet-omat-l``. See :ref:`models` for the full list.

.. code-block:: python

   from upet.calculator import UPETCalculator
   from ase.build import bulk

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   calculator = UPETCalculator(model="pet-mad-s", version="1.5.0", device="cpu")
   atoms.calc = calculator

   energy = atoms.get_potential_energy()
   forces = atoms.get_forces()

These ASE methods are ideal for single-structure evaluations, but
inefficient when you need to run a pre-defined dataset through the model.
For efficient batched evaluation, see :ref:`usage_metatrain` and the
`README_BATCHED.md
<https://github.com/lab-cosmo/upet/blob/main/docs/README_BATCHED.md>`_
reference.

If the ``version`` argument is not specified, the latest available
version of the model is downloaded and used by default:

.. code-block:: python

   from upet.calculator import UPETCalculator

   # uses the latest version of the PET-MAD-S model by default
   calculator = UPETCalculator(model="pet-mad-s", device="cpu")

You can list the available versions for any model with
:py:func:`upet.list_upet`:

.. code-block:: python

   from upet import list_upet

   list_upet(model="pet-mad", size="s")  # PET-MAD of size S
   list_upet(model="pet-mad")             # all PET-MAD sizes
   list_upet()                            # all available UPET models


Loading from a local checkpoint
-------------------------------

You can download a checkpoint from the `HuggingFace repository
<https://huggingface.co/lab-cosmo/upet>`_ directly:

.. code-block:: bash

   wget https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-mad-s-v1.5.0.ckpt

and then load the calculator from that local file (useful for fine-tuned
checkpoints or reproducibility):

.. code-block:: python

   from upet.calculator import UPETCalculator

   calculator = UPETCalculator(checkpoint_path="pet-mad-s-v1.5.0.ckpt", device="cpu")


.. _ase-non-conservative:

Non-conservative (direct) forces and stresses
---------------------------------------------

By default, UPET models compute conservative forces and stresses as
derivatives of the energy via automatic differentiation. UPET models also
support the direct prediction of forces and stresses as separate targets,
which typically gives a 2–3x speedup. However, as discussed in `this
preprint <https://arxiv.org/abs/2412.11569>`_, non-conservative forces and
stresses require additional care to avoid undesired effects during
molecular dynamics.

To use non-conservative forces and stresses, set ``non_conservative=True``
when constructing the calculator:

.. code-block:: python

   from upet.calculator import UPETCalculator
   from ase.build import bulk

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   calculator = UPETCalculator(
       model="pet-mad-s", version="1.5.0", device="cpu", non_conservative=True
   )
   atoms.calc = calculator

   energy = atoms.get_potential_energy()   # computed as usual
   forces = atoms.get_forces()             # predicted as a separate target
   stresses = atoms.get_stress()           # predicted as a separate target

More details on making direct-force MD simulations reliable are in the
`Atomistic Cookbook
<https://atomistic-cookbook.org/examples/pet-mad-nc/pet-mad-nc.html>`_.


.. _ase-uncertainty:

Uncertainty quantification
--------------------------

UPET models can also estimate the uncertainty of the energy prediction.
This is important when probing the model on data that may be substantially
different from its training distribution, or when propagating uncertainty
to derived observables (phase transition temperatures, diffusion
coefficients, and so on).

.. note::

   Uncertainty quantification is only available for a subset of
   checkpoints (currently ``pet-mad-s`` v1.0.2, ``pet-mad-xs`` v1.5.0
   and ``pet-mad-s`` v1.5.0). See :ref:`models` for the full list.

Use the ``get_energy_uncertainty`` and ``get_energy_ensemble`` methods of
:py:class:`~upet.calculator.UPETCalculator`:

.. code-block:: python

   from upet.calculator import UPETCalculator
   from ase.build import bulk

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   calculator = UPETCalculator(model="pet-mad-s", version="1.5.0", device="cpu")
   atoms.calc = calculator

   energy = atoms.get_potential_energy()
   energy_uncertainty = calculator.get_energy_uncertainty(atoms, per_atom=False)
   energy_ensemble = calculator.get_energy_ensemble(atoms, per_atom=False)

Both methods accept a ``per_atom`` flag, which controls whether the
quantity is reported per atom or for the whole system. For the
methodology, see the `LLPR <https://doi.org/10.1088/2632-2153/ad594a>`_
and `shallow ensembles
<https://doi.org/10.1088/2632-2153/ad805f>`_ papers.


Rotational averaging
--------------------

By design, UPET models are not exactly equivariant with respect to
rotations and inversions. The equivariance error is typically much smaller
than the overall model error, but in some cases (geometry optimization or
phonon calculations of highly symmetric structures) it can be beneficial
to enforce additional rotational averaging.

Pass a ``rotational_average_order`` to the calculator:

.. code-block:: python

   from upet.calculator import UPETCalculator
   from ase.build import bulk

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   calculator = UPETCalculator(
       model="pet-mad-s",
       version="1.5.0",
       device="cpu",
       rotational_average_order=3,
   )
   atoms.calc = calculator
   energy = atoms.get_potential_energy()
   forces = atoms.get_forces()
   stresses = atoms.get_stress()

Predictions are averaged over a Lebedev quadrature of the O(3) group at
the requested order (here 3). Higher orders are more accurate but
costlier.

By default, all transformed structures are evaluated in a single batch,
which can be memory-intensive for large systems. Reduce memory pressure
by setting ``rotational_average_batch_size`` to a smaller value:

.. code-block:: python

   from upet.calculator import UPETCalculator

   calculator = UPETCalculator(
       model="pet-mad-s",
       version="1.5.0",
       device="cpu",
       rotational_average_order=3,
       rotational_average_batch_size=8,
   )

After the evaluation, rotational averaging error statistics are available
on the calculator's ``results`` dictionary:

.. code-block:: python

   energy_rot_std = calculator.results["energy_rot_std"]
   forces_rot_std = calculator.results["forces_rot_std"]
   stresses_rot_std = calculator.results["stresses_rot_std"]


Empirical dispersion corrections (D3)
-------------------------------------

You can combine the UPET calculator with the torch-based D3 dispersion
correction of `pfnet-research <https://github.com/pfnet-research/torch-dftd>`_,
``torch-dftd``. Install it inside your UPET environment:

.. code-block:: bash

   pip install torch-dftd

Then combine both calculators with ``ase.calculators.mixing.SumCalculator``:

.. code-block:: python

   import torch
   from ase.calculators.mixing import SumCalculator
   from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
   from upet.calculator import UPETCalculator

   device = "cuda" if torch.cuda.is_available() else "cpu"

   calc_upet = UPETCalculator(model="pet-mad-s", version="1.5.0", device=device)
   dft_d3 = TorchDFTD3Calculator(device=device, xc="r2scan", damping="bj")

   combined_calc = SumCalculator([calc_upet, dft_d3])
   atoms.calc = combined_calc


.. _ase-pet-mad-dos:

DOS, Fermi levels and bandgaps (PET-MAD-DOS)
--------------------------------------------

The UPET package also exposes the **PET-MAD-DOS** model for predicting
electronic density of states of materials, their Fermi levels and
bandgaps, via :py:class:`~upet.calculator.PETMADDOSCalculator`:

.. code-block:: python

   from ase.build import bulk
   from upet.calculator import PETMADDOSCalculator

   atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   pet_mad_dos_calculator = PETMADDOSCalculator(version="latest", device="cpu")

   results = pet_mad_dos_calculator.calculate(atoms)
   print (f"The keys in results is: {results.keys()}")

Per-atom DOS and batched evaluation on a list of structures are also
supported:

.. code-block:: python

   # per-atom DOS
   results = pet_mad_dos_calculator.calculate(
      atoms,
      properties=["dos_raw_per_atom"]
   )

   # multiple structures
   atoms_1 = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
   atoms_2 = bulk("C", cubic=True, a=3.55, crystalstructure="diamond")
   results = pet_mad_dos_calculator.calculate([atoms_1, atoms_2])

Additionally, a denoising algorithm is available to mitigate unphysical oscillations
in the raw predicted DOS. The denoised DOS is non-negative and smoother compared to the
raw DOS. As the denoising algorithm is applied on the DOS of the system, the per atom
DOS is not supported when denoising is turned on.:

.. code-block:: python

   denoised_dos = results['dos_denoised']


Bandgap and Fermi level predictions are available via a dedicated CNN model
to predict these quantities directly from the predicted DOS.

.. code-block:: python

   bandgap = results['bandgap']
   fermi_level = results['efermi']


Dataset visualization with the PET-MAD featurizer
-------------------------------------------------

You can use the last-layer features of the PET-MAD model, together with a
pre-trained sketch-map dimensionality reduction, to obtain 2D and 3D
representations of a dataset (e.g. to identify structural or chemical
motifs). It can be used as a stand-alone feature builder or combined with
the `chemiscope viewer <https://chemiscope.org>`_ for interactive
visualization.

.. code-block:: python

   import ase.io
   import chemiscope
   from upet.explore import PETMADFeaturizer

   featurizer = PETMADFeaturizer(version="latest")

   # Load structures
   frames = ase.io.read("dataset.xyz", ":")

   # Compute features only
   features = featurizer(frames, None)

   # Or create an interactive Jupyter visualization
   chemiscope.explore(frames, featurize=featurizer)
