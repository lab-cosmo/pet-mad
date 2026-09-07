.. image:: /../static/images/upet-logo-with-text.svg
   :class: only-light
   :width: 600px

.. image:: /../static/images/upet-logo-with-text-dark.svg
   :class: only-dark
   :width: 600px

**UPET** is a family of universal interatomic potentials for advanced
materials modeling across the periodic table. These models are based on the
**Point Edge Transformer (PET)** architecture, trained on a variety of
popular atomistic datasets, and capable of predicting energies and forces
in complex atomistic workflows.

The package also ships **PET-MAD-DOS**, a universal model for predicting
the electronic density of states (DOS) of materials and molecules, as well
as their Fermi levels and bandgaps. PET-MAD-DOS uses a slightly modified
PET architecture and is trained on the MAD dataset.

.. note::

   The **PET-MAD-1.5** models, trained for 102 elements at the r2SCAN level
   of theory, are now available. These models are more robust, more
   accurate and faster than the previous PET-MAD models. We highly
   recommend using them for all applications, especially molecular
   dynamics simulations.

   .. code-block:: python

      from upet.calculator import UPETCalculator
      calculator = UPETCalculator(model="pet-mad-s", version="1.5.0", device="cuda")

.. note::

   Are you here to try our **Matbench model**? Here is all you need. Don't
   be scared by the parameter count — our model is :ref:`much faster <model-speeds>` than you
   might think. It is excellent for convex hull energies, geometry
   optimization and phonons, but we highly recommend the lighter and more
   universal PET-MAD for molecular dynamics.

   .. code-block:: python

      from upet.calculator import UPETCalculator
      calculator = UPETCalculator(model="pet-oam-xl", version="1.0.0", device="cuda")

.. warning::

   This repository is the successor of the PET-MAD repository, which is now
   deprecated. The package has been renamed to **UPET** to reflect the
   broader scope of the models and functionalities it provides, which now
   go beyond the original PET-MAD model. Please use version ``1.4.4`` of
   the ``pet-mad`` package if you need the old API. The `older version of
   the README
   <https://github.com/lab-cosmo/upet/blob/main/docs/README_OLD.md>`_ and
   a `migration guide
   <https://github.com/lab-cosmo/upet/blob/main/docs/UPET_MIGRATION_GUIDE.md>`_
   are available in the repository.


Key features
------------

- **Universality**: UPET models are generally applicable, and can be used
  for predicting energies and forces, as well as the density of states,
  Fermi levels, and bandgaps for a wide range of materials and molecules.
- **Accuracy**: UPET models achieve excellent accuracies in various types
  of atomistic simulations of organic and inorganic systems.
- **Efficiency**: UPET models are highly computationally efficient and
  have low memory usage, which makes them suitable for large-scale
  simulations.
- **Infrastructure**: Various MD engines are available for diverse
  research and application needs.
- **HPC compatibility**: Efficient in HPC environments for extensive
  simulations.


Maintainers
-----------

This project is `maintained
<https://github.com/lab-cosmo/.github/blob/main/Maintainers.md>`_ by
`@abmazitov <https://github.com/abmazitov>`_ and `@frostedoyster
<https://github.com/frostedoyster>`_, who will reply to issues and pull
requests opened on the repository as soon as possible. You can mention
them directly if you have not received an answer after a couple of days.


.. toctree::
   :maxdepth: 1
   :caption: Contents
   :hidden:

   quickstart
   installation
   usage/index
   models
   fine-tuning
   api
   generated_examples/index
   cookbook/index
   miscellaneous
   faq
   changelog
   cite
