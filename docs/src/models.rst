.. _models:

Available models
================

The following pre-trained UPET models are available:

.. list-table::
   :header-rows: 1
   :widths: 14 22 18 26 20

   * - Name
     - Level of theory
     - Available sizes
     - To be used for
     - Training set
   * - PET-MAD-1.5
     - r2SCAN
     - XS, S
     - materials & molecules (102 elements)
     - OMat → MAD-1.5
   * - PET-OAM
     - PBE (Materials Project)
     - L, XL
     - materials (89 elements)
     - OMat → sAlex+MPtrj
   * - PET-OMat
     - PBE
     - XS, S, M, L, XL
     - materials (89 elements)
     - OMat
   * - PET-OMATPES
     - r2SCAN
     - L
     - materials (89 elements)
     - OMat → MATPES
   * - PET-SPICE
     - ωB97M-D3
     - S, L
     - molecules (17 elements)
     - SPICE

Recommended usage:

- **PET-MAD v1.5.0** for molecular dynamics simulations of materials,
  surfaces, interfaces, solutions, metal complexes and other challenging
  systems.
- **PET-OAM** for materials discovery tasks (convex hull energies,
  geometry optimization, phonons, etc.).
- **PET-SPICE** for accurate and fast simulations of molecules and
  biomolecules.

Legacy models
-------------

For reproducibility or to cover specific use cases, we also provide a few
additional models. These are expected to have worse performance than the
models above.

.. list-table::
   :header-rows: 1
   :widths: 14 22 18 26 20

   * - Name
     - Level of theory
     - Available sizes
     - To be used for
     - Training set
   * - PET-MAD-1
     - PBESol
     - S
     - materials & molecules (85 elements)
     - MAD-1.0
   * - PET-OMAD
     - PBESol
     - XS, S, L
     - materials & molecules (85 elements)
     - OMat → MAD-1.0

All checkpoints are available on the `HuggingFace repository
<https://huggingface.co/lab-cosmo/upet>`_.
