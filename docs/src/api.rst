.. _api:

API reference
=============

Auto-generated reference for UPET's Python API

Top-level functions
-------------------

.. currentmodule:: upet

.. autofunction:: get_upet

.. autofunction:: list_upet

.. autofunction:: save_upet


Calculators
-----------

.. currentmodule:: upet.ase

.. autoclass:: UPETCalculator
   :members:


PET-MAD-DOS
-----------

.. currentmodule:: upet.ase.dos

.. autoclass:: PETMADDOSCalculator
   :members:


Featurizer
----------

.. currentmodule:: upet.ase.explore

.. autoclass:: PETMADFeaturizer
   :members:


nvalchemi-toolkit integration
------------------------------

Requires the optional ``nvalchemi`` extra (``pip install "upet[nvalchemi]"``);
see :ref:`installation`.

.. currentmodule:: upet.nvalchemi

.. autoclass:: UPETWrapper
   :members:
