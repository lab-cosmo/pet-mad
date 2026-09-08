.. _installation:

Installation
============

UPET requires Python 3.11 or later.

From PyPI
---------

The recommended way to install UPET is from PyPI:

.. code-block:: bash

   pip install upet

To install a specific released version, pass it to ``pip``:

.. code-block:: bash

   pip install "upet==0.2.2"

From GitHub
-----------

To install the latest development version directly from the GitHub
repository:

.. code-block:: bash

   pip install "upet @ git+https://github.com/lab-cosmo/upet.git"

You can also pin to a specific branch, tag or commit:

.. code-block:: bash

   pip install "upet @ git+https://github.com/lab-cosmo/upet.git@main"

With uv
-------

If you use `uv <https://docs.astral.sh/uv/>`_, the equivalent commands are:

.. code-block:: bash

   uv pip install upet
   uv pip install "upet @ git+https://github.com/lab-cosmo/upet.git"
