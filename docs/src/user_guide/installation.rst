Installation
============

PET-MAD can be installed using pip or conda. We recommend using conda for the best compatibility, especially when using LAMMPS.

Using pip
---------

You can install PET-MAD using pip:

.. code-block:: bash

   pip install pet-mad

Or directly from the GitHub repository:

.. code-block:: bash

   pip install git+https://github.com/lab-cosmo/pet-mad.git

Using conda
-----------

Alternatively, you can install PET-MAD using the conda package manager, which is especially important for running PET-MAD with LAMMPS.

.. warning::
   We strongly recommend installing PET-MAD using `Miniforge <https://github.com/conda-forge/miniforge>`_ as a base conda provider, because other conda providers (such as Anaconda) may yield undesired behavior when resolving dependencies and are usually slower than Miniforge. Smooth installation and use of PET-MAD is not guaranteed with other conda providers.

To install Miniforge on unix-like systems, run:

.. code-block:: bash

   wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh

Once Miniforge is installed, create a new conda environment and install PET-MAD with:

.. code-block:: bash

   conda create -n pet-mad
   conda activate pet-mad
   conda install -c metatensor -c conda-forge pet-mad

Development Installation
------------------------

For development purposes, you can install PET-MAD in editable mode:

.. code-block:: bash

   git clone https://github.com/lab-cosmo/pet-mad.git
   cd pet-mad
   pip install -e .

To build the documentation locally, install the documentation dependencies:

.. code-block:: bash

   pip install -e .[docs]

Requirements
------------

PET-MAD requires:

- Python >= 3.10
- PyTorch (installed automatically with dependencies)
- metatrain == 2025.9.1
- Additional dependencies for specific features (installed automatically)

Optional Dependencies
---------------------

For deprecated features (v1.0.0 compatibility):

.. code-block:: bash

   pip install pet-mad[deprecated]

For documentation building:

.. code-block:: bash

   pip install pet-mad[docs]
