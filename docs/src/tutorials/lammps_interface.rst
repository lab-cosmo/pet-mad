LAMMPS Interface Tutorial
==========================

This tutorial covers how to use PET-MAD with LAMMPS for large-scale molecular dynamics simulations.

Prerequisites
-------------

Before using PET-MAD with LAMMPS, you need:

1. LAMMPS compiled with metatomic support
2. PET-MAD installed via conda (recommended)
3. The PET-MAD model file in TorchScript format

Installation
------------

Install LAMMPS with Metatomic Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # First install PET-MAD via conda
   conda create -n pet-mad
   conda activate pet-mad
   conda install -c metatensor -c conda-forge pet-mad

   # Install LAMMPS with metatomic
   conda install -c conda-forge lammps

Alternatively, follow the detailed instructions at the `metatomic documentation <https://docs.metatensor.org/metatomic/latest/engines/lammps.html>`_.

Download PET-MAD Model
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Download and convert model to TorchScript format
   mtt export https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.0.2/models/pet-mad-v1.0.2.ckpt -o pet-mad-latest.pt

Or from Python:

.. code-block:: python

   import pet_mad
   pet_mad.save_pet_mad(version="latest", output="pet-mad-latest.pt")

Basic LAMMPS Setup
------------------

CPU Version
~~~~~~~~~~~

Create a LAMMPS input file (``lammps.in``):

.. code-block:: text

   # Basic LAMMPS input for PET-MAD
   units metal
   atom_style atomic

   # Read structure
   read_data silicon.data

   # Use PET-MAD potential
   pair_style metatomic pet-mad-latest.pt device cpu
   pair_coeff * * 14  # Silicon has atomic number 14

   # Neighbor settings
   neighbor 2.0 bin
   timestep 0.001  # 1 fs

   # Output settings
   dump myDump all xyz 10 trajectory.xyz
   dump_modify myDump element Si

   thermo_style multi
   thermo 1

   # Initial velocities for 300K
   velocity all create 300 87287 mom yes rot yes

   # NVT ensemble
   fix 1 all nvt temp 300 300 0.10

   # Run simulation
   run 100

GPU Version with KOKKOS
~~~~~~~~~~~~~~~~~~~~~~~

For GPU acceleration, modify the input file:

.. code-block:: text

   # Enable KOKKOS
   package kokkos newton on neigh half

   units metal
   atom_style atomic/kk

   read_data silicon.data

   # GPU version - device is automatically detected
   pair_style metatomic/kk pet-mad-latest.pt
   pair_coeff * * 14

   neighbor 2.0 bin
   timestep 0.001

   dump myDump all xyz 10 trajectory.xyz
   dump_modify myDump element Si

   thermo_style multi
   thermo 1

   velocity all create 300 87287 mom yes rot yes
   fix 1 all nvt temp 300 300 0.10

   # Use KOKKOS run style
   run_style verlet/kk
   run 100

Creating LAMMPS Data Files
--------------------------

Silicon Crystal Example
~~~~~~~~~~~~~~~~~~~~~~~

Create ``silicon.data``:

.. code-block:: text

   # LAMMPS data file for Silicon unit cell
   8 atoms
   1 atom types

   0.0  5.43  xlo xhi
   0.0  5.43  ylo yhi
   0.0  5.43  zlo zhi

   Masses

   1  28.084999992775295 # Si

   Atoms # atomic

   1   1   0       0       0
   2   1   1.3575  1.3575  1.3575
   3   1   0       2.715   2.715
   4   1   1.3575  4.0725  4.0725
   5   1   2.715   0       2.715
   6   1   4.0725  1.3575  4.0725
   7   1   2.715   2.715   0
   8   1   4.0725  4.0725  1.3575

Multi-element Systems
~~~~~~~~~~~~~~~~~~~~~

For systems with multiple elements:

.. code-block:: text

   # Example: SiC system
   pair_style metatomic pet-mad-latest.pt device cpu
   pair_coeff * * 14 6  # Si=14, C=6

   # Or use element mapping
   pair_coeff 1 1 14    # Atom type 1 is Silicon
   pair_coeff 2 2 6     # Atom type 2 is Carbon
   pair_coeff 1 2 14 6  # Mixed interactions

Running LAMMPS Simulations
---------------------------

Serial Execution
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # CPU version
   lmp -in lammps.in

   # GPU version with KOKKOS
   lmp -in lammps.in -k on g 1 -sf kk

Parallel Execution
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # CPU parallel (use carefully - see performance notes)
   mpirun -np 4 lmp -in lammps.in

   # GPU parallel (one MPI rank per GPU)
   mpirun -np 2 lmp -in lammps.in -k on g 2 -sf kk

Advanced Simulation Examples
----------------------------

Molecular Dynamics with Temperature Ramping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   units metal
   atom_style atomic

   read_data system.data

   pair_style metatomic pet-mad-latest.pt device cpu
   pair_coeff * * 14

   neighbor 2.0 bin
   timestep 0.001

   # Output
   dump 1 all xyz 100 heating.xyz
   thermo 100

   # Heat from 100K to 500K
   velocity all create 100 12345
   fix 1 all nvt temp 100 500 0.1

   run 10000  # Heat for 10 ps

   # Equilibrate at 500K
   unfix 1
   fix 2 all nvt temp 500 500 0.1

   run 50000  # Equilibrate for 50 ps

Pressure Control (NPT)
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   units metal
   atom_style atomic

   read_data system.data

   pair_style metatomic pet-mad-latest.pt device cpu
   pair_coeff * * 14

   neighbor 2.0 bin
   timestep 0.001

   # NPT ensemble
   velocity all create 300 54321
   fix 1 all npt temp 300 300 0.1 iso 0.0 0.0 1.0

   # Output
   dump 1 all xyz 1000 npt.xyz
   thermo 1000
   thermo_style custom step temp press vol density etotal

   run 100000

Deformation and Mechanical Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   units metal
   atom_style atomic

   read_data system.data

   pair_style metatomic pet-mad-latest.pt device cpu
   pair_coeff * * 14

   neighbor 2.0 bin
   timestep 0.001

   # Equilibrate first
   velocity all create 300 98765
   fix 1 all nvt temp 300 300 0.1
   run 10000

   # Apply uniaxial strain
   unfix 1
   fix 1 all nvt temp 300 300 0.1
   fix 2 all deform 1 x scale 1.01  # 1% strain per 1000 steps

   # Output stress-strain data
   variable strain equal "(lx - v_L0)/v_L0"
   variable stress equal "-pxx/10000"  # Convert to GPa
   fix output all print 100 "${strain} ${stress}" file stress_strain.dat

   run 10000

Surface and Interface Simulations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   units metal
   atom_style atomic
   boundary p p f  # Periodic in x,y; fixed in z (surface)

   read_data surface.data

   pair_style metatomic pet-mad-latest.pt device cpu
   pair_coeff * * 14

   neighbor 2.0 bin
   timestep 0.001

   # Fix bottom layer
   region bottom block INF INF INF INF INF 2.0
   group bottom region bottom
   fix freeze bottom setforce 0.0 0.0 0.0

   # Thermostat for mobile atoms
   group mobile subtract all bottom
   velocity mobile create 300 11111
   fix 1 mobile nvt temp 300 300 0.1

   dump 1 all xyz 1000 surface.xyz
   thermo 1000

   run 100000

Performance Optimization
------------------------

CPU Performance
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Set number of OpenMP threads
   export OMP_NUM_THREADS=4

   # For CPU, generally use single MPI task unless system is very large
   lmp -in input.in

.. note::
   For CPU calculations, use a single MPI task unless simulating very large systems (30+ Å box size). Multi-threading via OpenMP is usually more efficient.

GPU Performance
~~~~~~~~~~~~~~~

.. code-block:: bash

   # One MPI task per GPU
   mpirun -np 2 lmp -in input.in -k on g 2 -sf kk

   # For single GPU
   lmp -in input.in -k on g 1 -sf kk

Memory Considerations
~~~~~~~~~~~~~~~~~~~~~

For large systems:

.. code-block:: text

   # Reduce neighbor list frequency
   neighbor 2.0 bin
   neigh_modify every 10 delay 0 check yes

   # Use smaller cutoffs if appropriate
   # (PET-MAD has learned cutoffs, so this should be done carefully)

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **Model not found**: Ensure the ``.pt`` file is in the correct location
2. **Element mapping errors**: Check that atomic numbers match your system
3. **Performance issues**: See performance optimization section

Debug Mode
~~~~~~~~~~

.. code-block:: text

   # Add debug output
   log debug.log

   # Check pair style is working
   pair_write 1 1 1000 r 0.5 10.0 table.txt Si_Si

Error Messages
~~~~~~~~~~~~~~

Common error messages and solutions:

- **"Cannot find metatomic pair style"**: LAMMPS not compiled with metatomic support
- **"Model file not found"**: Check path to ``.pt`` file
- **"Unsupported element"**: PET-MAD supports elements 1-86 except Astatine

Integration with Analysis Tools
-------------------------------

With OVITO
~~~~~~~~~~

.. code-block:: python

   from ovito.io import import_file
   from ovito.modifiers import *

   # Load trajectory
   pipeline = import_file("trajectory.xyz")

   # Add analysis modifiers
   pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=3.0))
   pipeline.modifiers.append(CommonNeighborAnalysisModifier())

   # Export results
   pipeline.compute()

With MDAnalysis
~~~~~~~~~~~~~~~

.. code-block:: python

   import MDAnalysis as mda

   # Load trajectory
   u = mda.Universe("system.data", "trajectory.xyz", format="XYZ")

   # Analyze trajectory
   for ts in u.trajectory:
       # Perform analysis
       pass

Post-Processing Examples
------------------------

Energy Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   # Read LAMMPS log file
   data = np.loadtxt("log.lammps", skiprows=3)

   time = data[:, 0] * 0.001  # Convert to ps
   temp = data[:, 1]
   energy = data[:, 2]

   plt.figure(figsize=(12, 4))

   plt.subplot(1, 2, 1)
   plt.plot(time, temp)
   plt.xlabel('Time (ps)')
   plt.ylabel('Temperature (K)')

   plt.subplot(1, 2, 2)
   plt.plot(time, energy)
   plt.xlabel('Time (ps)')
   plt.ylabel('Total Energy (eV)')

   plt.tight_layout()
   plt.show()

Radial Distribution Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ase.io import read
   import numpy as np

   # Read trajectory
   frames = read("trajectory.xyz", ":")

   def rdf(frames, rmax=10.0, nbins=100):
       """Calculate radial distribution function"""
       # Implementation would go here
       pass

   r, g_r = rdf(frames)

   plt.plot(r, g_r)
   plt.xlabel('Distance (Å)')
   plt.ylabel('g(r)')
   plt.show()

Best Practices
--------------

1. **Start Small**: Test with small systems first
2. **Equilibration**: Always equilibrate your system before production runs
3. **Timestep**: Use appropriate timesteps (typically 0.5-1.0 fs for PET-MAD)
4. **Monitoring**: Monitor energy, temperature, and pressure during runs
5. **Validation**: Compare results with known experimental or computational data

Example Workflow
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Prepare system
   python create_structure.py

   # 2. Energy minimization
   lmp -in minimize.in

   # 3. Equilibration
   lmp -in equilibrate.in

   # 4. Production run
   lmp -in production.in

   # 5. Analysis
   python analyze_results.py
