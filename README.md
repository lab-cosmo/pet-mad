<div align="center" width="600">
  <picture>
    <source srcset="https://github.com/lab-cosmo/pet-mad/raw/refs/heads/main/docs/static/pet-mad-logo-with-text-dark.svg" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/lab-cosmo/pet-mad/raw/refs/heads/main/docs/static/pet-mad-logo-with-text.svg" alt="Figure">
  </picture>
</div>

# PET-MAD: A Universal Interatomic Potential for Advanced Materials Modeling

This repository contains **PET-MAD** - a universal interatomic potential for advanced materials modeling across the periodic table. This model is based on the **Point Edge Transformer (PET)** model trained on the **Massive Atomic Diversity (MAD) Dataset** and is capable of predicting energies and forces in complex atomistic simulations.

## Key Features

- **Universality**: PET-MAD is a generally-applicable model that can be used for a wide range of materials and molecules.
- **Accuracy**: PET-MAD achieves high accuracy in various types of atomistic simulations of organic and inorganic systems, comparable with system-specific models, while being fast and efficient.
- **Efficiency**: PET-MAD achieves high computational efficiency and low memory usage, making it suitable for large-scale simulations.
- **Infrastructure**: Various MD engines are available for diverse research and application needs.
- **HPC Compatibility**: Efficient in HPC environments for extensive simulations.

## Installation

You can install PET-MAD with pip:

```bash
pip install git+https://github.com/lab-cosmo/pet-mad.git
```

Or directly from PyPI (available soon):

```bash
pip install pet-mad
```

Alternatively, you can install PET-MAD from Anaconda, particularly useful for users integrating it with **LAMMPS**. We recommend installing [`Miniforge`](https://github.com/conda-forge/miniforge), a minimal conda installer. To install Miniforge, on unix-like systems, run:

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Once Miniforge is installed, create a new conda environment and install PET-MAD with:

```bash
conda create -n pet-mad
conda activate pet-mad
conda install -c metatensor -c conda-forge pet-mad
```

## Usage

You can use the PET-MAD calculator, which is compatible with the Atomic Simulation Environment (ASE):

```python
from pet_mad.calculator import PETMADCalculator
from ase.build import bulk

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
atoms.calc = PETMADCalculator(version="latest", device="cpu")
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## Pre-trained Models

Currently, we provide the following pre-trained models:

- **`latest`**: PET-MAD model trained on the MAD dataset, which contains 95,595 structures, including 3D and 2D inorganic crystals, surfaces, molecular crystals, nanoclusters, and molecules.

## Interfaces for Atomistic Simulations

PET-MAD integrates with the following atomistic simulation engines:

- **Atomic Simulation Environment (ASE)**
- **LAMMPS**
- **i-PI**
- **OpenMM** (coming soon)
- **GROMACS** (coming soon)

## Running PET-MAD with LAMMPS

### 1. Install LAMMPS with PET-MAD Support

To use PET-MAD with LAMMPS, install PET-MAD from Anaconda (see above). Then, install **LAMMPS-METATENSOR**, which enables PET-MAD support:

```bash
conda install -c metatensor -c conda-forge lammps-metatensor
```

For GPU-accelerated LAMMPS:

```bash
conda install -c metatensor -c conda-forge lammps-metatensor=*=cuda*
```

Different MPI implementations are available:

- **`nompi`**: Serial version
- **`openmpi`**: OpenMPI
- **`mpich`**: MPICH

Example for GPU-accelerated OpenMPI version:

```bash
conda install -c metatensor -c conda-forge lammps-metatensor=*=cuda*openmpi*
```

Please note, that this version is not KOKKOS-enabled, so it provides limited performance on GPUs.
The KOKKOS-enabled version of LAMMPS-METATENSOR will be available soon.

To use system-installed MPI for HPC, install dummy MPI packages first:

```bash
conda install "mpich=x.y.z=external_*"
conda install "openmpi=x.y.z=external_*"
```

### 2. Run LAMMPS with PET-MAD

Fetch the PET-MAD checkpoint from the HuggingFace repository:

```bash
mtt export https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt
```

Prepare a **`lammps.in`** file:

```bash
# LAMMPS input file
units metal
atom_style atomic
read_data silicon.data
pair_style metatensor pet-mad-latest.pt device cpu extensions extensions/
pair_coeff * * 14
neighbor 2.0 bin
timestep 0.001
dump myDump all xyz 10 trajectory.xyz
dump_modify myDump element Si
thermo_style multi
thermo 1
velocity all create 300 87287 mom yes rot yes
fix 1 all nvt temp 300 300 0.10
run 100
```

Create a **`silicon.data`** file:

```bash
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
```

Run LAMMPS. By default, LAMMPS installs the `lmp_serial` executable for 
the serial version and `lmp_mpi` for the MPI version. Because of that,
the running command will be different depending on the version:

```bash
lmp_serial -in lammps.in  # Serial version
mpirun -np 1 lmp_mpi -in lammps.in  # MPI version
```

### 3. Important Notes

- For **CPU calculations**, use a single MPI task unless simulating large systems (30+ Å box size). Multi-threading can be enabled via:
  
  ```bash
  export OMP_NUM_THREADS=4
  ```

- For **GPU calculations**, use **one MPI task per GPU**.

## Examples

More examples for **ASE, i-PI, and LAMMPS** are available in the [Atomistic Cookbook](https://atomistic-cookbook.org/examples/pet-mad/pet-mad.html).

## Fine-tuning

PET-MAD can be fine-tuned using the [Metatrain](https://metatensor.github.io/metatrain/latest/advanced-concepts/fine-tuning.html) library.

## Documentation

Additional documentation can be found in the [Metatensor](https://docs.metatensor.org) and [Metatrain](https://metatensor.github.io/metatrain/latest/index.html) repositories.

- [Training a model](https://metatensor.github.io/metatrain/latest/getting-started/usage.html#training)
- [Fine-tuning](https://metatensor.github.io/metatrain/latest/advanced-concepts/fine-tuning.html)
- [LAMMPS interface](https://docs.metatensor.org/latest/atomistic/engines/lammps.html)
- [i-PI interface](https://docs.metatensor.org/latest/atomistic/engines/ipi.html)

## Citing PET-MAD

If you use PET-MAD in your research, please cite:

```bibtex
@misc{PET-MAD-2025,
      title={PET-MAD, a universal interatomic potential for advanced materials modeling}, 
      author={Arslan Mazitov and Filippo Bigi and Matthias Kellner and Paolo Pegolo and Davide Tisi and Guillaume Fraux and Sergey Pozdnyakov and Philip Loche and Michele Ceriotti},
      year={2025},
      eprint={2503.14118},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2503.14118}, 
}
