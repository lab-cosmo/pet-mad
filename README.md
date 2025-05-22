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

You can install PET-MAD using pip:

```bash
pip install pet-mad
```

Or directly from the GitHub repository:

```bash
pip install git+https://github.com/lab-cosmo/pet-mad.git
```

Alternatively, you can install PET-MAD using `conda` package manager, which is especially important
for running PET-MAD with **LAMMPS**.

> [!WARNING]
> We strongly recommend installing PET-MAD using [`Miniforge`](https://github.com/conda-forge/miniforge)
> as a base `conda` provider, because other `conda` providers (such as `Anaconda`) may yield undesired
> behavior when resolving dependencies and are usually slower than `Miniforge`. Smooth installation
> and use of PET-MAD is not guaranteed with other `conda` providers.

To install Miniforge on unix-like systems, run:

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

## Pre-trained Models

Currently, we provide the following pre-trained models:

- **`v1.1`** or **`latest`**: The updated PET-MAD model with an ability to run simulations using the non-conservative forces and stresses (temporarily disabled).
- **`v1.0.1`**: The updated PET-MAD model with a new, pure PyTorch backend and slightly improved performance.
- **`v1.0.0`**: PET-MAD model trained on the MAD dataset, which contains 95,595 structures, including 3D and 2D inorganic crystals, surfaces, molecular crystals, nanoclusters, and molecules.

## Usage

You can use the PET-MAD calculator, which is compatible with the Atomic Simulation Environment (ASE):

```python
from pet_mad.calculator import PETMADCalculator
from ase.build import bulk

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
pet_mad_calculator = PETMADCalculator(version="latest", device="cpu")
atoms.calc = pet_mad_calculator
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

These ASE methods are ideal for single-structure evaluations, but they are inefficient
for the evaluation on a large number of pre-defined structures. To perform efficient
evaluation in that case, read [here](docs/README_BATCHED.md).

Efficient evaluation of PET-MAD on a desired dataset is also available from the command line via 
[`metatrain`](https://github.com/metatensor/metatrain), which is installed as a depencecy of PET-MAD.
To evaluate the model, you first need to fetch the PET-MAD model from the HuggingFace repository:

```bash
mtt export https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt
```

This command will download the model and convert it to TorchScript format, also collecting the
libraries required to run the model in the `extensions` folder. Then you need to create the 
`options.yaml` file and specify the dataset you want to evaluate the model on 
(where the dataset is stored in `extxyz` format):

```yaml
systems: your-test-dataset.xyz
targets:
  energy:
    key: "energy"
    unit: "eV"
```

Then, you can use the `mtt eval` command to evaluate the model on a dataset:

```bash
mtt eval pet-mad-latest.pt options.yaml --batch-size=16 --extensions-dir=extensions --output=predictions.xyz
```

This will create a file called `predictions.xyz` with the predicted energies and forces for each 
structure in the dataset. More details on how to use `metatrain` can be found in the
[Metatrain documentation](https://metatensor.github.io/metatrain/latest/getting-started/usage.html#evaluation).

## Interfaces for Atomistic Simulations

PET-MAD integrates with the following atomistic simulation engines:

- **Atomic Simulation Environment (ASE)**
- **LAMMPS** (including the KOKKOS support)
- **i-PI**
- **OpenMM** (coming soon)
- **GROMACS** (coming soon)

## Running PET-MAD with LAMMPS

### 1. Install LAMMPS with PET-MAD Support

To use PET-MAD with LAMMPS, you need to first install PET-MAD from `conda`
(see the installation instructions above). Then, install **LAMMPS-METATENSOR**,
which enables PET-MAD support:

```bash
conda install -c metatensor -c conda-forge "lammps-metatensor=*=cpu*nompi*"
```

This command will install a serial CPU version of LAMMPS-METATENSOR, which only allows the molecular dynamics trajectory integration on CPU,
but supports both CPU and GPU modes of PET-MAD evaluation using PyTorch
primitives.

For full GPU-acceleration, you need to install KOKKOS-enabled ``cuda`` version of LAMMPS-METATENSOR. To do this, you need to know the so-called [compute capability](https://developer.nvidia.com/cuda-gpus) of your GPU.
Please find you GPU using the link above, or run the following command:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
```

Currently, the following compute capabilities are supported:

- VOLTA70
- AMPERE80
- AMPERE86
- ADA89
- HOPPER90

For example, if you have a **NVIDIA A100 GPU**, it's compute capability is `8.0` (AMPERE80).

To install the KOKKOS-enabled version of LAMMPS-METATENSOR, use the following command:

```bash
conda install -c metatensor -c conda-forge "lammps-metatensor=*=cuda*AMPERE80*nompi*"
```

Different MPI implementations of LAMMPS-METATENSOR are available:

- **`nompi`**: Serial version
- **`openmpi`**: OpenMPI
- **`mpich`**: MPICH

If you want to use the MPI version of LAMMPS-METATENSOR, you need to choose a desired MPI implementation and install the corresponding version of the package. Please note, that if you want to use the libraries from the system, please follow [these instructions](https://conda-forge.org/docs/user/tipsandtricks/#using-external-message-passing-interface-mpi-libraries) to install the dummy MPI packages first.

Finally, for installing the GPU-accelerated version of LAMMPS-METATENSOR, compatible with NVIDIA A100 GPU and using OpenMPI as the MPI provider, run:

```bash
conda install -c metatensor -c conda-forge "lammps-metatensor=*=cuda*AMPERE80*openmpi*"
```

### 2. Run LAMMPS with PET-MAD

Fetch the PET-MAD checkpoint from the HuggingFace repository:

```bash
mtt export https://huggingface.co/lab-cosmo/pet-mad/resolve/main/models/pet-mad-latest.ckpt
```

This will download the model and convert it to TorchScript format compatible with LAMMPS,
using the `metatensor` and `metatrain` libraries, which PET-MAD is based on. 

Prepare a **`lammps.in`** file using the metatensor `pair_style` and defining the
mapping from LAMMPS types in the data file to elements PET-MAD can handle using the
`pair_coeff * *` syntax. Here we use 14 for Si.

```
units metal
atom_style atomic

read_data silicon.data

pair_style metatensor pet-mad-latest.pt &
  device cpu # Change this to cuda evaluate PET-MAD on GPU
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

Create the **`silicon.data`** data file for a silicon system. 

```
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

```bash
lmp -in lammps.in  # For serial version
mpirun -np 1 lmp -in lammps.in  # For MPI version
```

### 3. Important Notes

- For **CPU calculations**, use a single MPI task unless simulating large systems (30+ Å box size). Multi-threading can be enabled via:
  
  ```bash
  export OMP_NUM_THREADS=4
  ```

- For **GPU calculations**, use **one MPI task per GPU**.

### 4. Running PET-MAD with empirical dispersion corrections

#### In **ASE**:
You can combine the PET-MAD calculator with the torch based implementation of the D3 dispersion correction of `pfnet-research` - `torch-dftd`:

Within the PET-MAD environment you can install `torch-dftd` via:

```bash
pip install torch-dftd
```

Then you can use the `D3Calculator` class to combine the two calculators:

```python
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from pet_mad.calculator import PETMADCalculator
from  ase.calculators.mixing import SumCalculator

device = "cuda:0" if torch.cuda.is_available() else "cpu"

calc_MAD = PETMADCalculator(version="latest", device=device)
dft_d3 = TorchDFTD3Calculator(device=device, xc="pbesol", damping="bj")

combined_calc = SumCalculator([calc_MAD, dft_d3])

# assign the calculator to the atoms object
# atoms.calc = combined_calc

```

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
      url={https://arxiv.org/abs/2503.14118}
}
