<div align="center" width="600">
  <picture>
    <source srcset="https://github.com/lab-cosmo/pet-mad/raw/refs/heads/main/docs/static/pet-mad-logo-with-text-dark.svg" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/lab-cosmo/pet-mad/raw/refs/heads/main/docs/static/pet-mad-logo-with-text.svg" alt="Figure">
  </picture>
</div>

# PET-MAD: Universal Models for Advanced Atomistic Simulations

This repository contains **PET-MAD** - a universal interatomic potential for
advanced materials modeling across the periodic table. This model is based on
the **Point Edge Transformer (PET)** model trained on the **Massive Atomic Diversity (MAD) Dataset** 
and is capable of predicting energies and forces in complex atomistic simulations.

In addition, it contains **PET-MAD-DOS** - a universal model for predicting
the density of states (DOS) of materials, as well as their Fermi levels and bandgaps.
**PET-MAD-DOS** is using a slightly modified **PET** architecture, and the same
**MAD** dataset. 

## Key Features

- **Universality**: PET-MAD models are generally-applicable, and can be used for
  predicting energies and forces, as well as the density of states, Fermi levels,
  and bandgaps for a wide range of materials and molecules.
- **Accuracy**: PET-MAD models achieve high accuracy in various types of atomistic
  simulations of organic and inorganic systems, comparable with system-specific
  models, while being fast and efficient.
- **Efficiency**: PET-MAD models are highly computationally efficient and have low 
  memory usage, what makes them suitable for large-scale simulations.
- **Infrastructure**: Various MD engines are available for diverse research and
  application needs.
- **HPC Compatibility**: Efficient in HPC environments for extensive simulations.

## Table of Contents
1. [Installation](#installation)
2. [Pre-trained Models](#pre-trained-models)
3. [Interfaces for Atomistic Simulations](#interfaces-for-atomistic-simulations)
4. [Usage](#usage)
    - [ASE Interface](#ase-interface)
        - [Basic usage](#basic-usage)
        - [Non-conservative (direct) forces and stresses prediction](#non-conservative-direct-forces-and-stresses-prediction)
    - [Evaluating PET-MAD on a dataset](#evaluating-pet-mad-on-a-dataset)
    - [Running PET-MAD with LAMMPS](#running-pet-mad-with-lammps)
    - [Uncertainty Quantification](#uncertainty-quantification)
    - [Running PET-MAD with empirical dispersion corrections](#running-pet-mad-with-empirical-dispersion-corrections)
    - [Calculating the DOS, Fermi levels, and bandgaps](#calculating-the-dos-fermi-levels-and-bandgaps)
    - [Dataset visualization with the PET-MAD featurizer](#dataset-visualization-with-the-pet-mad-featurizer)
5. [Examples](#examples)
6. [Fine-tuning](#fine-tuning)
7. [Documentation](#documentation)
8. [Citing PET-MAD](#citing-pet-mad)

## Installation

You can install PET-MAD using pip:

```bash
pip install pet-mad
```

Or directly from the GitHub repository:

```bash
pip install git+https://github.com/lab-cosmo/pet-mad.git
```

Alternatively, you can install PET-MAD using `conda` package manager, which is
especially important for running PET-MAD with **LAMMPS**.

> [!WARNING]
> We strongly recommend installing PET-MAD using
> [`Miniforge`](https://github.com/conda-forge/miniforge) as a base `conda`
> provider, because other `conda` providers (such as `Anaconda`) may yield
> undesired behavior when resolving dependencies and are usually slower than
> `Miniforge`. Smooth installation and use of PET-MAD is not guaranteed with
> other `conda` providers.

To install Miniforge on unix-like systems, run:

```bash
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

Once Miniforge is installed, create a new conda environment and install PET-MAD
with:

```bash
conda create -n pet-mad
conda activate pet-mad
conda install -c metatensor -c conda-forge pet-mad
```

## Pre-trained Models

Currently, we provide the following pre-trained models:

- **`v1.1.0`**: The dev version of the PET-MAD model with the non-conservative
  forces and stresses. This version has notably worse performance on molecular
  systems, and is not recommended for production use, as for now.
- **`v1.0.2`**: Stable PET-MAD model trained on the MAD dataset, which contains 95,595
  structures, including 3D and 2D inorganic crystals, surfaces, molecular
  crystals, nanoclusters, and molecules. Use this version in the case you want
  to repoduce the results from the [PET-MAD paper](https://arxiv.org/abs/2503.14118).

## Interfaces for Atomistic Simulations

PET-MAD integrates with the following atomistic simulation engines:

- **Atomic Simulation Environment (ASE)**
- **LAMMPS** (including the KOKKOS support)
- **i-PI**
- **OpenMM** (coming soon)
- **GROMACS** (coming soon)

## Usage

### ASE Interface

#### Basic usage

You can use the PET-MAD calculator, which is compatible with the Atomic
Simulation Environment (ASE):

```python
from pet_mad.calculator import PETMADCalculator
from ase.build import bulk

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
pet_mad_calculator = PETMADCalculator(version="latest", device="cpu")
atoms.calc = pet_mad_calculator
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

These ASE methods are ideal for single-structure evaluations, but they are
inefficient for the evaluation on a large number of pre-defined structures. To
perform efficient evaluation in that case, read [here](docs/README_BATCHED.md).


#### Non-conservative (direct) forces and stresses prediction

PET-MAD also supports the direct prediction of forces and stresses. In that case,
the forces and stresses are predicted as separate targets along with the energy
target, i.e. not computed as derivatives of the energy using the PyTorch
automatic differentiation. This approach typically leads to 2-3x speedup in the
evaluation time, since backward pass is disabled. However, as discussed in [this
preprint](https://arxiv.org/abs/2412.11569) it requires additional care to avoid
instabilities during the molecular dynamics simulations.

To use the non-conservative forces and stresses, you need to set the `non_conservative` parameter to `True` when initializing the `PETMADCalculator` class.

```python
from pet_mad.calculator import PETMADCalculator
from ase.build import bulk

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
pet_mad_calculator = PETMADCalculator(version="latest", device="cpu", non_conservative=True)
atoms.calc = pet_mad_calculator
energy = atoms.get_potential_energy() # energy is computed as usual
forces = atoms.get_forces() # forces now are predicted as a separate target
stresses = atoms.get_stress() # stresses now are predicted as a separate target
```

More details on how to make the direct forces MD simulations reliable are provided 
in the [Atomistic Cookbook](https://atomistic-cookbook.org/examples/pet-mad-nc/pet-mad-nc.html).

### Evaluating PET-MAD on a dataset

Efficient evaluation of PET-MAD on a desired dataset is also available from the
command line via [`metatrain`](https://github.com/metatensor/metatrain), which
is installed as a dependency of PET-MAD. To evaluate the model, you first need
to fetch the PET-MAD model from the HuggingFace repository:

```bash
mtt export https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.0.2/models/pet-mad-v1.0.2.ckpt -o pet-mad-latest.pt
```

Alternatively, you can also download the model from Python:

```py
import pet_mad

# Saving the latest version of PET-MAD to a TorchScript file
pet_mad.save_pet_mad(version="latest", output="pet-mad-latest.pt")

# you can also get a metatomic AtomisticModel for advance usage
model = pet_mad.get_pet_mad(version="latest")
```

This command will download the model and convert it to TorchScript format. Then
you need to create the `options.yaml` file and specify the dataset you want to
evaluate the model on (where the dataset is stored in `extxyz` format):

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

This will create a file called `predictions.xyz` with the predicted energies and
forces for each structure in the dataset. More details on how to use `metatrain`
can be found in the [Metatrain documentation](https://metatensor.github.io/metatrain/latest/getting-started/usage.html#evaluation).

### Uncertainty Quantification

PET-MAD can also be used to calculate the uncertainty of the energy prediction.
This feature is particularly important if you are interested in probing the model
on the data that is far away from the training data. Another important use case
is a propagation of the uncertainty of the energy prediction to other observables,
like phase transition temperatures, diffusion coefficients, etc.

To evaluate the uncertainty of the energy prediction, or to get an ensemble of energy
predictions, you can use the `get_energy_uncertainty` and `get_energy_ensemble` methods
of the `PETMADCalculator` class:

```python
from pet_mad.calculator import PETMADCalculator
from ase.build import bulk

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
pet_mad_calculator = PETMADCalculator(version="latest", device="cpu")
atoms.calc = pet_mad_calculator
energy = atoms.get_potential_energy()

energy_uncertainty = pet_mad_calculator.get_energy_uncertainty(atoms, per_atom=False)
energy_ensemble = pet_mad_calculator.get_energy_ensemble(atoms, per_atom=False)
```

Please note that the uncertainty quantification and ensemble prediction accepts the
`per_atom` flag, which indicates whether the uncertainty/ensemble should be computed
per atom or for the whole system. More details on the uncertainty quantification and shallow
ensemble method can be found in [this](https://doi.org/10.1088/2632-2153/ad594a) and
[this](https://doi.org/10.1088/2632-2153/ad805f) papers. 


## Running PET-MAD with LAMMPS

### 1. Install LAMMPS with metatomic support

To use PET-MAD with LAMMPS, you need to first install PET-MAD from `conda` (see
the installation instructions above). Then, follow the instructions
[here](https://docs.metatensor.org/metatomic/latest/engines/lammps.html#how-to-install-the-code) to install lammps-metatomic. We recomend you also use conda to install lammps.

### 2. Run LAMMPS with PET-MAD

#### 2.1. CPU version

Fetch the PET-MAD checkpoint from the HuggingFace repository:

```bash
mtt export https://huggingface.co/lab-cosmo/pet-mad/resolve/v1.0.2/models/pet-mad-v1.0.2.ckpt -o pet-mad-latest.pt
```

This will download the model and convert it to TorchScript format compatible
with LAMMPS, using the `metatomic` and `metatrain` libraries, which PET-MAD is
based on.

Prepare a lammps input file using `pair_style metatomic` and defining the
mapping from LAMMPS types in the data file to elements PET-MAD can handle using
`pair_coeff` syntax. Here we indicate that lammps atom type 1 is Silicon (atomic
number 14).

```
units metal
atom_style atomic

read_data silicon.data

pair_style metatomic pet-mad-latest.pt device cpu # Change device to 'cuda' evaluate PET-MAD on GPU
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

#### 2.2. KOKKOS-enabled GPU version

Running LAMMPS with KOKKOS and GPU support is similar to the CPU version, but
you need to change the `lammps.in` slightly and run `lmp` binary with a few
additional flags.

The updated `lammps.in` file looks like this:

```
package kokkos newton on neigh half

units metal
atom_style atomic/kk

read_data silicon.data

pair_style metatomic/kk pet-mad-latest.pt # This will use the same device as the kokkos simulation
pair_coeff * * 14

neighbor 2.0 bin
timestep 0.001

dump myDump all xyz 10 trajectory.xyz
dump_modify myDump element Si

thermo_style multi
thermo 1

velocity all create 300 87287 mom yes rot yes

fix 1 all nvt temp 300 300 0.10

run_style verlet/kk
run 100
```

The **silicon.data** file remains the same.

To run the KOKKOS-enabled version of LAMMPS, you need to run

```bash
lmp -in lammps.in -k on g 1 -sf kk # For serial version
mpirun -np 1 lmp -in lammps.in -k on g 1 -sf kk # For MPI version
```

Here, the `-k on g 1 -sf kk` flags are used to activate the KOKKOS
subroutines. Specifically `g 1` is used to specify, how many GPUs are the
simulation is parallelized over, so if running the large systems on two or more
GPUs, this number should be adjusted accordingly.


### 3. Important Notes

- For **CPU calculations**, use a single MPI task unless simulating large
  systems (30+ Å box size). Multi-threading can be enabled via:

  ```bash
  export OMP_NUM_THREADS=4
  ```

- For **GPU calculations**, use **one MPI task per GPU**.

## Running PET-MAD with empirical dispersion corrections

### In **ASE**:

You can combine the PET-MAD calculator with the torch based implementation of
the D3 dispersion correction of `pfnet-research` - `torch-dftd`:

Within the PET-MAD environment you can install `torch-dftd` via:

```bash
pip install torch-dftd
```

Then you can use the `D3Calculator` class to combine the two calculators:

```python
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from pet_mad.calculator import PETMADCalculator
from  ase.calculators.mixing import SumCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"

calc_MAD = PETMADCalculator(version="latest", device=device)
dft_d3 = TorchDFTD3Calculator(device=device, xc="pbesol", damping="bj")

combined_calc = SumCalculator([calc_MAD, dft_d3])

# assign the calculator to the atoms object
atoms.calc = combined_calc

```


## Calculating the DOS, Fermi levels, and bandgaps

PET-MAD packages also allows the use of the **PET-MAD-DOS** model to predict 
electronic density of states of materials, as well as their Fermi levels and
bandgaps. Similarly to the  **PET-MAD** model, the **PET-MAD-DOS** model is
also available in the **ASE** interface.

```python
from pet_mad.calculator import PETMADDOSCalculator

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
pet_mad_dos_calculator = PETMADDOSCalculator(version="latest", device="cpu")

energies, dos = pet_mad_dos_calculator.calculate_dos(atoms)
```

Predicting the densities of states for every atom in the crystal,
or a list of atoms, is also possible:

```python
# Calculating the DOS for every atom in the crystal
energies, dos_per_atom = pet_mad_dos_calculator.calculate_dos(atoms, per_atom=True)

# Calculating the DOS for a list of atoms
atoms_1 = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
atoms_2 = bulk("C", cubic=True, a=3.55, crystalstructure="diamond")

energies, dos = pet_mad_dos_calculator.calculate_dos([atoms_1, atoms_2], per_atom=False)
```

Finally, you can use the `calculate_bandgap` and `calculate_efermi` methods to
predict the bandgap and Fermi level for the crystal:

```python
bandgap = pet_mad_dos_calculator.calculate_bandgap(atoms)
fermi_level = pet_mad_dos_calculator.calculate_efermi(atoms)
```

You can also re-use the DOS calculated earlier:

```python
bandgap = pet_mad_dos_calculator.calculate_bandgap(atoms, dos=dos)
fermi_level = pet_mad_dos_calculator.calculate_efermi(atoms, dos=dos)
```

This option is also available for a list of `ase.Atoms` objects:

```python
bandgaps = pet_mad_dos_calculator.calculate_bandgap([atoms_1, atoms_2], dos=dos)
fermi_levels = pet_mad_dos_calculator.calculate_efermi([atoms_1, atoms_2], dos=dos)
```


## Dataset visualization with the PET-MAD featurizer
 
You can use PET-MAD last-layer features together with a pre-trained 
sketch-map dimensionality reduction to obtain 2D and 3D representations
of a dataset, e.g. to identify structural or chemical motifs.
This can be used as a stand-alone feature builder, or combined with
the [chemiscope viewer](https://chemiscope.org) to generate an 
interactive visualization. 

```python
import ase.io
import chemiscope
from pet_mad.explore import PETMADFeaturizer

featurizer = PETMADFeaturizer(version="latest")

# Load structures
frames = ase.io.read("dataset.xyz", ":")

# You can just compute features
features = featurizer(frames, None)

# Or create an interactive visualization in a Jupyter notebook
chemiscope.explore(
    frames,
    featurize=featurizer
)
```

## Examples

More examples for **ASE, i-PI, and LAMMPS** are available in the [Atomistic
Cookbook](https://atomistic-cookbook.org/examples/pet-mad/pet-mad.html).

## Fine-tuning

PET-MAD can be fine-tuned using the
[Metatrain](https://metatensor.github.io/metatrain/latest/advanced-concepts/fine-tuning.html)
library.

## Documentation

Additional documentation can be found in the
[metatensor](https://docs.metatensor.org),
[metatomic](https://docs.metatensor.org/metatomic) and
[metatrain](https://metatensor.github.io/metatrain/) repositories.

- [Training a model](https://metatensor.github.io/metatrain/latest/getting-started/usage.html#training)
- [Fine-tuning](https://metatensor.github.io/metatrain/latest/advanced-concepts/fine-tuning.html)
- [LAMMPS interface](https://docs.metatensor.org/metatomic/latest/engines/lammps.html)
- [i-PI interface](https://docs.metatensor.org/metatomic/latest/engines/ipi.html)

## Citing PET-MAD Models

If you use any of the PET-MAD models in your research, please cite the corresponding articles:

```bibtex
@misc{PET-MAD-2025,
  title={PET-MAD as a lightweight universal interatomic potential for advanced materials modeling},
  author={Mazitov, Arslan and Bigi, Filippo and Kellner, Matthias and Pegolo, Paolo and Tisi, Davide and Fraux, Guillaume and Pozdnyakov, Sergey and Loche, Philip and Ceriotti, Michele},
  journal={Nature Communications},
  volume={16},
  number={1},
  pages={10653},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
@misc{PET-MAD-DOS-2025,
      title={A universal machine learning model for the electronic density of states}, 
      author={Wei Bin How and Pol Febrer and Sanggyu Chong and Arslan Mazitov and Filippo Bigi and Matthias Kellner and Sergey Pozdnyakov and Michele Ceriotti},
      year={2025},
      eprint={2508.17418},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2508.17418}, 
}
