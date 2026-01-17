<div align="center" width="600">
  <picture>
    <source srcset="https://github.com/lab-cosmo/upet/raw/refs/heads/main/docs/static/upet-logo-with-text-dark.svg" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/lab-cosmo/upet/raw/refs/heads/main/docs/static/upet-logo-with-text.svg" alt="Figure">
  </picture>
</div>

> [!WARNING]
> This repository is a successor of the PET-MAD repository, which is now deprecated.
> The package has been renamed to **UPET** to reflect the broader scope of the models
> and functionalities provided, that go beyond the original PET-MAD model.
> Please use the version `1.4.4` of PET-MAD package if you want to use the old API.
> The older version of the README file with documentation is avaiable [here](docs/README_OLD.md).
> The migration guide from PET-MAD to UPET is available [here](docs/UPET_MIGRATION_GUIDE.md).

> [!NOTE]
> Are you here to try our **Matbench model? Here's all you need. Don't be scared by
> the parameter count: you'll see that our model is much faster than you think.**
> This model is excellent for convex hull energies, geometry optimization and phonons,
> but we highly recommend the lighter and more universal PET-MAD for molecular dynamics!
```py
from upet.calculator import UPETCalculator
calculator = UPETCalculator(model="pet-oam-xl", version="1.0.0", device="cuda")
```

# UPET: Universal Models for Advanced Atomistic Simulations

This repository contains **UPET** models - universal interatomic potentials for
advanced materials modeling across the periodic table. These models are based on
the **Point Edge Transformer (PET)**  architecture, trained on various popular 
materials datasets, and are capable of predicting energies and forces in complex
atomistic simulations.

In addition, it contains **PET-MAD-DOS** - a universal model for predicting
the density of states (DOS) of materials, as well as their Fermi levels and bandgaps.
**PET-MAD-DOS** is using a slightly modified **PET** architecture, and the **MAD** dataset. 

## Key Features

- **Universality**: UPET models are generally-applicable, and can be used for
  predicting energies and forces, as well as the density of states, Fermi levels,
  and bandgaps for a wide range of materials and molecules.
- **Accuracy**: UPET models achieve high accuracy in various types of atomistic
  simulations of organic and inorganic systems, comparable with system-specific
  models, while being fast and efficient.
- **Efficiency**: UPET models are highly computationally efficient and have low
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
    - [Evaluating UPET models on a dataset](#evaluating-upet-models-on-a-dataset)
    - [Running UPET models with LAMMPS](#running-upet-models-with-lammps)
    - [Uncertainty Quantification](#uncertainty-quantification)
    - [Rotational Averaging](#rotational-averaging)
    - [Running UPET models with empirical dispersion corrections](#running-upet-models-with-empirical-dispersion-corrections)
    - [Calculating the DOS, Fermi levels, and bandgaps](#calculating-the-dos-fermi-levels-and-bandgaps)
    - [Dataset visualization with the PET-MAD featurizer](#dataset-visualization-with-the-pet-mad-featurizer)
5. [Examples](#examples)
6. [Fine-tuning](#fine-tuning)
7. [Further Documentation](#further-documentation)
8. [FAQs](#faqs)
9. [Citing PET-MAD](#citing-pet-mad)

## Installation

You can install UPET using pip:

```bash
pip install upet
```

Or directly from the GitHub repository:

```bash
pip install git+https://github.com/lab-cosmo/upet.git
```

## Pre-trained Models

Currently, we provide the following pre-trained models:

| Name        | Level of theory         | Available sizes        | To be used for          | Training set          |
|:------------|:-----------------------:|:----------------------:|:-----------------------:|:---------------------:|
| PET-MAD     | PBEsol                  | S                      | materials and molecules | MAD                   |
| PET-OMAD    | PBEsol                  | XS, S, L               | materials and molecules | OMat -> MAD           |
| PET-OAM     | PBE (Materials Project) | L, XL                  | materials               | OMat -> sAlex+MPtrj   |
| PET-OMat    | PBE                     | XS, S, M, L, XL        | materials               | OMat                  |
| PET-OMATPES | r2SCAN                  | L                      | materials               | OMat -> MATPES        |
| PET-SPICE   | ωB97M-D3                | S, L                   | molecules               | SPICE                 | 

We recommend using the PET-MAD model for molecular dynamics simulations of materials, PET-OAM models for materials 
discovery tasks (convex hull energies, geometry optimization, phonons, etc), and PET-SPICE for accurate and fast 
simulations of biomolecules. PET-OMAD models are more accurate and potentially faster than PET-MAD,
but they were not tested as extensively yet. PET-OMATPES can be a good choice in case the accuracy of the PBE
functionals are not sufficient for your applications.

All the checkpoints are available on the HuggingFace [repository](https://huggingface.co/lab-cosmo/upet).

## Interfaces for Atomistic Simulations

UPET integrates with the following atomistic simulation engines:

- **Atomic Simulation Environment (ASE)**
- **LAMMPS** (including the KOKKOS support)
- **i-PI**
- **OpenMM** (coming soon)
- **GROMACS** (coming soon)

## Usage

### ASE Interface

#### Basic usage

In order to perform a simple evaluation of the UPET models on a desired
structure, you can use the UPET calculator compatible with the Atomic
Simulation Environment (ASE). Model name can be obtained from the
table above by combining the model name and the size, e.g., `pet-mad-s`, `pet-omat-l`, etc.

```python
from upet.calculator import UPETCalculator
from ase.build import bulk

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
calculator = UPETCalculator(model="pet-mad-s", version="1.0.2", device="cpu")
atoms.calc = calculator
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

These ASE methods are ideal for single-structure evaluations, but they are
inefficient for the evaluation on a large number of pre-defined structures. To
perform efficient evaluation in that case, read [here](docs/README_BATCHED.md).

#### Non-conservative (direct) forces and stresses prediction

UPET models also support the direct prediction of forces and stresses. In that case,
the forces and stresses are predicted as separate targets along with the energy
target, i.e. not computed as derivatives of the energy using the PyTorch
automatic differentiation. This approach typically leads to 2-3x speedup in the
evaluation time, since backward pass is disabled. However, as discussed in [this
preprint](https://arxiv.org/abs/2412.11569), the non-conservative forces and stresses require additional care to avoid
instabilities during the molecular dynamics simulations.

To use the non-conservative forces and stresses, you need to set the `non_conservative` parameter to `True` when initializing the `UPETCalculator` class.

```python
from upet.calculator import UPETCalculator
from ase.build import bulk

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
calculator = UPETCalculator(model="pet-mad-s", version="1.1.0", device="cpu", non_conservative=True)
atoms.calc = calculator
energy = atoms.get_potential_energy() # energy is computed as usual
forces = atoms.get_forces() # forces now are predicted as a separate target
stresses = atoms.get_stress() # stresses now are predicted as a separate target
```

More details on how to make the direct forces MD simulations reliable are provided 
in the [Atomistic Cookbook](https://atomistic-cookbook.org/examples/pet-mad-nc/pet-mad-nc.html).

### Evaluating UPET models on a dataset

Efficient evaluation of UPET models on a desired dataset is also available from the
command line via [`metatrain`](https://github.com/metatensor/metatrain), which
is installed as a dependency of UPET. To evaluate the model, you first need
to fetch the UPET model from the HuggingFace repository:

```bash
mtt export https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-mad-s-v1.0.2.ckpt -o model.pt
```

Alternatively, you can fetch and save the model using the UPET Python API:

```py
import upet

# Saving the latest version of UPET to a TorchScript file
upet.save_upet(
    model="pet-mad",
    size="s",
    version="1.0.2",
    output="model.pt",
)
```

Both these commands will download the model and convert it to TorchScript format. Next,
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
mtt eval model.pt options.yaml --batch-size=16 --output=predictions.xyz
```

This will create a file called `predictions.xyz` with the predicted energies and
forces for each structure in the dataset. More details on how to use `metatrain`
can be found in the [Metatrain documentation](https://metatensor.github.io/metatrain/latest/getting-started/usage.html#evaluation).

### Uncertainty Quantification

UPET models can also be used to calculate the uncertainty of the energy prediction.
This feature is particularly important if you are interested in probing the model
on the data that is substantially different from the training data. Another important 
use case is a propagation of the uncertainty of the energy prediction to other
observables, like phase transition temperatures, diffusion coefficients, etc.

To evaluate the uncertainty of the energy prediction, or to get an ensemble of energy
predictions, you can use the `get_energy_uncertainty` and `get_energy_ensemble` methods
of the `UPETCalculator` class:

```python
from upet.calculator import UPETCalculator
from ase.build import bulk

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
calculator = UPETCalculator(model="pet-mad-s", version="1.0.2", device="cpu", calculate_uncertainty=True, calculate_ensemble=True)
atoms.calc = calculator
energy = atoms.get_potential_energy()

energy_uncertainty = calculator.get_energy_uncertainty(atoms, per_atom=False)
energy_ensemble = calculator.get_energy_ensemble(atoms, per_atom=False)
```

Please note that the uncertainty quantification and ensemble prediction accepts the
`per_atom` flag, which indicates whether the uncertainty/ensemble should be computed
per atom or for the whole system. More details on the uncertainty quantification and shallow
ensemble method can be found in [this](https://doi.org/10.1088/2632-2153/ad594a) and
[this](https://doi.org/10.1088/2632-2153/ad805f) papers. 


### Rotational Averaging

By design, UPET models are not exactly equivariant w.r.t. rotations and inversions. Although
the equivariance error is typically much smaller than the overall model error, in some cases
(like geometry optimizations and phonon calculations of highly symmetrical structures)
it may be beneficial to enforce additional rotational averaging to improve the stability
of the calculation. This can be done by setting the `rotational_average_order` parameter
when initializing the `UPETCalculator` class:

```python
from upet.calculator import UPETCalculator
from ase.build import bulk

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
calculator = UPETCalculator(model="pet-mad-s", version="1.0.2", device="cpu", rotational_average_order=3)
atoms.calc = calculator
energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stresses = atoms.get_stress()
```

In this case, predictions will be averaged over a quadrature of the O(3) group based on a Lebedev grid of the 
specified order (here 3). Higher orders lead to more accurate equivariance, but also increase the computational cost.

By default, all the transformed structures are evaluated in a single batch, which may lead to high memory usage
for large systems. If you want to reduce the memory usage, you can set the `rotational_average_batch_size` 
parameter to a smaller value (eg. 8), which will evaluate the transformed structures in smaller batches:

```python
from upet.calculator import UPETCalculator
calculator = UPETCalculator(model="pet-mad-s", version="1.0.2", device="cpu", rotational_average_order=3, rotational_average_batch_size=8)
```

Finally, the rotational averaging error statistics are stored in the `results` dictionary of the calculator
after the energy/forces/stresses are computed:

```python
energy_rot_std = calculator.results['energy_rot_std']
forces_rot_std = calculator.results['forces_rot_std']
stresses_rot_std = calculator.results['stresses_rot_std']
```

## Running UPET models with LAMMPS

### 1. Install LAMMPS with metatomic support

To use UPET with LAMMPS, follow the instructions
[here](https://docs.metatensor.org/metatomic/latest/engines/lammps.html#how-to-install-the-code) 
to install lammps-metatomic. We recomend you also use conda to install prebuilt lammps binaries.

### 2. Run LAMMPS with UPET

#### 2.1. CPU version

Fetch the UPET checkpoint from the HuggingFace repository:

```bash
mtt export https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-mad-s-v1.0.2.ckpt -o model.pt
```

This will download the model and convert it to TorchScript format compatible
with LAMMPS, using the `metatomic` and `metatrain` libraries, which UPET is
based on.

Other pre-trained UPET models can be prepared in the same way, e.g.,
```bash
mtt export https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-omat-xs-v1.0.0.ckpt -o model.pt
mtt export https://huggingface.co/lab-cosmo/upet/resolve/main/models/pet-omatpes-l-v0.1.0.ckpt -o model.pt
```

Prepare a lammps input file using `pair_style metatomic` and defining the
mapping from LAMMPS types in the data file to elements UPET can handle using
`pair_coeff` syntax. Here we indicate that lammps atom type 1 is Silicon (atomic
number 14).

```
units metal
atom_style atomic

read_data silicon.data

pair_style metatomic model.pt device cpu # Change device to 'cuda' evaluate UPET on GPU
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

pair_style metatomic/kk model.pt # This will use the same device as the kokkos simulation
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

## Running UPET models with empirical dispersion corrections

### In **ASE**:

You can combine the UPET calculator with the torch based implementation of
the D3 dispersion correction of `pfnet-research` - `torch-dftd`:

Within the UPET environment you can install `torch-dftd` via:

```bash
pip install torch-dftd
```

Then you can use the `D3Calculator` class to combine the two calculators:

```python
from torch_dftd.torch_dftd3_calculator import TorchDFTD3Calculator
from upet.calculator import UPETCalculator
from ase.calculators.mixing import SumCalculator

device = "cuda" if torch.cuda.is_available() else "cpu"

calc_MAD = UPETCalculator(model="pet-mad-s", version="1.0.2", device=device)
dft_d3 = TorchDFTD3Calculator(device=device, xc="pbesol", damping="bj")

combined_calc = SumCalculator([calc_MAD, dft_d3])

# assign the calculator to the atoms object
atoms.calc = combined_calc

```


## Calculating the DOS, Fermi levels, and bandgaps

UPET package also allows the use of the **PET-MAD-DOS** model to predict
electronic density of states of materials, as well as their Fermi levels and
bandgaps. The **PET-MAD-DOS** model is also available in the **ASE** interface.

```python
from upet.calculator import PETMADDOSCalculator

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
 
You can use the last-layer features of the PET-MAD model together with a pre-trained 
sketch-map dimensionality reduction to obtain 2D and 3D representations
of a dataset, e.g. to identify structural or chemical motifs.
This can be used as a stand-alone feature builder, or combined with
the [chemiscope viewer](https://chemiscope.org) to generate an 
interactive visualization. 

```python
import ase.io
import chemiscope
from upet.explore import PETMADFeaturizer

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

UPET models can be fine-tuned using the
[Metatrain](https://docs.metatensor.org/metatrain/latest/generated_examples/0-beginner/02-fine-tuning.html)
library. At the moment, we recommend fine-tuning from our OMat models, because they are
pre-trained on a very large dataset and they come in all sizes (from XS to XL, allowing
you to choose a good trade-off for your application).

## Further Documentation

Additional documentation can be found in the
[metatensor](https://docs.metatensor.org),
[metatomic](https://docs.metatensor.org/metatomic) and
[metatrain](https://metatensor.github.io/metatrain/) repositories.

- [Training a model](https://docs.metatensor.org/metatrain/latest/generated_examples/0-beginner/00-basic-usage.html)
- [Fine-tuning](https://docs.metatensor.org/metatrain/latest/generated_examples/0-beginner/02-fine-tuning.html)
- [LAMMPS interface](https://docs.metatensor.org/metatomic/latest/engines/lammps.html)
- [i-PI interface](https://docs.metatensor.org/metatomic/latest/engines/ipi.html)

## FAQs

**What model should I use for my application?**
- For molecular dynamics simulations, we recommend using the **PET-MAD** model. Alternatively,
  you can use the **PET-OMAD** models, which are more accurate and potentially
  faster, but they were not tested as extensively yet.
- For materials discovery tasks (convex hull energies, geometry optimization, phonons, etc),
  we recommend using the **PET-OAM** models.
- For accurate and fast simulations of biomolecules, we recommend using the **PET-SPICE** models.
- In case the accuracy of the PBE functionals is not sufficient for your application,
  you can try the **PET-OMATPES** model for simulations of materials.
- If you want to fine-tune your own model, we recommend starting from the **PET-OMAT** checkpoints,
  and select an appropriate size (from XS to XL) for your needs.
- In any case, we recommend starting from the smaller models (XS or S) to benchmark your application,
  and then scaling up to larger models if you need more accuracy.

**The model is slow for my application. What should I do?**
- Make sure you run it on a GPU
- Use an S or XS model
- Simulate with LAMMPS (Kokkos-GPU version)
- Use non-conservative forces and stresses, preferably with multiple time stepping. Check out 
  [this example](https://atomistic-cookbook.org/examples/pet-mad-nc/pet-mad-nc.html) for details.
- Still too slow? Check out [FlashMD](https://github.com/lab-cosmo/flashmd) for a further 30x boost.

**My MD ran out of memory. How do I fix that?**
- Reduce the model size (XS models are the least memory-intensive)
- Reduce the structure size
- Try to use LAMMPS (Kokkos-GPU version) and run with multiple MPI tasks to enable domain decomposition
- As a last resort, use non-conservative forces and stresses
- If you hit a weird bug when running more than 65535 atoms on GPU, this is not an out-of-memory bug, but a
  pytorch bug. You can fix it by adding `torch.backends.cuda.enable_mem_efficient_sdp(False)`
  (after importing torch) near the top of your file.

**The model is not fully equivariant. Should I worry?**

Although our models are unconstrained, they are explicitly trained for equivariance, and the equivariance error
is, in the vast majority of cases, one to two orders of magnitude smaller than the machine-learning error with
respect to the target electronic structure method. Hence:
- Read [this paper](https://iopscience.iop.org/article/10.1088/2632-2153/ad86a0) which shows that the impact of non-equivariance on observables is often negligible.
  Proceed to the next two points **only** if you believe that you're seeing effects due to non-equivariance.
- For MD, activate random frame averaging (we are working on a tutorial)
- For geometry optimization, use a symmetrized calculator (see `rotational_average_order` parameter in the ASE calculator)

**The XL models are huge!**
There are two aspects to this:
- The number of parameters is large, but these parameters are used in a very sparse way and the evaluation cost is comparable to, and often lower than, that of other large models in the field.
- The listed cutoff radius may be large, but the cutoff strategy is adaptive, meaning that the model prunes the neighbor list internally. The effective cutoff for the vast majority of atomic environments in materials ends up being between 4 and 7 A in practice.

**If you are fine-tuning our models, please also see the [metatrain FAQs](https://docs.metatensor.org/metatrain/latest/faq.html)**

## Citing UPET Models

If you use any of the UPET models in your research, please cite the corresponding articles:

```bibtex
@misc{PET-MAD-2025,
      title={PET-MAD as a lightweight universal interatomic potential for advanced materials modeling},
      author={Mazitov, Arslan and Bigi, Filippo and Kellner, Matthias and Pegolo, Paolo and Tisi, Davide and Fraux, Guillaume and Pozdnyakov, Sergey and Loche, Philip and Ceriotti, Michele},
      journal={Nature Communications},
      volume={16},
      number={1},
      pages={10653},
      year={2025},
      publisher={Nature Publishing Group UK London},
      url={https://doi.org/10.1038/s41467-025-65662-7},
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
