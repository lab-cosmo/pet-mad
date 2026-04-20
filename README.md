<div align="center" width="600">
  <picture>
    <source srcset="https://github.com/lab-cosmo/upet/raw/refs/heads/main/docs/static/upet-logo-with-text-dark.svg" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/lab-cosmo/upet/raw/refs/heads/main/docs/static/upet-logo-with-text.svg" alt="Figure">
  </picture>
</div>

> [!NOTE]
> The PET-MAD-1.5 models trained for 102 elements at the r2SCAN level of theory are
> now available! These models are more robust, more accurate and faster than the
> previous PET-MAD models. We highly recommend using these models for all applications,
> especially molecular dynamics simulations. Try them out and let us know what you think!
```py
from upet.calculator import UPETCalculator
calculator = UPETCalculator(model="pet-mad-s", version="1.5.0", device="cuda")
```

> [!NOTE]
> Are you here to try our **Matbench model? Here's all you need.**
> This model is excellent for convex hull energies, geometry optimization and phonons,
> but we highly recommend the lighter and more universal PET-MAD for molecular dynamics!
```py
from upet.calculator import UPETCalculator
calculator = UPETCalculator(model="pet-oam-xl", version="1.0.0", device="cuda")
```

> [!WARNING]
> This repository is a successor of the PET-MAD repository, which is now deprecated.
> The package has been renamed to **UPET** to reflect the broader scope of the models
> and functionalities provided, that go beyond the original PET-MAD model.
> Please use version `1.4.4` of PET-MAD package if you want to use the old API.
> The [migration guide](docs/UPET_MIGRATION_GUIDE.md) and the
> [older version of the README](docs/README_OLD.md) are available in the repository.

# UPET: Universal Models for Advanced Atomistic Simulations

**UPET** is a family of universal interatomic potentials for advanced materials
modeling across the periodic table. These models are based on the **Point Edge
Transformer (PET)** architecture trained on various popular atomistic datasets,
and they are capable of predicting energies and forces in complex atomistic
workflows. The package also ships **PET-MAD-DOS**, a universal model for
predicting the electronic density of states (DOS) of materials and molecules,
as well as their Fermi levels and bandgaps.

📖 **Full documentation:** <https://lab-cosmo.github.io/upet/latest/>

## Installation

Install UPET from PyPI:

```bash
pip install upet
```

Or directly from the GitHub repository:

```bash
pip install upet@git+https://github.com/lab-cosmo/upet.git
```

See the [installation guide](https://lab-cosmo.github.io/upet/latest/installation.html)
for additional methods (specific versions, `uv`, etc.).

## Quick start

Run a single-point evaluation with the ASE-compatible `UPETCalculator`:

```python
from upet.calculator import UPETCalculator
from ase.build import bulk

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
calculator = UPETCalculator(model="pet-mad-s", version="1.5.0", device="cpu")
atoms.calc = calculator

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

The first call downloads the checkpoint from the
[HuggingFace repository](https://huggingface.co/lab-cosmo/upet) and caches it
locally, so subsequent calls are fast. If `version` is omitted, the latest
available version is used.

To list available models, sizes, and versions:

```python
from upet import list_upet

list_upet(model="pet-mad", size="s")
list_upet(model="pet-mad")
list_upet()
```

## Pre-trained models

UPET ships several pre-trained model families (PET-MAD, PET-OAM, PET-OMat,
PET-OMATPES, PET-SPICE) at multiple sizes. The full table with supported
elements, training sets, and recommended use cases is in the
[models documentation](https://lab-cosmo.github.io/upet/latest/models.html).
All checkpoints are available on the
[HuggingFace repository](https://huggingface.co/lab-cosmo/upet).

## Going further

The documentation covers the complete feature surface:

- [ASE workflows](https://lab-cosmo.github.io/upet/latest/usage/ase.html):
  non-conservative forces, uncertainty quantification, rotational averaging,
  dispersion corrections, and PET-MAD-DOS for DOS / Fermi levels / bandgaps.
- [Batched evaluation with metatrain](https://lab-cosmo.github.io/upet/latest/usage/metatrain.html).
- [LAMMPS](https://lab-cosmo.github.io/upet/latest/usage/lammps.html) (including
  KOKKOS GPU support),
  [i-PI](https://lab-cosmo.github.io/upet/latest/usage/ipi.html),
  [TorchSim](https://lab-cosmo.github.io/upet/latest/usage/torchsim.html), and
  [GROMACS](https://lab-cosmo.github.io/upet/latest/usage/gromacs.html) interfaces.
- [Fine-tuning](https://lab-cosmo.github.io/upet/latest/fine-tuning.html) and
  [example gallery](https://lab-cosmo.github.io/upet/latest/generated_examples/index.html).
- [FAQ and known issues](https://lab-cosmo.github.io/upet/latest/faq.html).
- [How to cite](https://lab-cosmo.github.io/upet/latest/cite.html) UPET and its
  underlying models.

More worked examples for **ASE, i-PI, and LAMMPS** are also available in the
[Atomistic Cookbook](https://atomistic-cookbook.org/examples/pet-mad/pet-mad.html).

## Maintainers

This project is [maintained](https://github.com/lab-cosmo/.github/blob/main/Maintainers.md) by [@abmazitov](https://github.com/abmazitov) and [@frostedoyster](https://github.com/frostedoyster), who will reply to issues and pull requests opened on this repository as soon as possible. You can mention them directly if you have not received an answer after a couple of days.
