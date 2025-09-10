<div align="center" width="600">
  <picture>
    <source srcset="https://github.com/lab-cosmo/pet-mad/raw/refs/heads/main/docs/src/static/pet-mad-logo-with-text-dark.svg" media="(prefers-color-scheme: dark)">
    <img src="https://github.com/lab-cosmo/pet-mad/raw/refs/heads/main/docs/src/static/pet-mad-logo-with-text.svg" alt="Figure">
  </picture>
</div>

# PET-MAD: Universal Models for Advanced Atomistic Simulations

**PET-MAD** is a universal interatomic potential for advanced materials modeling across the periodic table. This model is based on the **Point Edge Transformer (PET)** model trained on the **Massive Atomic Diversity (MAD) Dataset** and is capable of predicting energies and forces in complex atomistic simulations.

**PET-MAD-DOS** is a universal model for predicting the density of states (DOS) of materials, as well as their Fermi levels and bandgaps.

## Quick Start

Install PET-MAD:

```bash
pip install pet-mad
```

Use with ASE:

```python
from pet_mad.calculator import PETMADCalculator
from ase.build import bulk

atoms = bulk("Si", cubic=True, a=5.43, crystalstructure="diamond")
calculator = PETMADCalculator(version="latest", device="cpu")
atoms.calc = calculator

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

## Key Features

- **Universal**: Works across the periodic table for diverse materials and molecules
- **Accurate**: High accuracy comparable to system-specific models
- **Efficient**: Fast evaluation with low memory usage
- **Versatile**: Supports multiple MD engines (ASE, LAMMPS, i-PI)
- **Advanced**: Includes uncertainty quantification and DOS prediction

## Supported Simulation Engines

- **ASE** (Atomic Simulation Environment)
- **LAMMPS** (including KOKKOS support)
- **i-PI**
- **OpenMM** (coming soon)
- **GROMACS** (coming soon)

## Documentation

📚 **[Complete Documentation](https://pet-mad.readthedocs.io/)** - Comprehensive guides and tutorials

### Quick Links

- **[Installation Guide](https://pet-mad.readthedocs.io/en/latest/user_guide/installation.html)** - Detailed installation instructions
- **[Quick Start](https://pet-mad.readthedocs.io/en/latest/user_guide/quickstart.html)** - Get started in minutes
- **[ASE Tutorial](https://pet-mad.readthedocs.io/en/latest/tutorials/ase_interface.html)** - Using PET-MAD with ASE
- **[LAMMPS Tutorial](https://pet-mad.readthedocs.io/en/latest/tutorials/lammps_interface.html)** - Using PET-MAD with LAMMPS
- **[DOS Calculations](https://pet-mad.readthedocs.io/en/latest/tutorials/dos_calculations.html)** - Electronic structure predictions
- **[API Reference](https://pet-mad.readthedocs.io/en/latest/api/calculator.html)** - Complete API documentation

## Installation

Install via pip:
```bash
pip install pet-mad
```

Or via conda (recommended for LAMMPS users):
```bash
conda create -n pet-mad
conda activate pet-mad
conda install -c metatensor -c conda-forge pet-mad
```

## Available Models

- **v1.0.2**: Stable PET-MAD model (recommended)
- **v1.1.0**: Development version with non-conservative forces
- **PET-MAD-DOS**: Universal density of states model

## Examples

Find comprehensive examples in the [Atomistic Cookbook](https://atomistic-cookbook.org/examples/pet-mad/pet-mad.html).

## Citation

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
