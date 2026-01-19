# Changelog

## Unreleased changes

## 0.1.1
- Fixed a few bugs in the code, improved documentation and added a support for
  specifying the "latest" version of the model in the `upet.get_upet` and 
  `upet.save_upet` functions.

## 0.1.0
- Initial release of the UPET package, succeeding PET-MAD. Updated package
  structure, calculator names, and model naming conventions to reflect the new
  UPET branding and functionalities. 
- Added a support for new models trained on popular datasets for atomistic machine 
  learning.

# Legacy Changelog

## 1.4.4

- Upgraded to metatrain v2025.12
- `PETMADCalculator.get_energy_uncertainty` and
  `PETMADCalculator.get_energy_ensemble` now take an `atoms` parameter (which
  defaults to the last Atoms used when computing the energy) and a `per_atom`
  parameter to get either per-atom or per-structuyre energy uncertainty and
  ensemble.

## 1.4.3

- Added the rotational averaging feature to `PETMADCalculator`.

## 1.4.2

### Changed

- Updated the metatrain dependency to version 2025.10.

## 1.4.1

### Fixed

- Fixed `PetMadFeaturizer` after the release of `PETMADCalculator` with uncertainty quantification

## 1.4.0

### Added

- Added `PETMADDOSCalculator` class for calculating the density of states, Fermi levels, and bandgaps.
- Added the uncertainty quantification feature to `PETMADCalculator`.
- Added the tests against the dev version of `metatrain`.

### Changed

- Changed the default `PET-MAD` model in `PETMADCalculator` to version 1.0.2, which is the one used in the paper.
