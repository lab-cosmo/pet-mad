# Changelog

## Unreleased changes

- Added the rotational averaging feature to `PETMADCalculator`.

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
