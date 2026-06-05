Changelog
=========

Unreleased changes
------------------

0.2.4
-----
- Pin the version of warp-lang in dependencies as a hotfix around an internal
  change.

0.2.3
-----
- Added sphinx documentation for the package.
- Switched to using ``metatomic_ase`` directly instead of ``metatomic.torch.ase_calculator``.
- Updated ``nvalchemi-toolkit-ops`` dependency to v0.3.0.

0.2.2
-----
- Added ``list_upet()`` function for listing available models.
- Pinned ``nvalchemi-toolkit-ops`` dependency version.
- Removed unused ``scipy`` dependency.
- Enabled Windows CI tests.

0.2.1
-----
- Added ``pet-mad-xs`` to the available models list.
- Added citation for PET-MAD-1.5.

0.2.0
-----
- Added PET-MAD v1.5.0 model.
- Deprecated the PET-MAD and PET-OMAD models trained on the original
  MAD-1 dataset.

0.1.2
-----
- Added speed benchmarks for the UPET models
- Added a support for loading the local checkpoints
- Added variants selection for running different heads of the model
- Modified the internal TorchScript exporting mechanism, so it saves
  a temporary model file on the disk.

0.1.1
-----
- Fixed a few bugs in the code, improved documentation and added a support for
  specifying the "latest" version of the model in the ``upet.get_upet`` and
  ``upet.save_upet`` functions.

0.1.0
-----
- Initial release of the UPET package, succeeding PET-MAD. Updated package
  structure, calculator names, and model naming conventions to reflect the new
  UPET branding and functionalities.
- Added a support for new models trained on popular datasets for atomistic machine
  learning.
