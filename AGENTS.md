# AGENTS.md

This file provides guidance to coding agents working with code in this repository.

## Common commands

All workflows are driven through `tox` (see `tox.ini`). Install it once with `pip install tox`, then:

- `tox -e lint` — ruff format check, ruff lint, mypy, and sphinx-lint on `src/` and `tests/` plus `README.md`.
- `tox -e format` — auto-apply `ruff format` and `ruff check --fix-only` to the same paths.
- `tox -e upet-tests` — run the main package tests (`tests/upet/`) against the pinned `metatrain` release.
- `tox -e upet-tests-dev` — same suite but installs `metatrain` from git `main` first; used by the weekly CI job.
- `tox -e pet-mad-dos-tests` — run the PET-MAD-DOS tests (`tests/pet_mad_dos/`).
- `tox -e build` — build the sdist + wheel and run `twine check` and `check-manifest`.
- `tox -e docs` — build the Sphinx HTML docs into `docs/build/html` (runs with `--fail-on-warning`).
- Single test: `tox -e upet-tests -- -k test_name` (everything after `--` is forwarded as `{posargs}` to pytest; `changedir` is already `tests/upet`, so relative paths resolve there).

`pytest` is configured with `filterwarnings = ["error", ...]` in `pyproject.toml`, so any new warning that isn't explicitly ignored will fail tests — add entries to that allowlist rather than silencing warnings inline.

CI (`.github/workflows/`) runs `tox -e upet-tests` and `tox -e pet-mad-dos-tests` on push/PR across Linux/macOS/Windows + Python 3.11/3.13 with `PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu`. Use the same env var locally if you hit CUDA-wheel install issues. `HF_TOKEN` is needed to exercise tests that pull gated checkpoints.

## Architecture

`upet` is a thin user-facing wrapper around `metatrain` / `metatomic` that ships pre-trained PET-family interatomic potentials hosted on HuggingFace (`lab-cosmo/upet`).

Package layout (`src/upet/`):

- `__init__.py` — exports `get_upet`, `list_upet`, `save_upet`; also applies global side effects that matter for every entry point: warning filters for `nvalchemi`/`warp`, and `torch.jit.set_fusion_strategy([("DYNAMIC", 10)])` to disable static CUDA-kernel fusion (statically-fused kernels cannot allocate tensors at runtime, which breaks variable-size atomistic batches on CUDA 13+). Do not remove those without understanding the impact.
- `_version.py` — single source of truth for the model registry: `UPET_AVAILABLE_MODELS`, `UPET_NO_NC_SUPPORT_MODELS` (models without non-conservative forces), `UPET_UQ_SUPPORTED_MODELS` (uncertainty quantification), `DEPRECATED_MODELS`, and PET-MAD-DOS versions. Adding a new checkpoint to HF usually means editing this file.
- `_models.py` — resolves `(model, size, version)` → local checkpoint path via HuggingFace. `CHECKPOINT_NAME_PATTERN` defines the canonical name format `pet-{family}-{size}-v{X.Y.Z}.ckpt`; everything else (`get_available_models`, `get_sizes_for_model`, `get_versions_for_model`, `upet_resolve_model`, `parse_checkpoint_filename`) parses names against it. `_get_upet_repo_files()` is `lru_cache`d — tests that rely on a fresh HF listing must clear it.
- `_metadata.py` — per-model metadata (cutoffs, supported elements, etc.) attached to loaded models.
- `calculator.py` — the primary user API. `UPETCalculator` is an ASE `Calculator` wrapping `metatomic_ase.MetatomicCalculator`/`SymmetrizedCalculator`; it accepts either `model`+`version`+`device` or a local `checkpoint_path`, and exposes extras like `non_conservative`, `rotational_average_order`, `get_energy_uncertainty`, `get_energy_ensemble`. `PETMADDOSCalculator` uses `BandgapModel` (from `modules.py`) plus a DOS head for `calculate_dos` / `calculate_bandgap` / `calculate_efermi`.
- `modules.py` — small torch modules used by the DOS pipeline (notably `BandgapModel`, a 1D CNN over DOS).
- `utils.py` — shared helpers (HF URL construction, electron counting, Fermi–Dirac, etc.).
- `explore/` — dataset-level tools: `PETMADFeaturizer` (last-layer features + sketch-map) intended for use with `chemiscope`.

Tests (`tests/`):

- `tests/upet/` covers the calculator, MD, non-conservative forces, uncertainty, rotational averaging, featurizer, offline checkpoints, and metadata — parametrized over `UPET_AVAILABLE_MODELS`.
- `tests/pet_mad_dos/` covers DOS-specific paths.
- Each suite has its own `changedir` in `tox.ini`; tests pull real checkpoints from HuggingFace, so they are network-bound by design.

External dependencies to keep in mind: `metatrain` (pinned to `>=2026.2,<2026.3`), `metatomic-ase`, `nvalchemi-toolkit-ops` (pinned exactly to `0.2.0`), and `huggingface_hub`. Version bumps to these usually require matching updates to the warning allowlist in `pyproject.toml` and sometimes to `_version.py`.

## Documentation

User-facing documentation lives in `docs/` and is built with Sphinx + the `furo` theme. The canonical hosted version is <https://lab-cosmo.github.io/upet/latest/>, deployed from `.github/workflows/docs.yml` to the `gh-pages` branch (one directory per tag, plus `latest/` for `main`). A Read the Docs build is also configured via `.readthedocs.yml` as a secondary target.

- `docs/src/` — reST sources. `index.rst` is the landing page; the top-level pages are `quickstart`, `installation`, `usage/index` (one file per engine: `ase`, `metatrain`, `lammps`, `ipi`, `torchsim`, `gromacs`), `models`, `fine-tuning`, `miscellaneous`, `faq`, `cite`.
- `docs/src/conf.py` — Sphinx config. Uses `sphinx.ext.autodoc` + `sphinx.ext.intersphinx` (mapping to python/numpy/torch/ase/metatensor/metatomic/metatrain) and `sphinx_gallery` to turn `examples/` into `docs/src/generated_examples/`.
- `docs/src/generated_examples/` and `docs/src/sg_execution_times.rst` are sphinx-gallery outputs — do not hand-edit; regenerate by running `tox -e docs`.
- `docs/src/cookbook/` (sphinx-gallery output) and `docs/_cookbook_src/` (symlink wrapper, kept outside `docs/src/` so Sphinx doesn't scan its READMEs and trip "doc not in toctree" warnings) are build artifacts (gitignored). The cookbook gallery embeds PET recipes from the `external/atomistic-cookbook` git submodule (pinned SHA) without copying source. `conf.py` rebuilds the wrapper on each build and points `sphinx_gallery_conf.examples_dirs` at it. The `filename_pattern` regex (`/plot_[^/]+\.py$`) only executes upet's local `plot_*.py` examples; cookbook recipes (`pet-mad.py`, etc.) render source-only — no execution, no figures. Missing submodule or unknown recipe in `COOKBOOK_RECIPES` raises a clear `RuntimeError` from `conf.py`. To bump the cookbook pin: `cd external/atomistic-cookbook && git fetch && git checkout <sha>` then commit. To add/remove recipes: edit `COOKBOOK_RECIPES` in `conf.py`. CI / RTD pull the submodule via `submodules: recursive` / `submodules.include: all`.
- `docs/requirements.txt` — pinned doc-build deps (used by both RTD and the `docs` tox env).
- `docs/README_OLD.md`, `docs/UPET_MIGRATION_GUIDE.md`, `docs/SPEED.md`, `docs/README_BATCHED.md`, `docs/CHANGELOG.md` — legacy standalone markdown files, still linked from the Sphinx tree and the README.

Local builds: `tox -e docs` (uses `--fail-on-warning`, so any autodoc/intersphinx/sphinx-lint warning breaks the build — either fix the source or add a targeted ignore). The `tox -e lint` env also runs `sphinx-lint` on `docs/src/` and `README.md`.

`README.md` is intentionally a short quickstart that defers to the RTD site for details. When adding new user-facing features, prefer extending the Sphinx docs and linking from the README rather than inlining content there.
