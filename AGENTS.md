# AGENTS.md

This file provides guidance to coding agents working with code in this repository.

## Common commands

All workflows are driven through `tox` (see `tox.ini`). Install it once with `pip install tox`, then:

- `tox -e lint` — ruff format check, ruff lint, mypy, and sphinx-lint on `src/` and `tests/` plus `README.md`.
- `tox -e format` — auto-apply `ruff format` and `ruff check --fix-only` to the same paths.
- `tox -e core-tests` — backend-agnostic tests (`tests/core/`): model registry, metadata, deprecated shims.
- `tox -e ase-tests` — ASE calculator tests (`tests/ase/`, excluding `dos/` and `explore/`).
- `tox -e ase-dos-tests` — PET-MAD-DOS tests (`tests/ase/dos/`).
- `tox -e ase-explore-tests` — PET-MAD explorer tests (`tests/ase/explore/`).
- `tox -e nvalchemi-tests` — NVAlchemi wrapper tests (`tests/nvalchemi/`); installs the optional `nvalchemi` extra, without which the whole suite `importorskip`s itself.
  On macOS the `test_wrapper_compile.py` tests cannot run: they make inductor build a C++ kernel that loads a second `libomp.dylib`, and OpenMP aborts the whole interpreter (`OMP: Error #15`) rather than failing a single test. Setting `KMP_DUPLICATE_LIB_OK=TRUE` only trades the abort for a deadlock. Deselect them locally (`tox -e nvalchemi-tests -- -k 'not CompiledBackend'`) and rely on the Linux CI job for that coverage.
- `tox -e {core,ase,ase-dos,ase-explore}-tests-dev` — the same suites, but installing `metatrain` from git `main` first; used by the weekly CI job.
- `tox -e build` — build the sdist + wheel and run `twine check` and `check-manifest`.
- `tox -e docs` — build the Sphinx HTML docs into `docs/build/html` (runs with `--fail-on-warning`).
- Single test: `tox -e ase-tests -- -k test_name` (everything after `--` is forwarded as `{posargs}` to pytest; each suite has its own `changedir`, so relative paths resolve inside that suite's directory).

`pytest` is configured with `filterwarnings = ["error", ...]` in `pyproject.toml`, so any new warning that isn't explicitly ignored will fail tests — add entries to that allowlist rather than silencing warnings inline.

CI (`.github/workflows/tests.yml`) runs one job per suite, named `<suite> (<os>, py<version>)`, on push/PR. `ase-tests` covers the full matrix (Linux 3.11/3.13, macOS 3.13, Windows 3.13); the other suites run once on Linux + 3.13. All of them set `PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu` — use the same env var locally if you hit CUDA-wheel install issues — and `HF_TOKEN`, which is needed to exercise tests that pull gated checkpoints. `tests-dev.yml` runs the `-dev` variants weekly.

## Architecture

`upet` is a thin user-facing wrapper around `metatrain` / `metatomic` that ships pre-trained PET-family interatomic potentials hosted on HuggingFace (`lab-cosmo/upet`).

Package layout (`src/upet/`):

- `__init__.py` — exports `get_upet`, `list_upet`, `save_upet`; also applies global side effects that matter for every entry point: warning filters for `nvalchemi`/`warp`, and `torch.jit.set_fusion_strategy([("DYNAMIC", 10)])` to disable static CUDA-kernel fusion (statically-fused kernels cannot allocate tensors at runtime, which breaks variable-size atomistic batches on CUDA 13+). Do not remove those without understanding the impact.
- `_version.py` — single source of truth for the model registry: `UPET_AVAILABLE_MODELS`, `UPET_NO_NC_SUPPORT_MODELS` (models without non-conservative forces), `UPET_UQ_SUPPORTED_MODELS` (uncertainty quantification), `DEPRECATED_MODELS`, and PET-MAD-DOS versions. Adding a new checkpoint to HF usually means editing this file.
- `_models.py` — core, ecosystem-agnostic model registry shared by both connectors below: resolves `(model, size, version)` → local checkpoint path via HuggingFace. `CHECKPOINT_NAME_PATTERN` defines the canonical name format `pet-{family}-{size}-v{X.Y.Z}.ckpt`; everything else (`get_available_models`, `get_sizes_for_model`, `get_versions_for_model`, `upet_resolve_model`, `parse_checkpoint_filename`, `_resolve_and_download_checkpoint`) parses names against it. `_get_upet_repo_files()` is `lru_cache`d — tests that rely on a fresh HF listing must clear it.
- `_metadata.py` — per-model metadata (cutoffs, supported elements, etc.) attached to loaded models.
- `utils.py` — core shared helpers (HF URL construction).
- `calculator.py` / `explore/` — deprecated shims kept for backwards compatibility; both just re-export from `ase/` (below) and emit a `DeprecationWarning` on import. New code should not add to these.
- `ase/` — primary building block #1: the ASE ecosystem connector.
  - `calculator.py` — the primary user API. `UPETCalculator` is an ASE `Calculator` wrapping `metatomic_ase.MetatomicCalculator`/`SymmetrizedCalculator`; it accepts either `model`+`version`+`device` or a local `checkpoint_path`, and exposes extras like `non_conservative`, `rotational_average_order`, `get_energy_uncertainty`, `get_energy_ensemble`.
  - `dos/` — byproduct: PET-MAD-DOS. `calculator.py` has `PETMADDOSCalculator` (`calculate_dos` / `calculate_bandgap` / `calculate_efermi`); `_models.py` has the DOS-specific model loaders plus `CNNModel` (the bandgap/Fermi-level CNN); `utils.py` has DOS math helpers (electron counting, Fermi–Dirac, eigenvalue broadening).
  - `explore/` — byproduct: dataset-level tools. `PETMADFeaturizer` (last-layer features + sketch-map, intended for use with `chemiscope`) and `MADExplorer`.
- `nvalchemi/` — primary building block #2: the nvalchemi-toolkit ecosystem connector (optional `nvalchemi` extra). `wrapper.py` has `UPETWrapper`, a pure-torch `BaseModelMixin` wrapping `metatrain.pet.modules.backend.PETBackend`; `utils.py` has its checkpoint-decoding helpers.

Tests (`tests/`):

`tests/` mirrors `src/upet/`, one directory (and one tox environment) per source sub-package:

- `tests/core/` — ecosystem-agnostic code: model registry (`test_models.py`), checkpoint-name parsing (`test_checkpoint_names.py`), metadata (`test_metadata.py`), and the deprecated top-level shims (`test_deprecated_imports.py`).
- `tests/ase/` — `UPETCalculator`: energies/forces/stress, MD, non-conservative forces, uncertainty, rotational averaging — parametrized over `UPET_AVAILABLE_MODELS`. `_utils.py` holds the shared non-conservative-support expectations, imported as `from _utils import ...` (which works because pytest puts each test file's own directory on `sys.path`).
- `tests/ase/dos/` and `tests/ase/explore/` — the DOS and explorer byproducts.
- `tests/nvalchemi/` — `UPETWrapper`, split by concern: `test_wrapper_construction.py` (`__init__`, buffers, `ModelConfig`, properties), `test_wrapper_adapters.py` (`adapt_input` / `adapt_output`), `test_wrapper_forward.py` (`forward`, `compute_embeddings`), `test_wrapper_compile.py` (`torch.compile` parity), `test_wrapper_export.py` (`export_model`) and `test_wrapper_checkpoint.py` (integration against a real HuggingFace checkpoint). Shared constants, hypers and `AtomicData` builders live in `_helpers.py`; the fixtures in `conftest.py`. Every `test_wrapper_*.py` opens with `pytest.importorskip("nvalchemi")`, so the whole directory skips without the extra — `conftest.py` therefore imports `nvalchemi` only inside fixture bodies, never at module scope.
- Test file basenames are unique across the whole tree (hence `test_dos_metadata.py`, not a second `test_metadata.py`): there are no `__init__.py` files, so pytest imports test modules into a flat namespace and duplicate basenames collide when running `pytest tests/` from the repository root.
- Each suite has its own `changedir` in `tox.ini`; tests pull real checkpoints from HuggingFace, so they are network-bound by design.

External dependencies to keep in mind: `metatrain` (pinned to `>=2026.4,<2026.5`), `metatomic-ase`, `nvalchemi-toolkit-ops` (pinned to `>=0.4.0,<0.5.0`), and `huggingface_hub`. Version bumps to these usually require matching updates to the warning allowlist in `pyproject.toml` and sometimes to `_version.py`.

## Documentation

User-facing documentation lives in `docs/` and is built with Sphinx + the `furo` theme. The canonical hosted version is <https://lab-cosmo.github.io/upet/latest/>, deployed from `.github/workflows/docs.yml` to the `gh-pages` branch (one directory per tag, plus `latest/` for `main`). A Read the Docs build is also configured via `.readthedocs.yml` as a secondary target.

- `docs/src/` — reST sources. `index.rst` is the landing page; the top-level pages are `quickstart`, `installation`, `usage/index` (one file per engine: `ase`, `metatrain`, `lammps`, `ipi`, `torchsim`, `gromacs`), `models`, `fine-tuning`, `miscellaneous`, `faq`, `cite`.
- `docs/src/conf.py` — Sphinx config. Uses `sphinx.ext.autodoc` + `sphinx.ext.intersphinx` (mapping to python/numpy/torch/ase/metatensor/metatomic/metatrain) and `sphinx_gallery` to turn `examples/` into `docs/src/generated_examples/`.
- `docs/src/generated_examples/` and `docs/src/sg_execution_times.rst` are sphinx-gallery outputs — do not hand-edit; regenerate by running `tox -e docs`.
- `docs/requirements.txt` — pinned doc-build deps (used by both RTD and the `docs` tox env).
- `docs/README_OLD.md`, `docs/UPET_MIGRATION_GUIDE.md`, `docs/SPEED.md`, `docs/README_BATCHED.md`, `docs/CHANGELOG.md` — legacy standalone markdown files, still linked from the Sphinx tree and the README.

Local builds: `tox -e docs` (uses `--fail-on-warning`, so any autodoc/intersphinx/sphinx-lint warning breaks the build — either fix the source or add a targeted ignore). The `tox -e lint` env also runs `sphinx-lint` on `docs/src/` and `README.md`.

`README.md` is intentionally a short quickstart that defers to the RTD site for details. When adding new user-facing features, prefer extending the Sphinx docs and linking from the README rather than inlining content there.
