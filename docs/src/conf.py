import os
import shutil
import tomllib
from datetime import datetime

import upet


ROOT = os.path.abspath(os.path.join("..", ".."))

# ----------------------------------------------------------------------------
# Cookbook recipes
#
# We embed PET-related examples from the atomistic-cookbook (vendored as a git
# submodule under external/atomistic-cookbook) without duplicating their source.
# A wrapper directory is assembled at build time with symlinks into the
# submodule so sphinx-gallery sees a single gallery with one subsection per
# recipe. The wrapper lives at docs/_cookbook_src/ (outside Sphinx's source
# dir) and is gitignored.
#
# Recipes are rendered (code + docstring narrative) but NOT executed: the
# global sphinx_gallery_conf.filename_pattern below only matches files whose
# basename starts with `plot_`, which catches upet's local examples.
# Cookbook files (e.g. `pet-mad.py`) do not match and are rendered source-only.
# ----------------------------------------------------------------------------
COOKBOOK_ROOT = os.path.join(ROOT, "external", "atomistic-cookbook", "examples")
COOKBOOK_RECIPES = [
    "pet-mad",
    "pet-mad-dos",
    "pet-mad-nc",
    "pet-mad-uq",
    "pet-finetuning",
    "pet-phonons",
    "pet-relaxation",
    "eon-pet-neb",
    "dos-align",
    "flashmd",
    "heat-capacity",
]
# Wrapper lives OUTSIDE docs/src/ so Sphinx never scans it as source (which
# would trip "document isn't included in any toctree" warnings under
# --fail-on-warning). Sphinx-gallery still reads it via the absolute path
# in `examples_dirs` below.
COOKBOOK_WRAPPER = os.path.join(ROOT, "docs", "_cookbook_src")

if not os.path.isdir(COOKBOOK_ROOT):
    raise RuntimeError(
        f"atomistic-cookbook submodule not found at {COOKBOOK_ROOT!r}. "
        "Run `git submodule update --init --recursive` from the repo root "
        "before building the docs."
    )

# wipe & recreate so removing a recipe from COOKBOOK_RECIPES doesn't leave
# a stale symlink that sphinx-gallery would still pick up
if os.path.isdir(COOKBOOK_WRAPPER):
    shutil.rmtree(COOKBOOK_WRAPPER)
os.makedirs(COOKBOOK_WRAPPER)
with open(os.path.join(COOKBOOK_WRAPPER, "README.rst"), "w") as _fp:
    _fp.write(
        "Cookbook recipes\n"
        "================\n\n"
        "End-to-end PET / UPET tutorials from the `atomistic cookbook "
        "<https://atomistic-cookbook.org/>`_. Source is rendered inline "
        "from the pinned submodule at ``external/atomistic-cookbook``; "
        "outputs (figures, console) are not regenerated here — see the "
        "cookbook site for the executed versions.\n"
    )
for _name in COOKBOOK_RECIPES:
    _src = os.path.join(COOKBOOK_ROOT, _name)
    if not os.path.isdir(_src):
        raise RuntimeError(
            f"cookbook recipe {_name!r} listed in COOKBOOK_RECIPES but not "
            f"found at {_src!r} — check the recipe name or bump the submodule."
        )
    os.symlink(_src, os.path.join(COOKBOOK_WRAPPER, _name))


# -- Project information -----------------------------------------------------

master_doc = "index"

with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fp:
    project_dict = tomllib.load(fp)["project"]

project = project_dict["name"]
author = ", ".join(a["name"] for a in project_dict["authors"])
copyright = f"{datetime.now().date().year}, {author}"
release = upet.__version__


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_toggleprompt",
    "sphinx_gallery.gen_gallery",
]

exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    "generated_examples/sg_execution_times.rst",
    "cookbook/sg_execution_times.rst",
    "sg_execution_times.rst",
]

python_use_unqualified_type_names = True

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_format = "short"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "ase": ("https://ase-lib.org/", None),
    "metatensor": ("https://docs.metatensor.org/latest/", None),
    "metatomic": ("https://docs.metatensor.org/metatomic/latest/", None),
    "metatrain": ("https://docs.metatensor.org/metatrain/latest/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": [os.path.join(ROOT, "examples"), COOKBOOK_WRAPPER],
    "gallery_dirs": ["generated_examples", "cookbook"],
    # Only files whose basename starts with `plot_` execute. Upet's local
    # examples follow this convention; cookbook recipes (e.g. `pet-mad.py`)
    # don't, so they render source-only. Sphinx-gallery uses re.search against
    # the absolute path, so the leading `/` anchors `plot_` to a directory
    # boundary rather than to start-of-string.
    "filename_pattern": r"/plot_[^/]+\.py$",
    # Cookbook recipes reference adjacent yaml configs and figures via
    # `.. literalinclude::` / `.. image::`; mirror cookbook's copy regex so
    # those side files land next to the generated rst.
    "copyfile_regex": r".*\.(cp2k|jpg|jpeg|mdp|png|sh|xyz|yaml|yml|zip)",
    "within_subsection_order": "FileNameSortKey",
    "default_thumb_file": os.path.join(
        ROOT, "docs", "static", "images", "upet-logo-with-text.svg"
    ),
    "min_reported_time": 5,
    "remove_config_comments": True,
}


# -- Options for HTML output -------------------------------------------------

html_title = "UPET"
html_theme = "furo"

html_static_path = [os.path.join(ROOT, "docs", "static")]

html_theme_options = {
    "light_logo": "images/upet-logo-with-text.svg",
    "dark_logo": "images/upet-logo-with-text-dark.svg",
    "sidebar_hide_name": True,
    "footer_icons": [
        {
            "name": "GitHub",
            "url": project_dict["urls"]["repository"],
            "html": "",
            "class": "fa-brands fa-github fa-2x",
        },
    ],
}

html_css_files = [
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/fontawesome.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/solid.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/brands.min.css",
    "css/custom.css",
]
