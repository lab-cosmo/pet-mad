# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys


sys.path.insert(0, os.path.abspath("../../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PET-MAD"
copyright = "2025, Lab-COSMO"
author = "Arslan Mazitov, Filippo Bigi, Matthias Kellner, Paolo Pegolo, Davide Tisi, "
"Guillaume Fraux, Sergey Pozdnyakov, Philip Loche, Michele Ceriotti"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
try:
    from pet_mad import __version__

    version = __version__
    release = __version__
except ImportError:
    version = "1.4.1"
    release = "1.4.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["static"]

# Logo configuration for light and dark modes
html_theme_options = {
    "sidebar_hide_name": True,
    "light_logo": "pet-mad-logo-with-text.svg",
    "dark_logo": "pet-mad-logo-with-text-dark.svg",
}

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

# Autosummary settings
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}

# MyST parser settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# nbsphinx settings
nbsphinx_execute = "never"  # Don't execute notebooks during build
