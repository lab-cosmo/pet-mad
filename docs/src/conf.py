import os
import tomllib
from datetime import datetime

import upet


ROOT = os.path.abspath(os.path.join("..", ".."))


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
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_toggleprompt",
    "sphinx_gallery.gen_gallery",
]

exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    "generated_examples/sg_execution_times.rst",
    "sg_execution_times.rst",
]

python_use_unqualified_type_names = True

autoclass_content = "both"
autodoc_member_order = "bysource"
autodoc_typehints = "both"
autodoc_typehints_format = "short"
# `upet.nvalchemi` requires the optional `nvalchemi` extra, which isn't
# installed in the docs build environment; mock it so autodoc can still
# extract docstrings/signatures without importing it.
autodoc_mock_imports = ["nvalchemi"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "ase": ("https://docs.ase-lib.org/", None),
    "metatensor": ("https://docs.metatensor.org/latest/", None),
    "metatomic": ("https://docs.metatensor.org/metatomic/latest/", None),
    "metatrain": ("https://docs.metatensor.org/metatrain/latest/", None),
}

sphinx_gallery_conf = {
    "examples_dirs": os.path.join(ROOT, "examples"),
    "gallery_dirs": "generated_examples",
    "filename_pattern": r"/*\.py",
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
