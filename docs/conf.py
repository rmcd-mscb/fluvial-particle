"""Sphinx configuration."""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../../"))

project = "Fluvial Particle"
author = "Richard McDonald"
copyright = f"{datetime.now().year}, {author}"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "sphinx_rtd_theme",
    "myst_parser",
]

myst_enable_extensions = ["colon_fence"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
autodoc_typehints = "description"
html_theme = "sphinx_rtd_theme"
