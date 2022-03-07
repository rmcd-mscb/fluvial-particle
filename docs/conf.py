"""Sphinx configuration."""
from datetime import datetime


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

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
autodoc_typehints = "description"
html_theme = "sphinx_rtd_theme"
