"""Sphinx configuration."""

import sys
from datetime import datetime
from pathlib import Path


sys.path.insert(0, str(Path("..").resolve()))
sys.path.insert(0, str(Path("../..").resolve()))
sys.path.insert(0, str(Path("../src").resolve()))

project = "Fluvial Particle"
author = "Richard McDonald"
copyright = f"{datetime.now().year}, {author}"

# Get version from package
try:
    from fluvial_particle import __version__

    version = __version__
    release = __version__
except ImportError:
    version = "0.0.4"
    release = "0.0.4"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "sphinx_rtd_theme",
    "myst_parser",
]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_admonition",
    "html_image",
    "linkify",
    "replacements",
    "smartquotes",
    "strikethrough",
    "substitution",
    "tasklist",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
autodoc_typehints = "description"
html_theme = "sphinx_rtd_theme"
