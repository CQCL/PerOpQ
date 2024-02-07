"""Sphinx configuration."""

project = "PerOpQ"
copyright = "2023, Quantinuum"
author = "Quantinuum"

extensions = [
    "sphinx.ext.autodoc",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "myst_parser",
]
autoapi_dirs = ["../src"]
autodoc_typehints = "description"
html_theme = "furo"
