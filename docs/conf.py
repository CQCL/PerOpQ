"""Sphinx configuration."""
project = "PerOpQ"
copyright = "2023, Quantinuum"
author = "Quantinuum"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
