"""Sphinx configuration for Gordion."""

project = "Gordion"
author = "Gordion contributors"
extensions = ["sphinx.ext.autodoc"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
root_doc = "index"
language = "en"
html_theme = "sphinx_rtd_theme"