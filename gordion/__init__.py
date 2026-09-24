"""Gordion package.

The project layout bundles the implementation in a nested package directory, so
this module re-exports the main scientific workflow for package imports.
"""

from .main import *
from .paths import get_data_path, get_repository_path, get_visuals_path
from .variability import compute_variability_products, plot_variability_summary
