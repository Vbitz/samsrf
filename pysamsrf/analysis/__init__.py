"""Analysis and visualization functions for PySamSrf."""

from .plotting import plot_prf, plot_eccentricity, plot_polar, plot_surface
from .statistics import calculate_statistics, compare_models, bootstrap_confidence
from .visual_field import visual_field_coverage, calculate_cortical_magnification

__all__ = [
    'plot_prf', 'plot_eccentricity', 'plot_polar', 'plot_surface',
    'calculate_statistics', 'compare_models', 'bootstrap_confidence',
    'visual_field_coverage', 'calculate_cortical_magnification'
]