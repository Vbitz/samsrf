"""
PySamSrf - Python Population Receptive Field Analysis

A Python implementation of the SamSrf MATLAB toolbox for population receptive field (pRF) analysis.
Supports forward-modeling, reverse correlation, and connective field analysis approaches.

Author: Translated from MATLAB SamSrf by Sam Schwarzkopf
"""

__version__ = "1.0.0"
__author__ = "PySamSrf Developers"

from .core.data_structures import SrfData, Model, PRFResults
from .core.prf_functions import gaussian_rf, predict_timecourse
from .core.hrf_functions import doublegamma, canonical_hrf
from .fitting.fit_prf import fit_prf

# Import I/O functions
try:
    from .io.surface_io import load_surface, save_surface
    from .io.volume_io import load_volume, save_volume
    _io_available = True
except ImportError:
    _io_available = False

# Import analysis functions
try:
    from .analysis.plotting import plot_prf, plot_eccentricity
    _analysis_available = True
except ImportError:
    _analysis_available = False

__all__ = [
    'SrfData', 'Model', 'PRFResults',
    'gaussian_rf', 'predict_timecourse', 
    'doublegamma', 'canonical_hrf',
    'fit_prf'
]

# Add I/O functions if available
if _io_available:
    __all__.extend(['load_surface', 'save_surface', 'load_volume', 'save_volume'])

# Add analysis functions if available  
if _analysis_available:
    __all__.extend(['plot_prf', 'plot_eccentricity'])