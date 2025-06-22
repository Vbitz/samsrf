"""
Simulation functions for generating synthetic pRF data.

This module provides functions to create synthetic pRF experiments
for testing and validation purposes.
"""

import numpy as np
from typing import Tuple, Optional

from ..core.data_structures import SrfData
from ..core.prf_functions import gaussian_rf, predict_timecourse
from ..core.hrf_functions import canonical_hrf, convolve_hrf


def simulate_prf_experiment(x0_true: np.ndarray, y0_true: np.ndarray, 
                           sigma_true: np.ndarray,
                           n_timepoints: int = 200,
                           noise_level: float = 0.1,
                           tr: float = 2.0) -> SrfData:
    """
    Simulate a complete pRF experiment with synthetic data.
    
    Parameters
    ----------
    x0_true, y0_true, sigma_true : np.ndarray
        Ground truth pRF parameters
    n_timepoints : int, default=200
        Number of timepoints to simulate
    noise_level : float, default=0.1
        Noise level as fraction of signal
    tr : float, default=2.0
        Repetition time in seconds
        
    Returns
    -------
    SrfData
        Synthetic surface data with ground truth and time series
    """
    n_vertices = len(x0_true)
    aperture_width = 100
    
    # Create synthetic apertures
    apertures = create_synthetic_apertures(aperture_width, n_timepoints)
    
    # Generate HRF
    hrf = canonical_hrf(tr, 'spm')
    
    # Simulate BOLD responses
    bold_responses = np.zeros((n_vertices, n_timepoints))
    
    for v in range(n_vertices):
        # Generate pRF
        rf = gaussian_rf(x0_true[v], y0_true[v], sigma_true[v], aperture_width)
        
        # Predict neural response
        neural = predict_timecourse(rf, apertures)
        
        # Convolve with HRF
        bold = convolve_hrf(neural, hrf)[:n_timepoints]
        
        # Add noise
        signal_std = np.std(bold)
        noise = np.random.normal(0, signal_std * noise_level, n_timepoints)
        bold_responses[v, :] = bold + noise
    
    # Create SrfData structure
    srf = SrfData("synthetic")
    srf.y = bold_responses
    
    # Add ground truth parameters
    ground_truth = np.vstack([x0_true, y0_true, sigma_true])
    srf.add_data(ground_truth, ['x0_true', 'y0_true', 'sigma_true'])
    
    # Add dummy vertex coordinates
    srf.vertices = np.random.rand(n_vertices, 3) * 100
    
    return srf


def create_synthetic_apertures(aperture_width: int, n_timepoints: int,
                              stimulus_type: str = 'expanding_rings') -> np.ndarray:
    """
    Create synthetic stimulus apertures.
    
    Parameters
    ----------
    aperture_width : int
        Width of aperture grid in pixels
    n_timepoints : int
        Number of timepoints
    stimulus_type : str, default='expanding_rings'
        Type of stimulus ('expanding_rings', 'bars', 'random')
        
    Returns
    -------
    np.ndarray
        Apertures array (aperture_pixels x timepoints)
    """
    apertures = np.zeros((aperture_width**2, n_timepoints))
    
    if stimulus_type == 'expanding_rings':
        # Expanding ring stimulus
        for t in range(n_timepoints):
            radius = 0.5 + (t / n_timepoints) * 8
            y, x = np.mgrid[-50:50, -50:50]
            x = x * 10 / 50  # Scale to degrees
            y = y * 10 / 50
            ring = (np.sqrt(x**2 + y**2) < radius).flatten()
            apertures[:, t] = ring
            
    elif stimulus_type == 'bars':
        # Sweeping bar stimulus
        for t in range(n_timepoints):
            bar_pos = -50 + (100 * t / n_timepoints)
            y, x = np.mgrid[-50:50, -50:50]
            bar = (np.abs(x - bar_pos) < 5).flatten()
            apertures[:, t] = bar
            
    elif stimulus_type == 'random':
        # Random dot stimulus
        apertures = np.random.binomial(1, 0.3, (aperture_width**2, n_timepoints))
    
    return apertures