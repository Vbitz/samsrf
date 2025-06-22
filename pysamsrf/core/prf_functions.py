"""
Core population receptive field (pRF) functions.

This module contains the fundamental functions for pRF modeling,
including receptive field generation and time course prediction.
"""

import numpy as np
from typing import Tuple, Optional, Union
import warnings


def gaussian_rf(x0: float, y0: float, sigma: float, 
                aperture_width: float = 200) -> np.ndarray:
    """
    Generate 2D Gaussian population receptive field.
    
    Equivalent to MATLAB prf_gaussian_rf function.
    
    Parameters
    ----------
    x0 : float
        Horizontal center position in degrees of visual angle
    y0 : float
        Vertical center position in degrees of visual angle  
    sigma : float
        Standard deviation (size) in degrees of visual angle
    aperture_width : float, default=200
        Width of aperture grid in pixels
        
    Returns
    -------
    np.ndarray
        2D Gaussian receptive field (aperture_width x aperture_width)
    """
    # Create coordinate grids
    extent = aperture_width / 2
    x = np.linspace(-extent, extent, int(aperture_width))
    y = np.linspace(-extent, extent, int(aperture_width))
    X, Y = np.meshgrid(x, y)
    
    # Generate 2D Gaussian
    rf = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    
    return rf


def predict_timecourse(rf: np.ndarray, apertures: np.ndarray,
                      percent_overlap: bool = True) -> np.ndarray:
    """
    Predict neural time course from pRF and stimulus apertures.
    
    Equivalent to MATLAB prf_predict_timecourse function.
    
    Parameters
    ----------
    rf : np.ndarray
        Population receptive field (2D array)
    apertures : np.ndarray
        Stimulus apertures (aperture_size x n_timepoints)
    percent_overlap : bool, default=True
        If True, use SamSrf percent overlap model
        If False, use Dumoulin & Wandell mean response model
        
    Returns
    -------
    np.ndarray
        Predicted neural response time course (n_timepoints,)
    """
    if rf.size != apertures.shape[0]:
        raise ValueError("pRF and aperture dimensions must match")
    
    # Flatten RF for computation
    rf_flat = rf.flatten()
    
    # Calculate response for each time point
    timecourse = np.zeros(apertures.shape[1])
    
    for t in range(apertures.shape[1]):
        aperture_t = apertures[:, t]
        
        if percent_overlap:
            # SamSrf model: percent of pRF overlapping with stimulus
            overlap = np.sum(rf_flat * aperture_t)
            total_rf = np.sum(np.abs(rf_flat))
            if total_rf > 0:
                timecourse[t] = (overlap / total_rf) * 100
            else:
                timecourse[t] = 0
        else:
            # Dumoulin & Wandell model: mean response in aperture
            timecourse[t] = np.mean(rf_flat * aperture_t)
    
    return timecourse


def dog_rf(x0: float, y0: float, sigma_center: float, sigma_surround: float,
           amplitude_ratio: float, aperture_width: float = 200) -> np.ndarray:
    """
    Generate difference-of-Gaussians (DoG) receptive field.
    
    Parameters
    ----------
    x0, y0 : float
        Center position in degrees of visual angle
    sigma_center : float
        Center Gaussian standard deviation
    sigma_surround : float
        Surround Gaussian standard deviation
    amplitude_ratio : float
        Ratio of surround to center amplitude
    aperture_width : float, default=200
        Width of aperture grid in pixels
        
    Returns
    -------
    np.ndarray
        DoG receptive field
    """
    # Generate center and surround Gaussians
    center = gaussian_rf(x0, y0, sigma_center, aperture_width)
    surround = gaussian_rf(x0, y0, sigma_surround, aperture_width)
    
    # Combine with amplitude ratio
    dog = center - amplitude_ratio * surround
    
    return dog


def multivariate_gaussian_rf(x0: float, y0: float, sigma_x: float, sigma_y: float,
                           theta: float, aperture_width: float = 200) -> np.ndarray:
    """
    Generate multivariate (elliptical) Gaussian receptive field.
    
    Parameters
    ----------
    x0, y0 : float
        Center position in degrees of visual angle
    sigma_x, sigma_y : float
        Standard deviations along major axes
    theta : float
        Rotation angle in degrees
    aperture_width : float, default=200
        Width of aperture grid in pixels
        
    Returns
    -------
    np.ndarray
        Elliptical Gaussian receptive field
    """
    # Create coordinate grids
    extent = aperture_width / 2
    x = np.linspace(-extent, extent, int(aperture_width))
    y = np.linspace(-extent, extent, int(aperture_width))
    X, Y = np.meshgrid(x, y)
    
    # Translate to center
    X_centered = X - x0
    Y_centered = Y - y0
    
    # Rotation matrix
    theta_rad = np.radians(theta)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)
    
    # Rotate coordinates
    X_rot = cos_theta * X_centered + sin_theta * Y_centered
    Y_rot = -sin_theta * X_centered + cos_theta * Y_centered
    
    # Generate elliptical Gaussian
    rf = np.exp(-(X_rot**2 / (2 * sigma_x**2) + Y_rot**2 / (2 * sigma_y**2)))
    
    return rf


def generate_prf_grid(model_params: dict, aperture_width: float = 200) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate grid of pRFs for coarse fitting search space.
    
    Parameters
    ----------
    model_params : dict
        Dictionary containing parameter grids (param1, param2, etc.)
    aperture_width : float, default=200
        Width of aperture grid in pixels
        
    Returns
    -------
    prf_grid : np.ndarray
        Array of pRFs (n_prfs x aperture_pixels)
    param_grid : np.ndarray  
        Parameter combinations (n_prfs x n_params)
    """
    # Extract parameter grids
    param_grids = []
    param_names = []
    
    for i in range(1, 11):  # param1 to param10
        param_key = f"param{i}"
        if param_key in model_params:
            param_values = model_params[param_key]
            if not isinstance(param_values, np.ndarray):
                param_values = np.array([param_values])
            # Only include non-zero parameters
            if not (len(param_values) == 1 and param_values[0] == 0):
                param_grids.append(param_values)
                param_names.append(f"param{i}")
    
    if not param_grids:
        raise ValueError("No valid parameter grids found")
    
    # Create meshgrid for all parameter combinations
    grids = np.meshgrid(*param_grids, indexing='ij')
    param_combinations = np.stack([g.flatten() for g in grids], axis=1)
    
    # Check if using polar coordinates
    polar_search = model_params.get('polar_search_space', True)
    
    if polar_search and len(param_grids) >= 2:
        # Convert polar to Cartesian for first two parameters
        angles = np.radians(param_combinations[:, 0])  # Convert degrees to radians
        eccentricities = param_combinations[:, 1]
        
        x0 = eccentricities * np.cos(angles)
        y0 = eccentricities * np.sin(angles)
        
        # Replace first two columns with Cartesian coordinates
        param_combinations[:, 0] = x0
        param_combinations[:, 1] = y0
    
    # Generate pRF for each parameter combination
    n_combinations = param_combinations.shape[0]
    aperture_pixels = int(aperture_width * aperture_width)
    prf_grid = np.zeros((n_combinations, aperture_pixels))
    
    for i, params in enumerate(param_combinations):
        # Use first 3 parameters for basic Gaussian (x0, y0, sigma)
        if len(params) >= 3:
            x0, y0, sigma = params[0], params[1], params[2]
            rf = gaussian_rf(x0, y0, sigma, aperture_width)
            prf_grid[i, :] = rf.flatten()
        else:
            warnings.warn(f"Insufficient parameters for pRF {i}, skipping")
    
    return prf_grid, param_combinations


def center_prf(rf: np.ndarray) -> Tuple[float, float]:
    """
    Find center of mass of a receptive field.
    
    Parameters
    ----------
    rf : np.ndarray
        2D receptive field
        
    Returns
    -------
    x_center, y_center : float
        Center of mass coordinates
    """
    # Create coordinate grids
    height, width = rf.shape
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Calculate center of mass
    total_mass = np.sum(rf)
    if total_mass > 0:
        x_center = np.sum(x_coords * rf) / total_mass
        y_center = np.sum(y_coords * rf) / total_mass
    else:
        x_center = width / 2
        y_center = height / 2
    
    return float(x_center), float(y_center)


def prf_size(rf: np.ndarray, threshold: float = 0.5) -> float:
    """
    Calculate effective size of receptive field.
    
    Parameters
    ----------
    rf : np.ndarray
        2D receptive field
    threshold : float, default=0.5
        Threshold for size calculation (fraction of maximum)
        
    Returns
    -------
    float
        Effective pRF size in pixels
    """
    # Normalize RF
    rf_norm = rf / np.max(rf) if np.max(rf) > 0 else rf
    
    # Find pixels above threshold
    above_threshold = rf_norm >= threshold
    
    if np.any(above_threshold):
        # Calculate area above threshold
        area = np.sum(above_threshold)
        # Convert to equivalent radius
        radius = np.sqrt(area / np.pi)
    else:
        radius = 0.0
    
    return float(radius)


def rotate_prf(rf: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate receptive field by specified angle.
    
    Parameters
    ----------
    rf : np.ndarray
        2D receptive field
    angle : float
        Rotation angle in degrees
        
    Returns
    -------
    np.ndarray
        Rotated receptive field
    """
    from scipy.ndimage import rotate
    
    # Rotate with appropriate settings to preserve intensity
    rotated = rotate(rf, angle, reshape=False, order=1, mode='constant', cval=0)
    
    return rotated