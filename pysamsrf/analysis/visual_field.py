"""
Visual field analysis functions.

This module provides functions for analyzing visual field properties
from pRF data, including coverage and cortical magnification.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from ..core.data_structures import SrfData


def visual_field_coverage(srf: SrfData, 
                         r_squared_threshold: float = 0.2,
                         grid_resolution: int = 50) -> Dict[str, Any]:
    """
    Calculate visual field coverage from pRF data.
    
    Parameters
    ----------
    srf : SrfData
        Surface data with pRF parameters
    r_squared_threshold : float, default=0.2
        Minimum R² for inclusion
    grid_resolution : int, default=50
        Resolution of coverage grid
        
    Returns
    -------
    dict
        Visual field coverage analysis
    """
    # Get pRF positions
    x0 = srf.get_parameter('x0')
    y0 = srf.get_parameter('y0')
    r_squared = srf.get_parameter('R^2')
    
    if x0 is None or y0 is None:
        raise ValueError("pRF position parameters required")
    
    # Apply R² threshold
    if r_squared is not None:
        valid_mask = r_squared >= r_squared_threshold
    else:
        valid_mask = np.ones(len(x0), dtype=bool)
    
    valid_x0 = x0[valid_mask]
    valid_y0 = y0[valid_mask]
    
    # Calculate field extent
    max_extent = max(np.max(np.abs(valid_x0)), np.max(np.abs(valid_y0))) * 1.1
    
    # Create coverage grid
    coverage_grid, xedges, yedges = np.histogram2d(
        valid_x0, valid_y0, bins=grid_resolution,
        range=[[-max_extent, max_extent], [-max_extent, max_extent]]
    )
    
    # Calculate coverage metrics
    total_grid_points = grid_resolution * grid_resolution
    covered_points = np.sum(coverage_grid > 0)
    coverage_fraction = covered_points / total_grid_points
    
    # Calculate eccentricity coverage
    eccentricity = np.sqrt(valid_x0**2 + valid_y0**2)
    
    return {
        'coverage_fraction': float(coverage_fraction),
        'covered_points': int(covered_points),
        'total_points': int(total_grid_points),
        'max_eccentricity': float(np.max(eccentricity)),
        'mean_eccentricity': float(np.mean(eccentricity)),
        'coverage_grid': coverage_grid,
        'grid_edges': (xedges, yedges),
        'field_extent': float(max_extent)
    }


def calculate_cortical_magnification(srf: SrfData,
                                   r_squared_threshold: float = 0.2) -> Dict[str, Any]:
    """
    Calculate cortical magnification from pRF data.
    
    Parameters
    ----------
    srf : SrfData
        Surface data with pRF parameters and surface geometry
    r_squared_threshold : float, default=0.2
        Minimum R² for inclusion
        
    Returns
    -------
    dict
        Cortical magnification analysis
    """
    # Get pRF data
    x0 = srf.get_parameter('x0')
    y0 = srf.get_parameter('y0')
    r_squared = srf.get_parameter('R^2')
    
    if any(param is None for param in [x0, y0]):
        raise ValueError("pRF position parameters required")
    
    if srf.vertices is None:
        raise ValueError("Surface vertices required for cortical magnification")
    
    # Apply R² threshold
    if r_squared is not None:
        valid_mask = r_squared >= r_squared_threshold
    else:
        valid_mask = np.ones(len(x0), dtype=bool)
    
    valid_x0 = x0[valid_mask]
    valid_y0 = y0[valid_mask]
    valid_vertices = srf.vertices[valid_mask]
    
    # Calculate eccentricity
    eccentricity = np.sqrt(valid_x0**2 + valid_y0**2)
    
    # Bin by eccentricity
    ecc_bins = np.linspace(0, np.max(eccentricity), 10)
    magnification = []
    bin_centers = []
    
    for i in range(len(ecc_bins) - 1):
        bin_mask = (eccentricity >= ecc_bins[i]) & (eccentricity < ecc_bins[i+1])
        
        if np.sum(bin_mask) < 3:  # Need at least 3 points
            continue
        
        bin_vertices = valid_vertices[bin_mask]
        bin_visual_pos = np.column_stack([valid_x0[bin_mask], valid_y0[bin_mask]])
        
        # Calculate local cortical magnification
        # This is simplified - real calculation would be more sophisticated
        cortical_distances = np.sqrt(np.sum(np.diff(bin_vertices, axis=0)**2, axis=1))
        visual_distances = np.sqrt(np.sum(np.diff(bin_visual_pos, axis=0)**2, axis=1))
        
        if len(visual_distances) > 0 and np.mean(visual_distances) > 0:
            local_mag = np.mean(cortical_distances) / np.mean(visual_distances)
            magnification.append(local_mag)
            bin_centers.append((ecc_bins[i] + ecc_bins[i+1]) / 2)
    
    return {
        'eccentricity_bins': bin_centers,
        'cortical_magnification': magnification,
        'mean_magnification': float(np.mean(magnification)) if magnification else 0,
        'implementation': 'simplified'  # Note that this is a simplified calculation
    }