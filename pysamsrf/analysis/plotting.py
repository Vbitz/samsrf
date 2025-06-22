"""
Plotting and visualization functions.

This module provides comprehensive plotting capabilities for pRF analysis
results, equivalent to MATLAB plotting functions in SamSrf.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import warnings
from typing import Optional, Union, List, Tuple, Dict, Any
from pathlib import Path

from ..core.data_structures import SrfData
from ..core.prf_functions import gaussian_rf


def plot_prf(srf: SrfData, vertex_idx: int, 
            aperture_width: float = 10.0,
            show_data: bool = True,
            ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot population receptive field for a single vertex.
    
    Parameters
    ----------
    srf : SrfData
        Surface data containing pRF parameters
    vertex_idx : int
        Index of vertex to plot
    aperture_width : float, default=10.0
        Width of visual field in degrees
    show_data : bool, default=True
        Whether to show fitted parameters as text
    ax : plt.Axes, optional
        Matplotlib axes to plot on
        
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get pRF parameters
    x0 = srf.get_parameter('x0')
    y0 = srf.get_parameter('y0')
    sigma = srf.get_parameter('sigma')
    r_squared = srf.get_parameter('R^2')
    
    if any(param is None for param in [x0, y0, sigma]):
        raise ValueError("Required pRF parameters (x0, y0, sigma) not found")
    
    # Get parameters for this vertex
    x0_v = x0[vertex_idx]
    y0_v = y0[vertex_idx]
    sigma_v = sigma[vertex_idx]
    r2_v = r_squared[vertex_idx] if r_squared is not None else 0
    
    # Generate pRF
    grid_size = 200
    rf = gaussian_rf(x0_v, y0_v, sigma_v, grid_size)
    
    # Create coordinate grids for plotting
    extent = aperture_width / 2
    x_coords = np.linspace(-extent, extent, grid_size)
    y_coords = np.linspace(-extent, extent, grid_size)
    
    # Plot pRF
    im = ax.imshow(rf, extent=[-extent, extent, -extent, extent], 
                   origin='lower', cmap='hot', alpha=0.8)
    
    # Add contour lines
    levels = [0.6, 0.3, 0.1]  # 60%, 30%, 10% of maximum
    ax.contour(x_coords, y_coords, rf, levels=levels, colors='white', alpha=0.7)
    
    # Mark center
    ax.plot(x0_v, y0_v, 'w+', markersize=15, markeredgewidth=3)
    
    # Add sigma circle
    circle = patches.Circle((x0_v, y0_v), sigma_v, fill=False, 
                           color='white', linestyle='--', linewidth=2)
    ax.add_patch(circle)
    
    # Formatting
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_xlabel('Horizontal position (degrees)')
    ax.set_ylabel('Vertical position (degrees)')
    ax.set_title(f'pRF - Vertex {vertex_idx}')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='pRF response')
    
    # Show parameters as text
    if show_data:
        text_str = f'x₀ = {x0_v:.2f}°\ny₀ = {y0_v:.2f}°\nσ = {sigma_v:.2f}°\nR² = {r2_v:.3f}'
        ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return ax


def plot_eccentricity(srf: SrfData, parameter: str, 
                     bins: Optional[Union[List[float], np.ndarray]] = None,
                     roi: Optional[str] = None,
                     r_squared_threshold: float = 0.05,
                     statistic: str = 'median',
                     ax: Optional[plt.Axes] = None) -> Tuple[plt.Axes, np.ndarray]:
    """
    Plot parameter values by eccentricity.
    
    Equivalent to MATLAB samsrf_plot function.
    
    Parameters
    ----------
    srf : SrfData
        Surface data
    parameter : str
        Parameter name to plot
    bins : list or array, optional
        Eccentricity bins (if None, uses default binning)
    roi : str, optional
        ROI to restrict analysis to
    r_squared_threshold : float, default=0.05
        Minimum R² for inclusion
    statistic : str, default='median'
        Statistic to plot ('median', 'mean')
    ax : plt.Axes, optional
        Matplotlib axes to plot on
        
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    np.ndarray
        Plotted data (bin centers, values, errors)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get data
    param_data = srf.get_parameter(parameter)
    if param_data is None:
        raise ValueError(f"Parameter '{parameter}' not found")
    
    # Calculate eccentricity
    x0 = srf.get_parameter('x0')
    y0 = srf.get_parameter('y0')
    if x0 is None or y0 is None:
        raise ValueError("x0 and y0 parameters required for eccentricity calculation")
    
    eccentricity = np.sqrt(x0**2 + y0**2)
    
    # Apply R² threshold
    r_squared = srf.get_parameter('R^2')
    if r_squared is not None:
        valid_mask = r_squared >= r_squared_threshold
    else:
        valid_mask = np.ones(len(param_data), dtype=bool)
        warnings.warn("No R² data found, using all vertices")
    
    # Apply ROI mask (placeholder - would need ROI loading)
    if roi is not None:
        warnings.warn("ROI filtering not yet implemented")
    
    # Filter data
    valid_ecc = eccentricity[valid_mask]
    valid_param = param_data[valid_mask]
    
    # Set up bins
    if bins is None:
        bins = np.arange(0, np.max(valid_ecc) + 1, 1)
    
    # Calculate binned statistics
    bin_centers = []
    bin_values = []
    bin_errors = []
    
    for i in range(len(bins) - 1):
        bin_mask = (valid_ecc >= bins[i]) & (valid_ecc < bins[i + 1])
        
        if np.sum(bin_mask) > 0:
            bin_data = valid_param[bin_mask]
            
            if statistic == 'median':
                value = np.median(bin_data)
                # Calculate median absolute deviation
                mad = np.median(np.abs(bin_data - value))
                error = 1.4826 * mad  # Convert to approximate standard error
            elif statistic == 'mean':
                value = np.mean(bin_data)
                error = np.std(bin_data) / np.sqrt(len(bin_data))
            else:
                raise ValueError(f"Unknown statistic: {statistic}")
            
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_values.append(value)
            bin_errors.append(error)
    
    bin_centers = np.array(bin_centers)
    bin_values = np.array(bin_values)
    bin_errors = np.array(bin_errors)
    
    # Plot
    ax.errorbar(bin_centers, bin_values, yerr=bin_errors, 
               marker='o', linestyle='-', capsize=5)
    
    ax.set_xlabel('Eccentricity (degrees)')
    ax.set_ylabel(f'{parameter}')
    ax.set_title(f'{parameter} by Eccentricity ({statistic})')
    ax.grid(True, alpha=0.3)
    
    # Return results
    results = np.column_stack([bin_centers, bin_values, bin_errors])
    return ax, results


def plot_polar(srf: SrfData, parameter: str,
              r_squared_threshold: float = 0.05,
              ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot parameter values in polar coordinates.
    
    Parameters
    ----------
    srf : SrfData
        Surface data
    parameter : str
        Parameter name to plot
    r_squared_threshold : float, default=0.05
        Minimum R² for inclusion
    ax : plt.Axes, optional
        Matplotlib axes to plot on (should be polar projection)
        
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    # Get data
    param_data = srf.get_parameter(parameter)
    x0 = srf.get_parameter('x0')
    y0 = srf.get_parameter('y0')
    r_squared = srf.get_parameter('R^2')
    
    if any(data is None for data in [param_data, x0, y0]):
        raise ValueError("Required parameters not found")
    
    # Apply R² threshold
    if r_squared is not None:
        valid_mask = r_squared >= r_squared_threshold
    else:
        valid_mask = np.ones(len(param_data), dtype=bool)
    
    # Calculate polar coordinates
    eccentricity = np.sqrt(x0**2 + y0**2)
    polar_angle = np.arctan2(y0, x0)
    
    # Filter data
    valid_ecc = eccentricity[valid_mask]
    valid_angle = polar_angle[valid_mask]
    valid_param = param_data[valid_mask]
    
    # Create polar scatter plot
    scatter = ax.scatter(valid_angle, valid_ecc, c=valid_param, 
                        s=30, alpha=0.7, cmap='viridis')
    
    # Formatting
    ax.set_theta_zero_location('E')  # 0° at right (typical for visual field)
    ax.set_theta_direction(-1)       # Clockwise (typical for visual field)
    ax.set_xlabel('Eccentricity (degrees)')
    ax.set_title(f'{parameter} in Visual Field Coordinates')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label=parameter)
    
    return ax


def plot_surface(srf: SrfData, parameter: str,
                r_squared_threshold: float = 0.05,
                colormap: str = 'viridis',
                view_angle: Tuple[float, float] = (45, 45),
                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot parameter data on 3D surface.
    
    Parameters
    ----------
    srf : SrfData
        Surface data with geometry
    parameter : str
        Parameter name to plot
    r_squared_threshold : float, default=0.05
        Minimum R² for inclusion
    colormap : str, default='viridis'
        Colormap name
    view_angle : tuple, default=(45, 45)
        3D viewing angles (elevation, azimuth)
    ax : plt.Axes, optional
        Matplotlib 3D axes to plot on
        
    Returns
    -------
    plt.Axes
        Matplotlib 3D axes object
    """
    if srf.vertices is None or srf.faces is None:
        raise ValueError("Surface geometry (vertices and faces) required")
    
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Get data
    param_data = srf.get_parameter(parameter)
    if param_data is None:
        raise ValueError(f"Parameter '{parameter}' not found")
    
    # Apply R² threshold
    r_squared = srf.get_parameter('R^2')
    if r_squared is not None:
        valid_mask = r_squared >= r_squared_threshold
        # Set invalid vertices to NaN for proper visualization
        plot_data = param_data.copy()
        plot_data[~valid_mask] = np.nan
    else:
        plot_data = param_data
    
    # Plot surface
    surface = ax.plot_trisurf(srf.vertices[:, 0], srf.vertices[:, 1], srf.vertices[:, 2],
                             triangles=srf.faces, facecolors=plt.cm.get_cmap(colormap)(plot_data),
                             alpha=0.9, shade=True)
    
    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # Formatting
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f'{parameter} on Surface')
    
    # Add colorbar (approximate)
    mappable = plt.cm.ScalarMappable(cmap=colormap)
    mappable.set_array(plot_data[~np.isnan(plot_data)])
    plt.colorbar(mappable, ax=ax, label=parameter, shrink=0.8)
    
    return ax


def plot_visual_field_coverage(srf: SrfData, roi: Optional[str] = None,
                              r_squared_threshold: float = 0.2,
                              bins: int = 50,
                              ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot visual field coverage as 2D histogram.
    
    Parameters
    ----------
    srf : SrfData
        Surface data
    roi : str, optional
        ROI to analyze
    r_squared_threshold : float, default=0.2
        Minimum R² for inclusion
    bins : int, default=50
        Number of bins for histogram
    ax : plt.Axes, optional
        Matplotlib axes to plot on
        
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get pRF positions
    x0 = srf.get_parameter('x0')
    y0 = srf.get_parameter('y0')
    r_squared = srf.get_parameter('R^2')
    
    if x0 is None or y0 is None:
        raise ValueError("pRF position parameters (x0, y0) required")
    
    # Apply R² threshold
    if r_squared is not None:
        valid_mask = r_squared >= r_squared_threshold
    else:
        valid_mask = np.ones(len(x0), dtype=bool)
        warnings.warn("No R² data found, using all vertices")
    
    valid_x0 = x0[valid_mask]
    valid_y0 = y0[valid_mask]
    
    # Create 2D histogram
    max_extent = max(np.max(np.abs(valid_x0)), np.max(np.abs(valid_y0))) * 1.1
    
    hist, xedges, yedges = np.histogram2d(valid_x0, valid_y0, bins=bins,
                                         range=[[-max_extent, max_extent],
                                               [-max_extent, max_extent]])
    
    # Plot
    im = ax.imshow(hist.T, extent=[-max_extent, max_extent, -max_extent, max_extent],
                   origin='lower', cmap='hot', alpha=0.8)
    
    # Add contour lines
    X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, 
                       (yedges[:-1] + yedges[1:]) / 2)
    ax.contour(X, Y, hist.T, colors='white', alpha=0.5)
    
    # Formatting
    ax.set_xlabel('Horizontal position (degrees)')
    ax.set_ylabel('Vertical position (degrees)')
    ax.set_title('Visual Field Coverage')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add colorbar
    plt.colorbar(im, ax=ax, label='Number of pRFs')
    
    return ax


def plot_prf_size_distribution(srf: SrfData, parameter: str = 'sigma',
                              r_squared_threshold: float = 0.05,
                              bins: int = 50,
                              ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot distribution of pRF sizes.
    
    Parameters
    ----------
    srf : SrfData
        Surface data
    parameter : str, default='sigma'
        Size parameter to plot
    r_squared_threshold : float, default=0.05
        Minimum R² for inclusion
    bins : int, default=50
        Number of histogram bins
    ax : plt.Axes, optional
        Matplotlib axes to plot on
        
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get data
    size_data = srf.get_parameter(parameter)
    r_squared = srf.get_parameter('R^2')
    
    if size_data is None:
        raise ValueError(f"Parameter '{parameter}' not found")
    
    # Apply R² threshold
    if r_squared is not None:
        valid_mask = r_squared >= r_squared_threshold
        valid_sizes = size_data[valid_mask]
    else:
        valid_sizes = size_data
        warnings.warn("No R² data found, using all vertices")
    
    # Plot histogram
    counts, bins_edges, patches = ax.hist(valid_sizes, bins=bins, alpha=0.7, 
                                         color='skyblue', edgecolor='black')
    
    # Add statistics
    mean_size = np.mean(valid_sizes)
    median_size = np.median(valid_sizes)
    
    ax.axvline(mean_size, color='red', linestyle='--', linewidth=2, 
              label=f'Mean: {mean_size:.2f}')
    ax.axvline(median_size, color='orange', linestyle='--', linewidth=2,
              label=f'Median: {median_size:.2f}')
    
    # Formatting
    ax.set_xlabel(f'{parameter}')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of {parameter}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return ax


def create_colormap(name: str) -> LinearSegmentedColormap:
    """
    Create custom colormaps used in SamSrf.
    
    Parameters
    ----------
    name : str
        Colormap name ('polar', 'eccentricity', etc.)
        
    Returns
    -------
    LinearSegmentedColormap
        Custom colormap
    """
    if name == 'polar':
        # HSV-style colormap for polar angle
        colors = ['red', 'yellow', 'green', 'cyan', 'blue', 'magenta', 'red']
        return LinearSegmentedColormap.from_list('polar', colors)
    
    elif name == 'eccentricity':
        # Hot colormap for eccentricity
        colors = ['black', 'red', 'orange', 'yellow', 'white']
        return LinearSegmentedColormap.from_list('eccentricity', colors)
    
    elif name == 'r_squared':
        # Cool to warm for R²
        colors = ['darkblue', 'blue', 'cyan', 'yellow', 'red']
        return LinearSegmentedColormap.from_list('r_squared', colors)
    
    else:
        # Default to matplotlib colormap
        return plt.cm.get_cmap(name)


def save_plot(filename: Union[str, Path], dpi: int = 300, **kwargs) -> None:
    """
    Save current plot with publication-quality settings.
    
    Parameters
    ----------
    filename : str or Path
        Output filename
    dpi : int, default=300
        Resolution in dots per inch
    **kwargs
        Additional arguments to matplotlib savefig
    """
    plt.savefig(filename, dpi=dpi, bbox_inches='tight', **kwargs)


def plot_comparison(srf1: SrfData, srf2: SrfData, parameter: str,
                   labels: List[str] = ['Dataset 1', 'Dataset 2'],
                   r_squared_threshold: float = 0.05,
                   ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot comparison between two datasets.
    
    Parameters
    ----------
    srf1, srf2 : SrfData
        Surface data to compare
    parameter : str
        Parameter to compare
    labels : list, default=['Dataset 1', 'Dataset 2']
        Labels for datasets
    r_squared_threshold : float, default=0.05
        Minimum R² for inclusion
    ax : plt.Axes, optional
        Matplotlib axes to plot on
        
    Returns
    -------
    plt.Axes
        Matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Get data from both datasets
    data1 = srf1.get_parameter(parameter)
    data2 = srf2.get_parameter(parameter)
    r2_1 = srf1.get_parameter('R^2')
    r2_2 = srf2.get_parameter('R^2')
    
    if data1 is None or data2 is None:
        raise ValueError(f"Parameter '{parameter}' not found in one or both datasets")
    
    # Apply R² thresholds
    if r2_1 is not None:
        mask1 = r2_1 >= r_squared_threshold
        valid_data1 = data1[mask1]
    else:
        valid_data1 = data1
    
    if r2_2 is not None:
        mask2 = r2_2 >= r_squared_threshold
        valid_data2 = data2[mask2]
    else:
        valid_data2 = data2
    
    # Find common range for plotting
    all_data = np.concatenate([valid_data1, valid_data2])
    data_range = [np.min(all_data), np.max(all_data)]
    
    # Scatter plot
    ax.scatter(valid_data1, valid_data2, alpha=0.6, s=20)
    
    # Add unity line
    ax.plot(data_range, data_range, 'r--', linewidth=2, label='Unity line')
    
    # Calculate correlation
    if len(valid_data1) == len(valid_data2):
        correlation = np.corrcoef(valid_data1, valid_data2)[0, 1]
        ax.text(0.05, 0.95, f'r = {correlation:.3f}', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Formatting
    ax.set_xlabel(f'{parameter} - {labels[0]}')
    ax.set_ylabel(f'{parameter} - {labels[1]}')
    ax.set_title(f'{parameter} Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return ax