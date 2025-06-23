"""
Aperture/stimulus data input/output functions.

This module handles loading of stimulus aperture files used in pRF fitting,
supporting MATLAB .mat files and other formats.
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import scipy.io as sio
import h5py


def load_apertures(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load stimulus aperture file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to aperture file (.mat or .npz)
        
    Returns
    -------
    dict
        Dictionary containing aperture data with keys:
        - 'apertures': stimulus aperture matrix (n_timepoints x n_pixels)
        - 'resolution': aperture resolution [width, height]
        - 'timing': stimulus timing information (if available)
        - 'metadata': additional metadata
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.mat':
        return _load_matlab_apertures(filepath)
    elif suffix == '.npz':
        return _load_numpy_apertures(filepath)
    else:
        raise ValueError(f"Unsupported aperture format: {suffix}")


def save_apertures(apertures: np.ndarray, filepath: Union[str, Path],
                  resolution: Optional[List[int]] = None,
                  timing: Optional[np.ndarray] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save stimulus apertures to file.
    
    Parameters
    ----------
    apertures : np.ndarray
        Aperture matrix (n_timepoints x n_pixels)
    filepath : str or Path
        Output file path
    resolution : list, optional
        Aperture resolution [width, height]
    timing : np.ndarray, optional
        Stimulus timing information
    metadata : dict, optional
        Additional metadata to save
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.npz':
        _save_numpy_apertures(apertures, filepath, resolution, timing, metadata)
    elif suffix == '.mat':
        _save_matlab_apertures(apertures, filepath, resolution, timing, metadata)
    else:
        raise ValueError(f"Unsupported output format: {suffix}")


def _load_matlab_apertures(filepath: Path) -> Dict[str, Any]:
    """Load apertures from MATLAB .mat file."""
    try:
        # Try loading with scipy.io first (for older MATLAB files)
        mat_data = sio.loadmat(str(filepath), struct_as_record=False, squeeze_me=True)
        
        # Look for common aperture variable names
        aperture_keys = ['ApFrm', 'ApFrms', 'apertures', 'Apertures', 'stim', 'stimulus']
        apertures = None
        
        for key in aperture_keys:
            if key in mat_data:
                apertures = mat_data[key]
                break
        
        if apertures is None:
            # Look for the first large 2D array
            for key, value in mat_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray) and value.ndim == 2:
                    apertures = value
                    break
        
        if apertures is None:
            raise ValueError("No aperture data found in MATLAB file")
        
        # Extract metadata
        result = {
            'apertures': apertures,
            'resolution': None,
            'timing': None,
            'metadata': {}
        }
        
        # Look for resolution information
        if 'resolution' in mat_data:
            result['resolution'] = mat_data['resolution']
        elif 'res' in mat_data:
            result['resolution'] = mat_data['res']
        elif apertures.shape[1] == 10201:  # Common 101x101 resolution
            result['resolution'] = [101, 101]
        else:
            # Try to infer square resolution
            n_pixels = apertures.shape[1]
            side = int(np.sqrt(n_pixels))
            if side * side == n_pixels:
                result['resolution'] = [side, side]
        
        # Look for timing information
        if 'timing' in mat_data:
            result['timing'] = mat_data['timing']
        elif 'time' in mat_data:
            result['timing'] = mat_data['time']
        elif 'TR' in mat_data:
            # Create timing array from TR
            tr = float(mat_data['TR'])
            n_timepoints = apertures.shape[0]
            result['timing'] = np.arange(n_timepoints) * tr
            result['metadata']['TR'] = tr
        
        # Store any other metadata
        for key, value in mat_data.items():
            if not key.startswith('__') and key not in ['ApFrm', 'ApFrms', 'apertures', 
                                                         'resolution', 'timing', 'time', 'TR']:
                if isinstance(value, (int, float, str, np.ndarray)):
                    result['metadata'][key] = value
        
        return result
        
    except NotImplementedError:
        # Try loading with h5py for newer MATLAB v7.3 files
        return _load_matlab_v73_apertures(filepath)
    except Exception as e:
        raise IOError(f"Failed to load MATLAB aperture file {filepath}: {e}")


def _load_matlab_v73_apertures(filepath: Path) -> Dict[str, Any]:
    """Load apertures from MATLAB v7.3 format using h5py."""
    try:
        with h5py.File(str(filepath), 'r') as f:
            # Look for aperture data
            aperture_keys = ['ApFrm', 'ApFrms', 'apertures', 'Apertures', 'stim', 'stimulus']
            apertures = None
            
            for key in aperture_keys:
                if key in f:
                    apertures = f[key][:]
                    # Handle MATLAB's column-major order
                    if apertures.ndim == 2:
                        apertures = apertures.T
                    break
            
            if apertures is None:
                raise ValueError("No aperture data found in MATLAB v7.3 file")
            
            result = {
                'apertures': apertures,
                'resolution': None,
                'timing': None,
                'metadata': {}
            }
            
            # Look for resolution
            if 'resolution' in f:
                result['resolution'] = f['resolution'][:]
            elif 'res' in f:
                result['resolution'] = f['res'][:]
            
            # Look for timing
            if 'timing' in f:
                result['timing'] = f['timing'][:]
            elif 'time' in f:
                result['timing'] = f['time'][:]
            elif 'TR' in f:
                tr = float(f['TR'][()])
                n_timepoints = apertures.shape[0]
                result['timing'] = np.arange(n_timepoints) * tr
                result['metadata']['TR'] = tr
            
            return result
            
    except Exception as e:
        raise IOError(f"Failed to load MATLAB v7.3 aperture file {filepath}: {e}")


def _load_numpy_apertures(filepath: Path) -> Dict[str, Any]:
    """Load apertures from numpy .npz file."""
    try:
        data = np.load(str(filepath))
        
        if 'apertures' not in data:
            raise ValueError("No 'apertures' array found in .npz file")
        
        result = {
            'apertures': data['apertures'],
            'resolution': data.get('resolution', None),
            'timing': data.get('timing', None),
            'metadata': {}
        }
        
        # Load any additional arrays as metadata
        for key in data.files:
            if key not in ['apertures', 'resolution', 'timing']:
                result['metadata'][key] = data[key]
        
        return result
        
    except Exception as e:
        raise IOError(f"Failed to load numpy aperture file {filepath}: {e}")


def _save_numpy_apertures(apertures: np.ndarray, filepath: Path,
                         resolution: Optional[List[int]] = None,
                         timing: Optional[np.ndarray] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save apertures to numpy .npz format."""
    save_dict = {'apertures': apertures}
    
    if resolution is not None:
        save_dict['resolution'] = np.array(resolution)
    
    if timing is not None:
        save_dict['timing'] = timing
    
    if metadata is not None:
        for key, value in metadata.items():
            if isinstance(value, (np.ndarray, int, float, str)):
                save_dict[key] = value
    
    np.savez_compressed(str(filepath), **save_dict)


def _save_matlab_apertures(apertures: np.ndarray, filepath: Path,
                          resolution: Optional[List[int]] = None,
                          timing: Optional[np.ndarray] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> None:
    """Save apertures to MATLAB .mat format."""
    save_dict = {'ApFrm': apertures}
    
    if resolution is not None:
        save_dict['resolution'] = np.array(resolution)
    
    if timing is not None:
        save_dict['timing'] = timing
    
    if metadata is not None:
        for key, value in metadata.items():
            if isinstance(value, (np.ndarray, int, float, str)):
                save_dict[key] = value
    
    sio.savemat(str(filepath), save_dict)


def create_bar_apertures(n_steps: int, bar_width: float, resolution: int = 101,
                        orientations: Optional[List[float]] = None) -> np.ndarray:
    """
    Create sweeping bar apertures for pRF mapping.
    
    Parameters
    ----------
    n_steps : int
        Number of steps for each bar sweep
    bar_width : float
        Width of the bar in visual field units
    resolution : int, default=101
        Resolution of the aperture grid
    orientations : list, optional
        Bar orientations in degrees. Default: [0, 45, 90, 135, 180, 225, 270, 315]
        
    Returns
    -------
    np.ndarray
        Aperture matrix (n_timepoints x n_pixels)
    """
    if orientations is None:
        orientations = [0, 45, 90, 135, 180, 225, 270, 315]
    
    # Create coordinate grid
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Calculate bar width in pixels
    bar_width_pixels = bar_width * resolution / 2
    
    apertures = []
    
    for orientation in orientations:
        # Convert orientation to radians
        theta = np.radians(orientation)
        
        # Rotate coordinates
        x_rot = xx * np.cos(theta) - yy * np.sin(theta)
        
        # Create bar positions
        positions = np.linspace(-1 - bar_width/2, 1 + bar_width/2, n_steps)
        
        for pos in positions:
            # Create bar aperture
            aperture = np.abs(x_rot - pos) <= bar_width/2
            apertures.append(aperture.flatten())
    
    return np.array(apertures)


def create_wedge_apertures(n_steps: int, wedge_width: float, resolution: int = 101,
                          n_cycles: int = 1) -> np.ndarray:
    """
    Create rotating wedge apertures for pRF mapping.
    
    Parameters
    ----------
    n_steps : int
        Number of steps per cycle
    wedge_width : float
        Angular width of wedge in degrees
    resolution : int, default=101
        Resolution of the aperture grid
    n_cycles : int, default=1
        Number of full rotation cycles
        
    Returns
    -------
    np.ndarray
        Aperture matrix (n_timepoints x n_pixels)
    """
    # Create coordinate grid
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Convert to polar coordinates
    angle = np.arctan2(yy, xx)
    radius = np.sqrt(xx**2 + yy**2)
    
    # Convert wedge width to radians
    wedge_rad = np.radians(wedge_width)
    
    apertures = []
    
    for cycle in range(n_cycles):
        angles = np.linspace(0, 2*np.pi, n_steps, endpoint=False)
        
        for center_angle in angles:
            # Create wedge aperture
            angle_diff = np.angle(np.exp(1j * (angle - center_angle)))
            aperture = (np.abs(angle_diff) <= wedge_rad/2) & (radius <= 1)
            apertures.append(aperture.flatten())
    
    return np.array(apertures)


def create_ring_apertures(n_steps: int, ring_width: float, resolution: int = 101,
                         expand: bool = True) -> np.ndarray:
    """
    Create expanding/contracting ring apertures for pRF mapping.
    
    Parameters
    ----------
    n_steps : int
        Number of steps
    ring_width : float
        Width of the ring in visual field units
    resolution : int, default=101
        Resolution of the aperture grid
    expand : bool, default=True
        If True, rings expand; if False, rings contract
        
    Returns
    -------
    np.ndarray
        Aperture matrix (n_timepoints x n_pixels)
    """
    # Create coordinate grid
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    xx, yy = np.meshgrid(x, y)
    
    # Calculate radius
    radius = np.sqrt(xx**2 + yy**2)
    
    apertures = []
    
    # Create ring positions
    if expand:
        positions = np.linspace(0, 1 + ring_width, n_steps)
    else:
        positions = np.linspace(1 + ring_width, 0, n_steps)
    
    for pos in positions:
        # Create ring aperture
        aperture = (radius >= pos - ring_width/2) & (radius <= pos + ring_width/2)
        apertures.append(aperture.flatten())
    
    return np.array(apertures)


def validate_apertures(apertures: np.ndarray, resolution: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Validate aperture data.
    
    Parameters
    ----------
    apertures : np.ndarray
        Aperture matrix to validate
    resolution : list, optional
        Expected resolution [width, height]
        
    Returns
    -------
    dict
        Validation results
    """
    validation = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check dimensions
    if apertures.ndim != 2:
        validation['errors'].append("Apertures must be 2D array")
        validation['is_valid'] = False
    else:
        n_timepoints, n_pixels = apertures.shape
        validation['n_timepoints'] = n_timepoints
        validation['n_pixels'] = n_pixels
        
        # Check if resolution matches
        if resolution is not None:
            expected_pixels = resolution[0] * resolution[1]
            if n_pixels != expected_pixels:
                validation['errors'].append(f"Pixel count {n_pixels} doesn't match resolution {resolution}")
                validation['is_valid'] = False
        
        # Check data range
        if np.any(apertures < 0) or np.any(apertures > 1):
            validation['warnings'].append("Aperture values outside [0, 1] range")
        
        # Check for empty frames
        empty_frames = np.sum(apertures, axis=1) == 0
        if np.any(empty_frames):
            validation['warnings'].append(f"{np.sum(empty_frames)} empty frames detected")
    
    return validation