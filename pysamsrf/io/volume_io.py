"""
Volume data input/output functions.

This module handles loading and saving of volumetric neuroimaging data
in NIfTI and other formats.
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
import nibabel as nib

from ..core.data_structures import SrfData


def load_volume(filepath: Union[str, Path], 
               mask: Optional[Union[str, Path, np.ndarray]] = None) -> SrfData:
    """
    Load volumetric data from NIfTI files.
    
    Parameters
    ----------
    filepath : str or Path
        Path to NIfTI file
    mask : str, Path, or np.ndarray, optional
        Brain mask to apply (path to mask file or boolean array)
        
    Returns
    -------
    SrfData
        Volume data in SrfData format
    """
    filepath = Path(filepath)
    
    try:
        img = nib.load(str(filepath))
    except Exception as e:
        raise IOError(f"Failed to load NIfTI file {filepath}: {e}")
    
    data = img.get_fdata()
    affine = img.affine
    header = img.header
    
    # Apply mask if provided
    if mask is not None:
        if isinstance(mask, (str, Path)):
            mask_img = nib.load(str(mask))
            mask_data = mask_img.get_fdata().astype(bool)
        else:
            mask_data = mask.astype(bool)
        
        # Check mask dimensions
        if mask_data.shape[:3] != data.shape[:3]:
            raise ValueError("Mask dimensions don't match data dimensions")
    else:
        # Create mask from non-zero values
        if data.ndim == 4:
            mask_data = np.any(data != 0, axis=3)
        else:
            mask_data = data != 0
    
    # Extract masked voxels
    voxel_indices = np.where(mask_data)
    n_voxels = len(voxel_indices[0])
    
    # Create SrfData structure
    srf = SrfData('vol')
    
    # Store voxel coordinates as "vertices"
    srf.vertices = np.column_stack(voxel_indices).astype(float)
    
    # Store volume data
    if data.ndim == 4:
        # Time series data
        n_timepoints = data.shape[3]
        srf.y = np.zeros((n_voxels, n_timepoints))
        
        for t in range(n_timepoints):
            srf.y[:, t] = data[voxel_indices + (t,)]
    else:
        # Single volume (parameter map)
        param_data = data[voxel_indices]
        srf.add_data(param_data.reshape(1, -1), ['volume_data'])
    
    # Store metadata
    srf._volume_shape = data.shape[:3]
    srf._affine = affine
    srf._header = header
    srf._mask = mask_data
    srf.functional = [str(filepath)]
    
    return srf


def save_volume(srf: SrfData, filepath: Union[str, Path],
               parameter: Optional[str] = None,
               template: Optional[Union[str, Path]] = None) -> None:
    """
    Save volume data to NIfTI format.
    
    Parameters
    ----------
    srf : SrfData
        Volume data to save
    filepath : str or Path
        Output file path
    parameter : str, optional
        Specific parameter to save (if None, saves first parameter)
    template : str or Path, optional
        Template NIfTI file for header/affine information
    """
    if srf.hemisphere != 'vol':
        warnings.warn("SrfData doesn't appear to be volume data")
    
    # Get volume shape and affine
    if hasattr(srf, '_volume_shape'):
        volume_shape = srf._volume_shape
    else:
        raise ValueError("Volume shape information not available")
    
    if hasattr(srf, '_affine'):
        affine = srf._affine
    else:
        # Create default affine
        affine = np.eye(4)
        warnings.warn("Using default affine matrix")
    
    # Get mask
    if hasattr(srf, '_mask'):
        mask = srf._mask
    else:
        # Reconstruct mask from vertex coordinates
        mask = np.zeros(volume_shape, dtype=bool)
        coords = srf.vertices.astype(int)
        mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    
    # Get data to save
    if parameter is not None:
        if parameter not in srf.values:
            raise ValueError(f"Parameter '{parameter}' not found")
        data_to_save = srf.get_parameter(parameter)
    elif srf.data is not None:
        data_to_save = srf.data[0, :]  # First parameter
    elif srf.y is not None:
        # Save time series (create 4D volume)
        n_timepoints = srf.y.shape[1]
        volume_data = np.zeros(volume_shape + (n_timepoints,))
        
        voxel_coords = srf.vertices.astype(int)
        for t in range(n_timepoints):
            volume_data[voxel_coords[:, 0], voxel_coords[:, 1], 
                       voxel_coords[:, 2], t] = srf.y[:, t]
        
        # Save 4D volume
        img = nib.Nifti1Image(volume_data, affine)
        nib.save(img, str(filepath))
        return
    else:
        raise ValueError("No data to save")
    
    # Create 3D volume
    volume_data = np.zeros(volume_shape)
    voxel_coords = srf.vertices.astype(int)
    volume_data[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = data_to_save
    
    # Create NIfTI image
    if template is not None:
        # Use template for header info
        template_img = nib.load(str(template))
        img = nib.Nifti1Image(volume_data, template_img.affine, template_img.header)
    else:
        img = nib.Nifti1Image(volume_data, affine)
    
    # Save to file
    nib.save(img, str(filepath))


def load_multiple_volumes(filepaths: list, 
                         mask: Optional[Union[str, Path, np.ndarray]] = None) -> SrfData:
    """
    Load and combine multiple volume files.
    
    Parameters
    ----------
    filepaths : list
        List of NIfTI file paths
    mask : str, Path, or np.ndarray, optional
        Brain mask to apply to all volumes
        
    Returns
    -------
    SrfData
        Combined volume data
    """
    if not filepaths:
        raise ValueError("No files provided")
    
    # Load first volume
    combined_srf = load_volume(filepaths[0], mask)
    
    # Combine remaining volumes
    for filepath in filepaths[1:]:
        srf = load_volume(filepath, mask)
        
        # Check compatibility
        if not np.array_equal(srf.vertices, combined_srf.vertices):
            raise ValueError(f"Volume {filepath} has different voxel coordinates")
        
        # Combine time series data
        if srf.y is not None:
            if combined_srf.y is None:
                combined_srf.y = srf.y
            else:
                combined_srf.y = np.concatenate([combined_srf.y, srf.y], axis=1)
        
        # Add to functional file list
        combined_srf.functional.extend(srf.functional)
    
    return combined_srf


def volume_to_surface_projection(volume_srf: SrfData, surface_srf: SrfData,
                                projection_method: str = 'nearest') -> SrfData:
    """
    Project volume data onto surface.
    
    Parameters
    ----------
    volume_srf : SrfData
        Volume data to project
    surface_srf : SrfData
        Target surface
    projection_method : str, default='nearest'
        Projection method ('nearest', 'trilinear')
        
    Returns
    -------
    SrfData
        Surface data with projected volume data
    """
    if volume_srf.hemisphere != 'vol':
        raise ValueError("First argument must be volume data")
    
    if surface_srf.vertices is None:
        raise ValueError("Surface must have vertex coordinates")
    
    # Get volume properties
    volume_shape = volume_srf._volume_shape
    affine = volume_srf._affine
    
    # Transform surface coordinates to voxel space
    inv_affine = np.linalg.inv(affine)
    
    # Add homogeneous coordinate
    surface_coords_homo = np.column_stack([
        surface_srf.vertices, 
        np.ones(surface_srf.vertices.shape[0])
    ])
    
    # Transform to voxel coordinates
    voxel_coords = (inv_affine @ surface_coords_homo.T).T[:, :3]
    
    # Project volume data to surface
    projected_srf = SrfData(surface_srf.hemisphere)
    projected_srf.vertices = surface_srf.vertices
    projected_srf.faces = surface_srf.faces
    projected_srf.curvature = surface_srf.curvature
    
    if projection_method == 'nearest':
        # Nearest neighbor interpolation
        voxel_indices = np.round(voxel_coords).astype(int)
        
        # Check bounds
        valid_mask = (
            (voxel_indices[:, 0] >= 0) & (voxel_indices[:, 0] < volume_shape[0]) &
            (voxel_indices[:, 1] >= 0) & (voxel_indices[:, 1] < volume_shape[1]) &
            (voxel_indices[:, 2] >= 0) & (voxel_indices[:, 2] < volume_shape[2])
        )
        
        # Find corresponding volume voxels
        n_surface_vertices = surface_srf.vertices.shape[0]
        
        if volume_srf.data is not None:
            # Project parameter data
            projected_data = np.zeros((volume_srf.data.shape[0], n_surface_vertices))
            
            for v in range(n_surface_vertices):
                if valid_mask[v]:
                    # Find closest volume voxel
                    vol_coords = voxel_indices[v, :]
                    
                    # Find this voxel in volume data
                    vol_vertex_coords = volume_srf.vertices.astype(int)
                    matches = np.where(
                        (vol_vertex_coords[:, 0] == vol_coords[0]) &
                        (vol_vertex_coords[:, 1] == vol_coords[1]) &
                        (vol_vertex_coords[:, 2] == vol_coords[2])
                    )[0]
                    
                    if len(matches) > 0:
                        projected_data[:, v] = volume_srf.data[:, matches[0]]
            
            projected_srf.add_data(projected_data, volume_srf.values)
        
        if volume_srf.y is not None:
            # Project time series data
            projected_timeseries = np.zeros((n_surface_vertices, volume_srf.y.shape[1]))
            
            for v in range(n_surface_vertices):
                if valid_mask[v]:
                    vol_coords = voxel_indices[v, :]
                    vol_vertex_coords = volume_srf.vertices.astype(int)
                    matches = np.where(
                        (vol_vertex_coords[:, 0] == vol_coords[0]) &
                        (vol_vertex_coords[:, 1] == vol_coords[1]) &
                        (vol_vertex_coords[:, 2] == vol_coords[2])
                    )[0]
                    
                    if len(matches) > 0:
                        projected_timeseries[v, :] = volume_srf.y[matches[0], :]
            
            projected_srf.y = projected_timeseries
    
    else:
        raise NotImplementedError(f"Projection method '{projection_method}' not implemented")
    
    return projected_srf


def create_volume_mask(shape: Tuple[int, int, int], 
                      coordinates: np.ndarray) -> np.ndarray:
    """
    Create a volume mask from coordinates.
    
    Parameters
    ----------
    shape : tuple
        Volume shape (nx, ny, nz)
    coordinates : np.ndarray
        Voxel coordinates (n_voxels x 3)
        
    Returns
    -------
    np.ndarray
        Boolean mask array
    """
    mask = np.zeros(shape, dtype=bool)
    coords = coordinates.astype(int)
    
    # Check bounds
    valid = (
        (coords[:, 0] >= 0) & (coords[:, 0] < shape[0]) &
        (coords[:, 1] >= 0) & (coords[:, 1] < shape[1]) &
        (coords[:, 2] >= 0) & (coords[:, 2] < shape[2])
    )
    
    valid_coords = coords[valid]
    mask[valid_coords[:, 0], valid_coords[:, 1], valid_coords[:, 2]] = True
    
    return mask


def extract_volume_roi(volume_data: np.ndarray, roi_mask: np.ndarray,
                      affine: Optional[np.ndarray] = None) -> SrfData:
    """
    Extract ROI from volume data.
    
    Parameters
    ----------
    volume_data : np.ndarray
        Volume data (3D or 4D)
    roi_mask : np.ndarray
        Boolean ROI mask
    affine : np.ndarray, optional
        Affine transformation matrix
        
    Returns
    -------
    SrfData
        ROI data in SrfData format
    """
    if roi_mask.shape != volume_data.shape[:3]:
        raise ValueError("ROI mask shape doesn't match volume data")
    
    # Get ROI coordinates
    roi_coords = np.where(roi_mask)
    n_voxels = len(roi_coords[0])
    
    # Create SrfData
    srf = SrfData('vol')
    srf.vertices = np.column_stack(roi_coords).astype(float)
    srf._volume_shape = volume_data.shape[:3]
    srf._mask = roi_mask
    
    if affine is not None:
        srf._affine = affine
    
    # Extract data
    if volume_data.ndim == 4:
        # Time series
        n_timepoints = volume_data.shape[3]
        srf.y = np.zeros((n_voxels, n_timepoints))
        
        for t in range(n_timepoints):
            srf.y[:, t] = volume_data[roi_coords + (t,)]
    else:
        # Single volume
        roi_data = volume_data[roi_coords]
        srf.add_data(roi_data.reshape(1, -1), ['roi_data'])
    
    return srf


def resample_volume(srf: SrfData, target_shape: Tuple[int, int, int],
                   target_affine: np.ndarray) -> SrfData:
    """
    Resample volume data to new resolution.
    
    Parameters
    ----------
    srf : SrfData
        Volume data to resample
    target_shape : tuple
        Target volume shape
    target_affine : np.ndarray
        Target affine matrix
        
    Returns
    -------
    SrfData
        Resampled volume data
    """
    if srf.hemisphere != 'vol':
        raise ValueError("Input must be volume data")
    
    # This would require more sophisticated resampling
    # For now, just raise an error to indicate it needs implementation
    raise NotImplementedError("Volume resampling not yet implemented")


def validate_volume_data(srf: SrfData) -> Dict[str, Any]:
    """
    Validate volume data integrity.
    
    Parameters
    ----------
    srf : SrfData
        Volume data to validate
        
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
    
    if srf.hemisphere != 'vol':
        validation['warnings'].append("Hemisphere is not 'vol'")
    
    # Check required volume attributes
    if not hasattr(srf, '_volume_shape'):
        validation['errors'].append("Missing volume shape information")
        validation['is_valid'] = False
    
    if not hasattr(srf, '_affine'):
        validation['warnings'].append("Missing affine transformation")
    
    # Check coordinates
    if srf.vertices is not None:
        if hasattr(srf, '_volume_shape'):
            coords = srf.vertices.astype(int)
            shape = srf._volume_shape
            
            out_of_bounds = (
                (coords[:, 0] < 0) | (coords[:, 0] >= shape[0]) |
                (coords[:, 1] < 0) | (coords[:, 1] >= shape[1]) |
                (coords[:, 2] < 0) | (coords[:, 2] >= shape[2])
            )
            
            if np.any(out_of_bounds):
                validation['warnings'].append(f"{np.sum(out_of_bounds)} voxels out of bounds")
    
    # Check data consistency
    if srf.vertices is not None:
        n_voxels = srf.vertices.shape[0]
        
        if srf.data is not None and srf.data.shape[1] != n_voxels:
            validation['errors'].append("Data size doesn't match number of voxels")
            validation['is_valid'] = False
        
        if srf.y is not None and srf.y.shape[0] != n_voxels:
            validation['errors'].append("Time series size doesn't match number of voxels")
            validation['is_valid'] = False
    
    validation['n_voxels'] = srf.vertices.shape[0] if srf.vertices is not None else 0
    validation['n_parameters'] = len(srf.values)
    validation['n_timepoints'] = srf.y.shape[1] if srf.y is not None else 0
    
    return validation