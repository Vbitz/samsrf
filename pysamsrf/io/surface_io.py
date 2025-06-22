"""
Surface data input/output functions.

This module handles loading and saving of surface data in various formats
including GIFTI, FreeSurfer, and other neuroimaging surface formats.
"""

import numpy as np
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
import nibabel as nib

from ..core.data_structures import SrfData


def load_surface(filepath: Union[str, Path], 
                load_data: bool = True) -> SrfData:
    """
    Load surface data from various formats.
    
    Parameters
    ----------
    filepath : str or Path
        Path to surface file (.gii, .srf, .surf, etc.)
    load_data : bool, default=True
        Whether to load functional data along with geometry
        
    Returns
    -------
    SrfData
        Loaded surface data structure
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.gii':
        return _load_gifti(filepath, load_data)
    elif suffix == '.srf':
        return _load_samsrf_format(filepath)
    elif suffix in ['.surf', '.pial', '.white', '.inflated']:
        return _load_freesurfer_surface(filepath)
    else:
        raise ValueError(f"Unsupported surface format: {suffix}")


def save_surface(srf: SrfData, filepath: Union[str, Path],
                format: Optional[str] = None) -> None:
    """
    Save surface data to file.
    
    Parameters
    ----------
    srf : SrfData
        Surface data to save
    filepath : str or Path
        Output file path
    format : str, optional
        Output format ('gifti', 'samsrf', 'freesurfer')
        If None, inferred from file extension
    """
    filepath = Path(filepath)
    
    if format is None:
        suffix = filepath.suffix.lower()
        if suffix == '.gii':
            format = 'gifti'
        elif suffix == '.srf':
            format = 'samsrf'
        elif suffix == '.surf':
            format = 'freesurfer'
        else:
            raise ValueError(f"Cannot infer format from extension: {suffix}")
    
    if format == 'gifti':
        _save_gifti(srf, filepath)
    elif format == 'samsrf':
        _save_samsrf_format(srf, filepath)
    elif format == 'freesurfer':
        _save_freesurfer_surface(srf, filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_label(filepath: Union[str, Path]) -> np.ndarray:
    """
    Load ROI label file.
    
    Parameters
    ----------
    filepath : str or Path
        Path to label file (.label, .annot, etc.)
        
    Returns
    -------
    np.ndarray
        Vertex indices in the ROI
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.label':
        return _load_freesurfer_label(filepath)
    elif suffix == '.annot':
        return _load_freesurfer_annotation(filepath)
    else:
        raise ValueError(f"Unsupported label format: {suffix}")


def save_label(vertices: np.ndarray, filepath: Union[str, Path],
              coordinates: Optional[np.ndarray] = None) -> None:
    """
    Save ROI label file.
    
    Parameters
    ----------
    vertices : np.ndarray
        Vertex indices in the ROI
    filepath : str or Path
        Output file path
    coordinates : np.ndarray, optional
        Vertex coordinates (required for FreeSurfer labels)
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.label':
        _save_freesurfer_label(vertices, filepath, coordinates)
    else:
        raise ValueError(f"Unsupported label format: {suffix}")


def _load_gifti(filepath: Path, load_data: bool = True) -> SrfData:
    """Load GIFTI surface file."""
    try:
        gii = nib.load(str(filepath))
    except Exception as e:
        raise IOError(f"Failed to load GIFTI file {filepath}: {e}")
    
    # Determine hemisphere from filename
    filename = filepath.name.lower()
    if filename.startswith('lh'):
        hemisphere = 'lh'
    elif filename.startswith('rh'):
        hemisphere = 'rh'
    elif 'left' in filename:
        hemisphere = 'lh'
    elif 'right' in filename:
        hemisphere = 'rh'
    else:
        hemisphere = 'unknown'
    
    srf = SrfData(hemisphere)
    
    # Load geometry and data from GIFTI
    for darray in gii.darrays:
        intent = darray.intent
        
        if intent == nib.nifti1.intent_codes['NIFTI_INTENT_POINTSET']:
            # Vertex coordinates
            srf.vertices = darray.data
            
        elif intent == nib.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE']:
            # Face connectivity
            srf.faces = darray.data
            
        elif intent == nib.nifti1.intent_codes['NIFTI_INTENT_SHAPE']:
            # Curvature or other shape metric
            srf.curvature = darray.data
            
        elif intent == nib.nifti1.intent_codes['NIFTI_INTENT_TIME_SERIES']:
            # Time series data
            if load_data:
                if srf.y is None:
                    srf.y = darray.data.T  # Transpose to (vertices, timepoints)
                else:
                    srf.y = np.concatenate([srf.y, darray.data.T], axis=1)
                    
        elif intent == nib.nifti1.intent_codes['NIFTI_INTENT_ESTIMATE']:
            # Parameter estimates
            if load_data:
                if hasattr(darray, 'meta') and 'Name' in darray.meta:
                    param_name = darray.meta['Name']
                else:
                    param_name = f"param_{len(srf.values)}"
                
                srf.set_parameter(param_name, darray.data)
    
    # Set functional file info
    srf.functional = [str(filepath)]
    
    return srf


def _save_gifti(srf: SrfData, filepath: Path) -> None:
    """Save surface data to GIFTI format."""
    darrays = []
    
    # Save vertex coordinates
    if srf.vertices is not None:
        coord_darray = nib.gifti.GiftiDataArray(
            data=srf.vertices.astype(np.float32),
            intent='NIFTI_INTENT_POINTSET',
            datatype='NIFTI_TYPE_FLOAT32'
        )
        darrays.append(coord_darray)
    
    # Save face connectivity
    if srf.faces is not None:
        face_darray = nib.gifti.GiftiDataArray(
            data=srf.faces.astype(np.int32),
            intent='NIFTI_INTENT_TRIANGLE',
            datatype='NIFTI_TYPE_INT32'
        )
        darrays.append(face_darray)
    
    # Save parameter data
    if srf.data is not None:
        for i, param_name in enumerate(srf.values):
            param_darray = nib.gifti.GiftiDataArray(
                data=srf.data[i, :].astype(np.float32),
                intent='NIFTI_INTENT_ESTIMATE',
                datatype='NIFTI_TYPE_FLOAT32'
            )
            param_darray.meta = nib.gifti.GiftiMetaData({'Name': param_name})
            darrays.append(param_darray)
    
    # Save time series data
    if srf.y is not None:
        ts_darray = nib.gifti.GiftiDataArray(
            data=srf.y.T.astype(np.float32),  # Transpose back to (timepoints, vertices)
            intent='NIFTI_INTENT_TIME_SERIES',
            datatype='NIFTI_TYPE_FLOAT32'
        )
        darrays.append(ts_darray)
    
    # Create GIFTI image
    gii = nib.gifti.GiftiImage(darrays=darrays)
    
    # Save to file
    nib.save(gii, str(filepath))


def _load_samsrf_format(filepath: Path) -> SrfData:
    """Load SamSrf .srf format (MATLAB .mat file)."""
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy is required to load .srf files")
    
    try:
        mat_data = loadmat(str(filepath), struct_as_record=False, squeeze_me=True)
    except Exception as e:
        raise IOError(f"Failed to load SRF file {filepath}: {e}")
    
    if 'Srf' not in mat_data:
        raise ValueError("SRF file does not contain 'Srf' structure")
    
    srf_struct = mat_data['Srf']
    srf = SrfData()
    
    # Load basic fields
    if hasattr(srf_struct, 'Hemisphere'):
        srf.hemisphere = str(srf_struct.Hemisphere)
    if hasattr(srf_struct, 'Version'):
        srf.version = f"Converted from SamSrf {srf_struct.Version}"
    
    # Load geometry
    if hasattr(srf_struct, 'Vertices'):
        srf.vertices = np.array(srf_struct.Vertices)
    if hasattr(srf_struct, 'Faces'):
        srf.faces = np.array(srf_struct.Faces) - 1  # Convert to 0-indexed
    if hasattr(srf_struct, 'Curvature'):
        srf.curvature = np.array(srf_struct.Curvature)
    
    # Load data
    if hasattr(srf_struct, 'Data'):
        srf.data = np.array(srf_struct.Data)
    if hasattr(srf_struct, 'Values'):
        if isinstance(srf_struct.Values, np.ndarray):
            srf.values = [str(v) for v in srf_struct.Values]
        else:
            srf.values = [str(srf_struct.Values)]
    
    # Load time series
    if hasattr(srf_struct, 'Y'):
        srf.y = np.array(srf_struct.Y)
    if hasattr(srf_struct, 'X'):
        srf.x = np.array(srf_struct.X)
    
    # Load additional fields
    if hasattr(srf_struct, 'Functional'):
        if isinstance(srf_struct.Functional, np.ndarray):
            srf.functional = [str(f) for f in srf_struct.Functional]
        else:
            srf.functional = [str(srf_struct.Functional)]
    
    return srf


def _save_samsrf_format(srf: SrfData, filepath: Path) -> None:
    """Save surface data to SamSrf .srf format."""
    try:
        from scipy.io import savemat
    except ImportError:
        raise ImportError("scipy is required to save .srf files")
    
    # Prepare data dictionary
    srf_dict = {
        'Hemisphere': srf.hemisphere,
        'Version': srf.version,
    }
    
    # Add geometry
    if srf.vertices is not None:
        srf_dict['Vertices'] = srf.vertices
    if srf.faces is not None:
        srf_dict['Faces'] = srf.faces + 1  # Convert to 1-indexed for MATLAB
    if srf.curvature is not None:
        srf_dict['Curvature'] = srf.curvature
    
    # Add data
    if srf.data is not None:
        srf_dict['Data'] = srf.data
    if srf.values:
        srf_dict['Values'] = srf.values
    
    # Add time series
    if srf.y is not None:
        srf_dict['Y'] = srf.y
    if srf.x is not None:
        srf_dict['X'] = srf.x
    
    # Add functional info
    if srf.functional:
        srf_dict['Functional'] = srf.functional
    
    # Save to file
    savemat(str(filepath), {'Srf': srf_dict}, oned_as='column')


def _load_freesurfer_surface(filepath: Path) -> SrfData:
    """Load FreeSurfer surface file."""
    try:
        vertices, faces = nib.freesurfer.read_geometry(str(filepath))
    except Exception as e:
        raise IOError(f"Failed to load FreeSurfer surface {filepath}: {e}")
    
    # Determine hemisphere
    filename = filepath.name.lower()
    if filename.startswith('lh'):
        hemisphere = 'lh'
    elif filename.startswith('rh'):
        hemisphere = 'rh'
    else:
        hemisphere = 'unknown'
    
    srf = SrfData(hemisphere)
    srf.vertices = vertices
    srf.faces = faces
    
    return srf


def _save_freesurfer_surface(srf: SrfData, filepath: Path) -> None:
    """Save surface to FreeSurfer format."""
    if srf.vertices is None or srf.faces is None:
        raise ValueError("Both vertices and faces are required for FreeSurfer format")
    
    try:
        nib.freesurfer.write_geometry(str(filepath), srf.vertices, srf.faces)
    except Exception as e:
        raise IOError(f"Failed to save FreeSurfer surface {filepath}: {e}")


def _load_freesurfer_label(filepath: Path) -> np.ndarray:
    """Load FreeSurfer label file."""
    try:
        vertex_ids = nib.freesurfer.read_label(str(filepath))
    except Exception as e:
        raise IOError(f"Failed to load FreeSurfer label {filepath}: {e}")
    
    return vertex_ids


def _save_freesurfer_label(vertices: np.ndarray, filepath: Path,
                          coordinates: Optional[np.ndarray] = None) -> None:
    """Save FreeSurfer label file."""
    if coordinates is None:
        # Create dummy coordinates
        coordinates = np.zeros((len(vertices), 3))
        values = np.zeros(len(vertices))
    else:
        values = np.zeros(len(vertices))
    
    try:
        nib.freesurfer.write_label(str(filepath), vertices, coordinates, values)
    except Exception as e:
        raise IOError(f"Failed to save FreeSurfer label {filepath}: {e}")


def _load_freesurfer_annotation(filepath: Path) -> np.ndarray:
    """Load FreeSurfer annotation file."""
    try:
        labels, ctab, names = nib.freesurfer.read_annot(str(filepath))
        # Return vertices with non-zero labels
        return np.where(labels > 0)[0]
    except Exception as e:
        raise IOError(f"Failed to load FreeSurfer annotation {filepath}: {e}")


def load_multiple_surfaces(filepaths: list, hemisphere: str = 'auto') -> SrfData:
    """
    Load and combine multiple surface files.
    
    Parameters
    ----------
    filepaths : list
        List of surface file paths
    hemisphere : str, default='auto'
        Hemisphere specification ('lh', 'rh', 'bi', 'auto')
        
    Returns
    -------
    SrfData
        Combined surface data
    """
    if not filepaths:
        raise ValueError("No files provided")
    
    # Load first file
    combined_srf = load_surface(filepaths[0])
    
    # Set hemisphere
    if hemisphere != 'auto':
        combined_srf.hemisphere = hemisphere
    
    # Combine remaining files
    for filepath in filepaths[1:]:
        srf = load_surface(filepath)
        
        # Combine time series data
        if srf.y is not None:
            if combined_srf.y is None:
                combined_srf.y = srf.y
            else:
                combined_srf.y = np.concatenate([combined_srf.y, srf.y], axis=1)
        
        # Add to functional file list
        if srf.functional:
            combined_srf.functional.extend(srf.functional)
    
    return combined_srf


def combine_hemispheres(lh_srf: SrfData, rh_srf: SrfData) -> SrfData:
    """
    Combine left and right hemisphere surfaces.
    
    Parameters
    ----------
    lh_srf : SrfData
        Left hemisphere surface data
    rh_srf : SrfData
        Right hemisphere surface data
        
    Returns
    -------
    SrfData
        Combined bilateral surface data
    """
    combined_srf = SrfData('bi')
    
    # Combine vertices
    if lh_srf.vertices is not None and rh_srf.vertices is not None:
        combined_srf.vertices = np.vstack([lh_srf.vertices, rh_srf.vertices])
    
    # Combine faces (adjust indices for right hemisphere)
    if lh_srf.faces is not None and rh_srf.faces is not None:
        n_lh_vertices = lh_srf.vertices.shape[0]
        rh_faces_adjusted = rh_srf.faces + n_lh_vertices
        combined_srf.faces = np.vstack([lh_srf.faces, rh_faces_adjusted])
    
    # Combine data
    if lh_srf.data is not None and rh_srf.data is not None:
        combined_srf.data = np.hstack([lh_srf.data, rh_srf.data])
        combined_srf.values = lh_srf.values  # Assume same parameters
    
    # Combine time series
    if lh_srf.y is not None and rh_srf.y is not None:
        combined_srf.y = np.vstack([lh_srf.y, rh_srf.y])
    
    # Combine other arrays
    for attr in ['curvature', 'area', 'thickness']:
        lh_data = getattr(lh_srf, attr)
        rh_data = getattr(rh_srf, attr)
        if lh_data is not None and rh_data is not None:
            setattr(combined_srf, attr, np.concatenate([lh_data, rh_data]))
    
    # Combine functional info
    combined_srf.functional = lh_srf.functional + rh_srf.functional
    
    return combined_srf


def validate_surface_data(srf: SrfData) -> Dict[str, Any]:
    """
    Validate surface data integrity.
    
    Parameters
    ----------
    srf : SrfData
        Surface data to validate
        
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
    
    # Check basic geometry
    if srf.vertices is None:
        validation['errors'].append("No vertex coordinates")
        validation['is_valid'] = False
    else:
        n_vertices = srf.vertices.shape[0]
        
        if srf.vertices.shape[1] != 3:
            validation['errors'].append("Vertices should have 3 coordinates")
            validation['is_valid'] = False
    
    if srf.faces is not None:
        if srf.faces.shape[1] != 3:
            validation['errors'].append("Faces should be triangular")
            validation['is_valid'] = False
        
        max_vertex_idx = np.max(srf.faces)
        if srf.vertices is not None and max_vertex_idx >= srf.vertices.shape[0]:
            validation['errors'].append("Face indices exceed vertex count")
            validation['is_valid'] = False
    
    # Check data consistency
    if srf.data is not None and srf.vertices is not None:
        if srf.data.shape[1] != srf.vertices.shape[0]:
            validation['errors'].append("Data and vertex count mismatch")
            validation['is_valid'] = False
        
        if len(srf.values) != srf.data.shape[0]:
            validation['warnings'].append("Number of value names doesn't match data rows")
    
    # Check time series
    if srf.y is not None and srf.vertices is not None:
        if srf.y.shape[0] != srf.vertices.shape[0]:
            validation['errors'].append("Time series and vertex count mismatch")
            validation['is_valid'] = False
    
    validation['n_vertices'] = srf.vertices.shape[0] if srf.vertices is not None else 0
    validation['n_faces'] = srf.faces.shape[0] if srf.faces is not None else 0
    validation['n_parameters'] = len(srf.values)
    validation['n_timepoints'] = srf.y.shape[1] if srf.y is not None else 0
    
    return validation