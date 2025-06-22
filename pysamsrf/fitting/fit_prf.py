"""
Main pRF fitting pipeline.

This module implements the complete population receptive field fitting workflow,
equivalent to the MATLAB samsrf_fit_prf function.
"""

import numpy as np
import warnings
from typing import Optional, Union, Dict, Any, Callable, Tuple
from pathlib import Path
import time

from ..core.data_structures import SrfData, Model, PRFResults
from ..core.prf_functions import gaussian_rf, predict_timecourse, generate_prf_grid
from ..core.hrf_functions import canonical_hrf, convolve_hrf, doublegamma
from .optimizers import (
    error_function, grid_search, nelder_mead_wrapper, 
    hooke_jeeves, multi_start_optimization
)


def fit_prf(model: Model, srf_data: Union[SrfData, str], 
           roi: Optional[str] = None, 
           apertures: Optional[np.ndarray] = None,
           parallel: bool = True, n_jobs: int = -1,
           progress_callback: Optional[Callable] = None,
           verbose: bool = True) -> PRFResults:
    """
    Fit population receptive field models to neuroimaging data.
    
    Main fitting function equivalent to MATLAB samsrf_fit_prf.
    
    Parameters
    ----------
    model : Model
        Model configuration containing pRF parameters and fitting options
    srf_data : SrfData or str
        Surface/volume data structure or path to data file
    roi : str, optional
        Path to ROI file for analysis restriction
    apertures : np.ndarray, optional
        Stimulus apertures (if not specified in model.aperture_file)
    parallel : bool, default=True
        Whether to use parallel processing
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all CPUs)
    progress_callback : callable, optional
        Function to call for progress updates
    verbose : bool, default=True
        Whether to print progress information
        
    Returns
    -------
    PRFResults
        Fitted pRF parameters and goodness of fit metrics
    """
    if verbose:
        print("=" * 60)
        print("PySamSrf: Population Receptive Field Analysis")
        print("=" * 60)
        print(f"Model: {model.name}")
        print(f"Algorithm: Forward-modeling pRF fitting")
    
    # Load data if path provided
    if isinstance(srf_data, str):
        srf_data = _load_srf_data(srf_data)
    
    # Apply ROI filtering if specified
    if roi is not None:
        roi_indices = _load_roi(roi)
        srf_data = srf_data.filter_by_roi(roi_indices)
        if verbose:
            print(f"ROI: {len(roi_indices)} vertices")
    
    # Load stimulus apertures
    if apertures is None:
        apertures = _load_apertures(model.aperture_file)
    
    if verbose:
        print(f"Data: {srf_data.y.shape[0]} vertices, {srf_data.y.shape[1]} timepoints")
        print(f"Apertures: {apertures.shape[1]} timepoints")
    
    # Prepare HRF
    hrf = _prepare_hrf(model)
    
    # Setup pRF function
    prf_function = _setup_prf_function(model)
    
    # Configure fitting
    model_config = _setup_model_config(model, apertures.shape[0])
    
    # Run fitting
    if verbose:
        print(f"Starting pRF fitting...")
        start_time = time.time()
    
    results = _fit_vertices(
        srf_data.y, apertures, hrf, prf_function, model, model_config,
        parallel=parallel, n_jobs=n_jobs, progress_callback=progress_callback,
        verbose=verbose
    )
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"Fitting completed in {elapsed:.1f} seconds")
        print(f"Mean R²: {np.mean(results.r_squared):.3f}")
        print(f"Vertices with R² > 0.2: {np.sum(results.r_squared > 0.2)}")
    
    return results


def coarse_fit(srf_data: SrfData, apertures: np.ndarray, hrf: np.ndarray,
              prf_function: Callable, model: Model, model_config: Dict[str, Any],
              parallel: bool = True, n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform coarse fitting using grid search.
    
    Parameters
    ----------
    srf_data : SrfData
        Neuroimaging data
    apertures : np.ndarray
        Stimulus apertures
    hrf : np.ndarray
        Hemodynamic response function
    prf_function : callable
        pRF model function
    model : Model
        Model configuration
    model_config : dict
        Model configuration dictionary
    parallel : bool, default=True
        Use parallel processing
    n_jobs : int, default=-1
        Number of parallel jobs
        
    Returns
    -------
    coarse_params : np.ndarray
        Coarse fit parameters (n_params x n_vertices)
    coarse_errors : np.ndarray
        Coarse fit errors (n_vertices,)
    """
    n_vertices = srf_data.y.shape[0]
    n_params = len(model.param_names)
    
    # Generate search space
    search_grids = model.get_search_space()
    
    # Initialize results
    coarse_params = np.zeros((n_params, n_vertices))
    coarse_errors = np.ones(n_vertices)  # Start with maximum error
    
    # Fit each vertex
    for v in range(n_vertices):
        observed_bold = srf_data.y[v, :]
        
        def error_func(params):
            return error_function(
                params, prf_function, apertures, hrf, observed_bold, model_config
            )
        
        try:
            best_params, best_error, _ = grid_search(
                search_grids, error_func, parallel=False  # Per-vertex parallel in outer loop
            )
            
            coarse_params[:, v] = best_params
            coarse_errors[v] = best_error
            
        except Exception as e:
            warnings.warn(f"Coarse fit failed for vertex {v}: {e}")
            # Keep default values (zeros and max error)
    
    return coarse_params, coarse_errors


def fine_fit(coarse_params: np.ndarray, coarse_errors: np.ndarray,
            srf_data: SrfData, apertures: np.ndarray, hrf: np.ndarray,
            prf_function: Callable, model: Model, model_config: Dict[str, Any],
            parallel: bool = True, n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform fine fitting using optimization algorithms.
    
    Parameters
    ----------
    coarse_params : np.ndarray
        Initial parameters from coarse fit
    coarse_errors : np.ndarray
        Coarse fit errors
    srf_data : SrfData
        Neuroimaging data
    apertures : np.ndarray
        Stimulus apertures
    hrf : np.ndarray
        Hemodynamic response function
    prf_function : callable
        pRF model function
    model : Model
        Model configuration
    model_config : dict
        Model configuration dictionary
    parallel : bool, default=True
        Use parallel processing
    n_jobs : int, default=-1
        Number of parallel jobs
        
    Returns
    -------
    fine_params : np.ndarray
        Fine fit parameters (n_params x n_vertices)
    fine_errors : np.ndarray
        Fine fit errors (n_vertices,)
    """
    n_vertices = srf_data.y.shape[0]
    n_params = len(model.param_names)
    
    # Initialize with coarse fit results
    fine_params = coarse_params.copy()
    fine_errors = coarse_errors.copy()
    
    # Set up parameter bounds
    bounds = _get_parameter_bounds(model)
    
    # Choose optimization method
    use_hooke_jeeves = model.hooke_jeeves_steps > 0
    
    # Fit each vertex
    for v in range(n_vertices):
        # Skip if coarse fit was very poor
        if coarse_errors[v] > 0.95:
            continue
            
        observed_bold = srf_data.y[v, :]
        initial_params = coarse_params[:, v]
        
        def error_func(params):
            return error_function(
                params, prf_function, apertures, hrf, observed_bold, model_config
            )
        
        try:
            if use_hooke_jeeves:
                best_params, best_error = hooke_jeeves(
                    initial_params, error_func, bounds=bounds,
                    max_iterations=model.hooke_jeeves_steps
                )
            else:
                best_params, best_error = nelder_mead_wrapper(
                    initial_params, error_func, bounds=bounds
                )
            
            # Only update if improvement
            if best_error < fine_errors[v]:
                fine_params[:, v] = best_params
                fine_errors[v] = best_error
                
        except Exception as e:
            warnings.warn(f"Fine fit failed for vertex {v}: {e}")
            # Keep coarse fit results
    
    return fine_params, fine_errors


def _load_srf_data(filepath: str) -> SrfData:
    """Load SrfData from file."""
    # This would load from various formats (MAT, GII, NII)
    # For now, assume it's implemented elsewhere
    raise NotImplementedError("Data loading not yet implemented")


def _load_roi(roi_path: str) -> np.ndarray:
    """Load ROI vertex indices."""
    # This would load ROI from label files or masks
    # For now, return placeholder
    if roi_path.endswith('.label'):
        # FreeSurfer label format
        raise NotImplementedError("Label loading not yet implemented")
    elif roi_path.endswith('.nii'):
        # NIfTI mask format
        raise NotImplementedError("NIfTI ROI loading not yet implemented")
    else:
        raise ValueError(f"Unknown ROI format: {roi_path}")


def _load_apertures(aperture_file: str) -> np.ndarray:
    """Load stimulus apertures from file."""
    if not aperture_file:
        raise ValueError("No aperture file specified")
    
    # This would load from MAT files or other formats
    # For now, create synthetic apertures for testing
    warnings.warn("Using synthetic apertures - implement proper loading")
    
    # Create expanding ring stimulus
    aperture_width = 100
    n_timepoints = 50
    apertures = np.zeros((aperture_width**2, n_timepoints))
    
    for t in range(n_timepoints):
        radius = 1 + (t / n_timepoints) * 8
        y, x = np.mgrid[-50:50, -50:50]
        x = x * 10 / 50  # Scale to degrees
        y = y * 10 / 50
        ring = (np.sqrt(x**2 + y**2) < radius).flatten()
        apertures[:, t] = ring
    
    return apertures


def _prepare_hrf(model: Model) -> np.ndarray:
    """Prepare hemodynamic response function."""
    if model.hrf is None:
        # de Haas canonical HRF
        return canonical_hrf(model.tr, 'de_haas')
    elif model.hrf == 0:
        # SPM canonical HRF
        return canonical_hrf(model.tr, 'spm')
    elif model.hrf == 1:
        # No HRF convolution
        return np.array([1.0])
    elif isinstance(model.hrf, str):
        # Load HRF from file
        raise NotImplementedError("HRF file loading not yet implemented")
    elif isinstance(model.hrf, np.ndarray):
        # User-provided HRF
        return model.hrf
    elif np.isinf(model.hrf):
        # Fit HRF on the fly (not implemented yet)
        warnings.warn("HRF fitting not implemented, using SPM canonical")
        return canonical_hrf(model.tr, 'spm')
    else:
        raise ValueError(f"Unknown HRF specification: {model.hrf}")


def _setup_prf_function(model: Model) -> Callable:
    """Setup pRF function from model."""
    if model.prf_function is not None:
        return model.prf_function
    
    # Default to Gaussian pRF
    def default_gaussian_prf(*params, aperture_width):
        if len(params) >= 3:
            return gaussian_rf(params[0], params[1], params[2], aperture_width)
        else:
            raise ValueError("Insufficient parameters for Gaussian pRF")
    
    return default_gaussian_prf


def _setup_model_config(model: Model, aperture_size: int) -> Dict[str, Any]:
    """Setup model configuration dictionary."""
    return {
        'aperture_width': int(np.sqrt(aperture_size)),
        'percent_overlap': True,  # Use SamSrf model by default
        'compressive_nonlinearity': model.compressive_nonlinearity,
        'downsample': None,  # No microtime resolution by default
    }


def _get_parameter_bounds(model: Model) -> list:
    """Get parameter bounds for optimization."""
    bounds = []
    
    # Default bounds based on typical pRF analysis
    default_bounds = {
        'x0': (-20, 20),
        'y0': (-20, 20), 
        'sigma': (0.1, 10),
        'amplitude': (0, 2),
        'baseline': (-1, 1),
        'css_exponent': (0.1, 2),
    }
    
    for i, param_name in enumerate(model.param_names):
        if param_name in default_bounds:
            bounds.append(default_bounds[param_name])
        elif model.only_positive[i]:
            bounds.append((0.01, 20))
        else:
            bounds.append((-20, 20))
    
    return bounds


def _fit_vertices(observed_data: np.ndarray, apertures: np.ndarray, hrf: np.ndarray,
                 prf_function: Callable, model: Model, model_config: Dict[str, Any],
                 parallel: bool = True, n_jobs: int = -1,
                 progress_callback: Optional[Callable] = None,
                 verbose: bool = True) -> PRFResults:
    """
    Fit pRF models to all vertices.
    
    Parameters
    ----------
    observed_data : np.ndarray
        BOLD time series (n_vertices x n_timepoints)
    apertures : np.ndarray
        Stimulus apertures (n_pixels x n_timepoints)
    hrf : np.ndarray
        Hemodynamic response function
    prf_function : callable
        pRF model function
    model : Model
        Model configuration
    model_config : dict
        Model configuration dictionary
    parallel : bool, default=True
        Use parallel processing
    n_jobs : int, default=-1
        Number of parallel jobs
    progress_callback : callable, optional
        Progress callback function
    verbose : bool, default=True
        Print progress information
        
    Returns
    -------
    PRFResults
        Fitted parameters and metrics
    """
    n_vertices, n_timepoints = observed_data.shape
    n_params = len(model.param_names)
    
    # Create SrfData structure for fitting functions
    srf_temp = SrfData()
    srf_temp.y = observed_data
    
    if verbose:
        print("Phase 1: Coarse fitting (grid search)")
    
    # Coarse fitting
    coarse_params, coarse_errors = coarse_fit(
        srf_temp, apertures, hrf, prf_function, model, model_config,
        parallel=parallel, n_jobs=n_jobs
    )
    
    if progress_callback:
        progress_callback(0.5, "Coarse fitting completed")
    
    # Fine fitting (unless coarse-only mode)
    if model.coarse_fit_only:
        if verbose:
            print("Coarse-fit only mode: skipping fine fitting")
        final_params = coarse_params
        final_errors = coarse_errors
    else:
        if verbose:
            print("Phase 2: Fine fitting (optimization)")
        
        final_params, final_errors = fine_fit(
            coarse_params, coarse_errors, srf_temp, apertures, hrf,
            prf_function, model, model_config, parallel=parallel, n_jobs=n_jobs
        )
    
    if progress_callback:
        progress_callback(0.8, "Parameter fitting completed")
    
    # Calculate final R²
    r_squared = 1 - final_errors
    
    # Generate final predictions and calculate additional metrics
    if verbose:
        print("Phase 3: Generating predictions and metrics")
    
    predicted_timecourses = np.zeros((n_vertices, n_timepoints))
    
    for v in range(n_vertices):
        if r_squared[v] > 0:  # Only for successful fits
            try:
                params = final_params[:, v]
                rf = prf_function(*params, model_config['aperture_width'])
                
                neural = predict_timecourse(
                    rf.reshape(model_config['aperture_width'], model_config['aperture_width']),
                    apertures,
                    percent_overlap=model_config['percent_overlap']
                )
                
                if model_config['compressive_nonlinearity'] and len(params) > 3:
                    css_exp = params[-1] 
                    neural = neural ** css_exp
                
                if len(hrf) > 1:
                    predicted = convolve_hrf(neural, hrf)[:n_timepoints]
                else:
                    predicted = neural[:n_timepoints]
                
                predicted_timecourses[v, :] = predicted
                
            except Exception as e:
                warnings.warn(f"Prediction failed for vertex {v}: {e}")
    
    if progress_callback:
        progress_callback(1.0, "Analysis completed")
    
    # Create results structure
    results = PRFResults(
        parameters=final_params,
        r_squared=r_squared,
        predicted=predicted_timecourses,
        observed=observed_data,
        coarse_fit=coarse_params,
    )
    
    return results


def fit_prf_parallel(vertices_chunk: np.ndarray, apertures: np.ndarray,
                    hrf: np.ndarray, prf_function: Callable, model: Model,
                    model_config: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit pRF models to a chunk of vertices (for parallel processing).
    
    Parameters
    ----------
    vertices_chunk : np.ndarray
        BOLD time series for vertex chunk (n_vertices_chunk x n_timepoints)
    apertures : np.ndarray
        Stimulus apertures
    hrf : np.ndarray
        Hemodynamic response function
    prf_function : callable
        pRF model function
    model : Model
        Model configuration
    model_config : dict
        Model configuration dictionary
        
    Returns
    -------
    chunk_params : np.ndarray
        Fitted parameters for chunk
    chunk_r_squared : np.ndarray
        R² values for chunk
    """
    n_vertices_chunk = vertices_chunk.shape[0]
    n_params = len(model.param_names)
    
    chunk_params = np.zeros((n_params, n_vertices_chunk))
    chunk_r_squared = np.zeros(n_vertices_chunk)
    
    # Process each vertex in chunk
    for i, vertex_data in enumerate(vertices_chunk):
        try:
            # Create temporary SrfData for single vertex
            srf_temp = SrfData()
            srf_temp.y = vertex_data.reshape(1, -1)
            
            # Coarse fit
            coarse_params, coarse_errors = coarse_fit(
                srf_temp, apertures, hrf, prf_function, model, model_config,
                parallel=False  # No nested parallelization
            )
            
            # Fine fit (if not coarse-only)
            if not model.coarse_fit_only:
                final_params, final_errors = fine_fit(
                    coarse_params, coarse_errors, srf_temp, apertures, hrf,
                    prf_function, model, model_config, parallel=False
                )
            else:
                final_params = coarse_params
                final_errors = coarse_errors
            
            chunk_params[:, i] = final_params[:, 0]
            chunk_r_squared[i] = 1 - final_errors[0]
            
        except Exception as e:
            warnings.warn(f"Fitting failed for vertex in chunk: {e}")
            # Keep default values (zeros)
    
    return chunk_params, chunk_r_squared


def validate_fit_quality(results: PRFResults, model: Model,
                        min_r_squared: float = 0.05) -> Dict[str, Any]:
    """
    Validate pRF fitting results and generate quality metrics.
    
    Parameters
    ----------
    results : PRFResults
        Fitting results
    model : Model
        Model configuration
    min_r_squared : float, default=0.05
        Minimum R² threshold for valid fits
        
    Returns
    -------
    dict
        Quality metrics and validation results
    """
    n_vertices = results.parameters.shape[1]
    
    # Basic quality metrics
    valid_fits = results.r_squared >= min_r_squared
    n_valid = np.sum(valid_fits)
    
    quality_metrics = {
        'n_vertices': n_vertices,
        'n_valid_fits': n_valid,
        'fraction_valid': n_valid / n_vertices,
        'mean_r_squared': np.mean(results.r_squared),
        'median_r_squared': np.median(results.r_squared),
        'mean_r_squared_valid': np.mean(results.r_squared[valid_fits]) if n_valid > 0 else 0,
    }
    
    # Parameter statistics (for valid fits only)
    if n_valid > 0:
        for i, param_name in enumerate(model.param_names):
            param_values = results.parameters[i, valid_fits]
            quality_metrics[f'{param_name}_mean'] = np.mean(param_values)
            quality_metrics[f'{param_name}_std'] = np.std(param_values)
            quality_metrics[f'{param_name}_range'] = [np.min(param_values), np.max(param_values)]
    
    # Check for potential issues
    warnings_list = []
    
    if quality_metrics['fraction_valid'] < 0.1:
        warnings_list.append("Very low fraction of valid fits (<10%)")
    
    if quality_metrics['mean_r_squared'] < 0.1:
        warnings_list.append("Very low mean R² (<0.1)")
    
    # Check for suspicious parameter values
    if n_valid > 0:
        sigma_values = results.parameters[2, valid_fits]  # Assuming sigma is 3rd parameter
        if np.mean(sigma_values) > 10:
            warnings_list.append("Unusually large pRF sizes (mean sigma > 10°)")
        if np.std(sigma_values) < 0.1:
            warnings_list.append("Very low variability in pRF sizes")
    
    quality_metrics['warnings'] = warnings_list
    
    return quality_metrics