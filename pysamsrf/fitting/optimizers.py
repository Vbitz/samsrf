"""
Optimization algorithms for pRF fitting.

This module implements various optimization algorithms used in
population receptive field parameter estimation.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Callable, Tuple, Optional, List, Dict, Any
import warnings
from ..core.prf_functions import predict_timecourse
from ..core.hrf_functions import convolve_hrf


def error_function(params: np.ndarray, prf_func: Callable, apertures: np.ndarray,
                  hrf: np.ndarray, observed_data: np.ndarray, 
                  model_config: Dict[str, Any]) -> float:
    """
    Error function for pRF parameter optimization.
    
    Equivalent to MATLAB prf_errfun function.
    
    Parameters
    ----------
    params : np.ndarray
        pRF parameters to evaluate
    prf_func : callable
        Function to generate pRF from parameters
    apertures : np.ndarray
        Stimulus apertures (aperture_pixels x timepoints)
    hrf : np.ndarray
        Hemodynamic response function
    observed_data : np.ndarray
        Observed BOLD time course
    model_config : dict
        Model configuration options
        
    Returns
    -------
    float
        Error value (1 - R²)
    """
    try:
        # Generate pRF with current parameters
        aperture_width = model_config.get('aperture_width', 200)
        rf = prf_func(*params, aperture_width)
        
        # Predict neural time course
        predicted_neural = predict_timecourse(
            rf.reshape(int(aperture_width), int(aperture_width)), 
            apertures,
            percent_overlap=model_config.get('percent_overlap', True)
        )
        
        # Apply compressive nonlinearity if specified
        if model_config.get('compressive_nonlinearity', False) and len(params) > 3:
            css_exponent = params[-1]
            if css_exponent > 0:
                predicted_neural = predicted_neural ** css_exponent
        
        # Convolve with HRF
        if len(hrf) > 1:
            predicted_bold = convolve_hrf(
                predicted_neural, hrf, 
                downsample=model_config.get('downsample', None)
            )
        else:
            predicted_bold = predicted_neural
        
        # Ensure same length as observed data
        min_length = min(len(predicted_bold), len(observed_data))
        predicted_bold = predicted_bold[:min_length]
        observed_trimmed = observed_data[:min_length]
        
        # Calculate correlation
        if np.std(predicted_bold) == 0 or np.std(observed_trimmed) == 0:
            correlation = 0.0
        else:
            correlation = np.corrcoef(predicted_bold, observed_trimmed)[0, 1]
        
        # Handle NaN correlations
        if np.isnan(correlation):
            correlation = 0.0
        
        # Return 1 - R² (minimize negative R²)
        r_squared = correlation ** 2
        return 1.0 - r_squared
        
    except Exception as e:
        # Return high error for invalid parameters
        return 1.0


def nelder_mead_wrapper(initial_params: np.ndarray, error_func: Callable,
                       bounds: Optional[List[Tuple[float, float]]] = None,
                       max_iterations: int = 1000, tolerance: float = 1e-6) -> Tuple[np.ndarray, float]:
    """
    Wrapper for SciPy's Nelder-Mead optimization.
    
    Parameters
    ----------
    initial_params : np.ndarray
        Initial parameter guess
    error_func : callable
        Error function to minimize
    bounds : list of tuples, optional
        Parameter bounds [(min, max), ...]
    max_iterations : int, default=1000
        Maximum number of iterations
    tolerance : float, default=1e-6
        Convergence tolerance
        
    Returns
    -------
    best_params : np.ndarray
        Optimized parameters
    best_error : float
        Final error value
    """
    # Configure optimization options
    options = {
        'maxiter': max_iterations,
        'xatol': tolerance,
        'fatol': tolerance,
        'adaptive': True
    }
    
    try:
        if bounds is not None:
            # Use bounded optimization
            result = minimize(
                error_func, initial_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iterations, 'ftol': tolerance}
            )
        else:
            # Use unbounded Nelder-Mead
            result = minimize(
                error_func, initial_params,
                method='Nelder-Mead',
                options=options
            )
        
        return result.x, result.fun
        
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}")
        return initial_params, error_func(initial_params)


def hooke_jeeves(initial_params: np.ndarray, error_func: Callable,
                step_sizes: Optional[np.ndarray] = None,
                bounds: Optional[List[Tuple[float, float]]] = None,
                max_iterations: int = 1000, tolerance: float = 1e-6,
                step_reduction: float = 0.5) -> Tuple[np.ndarray, float]:
    """
    Hooke-Jeeves pattern search optimization algorithm.
    
    Equivalent to MATLAB samsrf_hookejeeves function.
    Direct search method that doesn't require derivatives.
    
    Parameters
    ----------
    initial_params : np.ndarray
        Initial parameter guess
    error_func : callable
        Error function to minimize
    step_sizes : np.ndarray, optional
        Initial step sizes for each parameter
    bounds : list of tuples, optional
        Parameter bounds [(min, max), ...]
    max_iterations : int, default=1000
        Maximum number of iterations
    tolerance : float, default=1e-6
        Convergence tolerance
    step_reduction : float, default=0.5
        Factor to reduce step sizes when no improvement
        
    Returns
    -------
    best_params : np.ndarray
        Optimized parameters
    best_error : float
        Final error value
    """
    current_params = initial_params.copy()
    n_params = len(current_params)
    
    # Initialize step sizes
    if step_sizes is None:
        step_sizes = np.abs(current_params) * 0.1
        step_sizes[step_sizes == 0] = 0.1  # Default for zero parameters
    else:
        step_sizes = step_sizes.copy()
    
    # Set up bounds
    if bounds is None:
        bounds = [(-np.inf, np.inf)] * n_params
    
    # Evaluate initial point
    current_error = error_func(current_params)
    best_params = current_params.copy()
    best_error = current_error
    
    # Main optimization loop
    for iteration in range(max_iterations):
        improved = False
        
        # Exploratory moves: try each coordinate direction
        for i in range(n_params):
            # Try positive step
            test_params = current_params.copy()
            test_params[i] += step_sizes[i]
            
            # Apply bounds
            test_params[i] = np.clip(test_params[i], bounds[i][0], bounds[i][1])
            
            test_error = error_func(test_params)
            
            if test_error < current_error:
                current_params = test_params
                current_error = test_error
                improved = True
            else:
                # Try negative step
                test_params = current_params.copy()
                test_params[i] -= step_sizes[i]
                
                # Apply bounds
                test_params[i] = np.clip(test_params[i], bounds[i][0], bounds[i][1])
                
                test_error = error_func(test_params)
                
                if test_error < current_error:
                    current_params = test_params
                    current_error = test_error
                    improved = True
        
        # Pattern move: if improvement found, try larger step in same direction
        if improved:
            direction = current_params - best_params
            pattern_params = current_params + direction
            
            # Apply bounds
            for i in range(n_params):
                pattern_params[i] = np.clip(pattern_params[i], bounds[i][0], bounds[i][1])
            
            pattern_error = error_func(pattern_params)
            
            if pattern_error < current_error:
                best_params = current_params.copy()
                current_params = pattern_params
                current_error = pattern_error
            else:
                best_params = current_params.copy()
            
            best_error = current_error
        else:
            # No improvement: reduce step sizes
            step_sizes *= step_reduction
            
            # Check convergence
            if np.max(step_sizes) < tolerance:
                break
    
    return best_params, best_error


def grid_search(param_grids: List[np.ndarray], error_func: Callable,
               parallel: bool = True, n_jobs: int = -1) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Exhaustive grid search over parameter space.
    
    Used for coarse fitting stage in pRF analysis.
    
    Parameters
    ----------
    param_grids : list of arrays
        Parameter grids to search over
    error_func : callable
        Error function to evaluate
    parallel : bool, default=True
        Whether to use parallel processing
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all CPUs)
        
    Returns
    -------
    best_params : np.ndarray
        Best parameter combination
    best_error : float
        Best error value
    all_errors : np.ndarray
        Error values for all parameter combinations
    """
    # Create parameter combinations
    grids = np.meshgrid(*param_grids, indexing='ij')
    param_combinations = np.stack([g.flatten() for g in grids], axis=1)
    
    n_combinations = param_combinations.shape[0]
    
    if parallel and n_combinations > 100:
        # Use parallel processing for large searches
        try:
            from joblib import Parallel, delayed
            
            def evaluate_params(params):
                return error_func(params)
            
            # Parallel evaluation
            errors = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_params)(params) for params in param_combinations
            )
            errors = np.array(errors)
            
        except ImportError:
            warnings.warn("Joblib not available, using sequential processing")
            parallel = False
    
    if not parallel:
        # Sequential evaluation
        errors = np.zeros(n_combinations)
        for i, params in enumerate(param_combinations):
            errors[i] = error_func(params)
    
    # Find best parameters
    best_idx = np.argmin(errors)
    best_params = param_combinations[best_idx]
    best_error = errors[best_idx]
    
    return best_params, best_error, errors


def multi_start_optimization(error_func: Callable, param_bounds: List[Tuple[float, float]],
                           n_starts: int = 10, method: str = 'nelder_mead') -> Tuple[np.ndarray, float]:
    """
    Multi-start optimization to avoid local minima.
    
    Parameters
    ----------
    error_func : callable
        Error function to minimize
    param_bounds : list of tuples
        Parameter bounds [(min, max), ...]
    n_starts : int, default=10
        Number of random starting points
    method : str, default='nelder_mead'
        Optimization method ('nelder_mead' or 'hooke_jeeves')
        
    Returns
    -------
    best_params : np.ndarray
        Best parameters across all starts
    best_error : float
        Best error value
    """
    n_params = len(param_bounds)
    best_params = None
    best_error = np.inf
    
    for start in range(n_starts):
        # Generate random starting point
        initial_params = np.zeros(n_params)
        for i, (low, high) in enumerate(param_bounds):
            if np.isfinite(low) and np.isfinite(high):
                initial_params[i] = np.random.uniform(low, high)
            elif np.isfinite(low):
                initial_params[i] = low + np.random.exponential(1.0)
            elif np.isfinite(high):
                initial_params[i] = high - np.random.exponential(1.0)
            else:
                initial_params[i] = np.random.normal(0, 1)
        
        # Optimize from this starting point
        try:
            if method == 'nelder_mead':
                params, error = nelder_mead_wrapper(initial_params, error_func, param_bounds)
            elif method == 'hooke_jeeves':
                params, error = hooke_jeeves(initial_params, error_func, bounds=param_bounds)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
            
            # Update best if improved
            if error < best_error:
                best_params = params
                best_error = error
                
        except Exception as e:
            warnings.warn(f"Optimization start {start} failed: {e}")
            continue
    
    if best_params is None:
        raise RuntimeError("All optimization attempts failed")
    
    return best_params, best_error


def adaptive_grid_search(initial_grids: List[np.ndarray], error_func: Callable,
                        n_iterations: int = 3, refinement_factor: int = 3) -> Tuple[np.ndarray, float]:
    """
    Adaptive grid search that refines around best solutions.
    
    Parameters
    ----------
    initial_grids : list of arrays
        Initial parameter grids
    error_func : callable
        Error function to evaluate
    n_iterations : int, default=3
        Number of refinement iterations
    refinement_factor : int, default=3
        Factor by which to refine grid around best solution
        
    Returns
    -------
    best_params : np.ndarray
        Best parameters found
    best_error : float
        Best error value
    """
    current_grids = [grid.copy() for grid in initial_grids]
    
    for iteration in range(n_iterations):
        # Search current grid
        best_params, best_error, _ = grid_search(current_grids, error_func)
        
        if iteration < n_iterations - 1:  # Don't refine on last iteration
            # Refine grids around best solution
            refined_grids = []
            
            for i, (grid, best_val) in enumerate(zip(current_grids, best_params)):
                # Find closest grid point
                closest_idx = np.argmin(np.abs(grid - best_val))
                
                # Create refined grid around this point
                if closest_idx > 0 and closest_idx < len(grid) - 1:
                    # Use neighboring points to define new range
                    grid_spacing = np.diff(grid).mean()
                    new_min = grid[closest_idx - 1]
                    new_max = grid[closest_idx + 1]
                    
                    # Create finer grid
                    refined_grid = np.linspace(new_min, new_max, 
                                             len(grid) * refinement_factor)
                else:
                    # Use original grid if at boundary
                    refined_grid = grid
                
                refined_grids.append(refined_grid)
            
            current_grids = refined_grids
    
    return best_params, best_error