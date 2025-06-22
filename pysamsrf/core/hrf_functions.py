"""
Hemodynamic Response Function (HRF) modeling.

This module implements various HRF models and convolution functions
for population receptive field analysis.
"""

import numpy as np
from scipy import signal
from scipy.stats import gamma
from typing import Union, Optional, Tuple
import warnings


def doublegamma(tr: float, params: Optional[list] = None) -> np.ndarray:
    """
    Generate double-gamma hemodynamic response function.
    
    Equivalent to MATLAB samsrf_doublegamma function.
    Based on SPM canonical HRF implementation.
    
    Parameters
    ----------
    tr : float
        Repetition time in seconds
    params : list, optional
        HRF parameters [a1, a2, b1, b2, c, dt, hrf_length]
        Default: [6, 16, 1, 1, 6, 0, 32] (SPM canonical)
        
    Returns
    -------
    np.ndarray
        HRF time course
    """
    if params is None:
        # SPM canonical parameters
        params = [6, 16, 1, 1, 6, 0, 32]
    
    a1, a2, b1, b2, c, dt, hrf_length = params
    
    # Time vector
    time = np.arange(0, hrf_length + tr, tr)
    
    # Positive gamma function (neural response)
    hrf_pos = gamma.pdf(time, a1/b1, scale=b1)
    
    # Negative gamma function (vascular undershoot)  
    hrf_neg = gamma.pdf(time, a2/b2, scale=b2)
    
    # Combine responses
    hrf = hrf_pos - hrf_neg/c
    
    # Apply time shift
    if dt != 0:
        dt_samples = int(np.round(dt / tr))
        if dt_samples > 0:
            hrf = np.concatenate([np.zeros(dt_samples), hrf[:-dt_samples]])
        elif dt_samples < 0:
            hrf = np.concatenate([hrf[-dt_samples:], np.zeros(-dt_samples)])
    
    return hrf


def canonical_hrf(tr: float, hrf_type: str = 'spm') -> np.ndarray:
    """
    Generate canonical hemodynamic response function.
    
    Parameters
    ----------
    tr : float
        Repetition time in seconds
    hrf_type : str, default='spm'
        Type of canonical HRF ('spm', 'de_haas', 'glover')
        
    Returns
    -------
    np.ndarray
        Canonical HRF
    """
    if hrf_type.lower() == 'spm':
        # SPM canonical HRF
        return doublegamma(tr, [6, 16, 1, 1, 6, 0, 32])
    
    elif hrf_type.lower() == 'de_haas':
        # de Haas canonical HRF (SamSrf default for earlier versions)
        return doublegamma(tr, [5.4, 10.8, 1, 1, 6, 0, 32])
    
    elif hrf_type.lower() == 'glover':
        # Glover HRF parameters
        return doublegamma(tr, [6, 12, 0.9, 0.9, 0.35, 0, 32])
    
    else:
        raise ValueError(f"Unknown HRF type: {hrf_type}")


def convolve_hrf(timecourse: np.ndarray, hrf: np.ndarray, 
                 downsample: Optional[int] = None) -> np.ndarray:
    """
    Convolve neural time course with hemodynamic response function.
    
    Equivalent to MATLAB prf_convolve_hrf function.
    
    Parameters
    ----------
    timecourse : np.ndarray
        Neural response time course
    hrf : np.ndarray
        Hemodynamic response function
    downsample : int, optional
        Downsampling factor for microtime resolution
        
    Returns
    -------
    np.ndarray
        Convolved BOLD time course
    """
    # Handle downsampling (microtime resolution)
    if downsample is not None and downsample > 1:
        # Upsample time course
        upsampled = np.repeat(timecourse, downsample)
        
        # Convolve with HRF
        convolved = signal.convolve(upsampled, hrf, mode='full')
        
        # Extract relevant portion and downsample
        convolved = convolved[:len(upsampled)]
        convolved = convolved[::downsample]
        
        # Trim to original length
        convolved = convolved[:len(timecourse)]
    else:
        # Standard convolution
        convolved = signal.convolve(timecourse, hrf, mode='full')
        
        # Extract relevant portion (same length as input)
        convolved = convolved[:len(timecourse)]
    
    return convolved


def estimate_hrf_parameters(bold_data: np.ndarray, neural_data: np.ndarray,
                          tr: float, initial_params: Optional[list] = None) -> Tuple[np.ndarray, float]:
    """
    Estimate HRF parameters from BOLD and neural data.
    
    Parameters
    ----------
    bold_data : np.ndarray
        Observed BOLD time course
    neural_data : np.ndarray  
        Estimated neural time course
    tr : float
        Repetition time in seconds
    initial_params : list, optional
        Initial parameter guess [a1, a2, b1, b2, c]
        
    Returns
    -------
    fitted_params : np.ndarray
        Fitted HRF parameters
    r_squared : float
        Goodness of fit
    """
    from scipy.optimize import minimize
    
    if initial_params is None:
        initial_params = [6, 16, 1, 1, 6]  # SPM defaults (without dt and length)
    
    def objective(params):
        """Objective function for HRF parameter fitting."""
        try:
            # Generate HRF with current parameters
            hrf_params = list(params) + [0, 32]  # Add dt=0 and length=32
            hrf = doublegamma(tr, hrf_params)
            
            # Convolve neural signal with HRF
            predicted = convolve_hrf(neural_data, hrf)
            
            # Calculate correlation with observed BOLD
            correlation = np.corrcoef(predicted, bold_data)[0, 1]
            
            # Return negative correlation (for minimization)
            return -correlation if not np.isnan(correlation) else 1.0
            
        except:
            return 1.0  # Return poor fit on error
    
    # Parameter bounds (reasonable physiological ranges)
    bounds = [(3, 10),    # a1: peak time of positive response
              (10, 20),   # a2: peak time of negative response  
              (0.5, 2),   # b1: width of positive response
              (0.5, 2),   # b2: width of negative response
              (3, 10)]    # c: ratio of responses
    
    # Optimize parameters
    result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
    
    # Calculate final R²
    final_hrf = doublegamma(tr, list(result.x) + [0, 32])
    predicted_final = convolve_hrf(neural_data, final_hrf)
    r_squared = np.corrcoef(predicted_final, bold_data)[0, 1] ** 2
    
    return result.x, r_squared


def deconvolve_hrf(bold_data: np.ndarray, hrf: np.ndarray) -> np.ndarray:
    """
    Deconvolve BOLD signal to estimate neural activity.
    
    Parameters
    ----------
    bold_data : np.ndarray
        BOLD time course
    hrf : np.ndarray
        Hemodynamic response function
        
    Returns
    -------
    np.ndarray
        Estimated neural time course
    """
    # Use Wiener deconvolution for noise robustness
    # Add small regularization to prevent division by zero
    hrf_fft = np.fft.fft(hrf, len(bold_data))
    bold_fft = np.fft.fft(bold_data)
    
    # Wiener filter parameter (adjust for noise level)
    noise_power = 0.01
    
    # Deconvolution
    wiener_filter = np.conj(hrf_fft) / (np.abs(hrf_fft)**2 + noise_power)
    neural_fft = bold_fft * wiener_filter
    
    # Convert back to time domain
    neural_estimate = np.real(np.fft.ifft(neural_fft))
    
    return neural_estimate


def create_basis_set(tr: float, n_basis: int = 3, 
                     basis_type: str = 'canonical') -> np.ndarray:
    """
    Create HRF basis set for flexible modeling.
    
    Parameters
    ----------
    tr : float
        Repetition time in seconds
    n_basis : int, default=3
        Number of basis functions
    basis_type : str, default='canonical'
        Type of basis set ('canonical', 'fir', 'gamma')
        
    Returns
    -------
    np.ndarray
        Basis set matrix (time_points x n_basis)
    """
    if basis_type == 'canonical':
        # Canonical HRF + derivatives
        hrf = canonical_hrf(tr)
        
        if n_basis >= 1:
            basis = [hrf]
        if n_basis >= 2:
            # Temporal derivative
            dt = tr
            temporal_deriv = np.gradient(hrf, dt)
            basis.append(temporal_deriv)
        if n_basis >= 3:
            # Dispersion derivative
            hrf_disp = canonical_hrf(tr * 1.01)  # Slightly different timing
            disp_deriv = hrf_disp - hrf
            basis.append(disp_deriv)
        
        return np.column_stack(basis[:n_basis])
    
    elif basis_type == 'fir':
        # Finite impulse response basis
        n_timepoints = int(32 / tr)  # 32 second window
        basis = np.eye(n_timepoints)[:, :n_basis]
        return basis
    
    elif basis_type == 'gamma':
        # Multiple gamma functions with different parameters
        time = np.arange(0, 32 + tr, tr)
        basis = []
        
        for i in range(n_basis):
            # Vary peak time and width
            peak_time = 4 + i * 2  # 4, 6, 8, ... seconds
            width = 1 + i * 0.5    # 1, 1.5, 2, ... 
            
            gamma_func = gamma.pdf(time, peak_time/width, scale=width)
            basis.append(gamma_func)
        
        return np.column_stack(basis)
    
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")


def plot_hrf(hrf: np.ndarray, tr: float, title: str = "HRF") -> None:
    """
    Plot hemodynamic response function.
    
    Parameters
    ----------
    hrf : np.ndarray
        HRF time course
    tr : float
        Repetition time in seconds
    title : str, default="HRF"
        Plot title
    """
    try:
        import matplotlib.pyplot as plt
        
        time = np.arange(len(hrf)) * tr
        
        plt.figure(figsize=(8, 4))
        plt.plot(time, hrf, 'b-', linewidth=2)
        plt.xlabel('Time (s)')
        plt.ylabel('Response (a.u.)')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        
        # Mark peak
        peak_idx = np.argmax(hrf)
        plt.plot(time[peak_idx], hrf[peak_idx], 'ro', markersize=8)
        plt.text(time[peak_idx], hrf[peak_idx] + 0.02, 
                f'Peak: {time[peak_idx]:.1f}s', ha='center')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        warnings.warn("Matplotlib not available for plotting")


def validate_hrf(hrf: np.ndarray, tr: float) -> dict:
    """
    Validate HRF properties and return metrics.
    
    Parameters
    ----------
    hrf : np.ndarray
        HRF time course
    tr : float
        Repetition time in seconds
        
    Returns
    -------
    dict
        Validation metrics
    """
    time = np.arange(len(hrf)) * tr
    
    # Find peak
    peak_idx = np.argmax(hrf)
    peak_time = time[peak_idx]
    peak_amplitude = hrf[peak_idx]
    
    # Find undershoot
    undershoot_idx = np.argmin(hrf[peak_idx:]) + peak_idx
    undershoot_time = time[undershoot_idx]
    undershoot_amplitude = hrf[undershoot_idx]
    
    # Calculate metrics
    metrics = {
        'peak_time': peak_time,
        'peak_amplitude': peak_amplitude,
        'undershoot_time': undershoot_time,
        'undershoot_amplitude': undershoot_amplitude,
        'time_to_peak': peak_time,
        'fwhm': _calculate_fwhm(hrf, tr),
        'area_under_curve': np.trapz(hrf, dx=tr),
        'is_valid': _check_hrf_validity(hrf, peak_time, undershoot_amplitude)
    }
    
    return metrics


def _calculate_fwhm(hrf: np.ndarray, tr: float) -> float:
    """Calculate full width at half maximum of HRF."""
    peak_val = np.max(hrf)
    half_max = peak_val / 2
    
    # Find indices where HRF crosses half maximum
    above_half = hrf >= half_max
    if not np.any(above_half):
        return 0.0
    
    # Find first and last crossing points
    first_idx = np.where(above_half)[0][0]
    last_idx = np.where(above_half)[0][-1]
    
    fwhm = (last_idx - first_idx) * tr
    return fwhm


def _check_hrf_validity(hrf: np.ndarray, peak_time: float, 
                       undershoot_amplitude: float) -> bool:
    """Check if HRF has reasonable properties."""
    # Peak should occur between 3-8 seconds
    peak_valid = 3.0 <= peak_time <= 8.0
    
    # Should have positive peak
    positive_peak = np.max(hrf) > 0
    
    # Undershoot should be negative but not too large
    undershoot_valid = undershoot_amplitude < 0 and undershoot_amplitude > -0.5
    
    return peak_valid and positive_peak and undershoot_valid