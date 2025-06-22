"""
Statistical analysis functions for pRF data.

This module provides statistical analysis capabilities for
population receptive field analysis results.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from ..core.data_structures import SrfData


def calculate_statistics(srf: SrfData, 
                        r_squared_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Calculate summary statistics for pRF data.
    
    Parameters
    ----------
    srf : SrfData
        Surface data with pRF parameters
    r_squared_threshold : float, default=0.05
        Minimum R² for inclusion in statistics
        
    Returns
    -------
    dict
        Summary statistics
    """
    stats = {
        'n_vertices': srf.vertices.shape[0] if srf.vertices is not None else 0,
        'parameters': {}
    }
    
    # Apply R² threshold
    r_squared = srf.get_parameter('R^2')
    if r_squared is not None:
        valid_mask = r_squared >= r_squared_threshold
        stats['n_valid_vertices'] = np.sum(valid_mask)
        stats['fraction_valid'] = np.mean(valid_mask)
        stats['r_squared_stats'] = {
            'mean': float(np.mean(r_squared)),
            'median': float(np.median(r_squared)),
            'std': float(np.std(r_squared)),
            'min': float(np.min(r_squared)),
            'max': float(np.max(r_squared))
        }
    else:
        valid_mask = np.ones(stats['n_vertices'], dtype=bool)
        stats['n_valid_vertices'] = stats['n_vertices']
        stats['fraction_valid'] = 1.0
    
    # Calculate statistics for each parameter
    for param_name in srf.values:
        if param_name == 'R^2':
            continue  # Already handled above
            
        param_data = srf.get_parameter(param_name)
        if param_data is not None:
            valid_data = param_data[valid_mask]
            
            stats['parameters'][param_name] = {
                'mean': float(np.mean(valid_data)),
                'median': float(np.median(valid_data)),
                'std': float(np.std(valid_data)),
                'min': float(np.min(valid_data)),
                'max': float(np.max(valid_data)),
                'percentile_25': float(np.percentile(valid_data, 25)),
                'percentile_75': float(np.percentile(valid_data, 75))
            }
    
    return stats


def compare_models(srf1: SrfData, srf2: SrfData, 
                  model_names: List[str] = ['Model 1', 'Model 2']) -> Dict[str, Any]:
    """
    Compare two pRF models statistically.
    
    Parameters
    ----------
    srf1, srf2 : SrfData
        Surface data for two models
    model_names : list, default=['Model 1', 'Model 2']
        Names for the models
        
    Returns
    -------
    dict
        Comparison results
    """
    comparison = {
        'model_names': model_names,
        'r_squared_comparison': {},
        'parameter_comparison': {}
    }
    
    # Compare R² values
    r2_1 = srf1.get_parameter('R^2')
    r2_2 = srf2.get_parameter('R^2')
    
    if r2_1 is not None and r2_2 is not None:
        # Paired t-test for R² improvement
        from scipy.stats import ttest_rel
        
        t_stat, p_value = ttest_rel(r2_2, r2_1)  # Test if model 2 > model 1
        
        comparison['r_squared_comparison'] = {
            'mean_r2_model1': float(np.mean(r2_1)),
            'mean_r2_model2': float(np.mean(r2_2)),
            'mean_improvement': float(np.mean(r2_2 - r2_1)),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'better_model': model_names[1] if np.mean(r2_2) > np.mean(r2_1) else model_names[0]
        }
    
    return comparison


def bootstrap_confidence(data: np.ndarray, statistic_func: callable,
                        n_bootstrap: int = 1000, 
                        confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Calculate bootstrap confidence intervals.
    
    Parameters
    ----------
    data : np.ndarray
        Data to bootstrap
    statistic_func : callable
        Function to calculate statistic (e.g., np.mean)
    n_bootstrap : int, default=1000
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level
        
    Returns
    -------
    dict
        Bootstrap results with confidence intervals
    """
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    return {
        'statistic': float(statistic_func(data)),
        'bootstrap_mean': float(np.mean(bootstrap_stats)),
        'bootstrap_std': float(np.std(bootstrap_stats)),
        'confidence_interval': [
            float(np.percentile(bootstrap_stats, lower_percentile)),
            float(np.percentile(bootstrap_stats, upper_percentile))
        ],
        'confidence_level': confidence_level
    }