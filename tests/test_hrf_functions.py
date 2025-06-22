"""
Tests for HRF functions.

Tests the hemodynamic response function modeling including
canonical HRFs, convolution, and parameter estimation.
"""

import pytest
import numpy as np
from pysamsrf.core.hrf_functions import (
    doublegamma, canonical_hrf, convolve_hrf, estimate_hrf_parameters,
    deconvolve_hrf, create_basis_set, validate_hrf
)


class TestDoubleGamma:
    """Test double-gamma HRF generation."""
    
    def test_default_parameters(self):
        """Test with default SPM parameters."""
        tr = 2.0
        hrf = doublegamma(tr)
        
        # Should be reasonable length
        assert len(hrf) > 10
        
        # Should have positive peak
        assert np.max(hrf) > 0
        
        # Should have negative undershoot
        assert np.min(hrf) < 0
        
        # Peak should occur before undershoot
        peak_idx = np.argmax(hrf)
        undershoot_idx = np.argmin(hrf)
        assert peak_idx < undershoot_idx
        
    def test_custom_parameters(self):
        """Test with custom parameters."""
        tr = 1.0
        params = [5, 12, 1, 1, 4, 0, 24]  # Custom parameters
        
        hrf = doublegamma(tr, params)
        
        # Should be 24 seconds long plus one TR
        expected_length = int(24 / tr) + 1
        assert len(hrf) == expected_length
        
    def test_different_tr(self):
        """Test with different TR values."""
        hrf_1s = doublegamma(1.0)
        hrf_2s = doublegamma(2.0)
        
        # 1s TR should have more samples
        assert len(hrf_1s) > len(hrf_2s)
        
        # Both should have similar peak times in seconds
        time_1s = np.arange(len(hrf_1s)) * 1.0
        time_2s = np.arange(len(hrf_2s)) * 2.0
        
        peak_time_1s = time_1s[np.argmax(hrf_1s)]
        peak_time_2s = time_2s[np.argmax(hrf_2s)]
        
        assert abs(peak_time_1s - peak_time_2s) < 2.0  # Within 2 seconds
        
    def test_time_shift(self):
        """Test HRF with time shift."""
        tr = 1.0
        params_no_shift = [6, 16, 1, 1, 6, 0, 32]
        params_shift = [6, 16, 1, 1, 6, 2, 32]  # 2s shift
        
        hrf_no_shift = doublegamma(tr, params_no_shift)
        hrf_shift = doublegamma(tr, params_shift)
        
        # Shifted version should have later peak
        peak_no_shift = np.argmax(hrf_no_shift)
        peak_shift = np.argmax(hrf_shift)
        
        assert peak_shift > peak_no_shift


class TestCanonicalHRF:
    """Test canonical HRF implementations."""
    
    def test_spm_canonical(self):
        """Test SPM canonical HRF."""
        hrf = canonical_hrf(2.0, 'spm')
        
        # Should have reasonable peak time (4-8 seconds)
        time = np.arange(len(hrf)) * 2.0
        peak_time = time[np.argmax(hrf)]
        
        assert 4 <= peak_time <= 8
        
    def test_de_haas_canonical(self):
        """Test de Haas canonical HRF."""
        hrf = canonical_hrf(2.0, 'de_haas')
        
        # Should be different from SPM
        hrf_spm = canonical_hrf(2.0, 'spm')
        
        # Not identical
        assert not np.allclose(hrf, hrf_spm)
        
    def test_glover_canonical(self):
        """Test Glover canonical HRF."""
        hrf = canonical_hrf(2.0, 'glover')
        
        # Should be reasonable
        assert np.max(hrf) > 0
        assert np.min(hrf) < 0
        
    def test_invalid_type(self):
        """Test error for invalid HRF type."""
        with pytest.raises(ValueError):
            canonical_hrf(2.0, 'invalid_type')


class TestConvolveHRF:
    """Test HRF convolution functions."""
    
    def test_basic_convolution(self):
        """Test basic convolution."""
        # Simple neural signal
        neural = np.zeros(100)
        neural[20:30] = 1  # Brief activation
        
        # Simple HRF
        hrf = canonical_hrf(1.0, 'spm')
        
        # Convolve
        bold = convolve_hrf(neural, hrf)
        
        # Should be same length as input
        assert len(bold) == len(neural)
        
        # Peak should be delayed relative to neural peak
        neural_peak = np.argmax(neural)
        bold_peak = np.argmax(bold)
        
        assert bold_peak > neural_peak
        
    def test_convolution_with_downsampling(self):
        """Test convolution with microtime resolution."""
        neural = np.zeros(50)
        neural[10:15] = 1
        
        hrf = canonical_hrf(2.0, 'spm')
        
        # With downsampling
        bold_downsampled = convolve_hrf(neural, hrf, downsample=4)
        
        # Without downsampling  
        bold_normal = convolve_hrf(neural, hrf)
        
        # Should be same length
        assert len(bold_downsampled) == len(bold_normal)
        
        # Should be similar but potentially different due to temporal resolution
        correlation = np.corrcoef(bold_downsampled, bold_normal)[0, 1]
        assert correlation > 0.9  # Should be highly correlated
        
    def test_empty_neural_signal(self):
        """Test convolution with zero neural signal."""
        neural = np.zeros(50)
        hrf = canonical_hrf(2.0, 'spm')
        
        bold = convolve_hrf(neural, hrf)
        
        # Should be all zeros (or close to zero)
        assert np.max(np.abs(bold)) < 1e-10
        
    def test_constant_neural_signal(self):
        """Test convolution with constant neural signal."""
        neural = np.ones(100)
        hrf = canonical_hrf(1.0, 'spm')
        
        bold = convolve_hrf(neural, hrf)
        
        # Should reach steady state
        # Later part should be relatively constant
        assert np.std(bold[-20:]) < np.std(bold[:20])


class TestHRFParameterEstimation:
    """Test HRF parameter estimation."""
    
    def test_perfect_fit(self):
        """Test estimation with perfect synthetic data."""
        tr = 2.0
        true_params = [6, 16, 1, 1, 6]
        
        # Generate synthetic data
        true_hrf = doublegamma(tr, true_params + [0, 32])
        neural = np.random.rand(100)
        bold = convolve_hrf(neural, true_hrf)
        
        # Add small amount of noise
        bold += np.random.normal(0, 0.01, len(bold))
        
        # Estimate parameters
        estimated_params, r_squared = estimate_hrf_parameters(bold, neural, tr)
        
        # Should have good fit
        assert r_squared > 0.95
        
        # Parameters should be reasonably close
        for true_p, est_p in zip(true_params, estimated_params):
            assert abs(true_p - est_p) / true_p < 0.2  # Within 20%
            
    def test_noisy_data(self):
        """Test estimation with noisy data."""
        tr = 2.0
        
        # Generate synthetic data with noise
        hrf = canonical_hrf(tr, 'spm')
        neural = np.random.rand(200)
        bold = convolve_hrf(neural, hrf)
        
        # Add significant noise
        bold += np.random.normal(0, 0.1, len(bold))
        
        # Estimate parameters
        estimated_params, r_squared = estimate_hrf_parameters(bold, neural, tr)
        
        # Should still give reasonable fit (though not perfect)
        assert r_squared > 0.5
        assert len(estimated_params) == 5
        
    def test_custom_initial_params(self):
        """Test estimation with custom initial parameters."""
        tr = 1.0
        hrf = canonical_hrf(tr, 'spm')
        neural = np.random.rand(100)
        bold = convolve_hrf(neural, hrf)
        
        # Custom initial guess
        initial_params = [5, 15, 0.8, 0.8, 7]
        
        estimated_params, r_squared = estimate_hrf_parameters(
            bold, neural, tr, initial_params
        )
        
        assert len(estimated_params) == 5
        assert r_squared > 0.8


class TestDeconvolveHRF:
    """Test HRF deconvolution."""
    
    def test_basic_deconvolution(self):
        """Test basic deconvolution."""
        # Create known neural signal
        neural_true = np.zeros(100)
        neural_true[20:25] = 1
        neural_true[60:65] = 0.5
        
        # Generate BOLD via convolution
        hrf = canonical_hrf(1.0, 'spm')
        bold = convolve_hrf(neural_true, hrf)
        
        # Deconvolve
        neural_estimated = deconvolve_hrf(bold, hrf)
        
        # Should be same length
        assert len(neural_estimated) == len(bold)
        
        # Should have peaks at approximately correct locations
        # (allowing for some temporal blurring)
        peaks_true = [22, 62]  # Approximate peak locations
        
        for peak_true in peaks_true:
            # Check if there's a peak within ±5 samples
            region = neural_estimated[peak_true-5:peak_true+5]
            local_max = np.max(region)
            baseline = np.mean(neural_estimated[:10])
            
            assert local_max > baseline + 0.1  # Significant elevation
            
    def test_noisy_deconvolution(self):
        """Test deconvolution with noisy BOLD signal."""
        neural_true = np.zeros(50)
        neural_true[15:20] = 1
        
        hrf = canonical_hrf(2.0, 'spm')
        bold_clean = convolve_hrf(neural_true, hrf)
        
        # Add noise
        bold_noisy = bold_clean + np.random.normal(0, 0.05, len(bold_clean))
        
        # Deconvolve both
        neural_clean = deconvolve_hrf(bold_clean, hrf)
        neural_noisy = deconvolve_hrf(bold_noisy, hrf)
        
        # Noisy version should be more variable but still correlated
        correlation = np.corrcoef(neural_clean, neural_noisy)[0, 1]
        assert correlation > 0.8


class TestBasisSets:
    """Test HRF basis set creation."""
    
    def test_canonical_basis(self):
        """Test canonical basis set."""
        tr = 2.0
        basis = create_basis_set(tr, n_basis=3, basis_type='canonical')
        
        # Should be 3 functions
        assert basis.shape[1] == 3
        
        # First should be canonical HRF
        canonical = canonical_hrf(tr, 'spm')
        assert np.allclose(basis[:len(canonical), 0], canonical, atol=0.1)
        
    def test_fir_basis(self):
        """Test FIR basis set."""
        tr = 2.0
        n_basis = 5
        basis = create_basis_set(tr, n_basis=n_basis, basis_type='fir')
        
        # Should be identity matrix (first n_basis columns)
        assert basis.shape[1] == n_basis
        
        # Should be orthogonal
        correlation_matrix = np.corrcoef(basis.T)
        off_diagonal = correlation_matrix - np.eye(n_basis)
        assert np.max(np.abs(off_diagonal)) < 1e-10
        
    def test_gamma_basis(self):
        """Test gamma basis set."""
        tr = 1.0
        n_basis = 4
        basis = create_basis_set(tr, n_basis=n_basis, basis_type='gamma')
        
        assert basis.shape[1] == n_basis
        
        # Each basis function should have different peak times
        peak_times = []
        for i in range(n_basis):
            peak_idx = np.argmax(basis[:, i])
            peak_time = peak_idx * tr
            peak_times.append(peak_time)
        
        # Peak times should be increasing
        assert all(peak_times[i] < peak_times[i+1] for i in range(n_basis-1))
        
    def test_invalid_basis_type(self):
        """Test error for invalid basis type."""
        with pytest.raises(ValueError):
            create_basis_set(2.0, n_basis=3, basis_type='invalid')


class TestHRFValidation:
    """Test HRF validation functions."""
    
    def test_valid_hrf(self):
        """Test validation of valid HRF."""
        hrf = canonical_hrf(2.0, 'spm')
        metrics = validate_hrf(hrf, 2.0)
        
        assert metrics['is_valid'] == True
        assert 3 <= metrics['peak_time'] <= 8
        assert metrics['peak_amplitude'] > 0
        assert metrics['undershoot_amplitude'] < 0
        assert metrics['fwhm'] > 0
        
    def test_invalid_hrf_late_peak(self):
        """Test validation of HRF with too late peak."""
        # Create artificial HRF with late peak
        time = np.arange(0, 32, 2.0)
        hrf = np.exp(-(time - 15)**2 / 10)  # Peak at 15s
        
        metrics = validate_hrf(hrf, 2.0)
        
        assert metrics['is_valid'] == False
        assert metrics['peak_time'] > 8
        
    def test_invalid_hrf_no_undershoot(self):
        """Test validation of HRF without undershoot."""
        # Create HRF without negative undershoot
        time = np.arange(0, 32, 2.0)
        hrf = np.exp(-(time - 6)**2 / 5)  # Only positive
        
        metrics = validate_hrf(hrf, 2.0)
        
        assert metrics['is_valid'] == False
        assert metrics['undershoot_amplitude'] >= 0
        
    def test_fwhm_calculation(self):
        """Test FWHM calculation."""
        # Create simple Gaussian-like HRF
        time = np.arange(0, 20, 1.0)
        hrf = np.exp(-(time - 6)**2 / 4)
        
        metrics = validate_hrf(hrf, 1.0)
        
        # FWHM should be approximately 2 * sqrt(2 * ln(2)) * sigma ≈ 4.7
        expected_fwhm = 2 * np.sqrt(2 * np.log(2)) * 2  # sigma = 2
        assert abs(metrics['fwhm'] - expected_fwhm) < 1.0


class TestHRFIntegration:
    """Test integration between HRF functions."""
    
    def test_convolution_deconvolution_roundtrip(self):
        """Test that convolution followed by deconvolution recovers signal."""
        # Create neural signal
        neural = np.zeros(100)
        neural[20:25] = 1
        neural[50:55] = 0.8
        neural[80:85] = 0.6
        
        # Convolve and deconvolve
        hrf = canonical_hrf(1.0, 'spm')
        bold = convolve_hrf(neural, hrf)
        neural_recovered = deconvolve_hrf(bold, hrf)
        
        # Should be reasonably well recovered
        correlation = np.corrcoef(neural, neural_recovered)[0, 1]
        assert correlation > 0.7
        
    def test_parameter_estimation_workflow(self):
        """Test complete parameter estimation workflow."""
        tr = 2.0
        
        # Generate synthetic experiment
        neural = np.zeros(150)
        # Add some activation blocks
        neural[20:30] = 1
        neural[60:70] = 0.8
        neural[100:110] = 1.2
        
        # Use known HRF
        true_hrf = canonical_hrf(tr, 'spm')
        bold = convolve_hrf(neural, true_hrf)
        
        # Add noise
        bold += np.random.normal(0, 0.02, len(bold))
        
        # Estimate HRF parameters
        estimated_params, r_squared = estimate_hrf_parameters(bold, neural, tr)
        
        # Create estimated HRF
        estimated_hrf = doublegamma(tr, list(estimated_params) + [0, 32])
        
        # Test that estimated HRF gives good prediction
        bold_predicted = convolve_hrf(neural, estimated_hrf)
        prediction_r2 = np.corrcoef(bold, bold_predicted)[0, 1] ** 2
        
        assert prediction_r2 > 0.9
        assert r_squared > 0.9


if __name__ == "__main__":
    pytest.main([__file__])