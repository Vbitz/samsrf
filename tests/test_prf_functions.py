"""
Tests for core pRF functions.

Tests the fundamental population receptive field modeling functions
including Gaussian RF generation and time course prediction.
"""

import pytest
import numpy as np
from pysamsrf.core.prf_functions import (
    gaussian_rf, predict_timecourse, dog_rf, multivariate_gaussian_rf,
    generate_prf_grid, center_prf, prf_size, rotate_prf
)


class TestGaussianRF:
    """Test Gaussian receptive field generation."""
    
    def test_basic_gaussian(self):
        """Test basic Gaussian RF generation."""
        x0, y0, sigma = 0, 0, 1
        rf = gaussian_rf(x0, y0, sigma, aperture_width=200)
        
        # Check output shape
        assert rf.shape == (200, 200)
        
        # Check that maximum is at center
        center_idx = 100  # Middle of 200x200 grid
        assert rf[center_idx, center_idx] == np.max(rf)
        
        # Check that RF is normalized (max = 1)
        assert np.isclose(np.max(rf), 1.0)
        
    def test_off_center_gaussian(self):
        """Test Gaussian RF away from center."""
        x0, y0, sigma = 50, -30, 2
        rf = gaussian_rf(x0, y0, sigma, aperture_width=200)
        
        # Find actual maximum location
        max_idx = np.unravel_index(np.argmax(rf), rf.shape)
        max_y, max_x = max_idx
        
        # Convert back to degrees (approximate)
        extent = 100  # Half of aperture_width
        actual_x = (max_x - 100) * extent / 100
        actual_y = (max_y - 100) * extent / 100
        
        # Should be close to specified center (within grid resolution)
        assert abs(actual_x - x0) < 5  # Within reasonable tolerance
        assert abs(actual_y - y0) < 5
        
    def test_different_sizes(self):
        """Test Gaussians with different sizes."""
        x0, y0 = 0, 0
        
        # Small Gaussian
        rf_small = gaussian_rf(x0, y0, 0.5, aperture_width=200)
        
        # Large Gaussian  
        rf_large = gaussian_rf(x0, y0, 3.0, aperture_width=200)
        
        # Large Gaussian should be more spread out
        # Check values at distance from center
        center = 100
        offset = 20
        
        small_val = rf_small[center + offset, center]
        large_val = rf_large[center + offset, center]
        
        assert large_val > small_val
        
    def test_aperture_width_scaling(self):
        """Test different aperture widths."""
        x0, y0, sigma = 0, 0, 1
        
        rf_100 = gaussian_rf(x0, y0, sigma, aperture_width=100)
        rf_200 = gaussian_rf(x0, y0, sigma, aperture_width=200)
        
        assert rf_100.shape == (100, 100)
        assert rf_200.shape == (200, 200)
        
        # Both should have maximum of 1 at center
        assert np.isclose(np.max(rf_100), 1.0)
        assert np.isclose(np.max(rf_200), 1.0)


class TestPredictTimecourse:
    """Test time course prediction from pRF and apertures."""
    
    def test_basic_prediction(self):
        """Test basic time course prediction."""
        # Create simple Gaussian RF
        rf = gaussian_rf(0, 0, 2, aperture_width=100)
        
        # Create simple apertures (binary on/off)
        n_timepoints = 50
        apertures = np.zeros((100*100, n_timepoints))
        
        # First half: stimulus on, second half: off
        apertures[:, :25] = 1
        
        # Predict time course
        timecourse = predict_timecourse(rf, apertures)
        
        assert len(timecourse) == n_timepoints
        
        # Should be higher when stimulus is on
        mean_on = np.mean(timecourse[:25])
        mean_off = np.mean(timecourse[25:])
        assert mean_on > mean_off
        
    def test_percent_overlap_model(self):
        """Test percent overlap model."""
        rf = gaussian_rf(0, 0, 1, aperture_width=50)
        
        # Full field stimulation
        apertures_full = np.ones((50*50, 10))
        
        # Partial field stimulation
        apertures_partial = np.zeros((50*50, 10))
        apertures_partial[:1250, :] = 1  # Half the field
        
        tc_full = predict_timecourse(rf, apertures_full, percent_overlap=True)
        tc_partial = predict_timecourse(rf, apertures_partial, percent_overlap=True)
        
        # Full field should give higher response
        assert np.mean(tc_full) > np.mean(tc_partial)
        
        # Full field should give ~100% overlap
        assert np.max(tc_full) > 80  # Should be close to 100
        
    def test_dumoulin_wandell_model(self):
        """Test Dumoulin & Wandell model."""
        rf = gaussian_rf(0, 0, 1, aperture_width=50)
        
        # Binary apertures
        apertures = np.zeros((50*50, 20))
        apertures[:, ::2] = 1  # Every other timepoint
        
        tc_dw = predict_timecourse(rf, apertures, percent_overlap=False)
        
        # Should alternate between high and low values
        on_responses = tc_dw[::2]
        off_responses = tc_dw[1::2]
        
        assert np.mean(on_responses) > np.mean(off_responses)
        
    def test_dimension_mismatch_error(self):
        """Test error when RF and aperture dimensions don't match."""
        rf = gaussian_rf(0, 0, 1, aperture_width=50)  # 50x50 = 2500 pixels
        apertures = np.random.rand(1000, 10)  # Wrong size
        
        with pytest.raises(ValueError):
            predict_timecourse(rf, apertures)


class TestAdvancedRFModels:
    """Test advanced receptive field models."""
    
    def test_dog_rf(self):
        """Test difference-of-Gaussians RF."""
        x0, y0 = 0, 0
        sigma_center = 1
        sigma_surround = 3
        amplitude_ratio = 0.5
        
        rf = dog_rf(x0, y0, sigma_center, sigma_surround, amplitude_ratio)
        
        # Should have positive center
        center = rf.shape[0] // 2
        assert rf[center, center] > 0
        
        # Should have negative surround
        surround_offset = 30
        surround_val = rf[center + surround_offset, center]
        assert surround_val < 0
        
    def test_multivariate_gaussian(self):
        """Test elliptical Gaussian RF."""
        x0, y0 = 0, 0
        sigma_x, sigma_y = 2, 1  # Elliptical
        theta = 45  # Rotated
        
        rf = multivariate_gaussian_rf(x0, y0, sigma_x, sigma_y, theta)
        
        # Check that it's not circular (should be elliptical)
        center = rf.shape[0] // 2
        
        # Check values along different axes
        val_horizontal = rf[center, center + 20]
        val_vertical = rf[center + 20, center]
        
        # Due to rotation and different sigmas, values should differ
        assert not np.isclose(val_horizontal, val_vertical, atol=0.1)
        
    def test_multivariate_no_rotation(self):
        """Test elliptical RF without rotation."""
        rf = multivariate_gaussian_rf(0, 0, 2, 1, 0)  # No rotation
        
        center = rf.shape[0] // 2
        
        # Should be wider horizontally than vertically
        val_horizontal = rf[center, center + 10]
        val_vertical = rf[center + 10, center]
        
        assert val_horizontal > val_vertical


class TestPRFGrid:
    """Test pRF grid generation for coarse fitting."""
    
    def test_basic_grid_generation(self):
        """Test basic grid generation."""
        model_params = {
            'param1': np.array([0, 90, 180, 270]),  # Angles
            'param2': np.array([1, 2, 4]),          # Eccentricities
            'param3': np.array([0.5, 1, 2]),        # Sigmas
            'polar_search_space': True
        }
        
        prf_grid, param_grid = generate_prf_grid(model_params)
        
        # Should have 4 * 3 * 3 = 36 combinations
        assert prf_grid.shape[0] == 36
        assert param_grid.shape[0] == 36
        assert param_grid.shape[1] == 3
        
        # First two columns should be Cartesian coordinates
        assert np.any(param_grid[:, 0] != model_params['param1'])  # Converted from polar
        
    def test_cartesian_grid(self):
        """Test Cartesian coordinate grid."""
        model_params = {
            'param1': np.array([-2, 0, 2]),  # X positions
            'param2': np.array([-1, 0, 1]),  # Y positions  
            'param3': np.array([1, 2]),      # Sigmas
            'polar_search_space': False
        }
        
        prf_grid, param_grid = generate_prf_grid(model_params)
        
        # Should preserve Cartesian coordinates
        assert np.any(np.isin(param_grid[:, 0], model_params['param1']))
        assert np.any(np.isin(param_grid[:, 1], model_params['param2']))
        
    def test_skip_zero_parameters(self):
        """Test skipping parameters set to zero."""
        model_params = {
            'param1': np.array([0, 90, 180]),
            'param2': np.array([1, 2]),
            'param3': np.array([1]),
            'param4': np.array([0]),  # Should be skipped
            'param5': np.array([0]),  # Should be skipped
            'polar_search_space': True
        }
        
        prf_grid, param_grid = generate_prf_grid(model_params)
        
        # Should only use first 3 parameters
        assert param_grid.shape[1] == 3


class TestPRFAnalysis:
    """Test pRF analysis and utility functions."""
    
    def test_center_prf(self):
        """Test pRF center of mass calculation."""
        # Create off-center Gaussian
        rf = gaussian_rf(20, -10, 2, aperture_width=200)
        
        x_center, y_center = center_prf(rf)
        
        # Should be close to actual center (in pixel coordinates)
        # Convert from degrees to pixels (approximate)
        expected_x = 100 + 20 * 100 / 100  # Rough conversion
        expected_y = 100 - 10 * 100 / 100
        
        # Should be reasonably close
        assert abs(x_center - expected_x) < 20
        assert abs(y_center - expected_y) < 20
        
    def test_prf_size(self):
        """Test pRF size calculation."""
        # Small pRF
        rf_small = gaussian_rf(0, 0, 0.5, aperture_width=100)
        size_small = prf_size(rf_small)
        
        # Large pRF
        rf_large = gaussian_rf(0, 0, 2.0, aperture_width=100)
        size_large = prf_size(rf_large)
        
        # Larger sigma should give larger size
        assert size_large > size_small
        
    def test_prf_size_threshold(self):
        """Test pRF size with different thresholds."""
        rf = gaussian_rf(0, 0, 1, aperture_width=100)
        
        size_50 = prf_size(rf, threshold=0.5)
        size_10 = prf_size(rf, threshold=0.1)
        
        # Lower threshold should give larger size
        assert size_10 > size_50
        
    def test_rotate_prf(self):
        """Test pRF rotation."""
        # Create elliptical pRF
        rf = multivariate_gaussian_rf(0, 0, 2, 1, 0, aperture_width=100)
        
        # Rotate by 90 degrees
        rf_rotated = rotate_prf(rf, 90)
        
        assert rf_rotated.shape == rf.shape
        
        # Check that orientation has changed
        center = rf.shape[0] // 2
        
        # Original should be wider horizontally
        orig_h = rf[center, center + 15]
        orig_v = rf[center + 15, center]
        
        # Rotated should be wider vertically
        rot_h = rf_rotated[center, center + 15]
        rot_v = rf_rotated[center + 15, center]
        
        # Relative orientations should flip
        assert (orig_h > orig_v) and (rot_v > rot_h)


class TestPRFValidation:
    """Test validation and edge cases."""
    
    def test_zero_sigma(self):
        """Test behavior with zero sigma."""
        rf = gaussian_rf(0, 0, 0, aperture_width=100)
        
        # Should be very concentrated (delta function)
        center = rf.shape[0] // 2
        assert rf[center, center] > 0.9
        
        # Should drop off very quickly
        assert rf[center + 1, center] < 0.1
        
    def test_large_sigma(self):
        """Test behavior with very large sigma."""
        rf = gaussian_rf(0, 0, 100, aperture_width=100)
        
        # Should be relatively flat
        center = rf.shape[0] // 2
        center_val = rf[center, center]
        edge_val = rf[center + 30, center]
        
        # Ratio should be small (relatively flat)
        ratio = center_val / edge_val
        assert ratio < 5  # Not too peaked
        
    def test_edge_positions(self):
        """Test pRF at edge of aperture."""
        # Place pRF at edge
        rf = gaussian_rf(90, 90, 2, aperture_width=200)
        
        # Should still be normalized
        assert np.max(rf) <= 1.0
        
        # Should be truncated but still reasonable
        assert np.sum(rf) > 0
        
    def test_negative_coordinates(self):
        """Test negative coordinates."""
        rf = gaussian_rf(-50, -30, 2, aperture_width=200)
        
        # Should work fine
        assert rf.shape == (200, 200)
        assert np.max(rf) > 0


if __name__ == "__main__":
    pytest.main([__file__])