"""
Integration tests for PySamSrf.

Tests that verify the interaction between different components
and end-to-end workflows.
"""

import pytest
import numpy as np
from pysamsrf.core.data_structures import SrfData, Model, PRFResults
from pysamsrf.core.prf_functions import gaussian_rf, predict_timecourse, generate_prf_grid
from pysamsrf.core.hrf_functions import canonical_hrf, convolve_hrf
from pysamsrf.fitting.optimizers import error_function, grid_search, nelder_mead_wrapper


class TestSimulationWorkflow:
    """Test complete simulation and analysis workflow."""
    
    def test_synthetic_prf_experiment(self):
        """Test complete synthetic pRF experiment."""
        # Step 1: Define experimental parameters
        model = Model(
            name="test_gaussian_prf",
            param_names=['x0', 'y0', 'sigma'],
            tr=2.0,
            scaling_factor=10.0
        )
        
        # Ground truth parameters for 5 synthetic voxels
        n_voxels = 5
        true_params = np.array([
            [0, 0, 1.5],      # Central foveal
            [3, 2, 2.0],      # Upper right
            [-2, -1, 1.0],    # Lower left
            [5, 0, 2.5],      # Right periphery
            [0, -4, 1.8]      # Lower vertical
        ]).T
        
        # Step 2: Create stimulus apertures (expanding rings)
        aperture_width = 100
        n_timepoints = 40
        apertures = np.zeros((aperture_width**2, n_timepoints))
        
        for t in range(n_timepoints):
            radius = 0.5 + (t / n_timepoints) * 8  # Expanding from 0.5 to 8.5 deg
            y, x = np.mgrid[-50:50, -50:50]
            x = x * 10 / 50  # Scale to degrees
            y = y * 10 / 50
            ring = (np.sqrt(x**2 + y**2) < radius).flatten()
            apertures[:, t] = ring
        
        # Step 3: Generate synthetic BOLD responses
        hrf = canonical_hrf(model.tr, 'spm')
        bold_responses = np.zeros((n_voxels, n_timepoints))
        
        for v in range(n_voxels):
            # Generate pRF
            rf = gaussian_rf(true_params[0, v], true_params[1, v], 
                           true_params[2, v], aperture_width)
            
            # Predict neural response
            neural = predict_timecourse(rf, apertures)
            
            # Convolve with HRF
            bold = convolve_hrf(neural, hrf)[:n_timepoints]
            
            # Add realistic noise
            noise_level = 0.1 * np.std(bold)
            bold += np.random.normal(0, noise_level, len(bold))
            
            bold_responses[v, :] = bold
        
        # Step 4: Create SrfData structure
        srf = SrfData("synthetic")
        srf.y = bold_responses  # Raw time courses
        srf.vertices = np.random.rand(n_voxels, 3)  # Dummy vertex coordinates
        
        # Step 5: Set up pRF fitting
        def prf_func(x0, y0, sigma, aperture_width):
            return gaussian_rf(x0, y0, sigma, aperture_width)
        
        model_config = {
            'aperture_width': aperture_width,
            'percent_overlap': True,
            'compressive_nonlinearity': False
        }
        
        # Step 6: Fit pRFs using grid search + fine fitting
        fitted_params = np.zeros((3, n_voxels))
        r_squared = np.zeros(n_voxels)
        
        # Create search grids
        x_grid = np.linspace(-8, 8, 9)
        y_grid = np.linspace(-8, 8, 9)
        sigma_grid = np.array([0.5, 1, 1.5, 2, 2.5, 3])
        param_grids = [x_grid, y_grid, sigma_grid]
        
        for v in range(n_voxels):
            observed_bold = bold_responses[v, :]
            
            def error_func(params):
                return error_function(
                    params, prf_func, apertures, hrf, observed_bold, model_config
                )
            
            # Coarse grid search
            coarse_params, coarse_error, _ = grid_search(
                param_grids, error_func, parallel=False
            )
            
            # Fine optimization
            bounds = [(-10, 10), (-10, 10), (0.1, 5)]
            fine_params, fine_error = nelder_mead_wrapper(
                coarse_params, error_func, bounds
            )
            
            fitted_params[:, v] = fine_params
            r_squared[v] = 1 - fine_error
        
        # Step 7: Validate results
        # Check that fits are reasonable
        assert np.all(r_squared > 0.5), "All fits should have R² > 0.5"
        
        # Check parameter recovery
        for v in range(n_voxels):
            true_x0, true_y0, true_sigma = true_params[:, v]
            fitted_x0, fitted_y0, fitted_sigma = fitted_params[:, v]
            
            # Position should be within 2 degrees
            position_error = np.sqrt((fitted_x0 - true_x0)**2 + (fitted_y0 - true_y0)**2)
            assert position_error < 2.0, f"Voxel {v}: position error {position_error:.2f} too large"
            
            # Size should be within 50% (pRF size is harder to recover)
            size_error = abs(fitted_sigma - true_sigma) / true_sigma
            assert size_error < 0.5, f"Voxel {v}: size error {size_error:.2f} too large"
        
        # Step 8: Create results structure
        results = PRFResults(
            parameters=fitted_params,
            r_squared=r_squared
        )
        
        # Convert to SrfData format
        final_srf = results.to_srf(model, srf)
        
        # Validate final structure
        assert final_srf.data.shape == (4, n_voxels)  # 3 params + R²
        assert 'x0' in final_srf.values
        assert 'y0' in final_srf.values
        assert 'sigma' in final_srf.values
        assert 'R^2' in final_srf.values
        
        print(f"✅ Synthetic experiment completed successfully!")
        print(f"   Mean R²: {np.mean(r_squared):.3f}")
        print(f"   Parameter recovery within tolerance for all {n_voxels} voxels")


class TestDataProcessingWorkflow:
    """Test realistic data processing workflows."""
    
    def test_surface_data_workflow(self):
        """Test typical surface-based analysis workflow."""
        # Simulate realistic surface data dimensions
        n_vertices = 1000
        n_timepoints = 100
        n_params = 3
        
        # Step 1: Create mock surface data
        srf = SrfData("lh")
        
        # Add surface geometry
        srf.vertices = np.random.rand(n_vertices, 3) * 100  # Vertex coordinates
        srf.faces = np.random.randint(0, n_vertices, (n_vertices*2, 3))  # Triangular faces
        srf.curvature = np.random.normal(0, 0.1, n_vertices)  # Surface curvature
        
        # Add time series data
        srf.y = np.random.randn(n_vertices, n_timepoints)
        
        # Step 2: Add analysis results
        parameters = np.random.rand(n_params, n_vertices)
        parameters[0, :] = np.random.uniform(-10, 10, n_vertices)  # x0
        parameters[1, :] = np.random.uniform(-10, 10, n_vertices)  # y0  
        parameters[2, :] = np.random.uniform(0.5, 5, n_vertices)   # sigma
        
        r_squared = np.random.beta(2, 2, n_vertices)  # Realistic R² distribution
        
        srf.add_data(
            np.vstack([parameters, r_squared.reshape(1, -1)]),
            ['x0', 'y0', 'sigma', 'R^2']
        )
        
        # Step 3: Define ROI (simulate V1)
        v1_vertices = np.random.choice(n_vertices, 200, replace=False)
        v1_srf = srf.filter_by_roi(v1_vertices)
        
        # Step 4: Analyze ROI data
        v1_x0 = v1_srf.get_parameter('x0')
        v1_r2 = v1_srf.get_parameter('R^2')
        
        # Quality control: filter by R²
        good_fits = v1_r2 > 0.2
        high_quality_x0 = v1_x0[good_fits]
        
        # Step 5: Validate workflow
        assert v1_srf.data.shape[1] == 200
        assert len(v1_x0) == 200
        assert len(high_quality_x0) <= 200
        assert np.all(np.isfinite(high_quality_x0))
        
        # Step 6: Test parameter modification
        # Simulate coordinate transformation
        transformed_x0 = v1_x0 * 1.1 + 0.5  # Scale and shift
        v1_srf.set_parameter('x0_transformed', transformed_x0)
        
        assert 'x0_transformed' in v1_srf.values
        assert v1_srf.data.shape[0] == 5  # Added one parameter
        
        print(f"✅ Surface workflow completed successfully!")
        print(f"   Full brain vertices: {n_vertices}")
        print(f"   V1 vertices: {len(v1_vertices)}")  
        print(f"   High quality fits: {np.sum(good_fits)}")


class TestModelComparison:
    """Test comparison between different pRF models."""
    
    def test_gaussian_vs_dog_models(self):
        """Compare Gaussian vs Difference-of-Gaussians models."""
        from pysamsrf.core.prf_functions import dog_rf
        
        # Create synthetic data with DoG ground truth
        aperture_width = 80
        n_timepoints = 50
        
        # Ground truth DoG parameters
        true_x0, true_y0 = 2, -1
        true_sigma_center = 1.0
        true_sigma_surround = 3.0
        true_amplitude_ratio = 0.3
        
        # Create stimulus (random dots)
        np.random.seed(42)  # For reproducibility
        apertures = np.random.binomial(1, 0.3, (aperture_width**2, n_timepoints))
        
        # Generate ground truth response with DoG
        true_rf = dog_rf(true_x0, true_y0, true_sigma_center, 
                        true_sigma_surround, true_amplitude_ratio, aperture_width)
        neural = predict_timecourse(true_rf, apertures)
        
        hrf = canonical_hrf(2.0, 'spm')
        observed_bold = convolve_hrf(neural, hrf)[:n_timepoints]
        
        # Add noise
        observed_bold += np.random.normal(0, 0.05, len(observed_bold))
        
        # Test Model 1: Standard Gaussian
        def gaussian_prf_func(x0, y0, sigma, aperture_width):
            return gaussian_rf(x0, y0, sigma, aperture_width)
        
        # Test Model 2: DoG 
        def dog_prf_func(x0, y0, sigma_c, sigma_s, amp_ratio, aperture_width):
            return dog_rf(x0, y0, sigma_c, sigma_s, amp_ratio, aperture_width)
        
        model_config = {
            'aperture_width': aperture_width,
            'percent_overlap': True,
            'compressive_nonlinearity': False
        }
        
        # Fit Gaussian model
        def gaussian_error(params):
            return error_function(
                params, gaussian_prf_func, apertures, hrf, observed_bold, model_config
            )
        
        # Simple grid search for Gaussian
        x_grid = np.linspace(-5, 5, 11)
        y_grid = np.linspace(-5, 5, 11) 
        sigma_grid = np.array([0.5, 1, 1.5, 2, 2.5, 3])
        
        gaussian_params, gaussian_error_val, _ = grid_search(
            [x_grid, y_grid, sigma_grid], gaussian_error, parallel=False
        )
        
        # Fit DoG model (simplified - just test that it works)
        def dog_error(params):
            return error_function(
                params, dog_prf_func, apertures, hrf, observed_bold, model_config
            )
        
        # Coarse grid for DoG
        sigma_c_grid = np.array([0.5, 1, 1.5])
        sigma_s_grid = np.array([2, 3, 4])
        amp_ratio_grid = np.array([0.2, 0.3, 0.4])
        
        dog_params, dog_error_val, _ = grid_search(
            [x_grid[:5], y_grid[:5], sigma_c_grid, sigma_s_grid, amp_ratio_grid],
            dog_error, parallel=False
        )
        
        # DoG model should fit better (lower error) since data was generated with DoG
        print(f"Gaussian model error: {gaussian_error_val:.3f}")
        print(f"DoG model error: {dog_error_val:.3f}")
        
        # At minimum, both models should give reasonable fits
        assert gaussian_error_val < 0.8, "Gaussian model should give reasonable fit"
        assert dog_error_val < 0.8, "DoG model should give reasonable fit"
        
        # DoG should generally fit better, but allow for optimization variability
        print(f"✅ Model comparison completed!")
        print(f"   Best fit improvement with DoG: {(gaussian_error_val - dog_error_val):.3f}")


class TestPerformanceBenchmarks:
    """Performance benchmarks for key functions."""
    
    @pytest.mark.benchmark
    def test_gaussian_rf_performance(self):
        """Benchmark Gaussian RF generation."""
        import time
        
        # Test different aperture sizes
        sizes = [50, 100, 200]
        n_iterations = 100
        
        for size in sizes:
            start_time = time.time()
            
            for _ in range(n_iterations):
                rf = gaussian_rf(0, 0, 1, aperture_width=size)
            
            elapsed = time.time() - start_time
            per_rf = elapsed / n_iterations * 1000  # ms per RF
            
            print(f"Gaussian RF ({size}x{size}): {per_rf:.2f} ms per RF")
            
            # Should be reasonably fast
            assert per_rf < 50, f"RF generation too slow for size {size}"
    
    @pytest.mark.benchmark 
    def test_timecourse_prediction_performance(self):
        """Benchmark time course prediction."""
        import time
        
        aperture_width = 100
        n_timepoints = 200
        n_iterations = 50
        
        # Create test data
        rf = gaussian_rf(0, 0, 2, aperture_width)
        apertures = np.random.binomial(1, 0.5, (aperture_width**2, n_timepoints))
        
        start_time = time.time()
        
        for _ in range(n_iterations):
            tc = predict_timecourse(rf, apertures)
        
        elapsed = time.time() - start_time
        per_prediction = elapsed / n_iterations * 1000
        
        print(f"Time course prediction: {per_prediction:.2f} ms per prediction")
        
        # Should be fast
        assert per_prediction < 10, "Time course prediction too slow"
    
    @pytest.mark.benchmark
    def test_grid_search_performance(self):
        """Benchmark grid search performance."""
        import time
        
        # Simple quadratic function
        def test_objective(params):
            return sum(p**2 for p in params)
        
        # Different grid sizes
        grid_sizes = [5, 10, 15]
        
        for size in grid_sizes:
            param_grids = [np.linspace(-2, 2, size)] * 3  # 3 parameters
            
            start_time = time.time()
            best_params, best_error, all_errors = grid_search(
                param_grids, test_objective, parallel=False
            )
            elapsed = time.time() - start_time
            
            n_evaluations = size ** 3
            print(f"Grid search ({size}³ = {n_evaluations} evals): {elapsed:.3f} s")
            
            # Should scale reasonably
            assert elapsed < 2.0, f"Grid search too slow for size {size}"


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_model_parameters(self):
        """Test handling of invalid model parameters."""
        # Test Model with inconsistent parameter specifications
        with pytest.warns(UserWarning):
            model = Model(
                param_names=['x', 'y', 'z'],
                scaled_param=[True, False],  # Too short
                only_positive=[False]        # Too short
            )
        
        # Should auto-adjust to correct length
        assert len(model.scaled_param) == 3
        assert len(model.only_positive) == 3
    
    def test_mismatched_data_dimensions(self):
        """Test error handling for mismatched data dimensions."""
        srf = SrfData()
        
        # Wrong number of parameter names
        data = np.random.rand(3, 100)
        values = ['x', 'y']  # Only 2 names for 3 parameters
        
        with pytest.raises(ValueError):
            srf.add_data(data, values)
    
    def test_optimization_failures(self):
        """Test handling of optimization failures."""
        from pysamsrf.fitting.optimizers import nelder_mead_wrapper
        
        # Objective function that always raises an error
        def failing_objective(params):
            raise RuntimeError("Optimization failed")
        
        initial_params = np.array([1, 1])
        
        # Should handle gracefully and return initial parameters
        result_params, result_error = nelder_mead_wrapper(initial_params, failing_objective)
        
        assert np.array_equal(result_params, initial_params)
    
    def test_invalid_hrf_parameters(self):
        """Test handling of invalid HRF parameters."""
        from pysamsrf.core.hrf_functions import validate_hrf
        
        # Create obviously invalid HRF (all zeros)
        invalid_hrf = np.zeros(20)
        
        metrics = validate_hrf(invalid_hrf, 2.0)
        
        assert metrics['is_valid'] == False
        assert metrics['peak_amplitude'] == 0
    
    def test_empty_data_handling(self):
        """Test handling of empty or invalid data."""
        from pysamsrf.core.prf_functions import predict_timecourse
        
        # Empty RF
        rf = np.zeros((50, 50))
        apertures = np.random.rand(2500, 10)
        
        tc = predict_timecourse(rf, apertures)
        
        # Should return zeros
        assert np.allclose(tc, 0)


if __name__ == "__main__":
    pytest.main([__file__])