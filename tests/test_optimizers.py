"""
Tests for optimization algorithms.

Tests the various optimization methods used in pRF fitting including
Nelder-Mead, Hooke-Jeeves, and grid search algorithms.
"""

import pytest
import numpy as np
from pysamsrf.fitting.optimizers import (
    error_function, nelder_mead_wrapper, hooke_jeeves, grid_search,
    multi_start_optimization, adaptive_grid_search
)
from pysamsrf.core.prf_functions import gaussian_rf, predict_timecourse
from pysamsrf.core.hrf_functions import canonical_hrf, convolve_hrf


class TestErrorFunction:
    """Test the pRF error function."""
    
    def setup_method(self):
        """Set up test data."""
        # Create synthetic ground truth
        self.true_params = [0, 0, 2]  # x0, y0, sigma
        self.aperture_width = 100
        
        # Create stimulus apertures (simple expanding ring)
        n_timepoints = 50
        apertures = np.zeros((self.aperture_width**2, n_timepoints))
        
        for t in range(n_timepoints):
            radius = t / n_timepoints * 40  # Expanding radius
            y, x = np.mgrid[-50:50, -50:50]
            ring = (np.sqrt(x**2 + y**2) < radius).flatten()
            apertures[:, t] = ring
            
        self.apertures = apertures
        
        # Generate ground truth BOLD signal
        true_rf = gaussian_rf(*self.true_params, self.aperture_width)
        neural = predict_timecourse(true_rf, apertures)
        hrf = canonical_hrf(2.0, 'spm')
        self.observed_bold = convolve_hrf(neural, hrf)[:n_timepoints]
        
        # HRF and model config
        self.hrf = hrf
        self.model_config = {
            'aperture_width': self.aperture_width,
            'percent_overlap': True,
            'compressive_nonlinearity': False
        }
        
    def test_perfect_parameters(self):
        """Test error function with perfect parameters."""
        # Define pRF function
        def prf_func(x0, y0, sigma, aperture_width):
            return gaussian_rf(x0, y0, sigma, aperture_width)
        
        # Test with true parameters
        error = error_function(
            self.true_params, prf_func, self.apertures, 
            self.hrf, self.observed_bold, self.model_config
        )
        
        # Should have very low error (high R²)
        assert error < 0.1  # R² > 0.9
        
    def test_wrong_parameters(self):
        """Test error function with wrong parameters."""
        def prf_func(x0, y0, sigma, aperture_width):
            return gaussian_rf(x0, y0, sigma, aperture_width)
        
        # Very different parameters
        wrong_params = [20, 20, 0.5]
        
        error = error_function(
            wrong_params, prf_func, self.apertures,
            self.hrf, self.observed_bold, self.model_config
        )
        
        # Should have high error
        assert error > 0.5  # R² < 0.5
        
    def test_compressive_nonlinearity(self):
        """Test error function with CSS model."""
        self.model_config['compressive_nonlinearity'] = True
        
        def prf_func(x0, y0, sigma, css_exp, aperture_width):
            return gaussian_rf(x0, y0, sigma, aperture_width)
        
        # Include CSS exponent
        params_with_css = self.true_params + [0.8]
        
        error = error_function(
            params_with_css, prf_func, self.apertures,
            self.hrf, self.observed_bold, self.model_config
        )
        
        # Should work without error
        assert isinstance(error, float)
        assert 0 <= error <= 1
        
    def test_invalid_parameters(self):
        """Test error function with invalid parameters."""
        def prf_func(x0, y0, sigma, aperture_width):
            # This will cause an error
            if sigma <= 0:
                raise ValueError("Invalid sigma")
            return gaussian_rf(x0, y0, sigma, aperture_width)
        
        invalid_params = [0, 0, -1]  # Negative sigma
        
        error = error_function(
            invalid_params, prf_func, self.apertures,
            self.hrf, self.observed_bold, self.model_config
        )
        
        # Should return maximum error for invalid parameters
        assert error == 1.0


class TestNelderMeadWrapper:
    """Test Nelder-Mead optimization wrapper."""
    
    def test_simple_optimization(self):
        """Test optimization of simple quadratic function."""
        # Minimize (x-2)² + (y-3)²
        def objective(params):
            x, y = params
            return (x - 2)**2 + (y - 3)**2
        
        initial_params = np.array([0, 0])
        
        best_params, best_error = nelder_mead_wrapper(initial_params, objective)
        
        # Should find minimum at (2, 3)
        assert abs(best_params[0] - 2) < 0.1
        assert abs(best_params[1] - 3) < 0.1
        assert best_error < 0.01
        
    def test_bounded_optimization(self):
        """Test optimization with bounds."""
        def objective(params):
            x, y = params
            return (x - 5)**2 + (y - 5)**2  # Minimum at (5, 5)
        
        initial_params = np.array([0, 0])
        bounds = [(0, 3), (0, 3)]  # Constrained region
        
        best_params, best_error = nelder_mead_wrapper(
            initial_params, objective, bounds
        )
        
        # Should find constrained minimum at (3, 3)
        assert abs(best_params[0] - 3) < 0.1
        assert abs(best_params[1] - 3) < 0.1
        
        # Should respect bounds
        assert 0 <= best_params[0] <= 3
        assert 0 <= best_params[1] <= 3
        
    def test_optimization_failure(self):
        """Test behavior when optimization fails."""
        def bad_objective(params):
            raise RuntimeError("Optimization error")
        
        initial_params = np.array([1, 1])
        
        best_params, best_error = nelder_mead_wrapper(initial_params, bad_objective)
        
        # Should return initial parameters
        assert np.array_equal(best_params, initial_params)


class TestHookeJeeves:
    """Test Hooke-Jeeves pattern search algorithm."""
    
    def test_simple_optimization(self):
        """Test optimization of simple function."""
        # Minimize (x-1)² + 2*(y-2)²
        def objective(params):
            x, y = params
            return (x - 1)**2 + 2*(y - 2)**2
        
        initial_params = np.array([0, 0])
        
        best_params, best_error = hooke_jeeves(initial_params, objective)
        
        # Should find minimum at (1, 2)
        assert abs(best_params[0] - 1) < 0.1
        assert abs(best_params[1] - 2) < 0.1
        
    def test_with_bounds(self):
        """Test Hooke-Jeeves with parameter bounds."""
        def objective(params):
            x, y = params
            return x**2 + y**2  # Minimum at (0, 0)
        
        initial_params = np.array([3, 4])
        bounds = [(1, 5), (2, 6)]  # Constrained away from minimum
        
        best_params, best_error = hooke_jeeves(
            initial_params, objective, bounds=bounds
        )
        
        # Should find constrained minimum at (1, 2)
        assert abs(best_params[0] - 1) < 0.1
        assert abs(best_params[1] - 2) < 0.1
        
        # Should respect bounds
        assert 1 <= best_params[0] <= 5
        assert 2 <= best_params[1] <= 6
        
    def test_custom_step_sizes(self):
        """Test with custom initial step sizes."""
        def objective(params):
            x, y = params
            return (x - 2)**2 + (y - 3)**2
        
        initial_params = np.array([0, 0])
        step_sizes = np.array([0.5, 0.5])
        
        best_params, best_error = hooke_jeeves(
            initial_params, objective, step_sizes=step_sizes
        )
        
        assert abs(best_params[0] - 2) < 0.1
        assert abs(best_params[1] - 3) < 0.1
        
    def test_convergence_criteria(self):
        """Test convergence with tight tolerance."""
        def objective(params):
            x, y = params
            return x**2 + y**2
        
        initial_params = np.array([1, 1])
        
        best_params, best_error = hooke_jeeves(
            initial_params, objective, tolerance=1e-8
        )
        
        # Should converge very close to minimum
        assert np.sqrt(best_params[0]**2 + best_params[1]**2) < 1e-4
        
    def test_no_improvement(self):
        """Test behavior when no improvement is possible."""
        def objective(params):
            # Flat function
            return 1.0
        
        initial_params = np.array([2, 3])
        
        best_params, best_error = hooke_jeeves(
            initial_params, objective, max_iterations=10
        )
        
        # Should return something close to initial parameters
        assert abs(best_params[0] - 2) < 1.0
        assert abs(best_params[1] - 3) < 1.0
        assert best_error == 1.0


class TestGridSearch:
    """Test grid search optimization."""
    
    def test_basic_grid_search(self):
        """Test basic grid search."""
        # Function with minimum at (1.5, 2.5)
        def objective(params):
            x, y = params
            return (x - 1.5)**2 + (y - 2.5)**2
        
        # Create grids
        x_grid = np.linspace(0, 3, 11)
        y_grid = np.linspace(1, 4, 11)
        param_grids = [x_grid, y_grid]
        
        best_params, best_error, all_errors = grid_search(
            param_grids, objective, parallel=False
        )
        
        # Should find grid point closest to (1.5, 2.5)
        assert abs(best_params[0] - 1.5) < 0.5
        assert abs(best_params[1] - 2.5) < 0.5
        
        # Should return all errors
        assert len(all_errors) == 11 * 11
        assert np.min(all_errors) == best_error
        
    def test_parallel_grid_search(self):
        """Test parallel grid search (if joblib available)."""
        def objective(params):
            x, y = params
            return x**2 + y**2
        
        x_grid = np.linspace(-2, 2, 5)
        y_grid = np.linspace(-2, 2, 5)
        param_grids = [x_grid, y_grid]
        
        try:
            best_params, best_error, all_errors = grid_search(
                param_grids, objective, parallel=True
            )
            
            # Should find minimum at (0, 0) or close
            assert abs(best_params[0]) < 1
            assert abs(best_params[1]) < 1
            
        except ImportError:
            # joblib not available, skip parallel test
            pass
            
    def test_single_parameter_grid(self):
        """Test grid search with single parameter."""
        def objective(params):
            x = params[0]
            return (x - 2)**2
        
        param_grids = [np.linspace(0, 4, 21)]
        
        best_params, best_error, all_errors = grid_search(
            param_grids, objective, parallel=False
        )
        
        assert abs(best_params[0] - 2) < 0.2
        assert len(all_errors) == 21


class TestMultiStartOptimization:
    """Test multi-start optimization."""
    
    def test_multi_start_nelder_mead(self):
        """Test multi-start with Nelder-Mead."""
        # Function with global minimum at (0, 0) and local minima
        def objective(params):
            x, y = params
            return (x**2 + y**2) + 0.5 * np.sin(5*x) * np.sin(5*y)
        
        bounds = [(-2, 2), (-2, 2)]
        
        best_params, best_error = multi_start_optimization(
            objective, bounds, n_starts=5, method='nelder_mead'
        )
        
        # Should find global minimum near (0, 0)
        assert abs(best_params[0]) < 0.5
        assert abs(best_params[1]) < 0.5
        
    def test_multi_start_hooke_jeeves(self):
        """Test multi-start with Hooke-Jeeves."""
        def objective(params):
            x, y = params
            return (x - 1)**2 + (y + 1)**2
        
        bounds = [(-3, 3), (-3, 3)]
        
        best_params, best_error = multi_start_optimization(
            objective, bounds, n_starts=3, method='hooke_jeeves'
        )
        
        # Should find minimum at (1, -1)
        assert abs(best_params[0] - 1) < 0.2
        assert abs(best_params[1] + 1) < 0.2
        
    def test_invalid_method(self):
        """Test error for invalid optimization method."""
        def objective(params):
            return sum(params**2)
        
        bounds = [(-1, 1), (-1, 1)]
        
        with pytest.raises(ValueError):
            multi_start_optimization(
                objective, bounds, n_starts=2, method='invalid_method'
            )


class TestAdaptiveGridSearch:
    """Test adaptive grid search."""
    
    def test_adaptive_refinement(self):
        """Test adaptive grid refinement."""
        # Function with sharp minimum at (0.7, 1.3)
        def objective(params):
            x, y = params
            return (x - 0.7)**2 + (y - 1.3)**2
        
        # Coarse initial grids
        initial_grids = [
            np.linspace(0, 2, 5),    # x grid
            np.linspace(0, 2, 5)     # y grid
        ]
        
        best_params, best_error = adaptive_grid_search(
            initial_grids, objective, n_iterations=3
        )
        
        # Should find minimum more accurately than single coarse grid
        assert abs(best_params[0] - 0.7) < 0.1
        assert abs(best_params[1] - 1.3) < 0.1
        
    def test_single_iteration(self):
        """Test adaptive search with single iteration."""
        def objective(params):
            x, y = params
            return x**2 + y**2
        
        initial_grids = [
            np.linspace(-1, 1, 3),
            np.linspace(-1, 1, 3)
        ]
        
        best_params, best_error = adaptive_grid_search(
            initial_grids, objective, n_iterations=1
        )
        
        # Should work but be less refined
        assert abs(best_params[0]) < 1
        assert abs(best_params[1]) < 1


class TestOptimizationIntegration:
    """Test optimization algorithms with realistic pRF problems."""
    
    def setup_method(self):
        """Set up realistic pRF optimization problem."""
        # Ground truth parameters
        self.true_x0 = 2.0
        self.true_y0 = -1.5
        self.true_sigma = 1.2
        
        # Create realistic stimulus
        aperture_width = 80
        n_timepoints = 60
        
        apertures = np.zeros((aperture_width**2, n_timepoints))
        
        # Sweeping bar stimulus
        for t in range(n_timepoints):
            bar_pos = -40 + (80 * t / n_timepoints)  # Sweep from left to right
            y, x = np.mgrid[-40:40, -40:40]
            bar = (np.abs(x - bar_pos) < 5).flatten()  # Vertical bar
            apertures[:, t] = bar
            
        self.apertures = apertures
        self.aperture_width = aperture_width
        
        # Generate ground truth response
        true_rf = gaussian_rf(self.true_x0, self.true_y0, self.true_sigma, aperture_width)
        neural = predict_timecourse(true_rf, apertures)
        hrf = canonical_hrf(2.0, 'spm')
        self.observed_bold = convolve_hrf(neural, hrf)[:n_timepoints]
        
        # Add realistic noise
        noise_level = 0.1 * np.std(self.observed_bold)
        self.observed_bold += np.random.normal(0, noise_level, len(self.observed_bold))
        
        self.hrf = hrf
        self.model_config = {
            'aperture_width': aperture_width,
            'percent_overlap': True,
            'compressive_nonlinearity': False
        }
        
    def test_prf_optimization_grid_to_fine(self):
        """Test complete pRF optimization: grid search + fine fitting."""
        def prf_func(x0, y0, sigma, aperture_width):
            return gaussian_rf(x0, y0, sigma, aperture_width)
        
        def error_func(params):
            return error_function(
                params, prf_func, self.apertures,
                self.hrf, self.observed_bold, self.model_config
            )
        
        # Coarse grid search
        x_grid = np.linspace(-10, 10, 11)
        y_grid = np.linspace(-10, 10, 11)
        sigma_grid = np.array([0.5, 1, 2, 3])
        
        param_grids = [x_grid, y_grid, sigma_grid]
        
        coarse_params, coarse_error, _ = grid_search(
            param_grids, error_func, parallel=False
        )
        
        # Fine optimization starting from coarse fit
        fine_params, fine_error = nelder_mead_wrapper(
            coarse_params, error_func,
            bounds=[(-15, 15), (-15, 15), (0.1, 5)]
        )
        
        # Fine fit should be better than coarse fit
        assert fine_error <= coarse_error
        
        # Should be reasonably close to ground truth
        assert abs(fine_params[0] - self.true_x0) < 1.0
        assert abs(fine_params[1] - self.true_y0) < 1.0
        assert abs(fine_params[2] - self.true_sigma) < 0.5
        
        # Should have good fit quality
        assert fine_error < 0.3  # R² > 0.7
        
    def test_optimization_robustness(self):
        """Test optimization robustness with different starting points."""
        def prf_func(x0, y0, sigma, aperture_width):
            return gaussian_rf(x0, y0, sigma, aperture_width)
        
        def error_func(params):
            return error_function(
                params, prf_func, self.apertures,
                self.hrf, self.observed_bold, self.model_config
            )
        
        # Try multiple starting points
        starting_points = [
            [0, 0, 1],
            [-5, 5, 2],
            [5, -5, 0.5],
            [10, 10, 3]
        ]
        
        results = []
        for start in starting_points:
            try:
                best_params, best_error = nelder_mead_wrapper(
                    np.array(start), error_func,
                    bounds=[(-15, 15), (-15, 15), (0.1, 5)]
                )
                results.append((best_params, best_error))
            except:
                continue
        
        # Should have at least some successful optimizations
        assert len(results) > 0
        
        # Best result should be reasonable
        best_result = min(results, key=lambda x: x[1])
        best_params, best_error = best_result
        
        assert best_error < 0.5  # Reasonable fit
        
        # At least one optimization should find approximate solution
        any_close = any(
            abs(params[0] - self.true_x0) < 2 and 
            abs(params[1] - self.true_y0) < 2 and
            abs(params[2] - self.true_sigma) < 1
            for params, _ in results
        )
        assert any_close


if __name__ == "__main__":
    pytest.main([__file__])