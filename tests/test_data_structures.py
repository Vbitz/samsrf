"""
Tests for core data structures.

Tests the SrfData, Model, and PRFResults classes to ensure proper
functionality and data integrity.
"""

import pytest
import numpy as np
from pysamsrf.core.data_structures import SrfData, Model, PRFResults


class TestSrfData:
    """Test SrfData class functionality."""
    
    def test_init_empty(self):
        """Test empty initialization."""
        srf = SrfData()
        assert srf.hemisphere == ""
        assert srf.data is None
        assert len(srf.values) == 0
        assert srf.vertices is None
        
    def test_init_with_hemisphere(self):
        """Test initialization with hemisphere."""
        srf = SrfData("lh")
        assert srf.hemisphere == "lh"
        
    def test_add_data(self):
        """Test adding parameter data."""
        srf = SrfData()
        data = np.random.rand(3, 100)  # 3 parameters, 100 vertices
        values = ['x0', 'y0', 'sigma']
        
        srf.add_data(data, values)
        
        assert np.array_equal(srf.data, data)
        assert srf.values == values
        
    def test_add_data_mismatch(self):
        """Test error when data shape doesn't match values."""
        srf = SrfData()
        data = np.random.rand(3, 100)
        values = ['x0', 'y0']  # Only 2 values for 3 parameters
        
        with pytest.raises(ValueError):
            srf.add_data(data, values)
            
    def test_get_parameter(self):
        """Test retrieving specific parameter."""
        srf = SrfData()
        data = np.random.rand(3, 100)
        values = ['x0', 'y0', 'sigma']
        srf.add_data(data, values)
        
        x0_data = srf.get_parameter('x0')
        assert np.array_equal(x0_data, data[0, :])
        
        # Test non-existent parameter
        assert srf.get_parameter('nonexistent') is None
        
    def test_set_parameter_existing(self):
        """Test setting existing parameter."""
        srf = SrfData()
        data = np.random.rand(3, 100)
        values = ['x0', 'y0', 'sigma']
        srf.add_data(data, values)
        
        new_x0 = np.random.rand(100)
        srf.set_parameter('x0', new_x0)
        
        assert np.array_equal(srf.data[0, :], new_x0)
        
    def test_set_parameter_new(self):
        """Test adding new parameter."""
        srf = SrfData()
        data = np.random.rand(3, 100)
        values = ['x0', 'y0', 'sigma']
        srf.add_data(data, values)
        
        new_param = np.random.rand(100)
        srf.set_parameter('r_squared', new_param)
        
        assert srf.values[-1] == 'r_squared'
        assert np.array_equal(srf.data[-1, :], new_param)
        assert srf.data.shape[0] == 4
        
    def test_filter_by_roi(self):
        """Test ROI filtering."""
        srf = SrfData("lh")
        
        # Add some data
        data = np.random.rand(3, 1000)
        values = ['x0', 'y0', 'sigma']
        vertices = np.random.rand(1000, 3)
        
        srf.add_data(data, values)
        srf.vertices = vertices
        
        # Filter to subset
        roi_indices = np.array([10, 50, 100, 200, 500])
        filtered = srf.filter_by_roi(roi_indices)
        
        assert filtered.hemisphere == "lh"
        assert filtered.data.shape[1] == len(roi_indices)
        assert filtered.vertices.shape[0] == len(roi_indices)
        assert np.array_equal(filtered.roi, roi_indices)
        assert np.array_equal(filtered.data, data[:, roi_indices])
        
    def test_repr(self):
        """Test string representation."""
        srf = SrfData("lh")
        srf.vertices = np.random.rand(1000, 3)
        srf.values = ['x0', 'y0', 'sigma']
        
        repr_str = repr(srf)
        assert "lh" in repr_str
        assert "1000" in repr_str
        assert "3" in repr_str


class TestModel:
    """Test Model class functionality."""
    
    def test_default_init(self):
        """Test default initialization."""
        model = Model()
        
        assert model.name == "pRF_Gaussian"
        assert model.param_names == ['x0', 'y0', 'sigma']
        assert model.scaled_param == [True, True, True]
        assert model.only_positive == [False, False, True]
        assert model.tr == 1.0
        assert model.hrf == 0
        
    def test_custom_init(self):
        """Test custom initialization."""
        model = Model(
            name="custom_pRF",
            param_names=['x', 'y', 's', 'amp'],
            tr=2.0,
            hrf=None
        )
        
        assert model.name == "custom_pRF"
        assert model.param_names == ['x', 'y', 's', 'amp']
        assert model.tr == 2.0
        assert model.hrf is None
        
    def test_get_search_space(self):
        """Test search space retrieval."""
        model = Model()
        search_space = model.get_search_space()
        
        assert len(search_space) == 3  # 3 parameters
        assert len(search_space[0]) > 1  # param1 should have multiple values
        
    def test_scale_search_space(self):
        """Test search space scaling."""
        model = Model()
        original_param2 = model.param2.copy()
        
        scaling_factor = 10.0
        model.scale_search_space(scaling_factor)
        
        # param2 and param3 should be scaled (scaled_param = [True, True, True])
        assert np.allclose(model.param2, original_param2 * scaling_factor)
        assert model.scaling_factor == scaling_factor
        
    def test_post_init_adjustment(self):
        """Test automatic adjustment of boolean fields."""
        model = Model(param_names=['x', 'y', 's', 'amp'])
        
        # Should automatically adjust scaled_param and only_positive
        assert len(model.scaled_param) == 4
        assert len(model.only_positive) == 4


class TestPRFResults:
    """Test PRFResults class functionality."""
    
    def test_init_empty(self):
        """Test empty initialization."""
        results = PRFResults()
        
        assert results.parameters is None
        assert results.r_squared is None
        assert results.predicted is None
        
    def test_init_with_data(self):
        """Test initialization with data."""
        n_vertices = 100
        n_params = 3
        
        parameters = np.random.rand(n_params, n_vertices)
        r_squared = np.random.rand(n_vertices)
        
        results = PRFResults(
            parameters=parameters,
            r_squared=r_squared
        )
        
        assert results.parameters.shape == (n_params, n_vertices)
        assert results.r_squared.shape == (n_vertices,)
        
    def test_dimension_validation(self):
        """Test validation of dimension consistency."""
        parameters = np.random.rand(3, 100)
        r_squared = np.random.rand(50)  # Wrong size
        
        with pytest.raises(ValueError):
            PRFResults(parameters=parameters, r_squared=r_squared)
            
    def test_to_srf(self):
        """Test conversion to SrfData."""
        # Create mock model
        model = Model()
        
        # Create results
        n_vertices = 100
        parameters = np.random.rand(3, n_vertices)
        r_squared = np.random.rand(n_vertices)
        
        results = PRFResults(
            parameters=parameters,
            r_squared=r_squared
        )
        
        # Convert to SrfData
        srf = results.to_srf(model)
        
        assert isinstance(srf, SrfData)
        assert srf.data.shape[0] == 4  # 3 parameters + R²
        assert srf.data.shape[1] == n_vertices
        assert 'R^2' in srf.values
        
    def test_to_srf_with_base(self):
        """Test conversion with base SrfData."""
        # Create base SrfData with geometry
        base_srf = SrfData("lh")
        base_srf.vertices = np.random.rand(100, 3)
        base_srf.faces = np.random.randint(0, 100, (200, 3))
        
        # Create results
        model = Model()
        results = PRFResults(
            parameters=np.random.rand(3, 100),
            r_squared=np.random.rand(100)
        )
        
        # Convert with base
        srf = results.to_srf(model, base_srf)
        
        assert srf.hemisphere == "lh"
        assert np.array_equal(srf.vertices, base_srf.vertices)
        assert np.array_equal(srf.faces, base_srf.faces)


class TestDataStructureIntegration:
    """Test integration between data structures."""
    
    def test_complete_workflow(self):
        """Test complete workflow from Model to SrfData."""
        # Create model
        model = Model(
            name="test_model",
            param_names=['x0', 'y0', 'sigma', 'amplitude'],
            tr=2.0
        )
        
        # Create mock results
        n_vertices = 50
        parameters = np.random.rand(4, n_vertices)
        r_squared = np.random.rand(n_vertices)
        
        # Simulate some realistic parameter values
        parameters[0, :] = np.random.uniform(-5, 5, n_vertices)  # x0
        parameters[1, :] = np.random.uniform(-5, 5, n_vertices)  # y0
        parameters[2, :] = np.random.uniform(0.5, 3, n_vertices)  # sigma
        parameters[3, :] = np.random.uniform(0, 1, n_vertices)  # amplitude
        
        results = PRFResults(
            parameters=parameters,
            r_squared=r_squared
        )
        
        # Convert to SrfData
        srf = results.to_srf(model)
        
        # Verify everything is consistent
        assert srf.data.shape == (5, n_vertices)  # 4 params + R²
        assert len(srf.values) == 5
        assert srf.values[:4] == model.param_names
        assert srf.values[4] == 'R^2'
        
        # Test parameter retrieval
        x0_data = srf.get_parameter('x0')
        assert np.array_equal(x0_data, parameters[0, :])
        
        r2_data = srf.get_parameter('R^2')
        assert np.array_equal(r2_data, r_squared)
        
    def test_roi_analysis_workflow(self):
        """Test workflow with ROI filtering."""
        # Create full brain data
        model = Model()
        full_vertices = 10000
        
        # Create SrfData with full brain
        srf = SrfData("lh")
        srf.vertices = np.random.rand(full_vertices, 3)
        
        # Add parameter data
        data = np.random.rand(3, full_vertices)
        values = ['x0', 'y0', 'sigma']
        srf.add_data(data, values)
        
        # Define ROI (e.g., V1)
        v1_indices = np.random.choice(full_vertices, 500, replace=False)
        v1_srf = srf.filter_by_roi(v1_indices)
        
        # Verify ROI data
        assert v1_srf.data.shape[1] == 500
        assert v1_srf.vertices.shape[0] == 500
        assert len(v1_srf.roi) == 500
        
        # Verify data consistency
        for i, param in enumerate(values):
            original_data = srf.get_parameter(param)
            roi_data = v1_srf.get_parameter(param)
            assert np.array_equal(roi_data, original_data[v1_indices])


if __name__ == "__main__":
    pytest.main([__file__])