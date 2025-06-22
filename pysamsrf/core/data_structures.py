"""
Core data structures for PySamSrf.

This module defines the primary data structures used throughout PySamSrf,
equivalent to the MATLAB Srf and Model structures in the original SamSrf.
"""

import numpy as np
import warnings
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path


class SrfData:
    """
    Surface/Volume data structure equivalent to MATLAB Srf structure.
    
    Contains neuroimaging data, surface geometry, and analysis results for
    population receptive field analysis.
    """
    
    def __init__(self, hemisphere: str = ""):
        """
        Initialize SrfData structure.
        
        Parameters
        ----------
        hemisphere : str
            Hemisphere identifier ('lh', 'rh', 'bi', 'vol', 'eeg')
        """
        # Core data fields
        self.data: Optional[np.ndarray] = None  # Parameters/timeseries (params x vertices)
        self.values: List[str] = []  # Parameter names corresponding to data rows
        self.hemisphere: str = hemisphere
        self.version: str = "PySamSrf v1.0.0"
        
        # Surface geometry
        self.vertices: Optional[np.ndarray] = None  # 3D coordinates (N x 3)
        self.faces: Optional[np.ndarray] = None     # Triangle connectivity (M x 3)
        
        # Anatomical properties
        self.curvature: Optional[np.ndarray] = None
        self.area: Optional[np.ndarray] = None
        self.thickness: Optional[np.ndarray] = None
        
        # Alternative surface coordinates
        self.inflated: Optional[np.ndarray] = None
        self.pial: Optional[np.ndarray] = None
        self.sphere: Optional[np.ndarray] = None
        
        # Analysis-specific fields
        self.x: Optional[np.ndarray] = None  # Predicted time courses
        self.y: Optional[np.ndarray] = None  # Raw time courses
        self.raw_data: Optional[np.ndarray] = None  # Unsmoothed data
        self.noise_ceiling: Optional[np.ndarray] = None
        self.roi: Optional[np.ndarray] = None  # ROI vertex indices
        self.ground_truth: Optional[np.ndarray] = None  # For simulations
        
        # File information
        self.functional: List[str] = []  # Source functional files
        self.structural: str = ""        # Source structural file
        
    def __repr__(self) -> str:
        """String representation of SrfData."""
        n_vertices = self.vertices.shape[0] if self.vertices is not None else 0
        n_params = len(self.values)
        return (f"SrfData(hemisphere='{self.hemisphere}', "
                f"vertices={n_vertices}, parameters={n_params})")
    
    def add_data(self, data: np.ndarray, values: List[str]) -> None:
        """
        Add parameter data to the structure.
        
        Parameters
        ----------
        data : np.ndarray
            Parameter values (n_params x n_vertices)
        values : List[str]
            Parameter names
        """
        if data.shape[0] != len(values):
            raise ValueError("Number of parameter rows must match number of value names")
            
        self.data = data
        self.values = values
    
    def get_parameter(self, param_name: str) -> Optional[np.ndarray]:
        """
        Get data for a specific parameter.
        
        Parameters
        ----------
        param_name : str
            Name of parameter to retrieve
            
        Returns
        -------
        np.ndarray or None
            Parameter data if found, None otherwise
        """
        if param_name in self.values and self.data is not None:
            idx = self.values.index(param_name)
            return self.data[idx, :]
        return None
    
    def set_parameter(self, param_name: str, data: np.ndarray) -> None:
        """
        Set data for a specific parameter.
        
        Parameters
        ----------
        param_name : str
            Name of parameter
        data : np.ndarray
            Parameter data (n_vertices,)
        """
        if param_name in self.values:
            idx = self.values.index(param_name)
            if self.data is not None:
                self.data[idx, :] = data
            else:
                raise ValueError("No data matrix initialized")
        else:
            # Add new parameter
            if self.data is None:
                self.data = data.reshape(1, -1)
                self.values = [param_name]
            else:
                self.data = np.vstack([self.data, data.reshape(1, -1)])
                self.values.append(param_name)
    
    def filter_by_roi(self, roi_indices: np.ndarray) -> 'SrfData':
        """
        Create filtered copy with only ROI vertices.
        
        Parameters
        ----------
        roi_indices : np.ndarray
            Vertex indices to include
            
        Returns
        -------
        SrfData
            Filtered copy
        """
        filtered = SrfData(self.hemisphere)
        
        # Copy basic info
        filtered.values = self.values.copy()
        filtered.version = self.version
        filtered.functional = self.functional.copy()
        filtered.structural = self.structural
        
        # Filter data arrays
        if self.data is not None:
            filtered.data = self.data[:, roi_indices]
        if self.vertices is not None:
            filtered.vertices = self.vertices[roi_indices, :]
        if self.curvature is not None:
            filtered.curvature = self.curvature[roi_indices]
        if self.area is not None:
            filtered.area = self.area[roi_indices]
        if self.thickness is not None:
            filtered.thickness = self.thickness[roi_indices]
            
        # Store ROI info
        filtered.roi = roi_indices
        
        return filtered


@dataclass
class Model:
    """
    Model configuration for pRF analysis.
    
    Equivalent to MATLAB Model structure, containing all parameters
    needed for population receptive field fitting.
    """
    
    # Basic model info
    name: str = "pRF_Gaussian"
    prf_function: Optional[callable] = None
    param_names: List[str] = field(default_factory=lambda: ['x0', 'y0', 'sigma'])
    scaled_param: List[bool] = field(default_factory=lambda: [True, True, True])
    only_positive: List[bool] = field(default_factory=lambda: [False, False, True])
    
    # Analysis parameters
    scaling_factor: float = np.nan
    tr: float = 1.0  # Repetition time
    hrf: Union[str, np.ndarray, int, None] = 0  # 0=SPM canonical, None=de Haas
    aperture_file: str = ""
    
    # Search space configuration
    polar_search_space: bool = True
    param1: np.ndarray = field(default_factory=lambda: np.arange(0, 360, 10))  # Polar angle
    param2: np.ndarray = field(default_factory=lambda: 2.0**np.arange(-5, 0.6, 0.2))  # Eccentricity
    param3: np.ndarray = field(default_factory=lambda: 2.0**np.arange(-5.6, 1, 0.2))  # Sigma
    param4: np.ndarray = field(default_factory=lambda: np.array([0]))
    param5: np.ndarray = field(default_factory=lambda: np.array([0]))
    param6: np.ndarray = field(default_factory=lambda: np.array([0]))
    param7: np.ndarray = field(default_factory=lambda: np.array([0]))
    param8: np.ndarray = field(default_factory=lambda: np.array([0]))
    param9: np.ndarray = field(default_factory=lambda: np.array([0]))
    param10: np.ndarray = field(default_factory=lambda: np.array([0]))
    
    # Fitting options
    coarse_fit_only: bool = False
    smoothed_coarse_fit: float = 0.0
    compressive_nonlinearity: bool = False
    hooke_jeeves_steps: int = 0
    noise_ceiling_threshold: float = 0.0
    
    # Advanced options
    seed_roi: str = ""  # For connective field analysis
    template: str = ""  # Template map for CF analysis
    fit_prf: int = 1  # For CF analysis: -1=summary, 0=convex hull, 1=Gaussian
    
    def __post_init__(self):
        """Validate model configuration after initialization."""
        if len(self.param_names) != len(self.scaled_param):
            self.scaled_param = [True] * len(self.param_names)
            warnings.warn("scaled_param length adjusted to match param_names")
            
        if len(self.param_names) != len(self.only_positive):
            self.only_positive = [False] * len(self.param_names)
            warnings.warn("only_positive length adjusted to match param_names")
    
    def get_search_space(self) -> List[np.ndarray]:
        """
        Get search space parameters as list.
        
        Returns
        -------
        List[np.ndarray]
            Search space grids for each parameter
        """
        search_params = []
        for i in range(len(self.param_names)):
            param_attr = f"param{i+1}"
            if hasattr(self, param_attr):
                param_values = getattr(self, param_attr)
                if not isinstance(param_values, np.ndarray):
                    param_values = np.array([param_values])
                search_params.append(param_values)
            else:
                search_params.append(np.array([0]))
        return search_params
    
    def scale_search_space(self, scaling_factor: float) -> None:
        """
        Scale search space parameters by scaling factor.
        
        Parameters
        ----------
        scaling_factor : float
            Scaling factor (typically eccentricity range)
        """
        self.scaling_factor = scaling_factor
        
        # Scale appropriate parameters
        for i, (param_name, should_scale) in enumerate(zip(self.param_names, self.scaled_param)):
            if should_scale:
                param_attr = f"param{i+1}"
                if hasattr(self, param_attr):
                    current_values = getattr(self, param_attr)
                    scaled_values = current_values * scaling_factor
                    setattr(self, param_attr, scaled_values)


@dataclass 
class PRFResults:
    """
    Results from pRF fitting analysis.
    
    Contains fitted parameters, goodness of fit metrics, and 
    other analysis outputs.
    """
    
    # Fitted parameters
    parameters: Optional[np.ndarray] = None  # Best-fit parameters (n_params x n_vertices)
    r_squared: Optional[np.ndarray] = None   # Goodness of fit
    residuals: Optional[np.ndarray] = None   # Model residuals
    
    # Time course data
    predicted: Optional[np.ndarray] = None   # Predicted time courses
    observed: Optional[np.ndarray] = None    # Observed time courses
    
    # Search space results
    coarse_fit: Optional[np.ndarray] = None  # Coarse fit parameters
    correlation_maps: Optional[np.ndarray] = None  # Search space correlations
    
    # Fitting metadata
    n_iterations: Optional[np.ndarray] = None  # Optimization iterations
    fit_time: Optional[np.ndarray] = None      # Fitting time per vertex
    convergence: Optional[np.ndarray] = None   # Convergence flags
    
    def __post_init__(self):
        """Validate results structure."""
        if self.parameters is not None and self.r_squared is not None:
            if self.parameters.shape[1] != self.r_squared.shape[0]:
                raise ValueError("Parameter and R² dimensions must match")
    
    def to_srf(self, model: Model, base_srf: Optional[SrfData] = None) -> SrfData:
        """
        Convert results to SrfData structure.
        
        Parameters
        ----------
        model : Model
            Model configuration used for fitting
        base_srf : SrfData, optional
            Base structure to copy geometry from
            
        Returns
        -------
        SrfData
            Results in SrfData format
        """
        if base_srf is not None:
            srf = SrfData(base_srf.hemisphere)
            srf.vertices = base_srf.vertices
            srf.faces = base_srf.faces
            srf.curvature = base_srf.curvature
            srf.area = base_srf.area
        else:
            srf = SrfData()
        
        # Add parameter data
        if self.parameters is not None:
            values = model.param_names.copy()
            data = self.parameters.copy()
            
            # Add R² if available
            if self.r_squared is not None:
                values.append('R^2')
                data = np.vstack([data, self.r_squared.reshape(1, -1)])
            
            srf.add_data(data, values)
        
        # Add time course data
        srf.x = self.predicted
        srf.y = self.observed
        
        return srf