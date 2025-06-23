# Data Loading in PySamSrf

This document describes the data loading functionality implemented in PySamSrf for loading neuroimaging data, stimulus apertures, and ROI definitions.

## Overview

PySamSrf provides comprehensive data loading utilities for various neuroimaging formats commonly used in pRF analysis:

- **Surface data**: GIFTI (.gii), FreeSurfer (.surf, .pial, .white, .inflated), SamSrf (.srf)
- **Volume data**: NIfTI (.nii, .nii.gz)
- **Aperture/stimulus data**: MATLAB (.mat), NumPy (.npz)
- **ROI data**: GIFTI labels (.gii), FreeSurfer labels (.label, .annot), NIfTI masks (.nii)

## Installation

### Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/samsrf.git
cd samsrf

# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate pysamsrf
```

### Using pip

```bash
# Install dependencies
pip install -r requirements.txt

# Install pysamsrf in development mode
pip install -e .
```

## Loading Surface Data

```python
from pysamsrf.io import load_surface, save_surface

# Load GIFTI surface
srf = load_surface('lh.prf_results.gii')

# Load FreeSurfer surface
srf = load_surface('lh.inflated')

# Load SamSrf MATLAB format
srf = load_surface('Srf_lh_pRF.srf')

# Save surface data
save_surface(srf, 'output.gii', format='gifti')
```

## Loading Volume Data

```python
from pysamsrf.io import load_volume, save_volume

# Load NIfTI volume
srf = load_volume('prf_results.nii.gz')

# Load with brain mask
srf = load_volume('prf_results.nii.gz', mask='brain_mask.nii.gz')

# Save volume data
save_volume(srf, 'output.nii.gz', parameter='R2')
```

## Loading Stimulus Apertures

```python
from pysamsrf.io import load_apertures, save_apertures

# Load MATLAB aperture file
aperture_data = load_apertures('aps_SweepingBars.mat')
apertures = aperture_data['apertures']
resolution = aperture_data['resolution']

# Load NumPy aperture file
aperture_data = load_apertures('apertures.npz')

# Save apertures
save_apertures(apertures, 'apertures.npz', 
               resolution=[101, 101], 
               timing=np.arange(200) * 2.0)  # TR=2s
```

### Creating Synthetic Apertures

PySamSrf also provides utilities for creating synthetic aperture stimuli:

```python
from pysamsrf.io.aperture_io import (
    create_bar_apertures, 
    create_wedge_apertures, 
    create_ring_apertures
)

# Create sweeping bar stimulus
bars = create_bar_apertures(n_steps=20, bar_width=2.0, resolution=101)

# Create rotating wedge stimulus
wedges = create_wedge_apertures(n_steps=32, wedge_width=45, n_cycles=2)

# Create expanding ring stimulus
rings = create_ring_apertures(n_steps=20, ring_width=1.5, expand=True)
```

## Loading ROI Data

```python
from pysamsrf.io import load_label

# Load FreeSurfer label
roi_vertices = load_label('V1.label')

# Load FreeSurfer annotation
roi_vertices = load_label('aparc.annot')

# For GIFTI ROI files, use the general _load_roi function
from pysamsrf.fitting.fit_prf import _load_roi
roi_vertices = _load_roi('V1_roi.gii')
```

## Integration with pRF Fitting

The data loading functions are integrated into the main pRF fitting pipeline:

```python
from pysamsrf.core.data_structures import Model
from pysamsrf.fitting.fit_prf import fit_prf

# Create model
model = Model(
    name='pRF_Gaussian',
    aperture_file='apertures.mat',
    tr=2.0
)

# Fit pRF - data loading is handled automatically
results = fit_prf(
    model=model,
    srf_data='lh.surface.gii',  # Can be path or SrfData object
    roi='V1.label'  # Optional ROI restriction (.label, .annot, .gii, .nii)
)
```

## Working with Model Apertures

The Model class can automatically load apertures:

```python
# Create model with aperture file
model = Model(aperture_file='stimulus_apertures.mat')

# Load apertures
model.load_apertures()

# Access loaded data
print(f"Aperture shape: {model.apertures.shape}")
print(f"Resolution: {model.aperture_resolution}")
```

## File Format Details

### MATLAB Aperture Files

PySamSrf supports both older (.mat) and newer (v7.3 HDF5) MATLAB files. Common variable names are automatically detected:
- ApFrm, ApFrms, apertures, Apertures, stim, stimulus

### Surface Data Formats

- **GIFTI**: Full support for geometry and functional data
- **FreeSurfer**: Geometry only (use separate functional files)
- **SamSrf**: Complete MATLAB structure compatibility

### Volume Data Formats

- **NIfTI**: 3D (parameter maps) and 4D (time series) support
- Automatic masking of zero voxels
- Preserves affine transformation and header information

## Troubleshooting

### Import Errors

If you encounter import errors, ensure all dependencies are installed:

```bash
pip install numpy scipy nibabel h5py pandas matplotlib joblib
```

### MATLAB File Loading

For MATLAB v7.3 files, h5py is required. Older formats use scipy.io.loadmat.

### Memory Issues

For large datasets, consider:
- Loading data in chunks
- Using ROI masks to reduce data size
- Enabling parallel processing with limited workers

## API Reference

### Surface I/O
- `load_surface(filepath, load_data=True)`: Load surface data
- `save_surface(srf, filepath, format=None)`: Save surface data
- `load_label(filepath)`: Load ROI labels
- `save_label(vertices, filepath, coordinates=None)`: Save ROI labels

### Volume I/O
- `load_volume(filepath, mask=None)`: Load volume data
- `save_volume(srf, filepath, parameter=None, template=None)`: Save volume data

### Aperture I/O
- `load_apertures(filepath)`: Load stimulus apertures
- `save_apertures(apertures, filepath, resolution=None, timing=None, metadata=None)`: Save apertures
- `create_bar_apertures(n_steps, bar_width, resolution=101, orientations=None)`: Create bar stimuli
- `create_wedge_apertures(n_steps, wedge_width, resolution=101, n_cycles=1)`: Create wedge stimuli
- `create_ring_apertures(n_steps, ring_width, resolution=101, expand=True)`: Create ring stimuli