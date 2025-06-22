"""
Command Line Interface for PySamSrf.

This module provides a command-line interface for running pRF analysis
and other PySamSrf functions from the terminal.
"""

import argparse
import sys
import json
from pathlib import Path
import numpy as np

from . import __version__
from .core.data_structures import SrfData, Model
from .fitting.fit_prf import fit_prf
from .io.surface_io import load_surface, save_surface
from .io.volume_io import load_volume, save_volume


def main():
    """Main entry point for PySamSrf CLI."""
    parser = argparse.ArgumentParser(
        description="PySamSrf: Population Receptive Field Analysis in Python",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pysamsrf fit --model gaussian_prf.json --data lh.func.gii --output results/
  pysamsrf simulate --n-vertices 100 --output synthetic_data.srf
  pysamsrf convert --input data.srf --output data.gii --format gifti
        """
    )
    
    parser.add_argument('--version', action='version', version=f'PySamSrf {__version__}')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # pRF fitting command
    fit_parser = subparsers.add_parser('fit', help='Fit pRF models to data')
    fit_parser.add_argument('--model', '-m', required=True, 
                           help='Model configuration file (.json)')
    fit_parser.add_argument('--data', '-d', required=True,
                           help='Input data file (.gii, .srf, .nii)')
    fit_parser.add_argument('--output', '-o', required=True,
                           help='Output directory or file')
    fit_parser.add_argument('--roi', '-r', 
                           help='ROI file for analysis restriction')
    fit_parser.add_argument('--apertures', '-a',
                           help='Apertures file (overrides model setting)')
    fit_parser.add_argument('--parallel', action='store_true',
                           help='Use parallel processing')
    fit_parser.add_argument('--verbose', '-v', action='store_true',
                           help='Verbose output')
    
    # Simulation command
    sim_parser = subparsers.add_parser('simulate', help='Simulate pRF data')
    sim_parser.add_argument('--n-vertices', type=int, default=100,
                           help='Number of vertices to simulate')
    sim_parser.add_argument('--n-timepoints', type=int, default=200,
                           help='Number of timepoints')
    sim_parser.add_argument('--noise-level', type=float, default=0.1,
                           help='Noise level (fraction of signal)')
    sim_parser.add_argument('--output', '-o', required=True,
                           help='Output file')
    sim_parser.add_argument('--seed', type=int, help='Random seed')
    
    # Conversion command
    conv_parser = subparsers.add_parser('convert', help='Convert between file formats')
    conv_parser.add_argument('--input', '-i', required=True,
                           help='Input file')
    conv_parser.add_argument('--output', '-o', required=True,
                           help='Output file')
    conv_parser.add_argument('--format', choices=['gifti', 'samsrf', 'freesurfer'],
                           help='Output format (auto-detect if not specified)')
    
    # Analysis command
    analysis_parser = subparsers.add_parser('analyze', help='Analyze pRF results')
    analysis_parser.add_argument('--input', '-i', required=True,
                                help='Input pRF results file')
    analysis_parser.add_argument('--output', '-o', required=True,
                                help='Output directory for plots and stats')
    analysis_parser.add_argument('--roi', '-r',
                                help='ROI file for analysis')
    analysis_parser.add_argument('--plots', nargs='+', 
                                choices=['eccentricity', 'polar', 'surface', 'coverage'],
                                default=['eccentricity', 'coverage'],
                                help='Types of plots to generate')
    
    # Validation command
    val_parser = subparsers.add_parser('validate', help='Validate installation and data')
    val_parser.add_argument('--test-data', action='store_true',
                           help='Run validation using test data')
    val_parser.add_argument('--benchmark', action='store_true',
                           help='Run performance benchmarks')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'fit':
            return cmd_fit(args)
        elif args.command == 'simulate':
            return cmd_simulate(args)
        elif args.command == 'convert':
            return cmd_convert(args)
        elif args.command == 'analyze':
            return cmd_analyze(args)
        elif args.command == 'validate':
            return cmd_validate(args)
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose if hasattr(args, 'verbose') else False:
            import traceback
            traceback.print_exc()
        return 1


def cmd_fit(args):
    """Handle pRF fitting command."""
    print(f"PySamSrf v{__version__} - pRF Fitting")
    print("=" * 40)
    
    # Load model configuration
    with open(args.model, 'r') as f:
        model_config = json.load(f)
    
    model = Model(**model_config)
    
    # Load data
    print(f"Loading data: {args.data}")
    if args.data.endswith('.gii') or args.data.endswith('.srf'):
        srf_data = load_surface(args.data)
    elif args.data.endswith('.nii') or args.data.endswith('.nii.gz'):
        srf_data = load_volume(args.data)
    else:
        raise ValueError(f"Unsupported data format: {args.data}")
    
    print(f"Data loaded: {srf_data.y.shape[0] if srf_data.y is not None else 0} vertices")
    
    # Load apertures if specified
    apertures = None
    if args.apertures:
        print(f"Loading apertures: {args.apertures}")
        # This would load apertures from file
        pass
    
    # Run fitting
    print("Starting pRF fitting...")
    results = fit_prf(
        model=model,
        srf_data=srf_data,
        roi=args.roi,
        apertures=apertures,
        parallel=args.parallel,
        verbose=args.verbose
    )
    
    # Save results
    output_path = Path(args.output)
    if output_path.is_dir():
        output_file = output_path / f"prf_results_{model.name}.srf"
    else:
        output_file = output_path
    
    print(f"Saving results: {output_file}")
    result_srf = results.to_srf(model, srf_data)
    save_surface(result_srf, output_file)
    
    # Print summary
    print("\nFitting completed!")
    print(f"Mean R²: {np.mean(results.r_squared):.3f}")
    print(f"Vertices with R² > 0.1: {np.sum(results.r_squared > 0.1)}")
    
    return 0


def cmd_simulate(args):
    """Handle simulation command."""
    print(f"PySamSrf v{__version__} - Data Simulation")
    print("=" * 40)
    
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed: {args.seed}")
    
    print(f"Simulating {args.n_vertices} vertices with {args.n_timepoints} timepoints")
    print(f"Noise level: {args.noise_level}")
    
    # Create synthetic pRF parameters
    from .simulation.simulate_prfs import simulate_prf_experiment
    
    # Generate ground truth parameters
    x0_true = np.random.uniform(-8, 8, args.n_vertices)
    y0_true = np.random.uniform(-8, 8, args.n_vertices) 
    sigma_true = np.random.uniform(0.5, 3, args.n_vertices)
    
    # Simulate experiment
    srf_data = simulate_prf_experiment(
        x0_true, y0_true, sigma_true,
        n_timepoints=args.n_timepoints,
        noise_level=args.noise_level
    )
    
    # Save results
    print(f"Saving synthetic data: {args.output}")
    save_surface(srf_data, args.output)
    
    print("Simulation completed!")
    return 0


def cmd_convert(args):
    """Handle file conversion command."""
    print(f"PySamSrf v{__version__} - File Conversion")
    print("=" * 40)
    
    print(f"Converting {args.input} -> {args.output}")
    
    # Load input file
    if args.input.endswith('.gii') or args.input.endswith('.srf'):
        data = load_surface(args.input)
    elif args.input.endswith('.nii') or args.input.endswith('.nii.gz'):
        data = load_volume(args.input)
    else:
        raise ValueError(f"Unsupported input format: {args.input}")
    
    # Save in output format
    if args.output.endswith('.gii') or args.output.endswith('.srf'):
        save_surface(data, args.output, format=args.format)
    elif args.output.endswith('.nii') or args.output.endswith('.nii.gz'):
        save_volume(data, args.output)
    else:
        raise ValueError(f"Unsupported output format: {args.output}")
    
    print("Conversion completed!")
    return 0


def cmd_analyze(args):
    """Handle analysis command."""
    print(f"PySamSrf v{__version__} - Results Analysis")
    print("=" * 40)
    
    # Load results
    print(f"Loading results: {args.input}")
    if args.input.endswith('.gii') or args.input.endswith('.srf'):
        srf_data = load_surface(args.input)
    else:
        raise ValueError(f"Unsupported input format: {args.input}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate requested plots
    from .analysis.plotting import plot_eccentricity, plot_visual_field_coverage, plot_polar
    import matplotlib.pyplot as plt
    
    for plot_type in args.plots:
        print(f"Generating {plot_type} plot...")
        
        if plot_type == 'eccentricity':
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_eccentricity(srf_data, 'sigma', ax=ax)
            plt.savefig(output_dir / 'eccentricity_sigma.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        elif plot_type == 'coverage':
            fig, ax = plt.subplots(figsize=(8, 8))
            plot_visual_field_coverage(srf_data, ax=ax)
            plt.savefig(output_dir / 'visual_field_coverage.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        elif plot_type == 'polar':
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
            plot_polar(srf_data, 'sigma', ax=ax)
            plt.savefig(output_dir / 'polar_sigma.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Generate summary statistics
    from .analysis.statistics import calculate_statistics
    
    stats = calculate_statistics(srf_data)
    with open(output_dir / 'summary_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    
    print(f"Analysis completed! Results saved to {output_dir}")
    return 0


def cmd_validate(args):
    """Handle validation command."""
    print(f"PySamSrf v{__version__} - Validation")
    print("=" * 40)
    
    if args.test_data:
        print("Running validation with test data...")
        
        # Create simple test data
        from .core.data_structures import SrfData, Model
        from .core.prf_functions import gaussian_rf
        
        # Test basic functionality
        print("✓ Testing data structures...")
        srf = SrfData("test")
        model = Model()
        
        print("✓ Testing pRF functions...")
        rf = gaussian_rf(0, 0, 1, 100)
        assert rf.shape == (100, 100)
        
        print("✓ Testing HRF functions...")
        from .core.hrf_functions import canonical_hrf
        hrf = canonical_hrf(2.0, 'spm')
        assert len(hrf) > 0
        
        print("All tests passed!")
    
    if args.benchmark:
        print("\nRunning performance benchmarks...")
        
        import time
        
        # Benchmark pRF generation
        start = time.time()
        for _ in range(100):
            rf = gaussian_rf(np.random.uniform(-5, 5), np.random.uniform(-5, 5), 
                           np.random.uniform(0.5, 3), 100)
        elapsed = time.time() - start
        print(f"pRF generation: {elapsed/100*1000:.2f} ms per RF")
        
        # Benchmark HRF convolution
        from .core.hrf_functions import convolve_hrf
        neural = np.random.rand(200)
        hrf = canonical_hrf(2.0, 'spm')
        
        start = time.time()
        for _ in range(100):
            bold = convolve_hrf(neural, hrf)
        elapsed = time.time() - start
        print(f"HRF convolution: {elapsed/100*1000:.2f} ms per convolution")
        
        print("Benchmarks completed!")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())