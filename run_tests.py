#!/usr/bin/env python3
"""
Comprehensive test runner for PySamSrf.

This script runs all tests and provides detailed reporting on
the status of the Python translation.
"""

import sys
import pytest
import argparse
from pathlib import Path


def main():
    """Run PySamSrf test suite."""
    parser = argparse.ArgumentParser(description="Run PySamSrf tests")
    parser.add_argument(
        "--coverage", 
        action="store_true", 
        help="Run tests with coverage reporting"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose test output"
    )
    parser.add_argument(
        "--module", "-m",
        type=str,
        help="Run tests for specific module (e.g., 'test_prf_functions')"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarks"
    )
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = []
    
    if args.verbose:
        pytest_args.append("-v")
    
    if args.coverage:
        pytest_args.extend([
            "--cov=pysamsrf",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    if args.module:
        pytest_args.append(f"tests/test_{args.module}.py")
    else:
        pytest_args.append("tests/")
    
    # Add markers for different test types
    if not args.benchmark:
        pytest_args.extend(["-m", "not benchmark"])
    
    print("=" * 60)
    print("PySamSrf Test Suite")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Test directory: {Path('tests').absolute()}")
    print(f"Running with args: {' '.join(pytest_args)}")
    print("=" * 60)
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
        if args.coverage:
            print("\nCoverage report generated in htmlcov/index.html")
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed!")
        print("=" * 60)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())