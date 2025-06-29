"""Setup script for PySamSrf."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="pysamsrf",
    version="1.0.0",
    author="PySamSrf Developers",
    author_email="",
    description="Python implementation of SamSrf for population receptive field analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pysamsrf/pysamsrf",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "black>=22.0.0",
            "flake8>=4.0.0", 
            "mypy>=0.910",
            "pytest-cov>=3.0.0",
        ],
        "speedup": [
            "numba>=0.56.0",
        ],
        "ml": [
            "scikit-learn>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pysamsrf=pysamsrf.cli:main",
        ],
    },
)