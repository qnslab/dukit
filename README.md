# dukit

Analysis toolkit for widefield quantum defect microscopy. Built for NV-center imaging
and extensible to other spin defects and quantitative hyperspectral microscopy.

## Capabilities

dukit handles the complete workflow from raw image stacks to quantitative analysis:

Image processing including drift correction, ROI cropping, rebinning, smoothing, and
background removal. Spectral fitting of ODMR, Rabi, and T1 data with built-in models
for Lorentzians, damped oscillations, and stretched exponentials. Magnetic field
quantification from defect resonance shifts. Vector magnetic field reconstruction via
Fourier propagation, matrix inversion, and Hamiltonian fitting methods. Simulation
of field distributions from arbitrary magnetic geometries. Interactive and
publication-quality visualization tools.

## Installation

```bash
pip install dukit
```

For GPU-accelerated fitting (NVIDIA, Windows or Linux):

```bash
pip install dukit[gpufit]
```

For CPU-parallel fitting:

```bash
pip install dukit[cpufit]
```

See INSTALL.md for detailed setup instructions including git configuration and
JupyterLab setup.

## Quick Start

```python
import dukit

# Load your microscope configuration
system = dukit.PyControl("/path/to/data")

# Load image stack and sweep parameters
image_stack, sweep_arr = system.read_image("experiment_001")

# Fit all pixels with a two-Lorentzian ODMR model
model = dukit.LinearLorentzians(n_lorentzians=2)
results = dukit.fit_all_pixels(image_stack, sweep_arr, model)

# Plot a fit parameter (e.g., resonance frequency)
dukit.plot.pl_param_image(results, param_name="f0")
```

## Documentation

API reference documentation is available at https://qnslab.github.io/dukit

The examples/ directory contains Jupyter notebooks demonstrating drift correction,
fitting workflows, magnetic simulation, and other common tasks.

For development guidelines, architecture overview, and instructions on adding new
model functions, see DEVDOCS.md.

## For External Labs

dukit uses a pluggable System class for hardware-specific data I/O. To adapt the
toolkit to your microscope setup, subclass dukit.systems.System and implement the
abstract methods read_image() and get_raw_pixel_size(). See dukit/systems.py for
implementation examples including support for multiple camera types and control
software.

## Project Status

Currently implemented: Image registration and drift correction, PL fitting via
scipy/cpufit/gpufit backends with uncertainty quantification, local and vector
magnetic field reconstruction from ODMR data (via Fourier propagation, matrix
inversion, and Hamiltonian fitting methods), magnetic sample simulation, and
comprehensive plotting utilities.

Features in development: Source reconstruction and aberration correction. These
capabilities are planned for future releases.

## Normal Usage

We typically use dukit within Jupyter notebooks, with one notebook per measurement.
This preserves the connection between analysis code and generated figures, supporting
reproducible research workflows.
