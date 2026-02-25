#!/usr/bin/env python
"""Simple test of the new plot_field_images function."""

import numpy as np
import dukit

# Create some test field data
nx, ny = 50, 50
x = np.linspace(-2, 2, nx)
y = np.linspace(-2, 2, ny)
X, Y = np.meshgrid(x, y)

# Create synthetic field components
Bx = np.sin(np.pi * X) * np.cos(np.pi * Y)
By = np.cos(np.pi * X) * np.sin(np.pi * Y)
Bz = np.exp(-(X**2 + Y**2) / 2)

# Test the new plotting function
field_data = {
    "Bx": Bx,
    "By": By,
    "Bz": Bz,
}

# Plot all fields
fig, ax = dukit.plot.plot_field_images(
    field_data,
    name="Test Fields",
    c_range_type="min_max",
    c_label="Field (a.u.)",
)

print("Test successful: plot_field_images works correctly")
print("Field shapes:", [f"{k}: {v.shape}" for k, v in field_data.items()])

# Test with field selection
fig2, ax2 = dukit.plot.plot_field_images(
    field_data,
    field_names=["Bx", "Bz"],
    name="Selected Fields",
    c_range_type="percentile",
    c_range_values=(5, 95),
)

print("Field selection test successful")

# Test with uncertainties
sigma_Bx = 0.1 * np.random.rand(ny, nx)
sigma_By = 0.1 * np.random.rand(ny, nx)
sigma_Bz = 0.05 * np.random.rand(ny, nx)

uncertainty_data = {
    "sigma_Bx": sigma_Bx,
    "sigma_By": sigma_By,
    "sigma_Bz": sigma_Bz,
}

fig3, ax3 = dukit.plot.plot_field_images(
    uncertainty_data,
    name="Field Uncertainties",
    c_range_type="strict_range",
    c_range_values=(0, 0.15),
    c_label="Uncertainty (a.u.)",
    c_map="viridis",
)

print("Uncertainty plotting test successful")