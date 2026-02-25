# Source Reconstruction - Design & Implementation Plan

**Status**: Not started - ready for implementation  
**Approach**: Copy-paste then edit from qdmpy source module  
**Estimated effort**: 2-3 hours  
**Complexity**: Low (pure Fourier operations, no parallelization/optimization)

---

## 1. EXECUTIVE SUMMARY

Port source reconstruction functionality from qdmpy to dukit. Source reconstruction inverts measured magnetic field maps to calculate their sources: **current density (Jx, Jy)** and **magnetization (Mz or Mpsi)**.

Unlike field reconstruction (which reconstructs Bx, By, Bz from B_defect), source reconstruction goes one step further to determine what created those fields.

**Key design principles** (following dukit patterns):
- Explicit function parameters (no options dict)
- Defect-agnostic naming: use `b_defect` not `b_nv`, `u_defect` not `u_nv`
- B-field input in **Gauss** (converted to Tesla internally)
- Output in physical units (A/m for current, μB/nm² for magnetization)
- Spherical coordinates for bias field (consistent with field module)
- u_defects normalized internally
- Flat dict output with metadata

---

## 2. QDMPY SOURCE FILES

Located at: `/home/samsc/me/rsc/qdm_fit/qdmpy_proj/qdmpy_git/src/qdmpy/source/`

| File | Purpose | Port? | Notes |
|------|---------|-------|-------|
| `current.py` | Current density algorithms | **YES** | Core reconstruction algorithms |
| `magnetization.py` | Magnetization algorithms | **YES** | Core reconstruction algorithms |
| `interface.py` | High-level dispatcher | **PARTIAL** | Replace options-dict with explicit API |
| `io.py` | Output directory handling | **NO** | Not needed - dukit doesn't use this pattern |
| `__init__.py` | Exports | **NEW** | Create fresh for dukit |

---

## 3. DUKIT FILE STRUCTURE

```
src/dukit/source/
├── __init__.py              # NEW - exports only
├── current.py               # COPY+EDIT from qdmpy
├── magnetization.py         # COPY+EDIT from qdmpy
└── reconstruct.py           # NEW - high-level interface (replaces interface.py)
```

---

## 4. API DESIGN

### 4.1 Current Density Functions

```python
def get_current_from_bxyz(
    Bx: npt.NDArray,
    By: npt.NDArray,
    Bz: npt.NDArray | None = None,
    pixel_size: float,
    standoff: float | None = None,
    nv_layer_thickness: float | None = None,
    nv_above_sample: bool = True,
    use_components: Literal["xy", "z", "xyz", "auto"] = "auto",
    pad_mode: str | None = "edge",
    pad_factor: int = 2,
    k_vector_epsilon: float = 1e-6,
    do_hanning_filter: bool = False,
    hanning_low_cutoff: float | None = None,
    hanning_high_cutoff: float | None = None,
) -> dict[str, npt.NDArray]:
    """
    Reconstruct current density Jx, Jy from magnetic field components.
    
    Parameters
    ----------
    Bx, By : npt.NDArray
        Magnetic field components in Gauss (2D arrays).
    Bz : npt.NDArray | None, optional
        Bz component. Required if use_components="z" or "xyz".
    pixel_size : float
        Effective pixel size in meters.
    standoff : float | None
        Distance from NV layer to sample in meters.
    nv_layer_thickness : float | None
        Thickness of NV layer in meters.
    nv_above_sample : bool, default=True
        True if NV layer is above sample (higher z).
    use_components : {"xy", "z", "xyz", "auto"}, default="auto"
        Which field components to use:
        - "xy": Use Bx, By only (most common)
        - "z": Use Bz only
        - "xyz": Combine all three (weighted average)
        - "auto": Use "xy" if Bx, By provided, else "z"
    pad_mode, pad_factor, k_vector_epsilon
        Fourier padding options.
    do_hanning_filter : bool, default=False
        Apply Hanning filter to reduce noise amplification.
    hanning_low_cutoff, hanning_high_cutoff : float | None
        Cutoff wavelengths in meters for Hanning filter.
    
    Returns
    -------
    dict with keys:
        - "Jx": Current density x-component (A/m)
        - "Jy": Current density y-component (A/m)
        - "Jnorm": Current density magnitude sqrt(Jx² + Jy²) (A/m)
        - "divperp_J": Perpendicular divergence (normalized)
        - "_metadata": Dict with parameters used
    
    Reference
    ---------
    D. A. Broadway et al., Phys. Rev. Applied 14, 024076 (2020)
    https://doi.org/10.1103/PhysRevApplied.14.024076
    """


def get_current_from_bdefect(
    b_defect: npt.NDArray,
    u_defect: npt.NDArray,
    pixel_size: float,
    standoff: float | None = None,
    nv_layer_thickness: float | None = None,
    nv_above_sample: bool = True,
    pad_mode: str | None = "edge",
    pad_factor: int = 2,
    k_vector_epsilon: float = 1e-6,
    do_hanning_filter: bool = False,
    hanning_low_cutoff: float | None = None,
    hanning_high_cutoff: float | None = None,
) -> dict[str, npt.NDArray]:
    """
    Reconstruct current density from single B_defect measurement.
    
    Parameters
    ----------
    b_defect : npt.NDArray
        2D array of B-field along defect axis (Gauss).
    u_defect : npt.NDArray
        Defect orientation unit vector, shape (3,). Normalized internally.
    
    Other parameters same as get_current_from_bxyz.
    
    Returns
    -------
    Same format as get_current_from_bxyz.
    """


def get_current_without_ft(
    Bx: npt.NDArray,
    By: npt.NDArray,
) -> dict[str, npt.NDArray]:
    """
    Approximate current density without Fourier propagation.
    
    Simple rescaling: J ≈ (2/μ₀) × B × (perpendicular conversion)
    Fast but less accurate - useful for quick checks.
    
    Parameters
    ----------
    Bx, By : npt.NDArray
        Magnetic field components in Gauss.
    
    Returns
    -------
    dict with keys "Jx", "Jy", "Jnorm" in A/m.
    """


def get_divperp_j(
    Jx: npt.NDArray,
    Jy: npt.NDArray,
    pixel_size: float,
    pad_mode: str | None = "edge",
    pad_factor: int = 2,
    k_vector_epsilon: float = 1e-6,
) -> npt.NDArray:
    """
    Calculate perpendicular divergence of current density.
    
    ∇⊥ · J = ∂Jx/∂x + ∂Jy/∂y
    
    Returns normalized divergence (divided by max absolute value).
    Useful for checking current conservation.
    """
```

### 4.2 Magnetization Functions

```python
def get_magnetization_from_bxyz(
    Bx: npt.NDArray,
    By: npt.NDArray,
    Bz: npt.NDArray | None = None,
    pixel_size: float,
    magnetization_angle: float | None = None,
    standoff: float | None = None,
    nv_layer_thickness: float | None = None,
    use_components: Literal["xy", "z", "auto"] = "auto",
    pad_mode: str | None = "edge",
    pad_factor: int = 2,
    k_vector_epsilon: float = 1e-6,
    do_hanning_filter: bool = False,
    hanning_low_cutoff: float | None = None,
    hanning_high_cutoff: float | None = None,
) -> dict[str, npt.NDArray]:
    """
    Reconstruct magnetization from magnetic field components.
    
    Parameters
    ----------
    magnetization_angle : float | None, default=None
        In-plane magnetization angle in degrees from +x towards +y.
        None = out-of-plane (Mz).
        Any value = in-plane at that angle (Mpsi).
    
    Other parameters same as get_current_from_bxyz.
    
    Returns
    -------
    dict with keys:
        - "Mz" or "Mpsi": Magnetization (μB/nm²)
        - "_metadata": Dict with parameters used
    
    Note
    ----
    If magnetization_angle is None, returns "Mz" (out-of-plane).
    If magnetization_angle is given, returns "Mpsi" (in-plane at angle).
    """


def get_magnetization_from_bdefect(
    b_defect: npt.NDArray,
    u_defect: npt.NDArray,
    pixel_size: float,
    magnetization_angle: float | None = None,
    standoff: float | None = None,
    nv_layer_thickness: float | None = None,
    nv_above_sample: bool = True,
    pad_mode: str | None = "edge",
    pad_factor: int = 2,
    k_vector_epsilon: float = 1e-6,
    do_hanning_filter: bool = False,
    hanning_low_cutoff: float | None = None,
    hanning_high_cutoff: float | None = None,
) -> dict[str, npt.NDArray]:
    """
    Reconstruct magnetization from single B_defect measurement.
    
    Parameters same as get_magnetization_from_bxyz with addition of:
    b_defect : B-field along defect axis (Gauss)
    u_defect : Defect orientation unit vector (3,)
    nv_above_sample : bool, default=True
    
    Returns
    -------
    Same format as get_magnetization_from_bxyz.
    """


def normalize_in_plane_mag(
    magnetization: npt.NDArray,
    angle_deg: float,
    edge_pixels: int = 10,
) -> npt.NDArray:
    """
    Normalize in-plane magnetization by subtracting line artifacts.
    
    For each line parallel to the magnetization direction, subtract
    the average of the edge pixels. This removes artifacts from
    reconstruction.
    
    Parameters
    ----------
    magnetization : npt.NDArray
        2D magnetization array.
    angle_deg : float
        In-plane magnetization angle in degrees.
    edge_pixels : int, default=10
        Number of pixels at each edge to use for averaging.
    
    Returns
    -------
    Normalized magnetization array.
    
    Reference
    ---------
    Adapted from D. Broadway's original implementation.
    """
```

### 4.3 High-Level Reconstruct Function

```python
def reconstruct_source(
    bxyz: dict[str, npt.NDArray],
    pixel_size: float,
    source_type: Literal["current", "magnetization"] = "current",
    standoff: float | None = None,
    nv_layer_thickness: float | None = None,
    nv_above_sample: bool = True,
    magnetization_angle: float | None = None,
    use_components: Literal["xy", "z", "auto"] = "auto",
    subtract_background: bool = False,
    background_method: str | None = None,
    background_params: dict | None = None,
    **fourier_kwargs,
) -> dict[str, npt.NDArray]:
    """
    High-level source reconstruction from Bxyz field.
    
    Parameters
    ----------
    bxyz : dict
        Dictionary with keys "Bx", "By", "Bz" (2D arrays in Gauss).
        Can also include "u_defects" for defect-based reconstruction.
    source_type : {"current", "magnetization"}, default="current"
        Type of source to reconstruct.
    subtract_background : bool, default=False
        Subtract background from result using dukit.itool.get_background.
    background_method : str | None
        Method for background subtraction (e.g., "poly", "gaussian").
    background_params : dict | None
        Parameters for background subtraction.
    **fourier_kwargs
        Additional arguments passed to fourier functions:
        - pad_mode, pad_factor, k_vector_epsilon
        - do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff
    
    Returns
    -------
    dict with reconstruction results (same as underlying functions).
    
    Example
    -------
    >>> import dukit
    >>> 
    >>> # First get Bxyz from field reconstruction
    >>> bxyz = dukit.field.get_bxyz_from_bdefects_inversion(
    ...     b_defects, u_defects, bias_field
    ... )
    >>> 
    >>> # Then reconstruct current density
    >>> current = dukit.source.reconstruct_source(
    ...     bxyz,
    ...     pixel_size=1e-6,
    ...     source_type="current",
    ...     standoff=100e-9,  # 100 nm standoff
    ...     use_components="xy",
    ... )
    >>> Jx = current["Jx"]
    >>> Jy = current["Jy"]
    """
```

---

## 5. IMPLEMENTATION STEPS

### Step 1: Copy Files from QDMPY

```bash
# Run from dukit repo root
cp /home/samsc/me/rsc/qdm_fit/qdmpy_proj/qdmpy_git/src/qdmpy/source/current.py src/dukit/source/
cp /home/samsc/me/rsc/qdm_fit/qdmpy_proj/qdmpy_git/src/qdmpy/source/magnetization.py src/dukit/source/
```

### Step 2: Edit current.py

**Changes needed:**

1. **Update imports:**
   ```python
   # Replace:
   from pyfftw.interfaces import numpy_fft
   import qdmpy.shared.fourier
   from qdmpy.shared.fourier import define_current_transform
   
   # With:
   from pyfftw.interfaces import numpy_fft
   import dukit.fourier
   from dukit.fourier import define_current_transform, MU_0
   ```

2. **Rename functions and update signatures:**
   ```python
   # Replace:
   def get_j_from_bxy(bfield, pad_mode, pad_factor, pixel_size, ...)
   
   # With:
   def get_current_from_bxyz(Bx, By, Bz=None, pixel_size=..., ...)
   ```

3. **Replace options-dict unpacking with explicit parameters**

4. **Update function bodies:**
   ```python
   # Replace:
   bx = copy(bfield[0]) * 1e-4
   by = copy(bfield[1]) * 1e-4
   
   # With (Bx, By already passed separately):
   bx = np.copy(Bx) * 1e-4  # Gauss -> Tesla
   by = np.copy(By) * 1e-4
   ```

5. **Replace qdmpy.shared.fourier → dukit.fourier:**
   ```python
   # Replace all:
   qdmpy.shared.fourier.pad_image → dukit.fourier.pad_image
   qdmpy.shared.fourier.define_k_vectors → dukit.fourier.define_k_vectors
   qdmpy.shared.fourier.set_naninf_to_zero → dukit.fourier.set_naninf_to_zero
   qdmpy.shared.fourier.unpad_image → dukit.fourier.unpad_image
   qdmpy.shared.fourier.hanning_filter_kspace → dukit.fourier.hanning_filter_kspace
   ```

6. **Update return values:**
   ```python
   # Replace:
   return jx_reg, jy_reg
   
   # With:
   return {
       "Jx": jx_reg,
       "Jy": jy_reg,
       "Jnorm": np.sqrt(jx_reg**2 + jy_reg**2),
       "_metadata": {...}
   }
   ```

### Step 3: Edit magnetization.py

**Same pattern as current.py:**

1. Update imports (dukit.fourier instead of qdmpy.shared.fourier)
2. Rename functions: `get_m_from_bxy` → `get_magnetization_from_bxyz`
3. Update signatures to explicit parameters
4. Replace array indexing with named parameters
5. Update return to dict format

**Special handling for magnetization_angle:**
```python
# Determine output key based on magnetization_angle
if magnetization_angle is None:
    result_key = "Mz"
else:
    result_key = "Mpsi"

return {
    result_key: m_reg * MAG_UNIT_CONV,  # dukit.fourier.MAG_UNIT_CONV
    "_metadata": {...}
}
```

### Step 4: Create reconstruct.py

New file combining the high-level interface logic from qdmpy's interface.py:

```python
"""High-level source reconstruction interface."""

import numpy as np
from typing import Literal

from dukit.source.current import (
    get_current_from_bxyz,
    get_current_from_bdefect,
    get_current_without_ft,
    get_divperp_j,
)
from dukit.source.magnetization import (
    get_magnetization_from_bxyz,
    get_magnetization_from_bdefect,
    normalize_in_plane_mag,
)
from dukit.itool import get_background


def reconstruct_source(...):
    """..."""
    ...


def normalize_in_plane_mag(...):
    """Ported from qdmpy.source.interface.in_plane_mag_normalise"""
    ...
```

### Step 5: Create __init__.py

```python
"""Source reconstruction module - invert B-field to current/magnetization."""

__pdoc__ = {
    "dukit.source.current": True,
    "dukit.source.magnetization": True,
    "dukit.source.reconstruct": True,
}

from dukit.source.current import (
    get_current_from_bxyz,
    get_current_from_bdefect,
    get_current_without_ft,
    get_divperp_j,
)

from dukit.source.magnetization import (
    get_magnetization_from_bxyz,
    get_magnetization_from_bdefect,
)

from dukit.source.reconstruct import (
    reconstruct_source,
    normalize_in_plane_mag,
)
```

### Step 6: Update dukit/__init__.py

Add to main exports:

```python
from dukit.source import (
    get_current_from_bxyz,
    get_current_from_bdefect,
    get_current_without_ft,
    get_divperp_j,
    get_magnetization_from_bxyz,
    get_magnetization_from_bdefect,
    reconstruct_source,
    normalize_in_plane_mag,
)
```

---

## 6. UNIT HANDLING

**Input**: Gauss (consistent with field reconstruction output)
**Internal**: Convert to Tesla (× 1e-4)
**Output**: Physical units

| Quantity | Input Unit | Internal Unit | Output Unit |
|----------|-----------|---------------|-------------|
| B-field | Gauss | Tesla | - |
| Current density | - | - | A/m |
| Magnetization | - | - | μB/nm² |

**Conversion constants** (already in dukit.fourier):
- `MU_0 = 1.25663706212e-6` (H/m)
- `MAG_UNIT_CONV = 1e-18 / 9.274010e-24` (A → μB/nm²)

---

## 7. DEFAULT PARAMETERS

Following dukit conventions:

```python
# Fourier defaults
pad_mode: str | None = "edge"
pad_factor: int = 2
k_vector_epsilon: float = 1e-6

# Hanning filter defaults  
do_hanning_filter: bool = False  # Off by default (user opts in)
hanning_low_cutoff: float | None = None
hanning_high_cutoff: float | None = None

# Geometry defaults
nv_above_sample: bool = True
standoff: float | None = None  # Must be provided for accurate results
nv_layer_thickness: float | None = None
```

---

## 8. TESTING STRATEGY

### 8.1 Import Test
```python
import dukit.source
from dukit.source import get_current_from_bxyz
```

### 8.2 Synthetic Data Test
Create simple current distribution, calculate B-field via forward model, reconstruct:

```python
# Create synthetic current loop
jx = np.zeros((100, 100))
jy = np.zeros((100, 100))
# ... set up circular current ...

# Forward calculate B-field (use dukit.magsim or simple dipole)
bx, by = forward_calculate(jx, jy, pixel_size=1e-6)

# Reconstruct
result = dukit.source.get_current_from_bxyz(
    bx * 1e4,  # Convert to Gauss for input
    by * 1e4,
    pixel_size=1e-6,
    standoff=100e-9,
)

# Compare
np.testing.assert_allclose(result["Jx"], jx, rtol=0.1)
np.testing.assert_allclose(result["Jy"], jy, rtol=0.1)
```

### 8.2 Basic Smoke Tests
Add basic smoke tests to the pytest suite:
```python
def test_imports():
    import dukit.source

def test_basic_current_reconstruction():
    # Create simple 10x10 test arrays
    Bx = np.random.randn(10, 10)
    By = np.random.randn(10, 10)
    result = dukit.source.get_current_from_bxyz(
        Bx, By, pixel_size=1e-6, standoff=100e-9
    )
    assert "Jx" in result
    assert "Jy" in result
    assert result["Jx"].shape == Bx.shape
```

### 8.3 Real Data Integration
Test with existing dukit field reconstruction output.

---

## 9. DOCUMENTATION

Each function needs:
- Full docstring with Parameters/Returns/Reference
- Example in docstring if non-trivial
- Reference to Broadway et al. 2020 paper

Example docstring format:
```python
def get_current_from_bxyz(...):
    """
    Reconstruct current density from magnetic field components.
    
    Uses Fourier-space inversion to calculate current density Jx, Jy
    from measured B-field components. Supports multiple reconstruction
    methods: from Bx/By (most common), from Bz only, or weighted
    combination of all three.
    
    Parameters
    ----------
    ...
    
    Returns
    -------
    ...
    
    Raises
    ------
    ValueError
        If use_components="z" but Bz not provided.
    
    Notes
    -----
    - Input B-field should be in Gauss (output of dukit.field)
    - Output current density is in A/m
    - Requires standoff for accurate absolute values
    - Zero-mean: DC current component cannot be determined
    
    Reference
    ---------
    D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner,
    N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg,
    Improved Current Density and Magnetization Reconstruction Through
    Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020).
    https://doi.org/10.1103/PhysRevApplied.14.024076
    """
```

---

## 10. WORKFLOW EXAMPLE

```python
import dukit
import numpy as np

# 1. Get frequencies from ODMR fitting
fit_results = dukit.fit_all_pixels("scipyfit", sig_norm, sweep_arr, 
                                   dukit.LinearLorentzians(4), ...)
freqs = dukit.get_fitres_params(fit_results, "pos")

# 2. Convert to B_defects
defect = dukit.NVEnsemble()
b_defects, _ = dukit.field.get_bdefects_from_frequencies(freqs, defect)

# 3. Reconstruct Bxyz (field reconstruction)
bias_field = (0.05, 45.0, 0.0)  # (mag_T, theta_deg, phi_deg)
u_defects = dukit.geom.get_u_defects(
    *dukit.field.spherical_to_cartesian(*bias_field),
    diamond_ori="<100>_<110>"
)
bxyz = dukit.field.get_bxyz_from_bdefects_inversion(b_defects, u_defects, bias_field)

# 4. Reconstruct current density (source reconstruction)
current = dukit.source.get_current_from_bxyz(
    bxyz["Bx"],
    bxyz["By"],
    Bz=bxyz.get("Bz"),
    pixel_size=1.5e-6,
    standoff=150e-9,
    nv_layer_thickness=20e-9,
    use_components="xy",
    do_hanning_filter=True,
    hanning_high_cutoff=500e-9,
)

# 5. Results
Jx = current["Jx"]  # A/m
Jy = current["Jy"]  # A/m
Jnorm = current["Jnorm"]  # A/m

# 6. Optional: Background subtraction
Jx_bg, _ = dukit.get_background(Jx, "poly", order=2)
Jx_sub = Jx - Jx_bg

# 7. Optional: Magnetization instead
mag = dukit.source.get_magnetization_from_bxyz(
    bxyz["Bx"],
    bxyz["By"],
    pixel_size=1.5e-6,
    standoff=150e-9,
    magnetization_angle=None,  # Out-of-plane
)
Mz = mag["Mz"]  # μB/nm²
```

---

## 11. DECISIONS SUMMARY

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **API style** | Explicit parameters | Consistent with dukit.field |
| **Input units** | Gauss | Match field reconstruction output |
| **Output units** | A/m (current), μB/nm² (mag) | Physical units, same as qdmpy |
| **Bz handling** | Optional parameter | Most workflows use Bx/By |
| **use_components** | "auto" default | Intuitive, uses what's available |
| **Hanning filter** | Default False | User must opt-in to filtering |
| **standoff** | Optional but recommended | Can work without but less accurate |
| **nv_layer_thickness** | Optional | Only needed for thick NV layers |
| **Output format** | Flat dict with metadata | Consistent with dukit |
| **divperp_J** | Included in current output | Useful for validation |
| **normalize_in_plane_mag** | Separate utility | Specialized use case |
| **Background subtraction** | In high-level function only | Use dukit.itool directly |

---

## 12. REFERENCES

- D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg, *Improved Current Density and Magnetization Reconstruction Through Vector Magnetic Field Measurements*, Phys. Rev. Applied **14**, 024076 (2020). https://doi.org/10.1103/PhysRevApplied.14.024076

- qdmpy source: https://github.com/casparvitch/qdmpy/tree/main/src/qdmpy/source
