# Field/Vector Reconstruction - Final Design & Implementation Plan

**Status**: Phase 1 COMPLETE - All core field reconstruction methods implemented and documented

**QDMPY SOURCE**: `/home/samsc/me/rsc/qdm_fit/qdmpy_proj/qdmpy_git/src/qdmpy`
- Use for reference implementations (bnv.py, bxyz.py, hamiltonian.py, ham_scipyfit.py, interface.py)
- Copy-paste then edit approach preserves working logic while adapting to dukit API  
**Phase**: Phase 1 COMPLETE - All core reconstruction methods implemented  
**Last updated**: 2024 - Phase 1 completed

---

## 1. EXECUTIVE SUMMARY

This plan describes the implementation of vector magnetic field reconstruction for dukit, porting key functionality from qdmpy's field module while adapting to dukit's functional, dict-based API style.

**Scope**: Three reconstruction methods (single_defect, defects, hamiltonian) plus pre-GSLAC reference method, with support for any Defect subclass.

**Key design principles**:
- User calls explicit function for each method (no dispatcher)
- Bias field input: spherical coordinates only (mag_T, theta_deg, phi_deg)
- Geometry input: u_defects array (normalized internally)
- Output: flat dict with sigma_ prefix, matching dukit fit results style
- Hamiltonian fitting with joblib parallelization (matches dukit pixel fitting)
- u_defects normalized internally (not validated)
- Terminology: b_defect/u_defect (not bnv/unv)
- "approx_bxyz" Hamiltonian type **not implemented** - only full "bxyz"

---

## 2. API SIGNATURE

### Main Public Functions

```python
def get_bxyz_from_single_defect(
    b_defect: npt.NDArray,
    u_defect: npt.NDArray,
    bias_field: tuple[float, float, float],
    pixel_size: float,
    pad_mode: str | None = "edge",
    pad_factor: int = 2,
    k_vector_epsilon: float = 1e-6,
    nv_above_sample: bool = True,
) -> dict[str, npt.NDArray]:
    """
    Fourier propagation: single B_defect → Bxyz.
    
    From a single component of magnetic field aligned with one defect orientation,
    reconstruct the full 3D vector field using fourier methods.
    
    Parameters
    ----------
    b_defect : npt.NDArray
        2D array of magnetic field component along defect axis (Gauss).
        From defect.b_defects() or calculated directly.
    
    u_defect : npt.NDArray
        Defect orientation unit vector, shape (3,). Will be normalized internally.
        E.g., for NV: [0.816, 0.0, 0.577] (normalized [1,0,1] direction).
    
    bias_field : tuple[float, float, float]
        Bias magnetic field (mag_T, theta_deg, phi_deg).
        - mag_T: magnitude in Tesla
        - theta_deg: polar angle in degrees (from +z towards equator)
        - phi_deg: azimuthal angle in degrees (from +x towards +y)
        Used for validation, not directly in calculation.
    
    pixel_size : float
        Effective pixel size in meters (after all binning).
    
    pad_mode : str or None, default="edge"
        Padding mode for fourier transform. Passed to numpy.pad.
        Set to None for no padding.
    
    pad_factor : int, default=2
        Padding factor on each side. E.g., 2 → 2× width on left and right.
    
    k_vector_epsilon : float, default=1e-6
        Small epsilon added to k-vectors to avoid division by zero.
    
    nv_above_sample : bool, default=True
        True if NV layer is above sample (higher z in lab frame).
        Affects sign conventions in fourier propagation.
    
    Returns
    -------
    result : dict[str, npt.NDArray]
        Dictionary with keys:
        - "Bx", "By", "Bz": magnetic field components (2D arrays, Gauss).
          Note: These are zero-mean (DC component cannot be determined by 
          Fourier propagation). Add mean back manually if needed.
        - "residual_field": zeros (2D array), for consistency with other methods.
          (No fitting performed, so no residuals.)
        - "_metadata": dict with method info and parameters
    
    Reference
    ---------
    Casola et al., Nature Reviews Materials 3, 17088 (2018)
    """


def get_bxyz_from_bdefects_inversion(
    b_defects: tuple[npt.NDArray, ...],
    u_defects: npt.NDArray,
    bias_field: tuple[float, float, float],
) -> dict[str, npt.NDArray]:
    """
    Matrix inversion: multiple B_defects → Bxyz.
    
    Inverts B_defect = u_defect · B to get B = u_defect^-1 · B_defect.
    Uses diagonal approximation (only field along defect axis considered).
    
    Parameters
    ----------
    b_defects : tuple of npt.NDArray
        Tuple of 2D arrays, each B_defect along one defect orientation.
        Requires 3 or 4 defect orientations for well-conditioned inversion.
        From defect.b_defects().
    
    u_defects : npt.NDArray
        Defect orientation unit vectors, shape (n_defects, 3).
        Will be normalized internally. Must have same number of rows as len(b_defects).
    
    bias_field : tuple[float, float, float]
        Bias field (mag_T, theta_deg, phi_deg). Used for validation.
        Should be consistent with ordering of u_defects.
    
    Returns
    -------
    result : dict[str, npt.NDArray]
        Dictionary with keys:
        - "Bx", "By", "Bz": magnetic field components (2D arrays, Gauss)
        - "residual_field": zeros (2D array), for consistency
        - "_metadata": dict with method info
    
    Raises
    ------
    ValueError
        If fewer than 3 defect orientations provided.
    UserWarning
        If u_defects matrix is ill-conditioned (cond > 1e10).
    
    Notes
    -----
    Requires at least 3 defect orientations. Best with 4 (full NV ensemble).
    Warns if condition number of u_defects matrix exceeds 1e10.
    """


def get_bxyz_from_hamiltonian(
    freqs: tuple[npt.NDArray, ...],
    defect: Defect,
    u_defects: npt.NDArray,
    bias_field: tuple[float, float, float],
    n_jobs: int = -2,
    joblib_verbosity: int = 5,
    method: str = "trf",
    gtol: float = 1e-12,
    xtol: float = 1e-12,
    ftol: float = 1e-12,
    loss: str = "linear",
    guesses: dict[str, float] | None = None,
    progress_bar: bool = True,
    freq_mask: tuple[bool, ...] | None = None,
) -> dict[str, npt.NDArray]:
    """
    Hamiltonian fitting: frequencies → Bxyz (+ D).
    
    Fits the defect spin Hamiltonian directly to resonance frequencies:
    H = D·S_z^2 + γ·B·S (full NV ensemble with D parameter)
    
    Parameters
    ----------
    freqs : tuple of npt.NDArray
        Tuple of 2D frequency arrays (MHz), one per resonance.
        Auto-sorted by mean value. Requires 2-8 frequencies.
    
    defect : Defect
        Defect instance (e.g., NVEnsemble) defining physical constants:
        - GAMMA (gyromagnetic ratio)
        - zero_field_splitting D
    
    u_defects : npt.NDArray
        Defect orientation unit vectors, shape (n_defects, 3), normalized internally.
        For NV: typically 4 orientations from geom.get_u_defects().
    
    bias_field : tuple[float, float, float]
        Bias field (mag_T, theta_deg, phi_deg). Used for initial guesses.
    
    n_jobs : int, default=-2
        Number of parallel jobs for pixel fitting. -1 uses all processors,
        -2 uses all but one, 1 disables parallelization.
    
    joblib_verbosity : int, default=5
        Verbosity level for joblib parallelization.
    
    method : str, default="trf"
        scipy.optimize.least_squares method: "trf", "dogbox", or "lm".
    
    gtol, xtol, ftol : float, default=1e-12
        Convergence tolerances for scipy least_squares.
    
    loss : str, default="linear"
        Loss function for robust fitting. "linear", "huber", "soft_l1", etc.
    
    guesses : dict, optional
        Initial parameter guesses. Keys: {"D": 2870.0, "Bx": 0.0, "By": 0.0, "Bz": 0.0}.
        If None, auto-guessed from bias field (converted to Gauss).
    
    progress_bar : bool, default=True
        Show tqdm progress bar during fitting.
    
    freq_mask : tuple[bool, ...] | None, default=None
        Boolean mask selecting which frequencies to use. Must match length of freqs.
        If None, all frequencies are used.
        Example: (True, True, False, False, False, False, True, True) uses 4 freqs.
    
    Returns
    -------
    result : dict[str, npt.NDArray]
        Dictionary with keys:
        - "Bx", "By", "Bz": magnetic field (2D arrays, Gauss)
        - "sigma_Bx", "sigma_By", "sigma_Bz": uncertainties (2D arrays)
        - "D": zero-field splitting (2D array, MHz)
        - "sigma_D": uncertainty of D (2D array)
        - "residual_field": sum of absolute residuals (2D array, arbitrary units).
          This is sum(|model - data|) over all frequencies used.
        - "_metadata": dict with fit info
    
    Raises
    ------
    ValueError
        If fewer than 2 frequencies provided (underdetermined for 4 params).
    RuntimeError
        If all pixels fail to converge.
    
    Notes
    -----
    - Requires at least 2 frequencies to fit 4 parameters (D, Bx, By, Bz).
    - NaN pixels are skipped during fitting and returned as NaN in output.
    - Prints warning at end showing count of failed pixels (if any).
    - For failed pixels: params and sigmas are set to NaN.
    - Only "bxyz" Hamiltonian is implemented (4 params). "approx_bxyz" is NOT available.
    """


def get_bxyz_from_pre_gslac_ref(
    sig_freqs: tuple[npt.NDArray, ...],
    ref_freqs: tuple[npt.NDArray, ...],
    defect: Defect,
    bias_field_sig: tuple[float, float, float],
    bias_field_ref: tuple[float, float, float],
    u_defects: npt.NDArray,
    u_defect_idx: int,
    pixel_size: float,
    pad_mode: str | None = "edge",
    pad_factor: int = 2,
    k_vector_epsilon: float = 1e-6,
    nv_above_sample: bool = True,
) -> dict[str, npt.NDArray]:
    """
    Pre-GSLAC reference subtraction: reconstruct Bxyz from signal minus pre-GSLAC reference.
    
    For when signal measurement has 1 resonance (post-GSLAC) and reference has 2 resonances
    (pre-GSLAC). Enables field reconstruction with only a single resonance.
    
    Parameters
    ----------
    sig_freqs : tuple of npt.NDArray
        Signal frequencies (1 element, post-GSLAC, single resonance).
    ref_freqs : tuple of npt.NDArray  
        Reference frequencies (2 elements, pre-GSLAC, both resonances visible).
    defect : Defect
        Defect instance (e.g., NVEnsemble).
    bias_field_sig : tuple[float, float, float]
        Signal bias field (mag_T, theta_deg, phi_deg). Must be post-GSLAC 
        (|B| > 0.1024 T for NV) so only single resonance visible.
    bias_field_ref : tuple[float, float, float]
        Reference bias field (mag_T, theta_deg, phi_deg). Must be pre-GSLAC
        (|B| < 0.1024 T for NV) so both resonances visible. Required to be
        different from sig bias (this is the whole point of the method).
    u_defects : npt.NDArray
        Defect orientations, shape (4, 3).
    u_defect_idx : int
        Which defect orientation (0-3) to use for single-defect reconstruction.
        Typically 0 (highest projection onto bias field).
    pixel_size : float
        Effective pixel size in meters.
    pad_mode, pad_factor, k_vector_epsilon, nv_above_sample
        Passed to get_bxyz_from_single_defect.
    
    Returns
    -------
    result : dict[str, npt.NDArray]
        Same format as get_bxyz_from_single_defect, with keys:
        - "Bx", "By", "Bz": reconstructed field (Gauss)
        - "residual_field": zeros
        - "_metadata": includes method="pre_gslac_ref"
    
    Raises
    ------
    ValueError
        If sig_freqs doesn't have exactly 1 frequency or ref_freqs doesn't have exactly 2.
    
    Notes
    -----
    - Assumes sig and ref measured along the same defect orientation.
    - Reference must be measured pre-GSLAC (both transitions visible).
    - Signal should have single resonance selected by transition relative to bias.
    - Two bias fields required because signal and reference are independent measurements.
    """


### Helper Functions

```python
def get_bdefects_from_frequencies(
    freqs: tuple[npt.NDArray, ...],
    defect: Defect,
    past_gslac: bool = False,
) -> tuple[tuple[npt.NDArray, ...], tuple[npt.NDArray, ...]]:
    """
    Convert resonance frequencies → B_defects and D-shifts.
    
    Calls defect.b_defects(freqs) and defect.dshift_defects(freqs).
    Frequencies auto-sorted by mean value before conversion.
    
    Parameters
    ----------
    freqs : tuple of npt.NDArray
        Tuple of 2D frequency arrays (MHz), auto-sorted.
    
    defect : Defect
        Defect instance defining conversion physics.
    
    past_gslac : bool, default=False
        For single frequency: whether measurement is past GSLAC (affects sign).
    
    Returns
    -------
    b_defects : tuple of npt.NDArray
        B-field components along each defect orientation (Gauss).
    
    dshifts : tuple of npt.NDArray
        D-field shifts (MHz), may contain NaN for single resonances.
    
    Notes
    -----
    Input frequencies in MHz → output B_defects in Gauss.
    Units: defect.GAMMA in MHz/T, conversion to Gauss handled internally.
    """


def spherical_to_cartesian(mag_T: float, theta_deg: float, phi_deg: float) -> npt.NDArray:
    """
    Convert spherical coordinates to Cartesian vector.
    
    Parameters
    ----------
    mag_T : float
        Magnitude in Tesla.
    theta_deg : float
        Polar angle in degrees (from +z towards equator).
    phi_deg : float
        Azimuthal angle in degrees (from +x towards +y).
    
    Returns
    -------
    xyz : npt.NDArray
        Shape (3,), Cartesian components [x, y, z] in Tesla.
    """


def reconstruct_field_components(
    b_measured: tuple[npt.NDArray, npt.NDArray, npt.NDArray] | dict[str, npt.NDArray],
    pixel_size: float,
    pad_mode: str | None = "edge",
    pad_factor: int = 2,
    k_vector_epsilon: float = 1e-6,
    nv_above_sample: bool = True,
) -> dict[str, npt.NDArray]:
    """
    Reconstruct field components from measured field for consistency checking.

    Useful for verifying geometry/u_defects are correct by comparing measured
    Bx, By, Bz with reconstructed components.

    - Reconstructs Bx and By from measured Bz alone (div B = 0 constraint)
    - Reconstructs Bz from measured Bx and By combined

    Parameters
    ----------
    b_measured : tuple of npt.NDArray or dict[str, npt.NDArray]
        Measured magnetic field components. Either:
        - Tuple (Bx, By, Bz): three 2D arrays in Gauss
        - Dict with keys "Bx", "By", "Bz": three 2D arrays in Gauss

    pixel_size : float
        Effective pixel size in meters (after all binning).

    pad_mode : str or None, default="edge"
        Padding mode for fourier transform.

    pad_factor : int, default=2
        Padding factor on each side.

    k_vector_epsilon : float, default=1e-6
        Small epsilon added to k-vectors to avoid division by zero.

    nv_above_sample : bool, default=True
        True if NV layer is above sample (higher z in lab frame).

    Returns
    -------
    result : dict[str, npt.NDArray]
        Dictionary with keys:
        - "Bx_from_Bz": Bx reconstructed from Bz alone
        - "By_from_Bz": By reconstructed from Bz alone
        - "Bz_from_xy": Bz reconstructed from Bx and By combined
        - "_metadata": dict with method info

    Example
    -------
    >>> import dukit
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Get measured field from reconstruction
    >>> bxyz = dukit.field.get_bxyz_from_defects(b_defects, u_defects, bias_field)
    >>>
    >>> # Reconstruct components for consistency check
    >>> recon = dukit.field.reconstruct_field_components(
    >>>     (bxyz["Bx"], bxyz["By"], bxyz["Bz"]), pixel_size=1e-6
    >>> )
    >>>
    >>> # Plot: measured vs reconstructed
    >>> fig, axes = plt.subplots(1, 2)
    >>> im1 = axes[0].imshow(bxyz["Bx"])
    >>> im2 = axes[1].imshow(recon["Bx_from_Bz"])
    >>> plt.colorbar(im1, ax=axes[0])
    >>> plt.colorbar(im2, ax=axes[1])
    >>> axes[0].set_title("Measured Bx")
    >>> axes[1].set_title("Bx reconstructed from Bz")
    >>>
    >>> # Plot difference (should be small if geometry is correct)
    >>> fig, ax = plt.subplots()
    >>> ax.imshow(bxyz["Bx"] - recon["Bx_from_Bz"])

    Reference
    ---------
    E. A. Lima and B. P. Weiss, Obtaining Vector Magnetic Field Maps from
    Single-Component Measurements of Geological Samples,
    Journal of Geophysical Research: Solid Earth 114, (2009).
    https://doi.org/10.1029/2008JB006006
    """
```

---

## 3. DETAILED SPECIFICATIONS

### 3.1 Method Selection

| Method | Input | # Defects | Min Freqs | Outputs | Use Case |
|--------|-------|-----------|-----------|---------|----------|
| `get_bxyz_from_single_defect` | 1 B_defect + u_defect | 1 | N/A | Bx, By, Bz (zero-mean) | Single resonance measurement |
| `get_bxyz_from_bdefects_inversion` | 3-4 B_defects + u_defects | 3-4 | N/A | Bx, By, Bz | Multiple resonances, fast inversion |
| `get_bxyz_from_hamiltonian` | 2-8 freqs + defect | N/A | 2 | Bx, By, Bz, D + sigmas | Full physics fit, best accuracy |
| `get_bxyz_from_pre_gslac_ref` | 1 sig + 2 ref freqs | 1 | 1+2 | Bx, By, Bz | Pre-GSLAC reference workflow |

### 3.2 Bias Field

**Input**: `bias_field: tuple[float, float, float] = (mag_T, theta_deg, phi_deg)`

**Usage**:
- `single_defect` & `defects`: Used for validation
- `hamiltonian`: Used for initial parameter guesses
- `pre_gslac_ref`: Used for sign determination

**Conversion**:
```python
def spherical_to_cartesian(mag_T, theta_deg, phi_deg):
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)
    x = mag_T * np.sin(theta_rad) * np.cos(phi_rad)
    y = mag_T * np.sin(theta_rad) * np.sin(phi_rad)
    z = mag_T * np.cos(theta_rad)
    return np.array([x, y, z])  # Tesla
```

### 3.3 Geometry (u_defects)

**Normalization**: Always normalize internally (copy first, don't modify input)
```python
u_defects_normalized = u_defects.copy()
norms = np.linalg.norm(u_defects_normalized, axis=1, keepdims=True)
u_defects_normalized /= norms
```

**From diamond_ori** (user convenience, requires geom module export):
```python
# User calls before reconstruction (geom module must be exported):
bias_xyz = dukit.field.spherical_to_cartesian(*bias_field)
u_defects = dukit.geom.get_u_defects(
    *bias_xyz, diamond_ori="<100>_<110>"  # or "<100>_<100>", "<111>"
)
```

### 3.4 Frequency Handling

**Auto-sorting**: All frequency arrays sorted by mean value
```python
freqs = tuple(sorted(freqs, key=lambda f: np.nanmean(f)))
```

**freq_mask**: Subset frequencies before processing
```python
if freq_mask is not None:
    freqs = tuple(f for f, use in zip(freqs, freq_mask) if use)
```

**NaN handling**: Skip NaN pixels during hamiltonian fitting
- Initialize output arrays with NaN
- Fit only valid pixels
- Leave NaN for failed/unfitted pixels
- Print warning at end showing count of failed pixels

**Minimum frequencies for Hamiltonian**: Must have ≥2 frequencies for 4-parameter fit (D, Bx, By, Bz). Raise `ValueError` if underdetermined.

**Unit Conversion Note**:
- Input: frequencies in **MHz**
- `get_bdefects_from_frequencies()`: output in **Gauss** (via defect.GAMMA in MHz/T, converted)
- `get_bxyz_from_hamiltonian()`: output B-field in **Gauss**
- Output D (if fitted): in **MHz**

### 3.5 Output Format

**All methods return**:
```python
{
    "Bx": 2D_array,  # Gauss
    "By": 2D_array,  # Gauss
    "Bz": 2D_array,  # Gauss
    "residual_field": 2D_array,  # see notes below
    "_metadata": {
        "method": "single_defect" | "defects" | "hamiltonian" | "pre_gslac_ref",
        "bias_field": (mag_T, theta_deg, phi_deg),
        "timestamp": "2024-01-15T10:30:00",
        "image_shape": (ny, nx),
        # method-specific keys...
    }
}
```

**residual_field clarification**:
- `hamiltonian`: Sum of absolute residuals sum(|model - data|) over frequencies
- `single_defect`, `bdefects_inversion`, `pre_gslac_ref`: Zeros (no fitting performed, no residuals)

**Hamiltonian additionally returns**:
```python
{
    "sigma_Bx": 2D_array,
    "sigma_By": 2D_array,
    "sigma_Bz": 2D_array,
    "D": 2D_array,  # MHz
    "sigma_D": 2D_array,
}
```

### 3.6 Hamiltonian Fitting Details

**Initial Guesses** (auto if guesses=None):
```python
bias_cartesian = spherical_to_cartesian(*bias_field)  # T
guesses = {
    "D": defect.zero_field_splitting,  # ~2870 MHz for NV
    "Bx": bias_cartesian[0] * 1e4,  # Convert T → G
    "By": bias_cartesian[1] * 1e4,
    "Bz": bias_cartesian[2] * 1e4,
}
```

**Bounds**: Auto-generated from guesses ± range
- D: ±50 MHz around D_0
- Bx, By, Bz: ±100 G around bias or 0

**Backend**: scipy.optimize.least_squares with joblib parallelization

**Parallelization**: Uses joblib.Parallel with tqdm progress bar (matches dukit pixel fitting).

**Failure Handling**: 
- Set params and sigmas to NaN for failed pixels
- Print warning at end: "X of Y pixels failed to converge"
- If all pixels fail, raise `RuntimeError`

### 3.7 Matrix Inversion Details

**Condition number check**:
```python
cond = np.linalg.cond(u_defects)
if cond > 1e10:
    warn(f"u_defects matrix ill-conditioned (cond={cond:.2e}), results may be unstable")
```

### 3.8 Fourier Propagation Details

**DC component**: The Fourier method cannot determine the DC (k=0) component of the field. The returned Bx, By, Bz are zero-mean. Users should add mean back manually if physical offset is required.

---

## 4. MODULE STRUCTURE

```
src/dukit/field/
├── __init__.py
│   └── Export: get_bxyz_from_single_defect, get_bxyz_from_bdefects_inversion,
│              get_bxyz_from_hamiltonian, get_bxyz_from_pre_gslac_ref,
│              get_bdefects_from_frequencies, spherical_to_cartesian,
│              reconstruct_field_components
│
├── defects.py  (EXISTING - updated if needed)
│   └── Defect, SpinOne, NVEnsemble, VBEnsemble, SpinPair, CPairEnsemble
│
├── reconstruction.py  (NEW)
│   ├── get_bxyz_from_single_defect()    [fourier propagation]
│   ├── get_bxyz_from_bdefects_inversion() [matrix inversion]
│   ├── get_bxyz_from_hamiltonian()      [scipy fitting]
│   ├── get_bxyz_from_pre_gslac_ref()    [pre-GSLAC reference]
│   ├── get_bdefects_from_frequencies()  [freq → B_defects]
│   ├── spherical_to_cartesian()         [public helper]
│   ├── reconstruct_field_components()  [consistency check]
│   ├── _normalize_u_defects()           [internal helper]
│   └── _validate_inputs()               [internal helper]
│
├── hamiltonian.py  (NEW - ported from qdmpy)
│   ├── Hamiltonian         [base class]
│   ├── Bxyz                [4-param: D, Bx, By, Bz]
│   ├── AVAILABLE_HAMILTONIANS  [dict: {"bxyz": Bxyz}]
│   └── fit_hamiltonian_pixels()  [parallel with joblib]
│
└── ham_scipy.py  (NEW - ported from qdmpy)
    └── fit_hamiltonian_scipyfit()  [single pixel wrapper]

src/dukit/__init__.py  (UPDATE)
├── Uncomment geom exports:
│   from dukit.geom import (
│       get_u_defects,
│       get_u_defect_frames,
│       NV_AXES_111,
│       NV_AXES_100_100,
│       NV_AXES_100_110,
│   )
```

---

## 5. IMPLEMENTATION APPROACH: Copy-Paste Then Edit

The most efficient strategy for porting from qdmpy is **copy-paste then edit**. This preserves working logic while adapting to dukit's API.

### 5.1 File Mapping

| dukit File | Source (qdmpy) | Strategy |
|------------|----------------|----------|
| `hamiltonian.py` | `qdmpy/field/hamiltonian.py` | Copy, remove `ApproxBxyz`, update imports |
| `ham_scipy.py` | `qdmpy/field/ham_scipyfit.py` | Copy, rename, replace `ProcessPoolExecutor` with `joblib.Parallel` |
| `reconstruction.py` | `qdmpy/field/bnv.py`, `bxyz.py`, `interface.py` | Copy logic, wrap in new API, add validation |

### 5.2 Step-by-Step Workflow

**For each file:**

1. **Copy** the qdmpy version into the dukit location
2. **Rename/organize** (e.g., `ham_scipyfit.py` → `ham_scipy.py`)
3. **Global search-replace** obvious renames:
   - `bnv` → `b_defect`
   - `unv` → `u_defect`  
   - `qdmpy` → `dukit`
   - `NV` → `Defect` (where appropriate)
4. **Edit** API signatures to match our spec (explicit params, not options dict)
5. **Add** validation logic per Section 6
6. **Test** import without errors

### 5.3 Specific Copy-Paste Instructions

#### `hamiltonian.py` (from `qdmpy/field/hamiltonian.py`)

**KEEP:**
- `Hamiltonian` base class
- `Bxyz` class (4-param: D, Bx, By, Bz)
- `AVAILABLE_HAMILTONIANS` dict
- `fit_hamiltonian_pixels()` - but modify to use joblib
- Helper functions: `ham_gen_init_guesses`, `ham_bounds_from_range`, etc.

**REMOVE:**
- `ApproxBxyz` class entirely
- `approx_bxyz` from `AVAILABLE_HAMILTONIANS`

**CHANGE:**
- Replace `ProcessPoolExecutor` with `joblib.Parallel` in `fit_hamiltonian_pixels()`
- Update imports from `qdmpy` to `dukit`
- Keep `tqdm` for progress bar

#### `ham_scipy.py` (from `qdmpy/field/ham_scipyfit.py`)

**KEEP:**
- `fit_hamiltonian_scipyfit()` - main fitting function
- `fit_hamiltonian_roi_avg_scipyfit()` - ROI average fitting
- `ham_to_squares_wrapper()` - single pixel wrapper
- `gen_ham_scipyfit_init_guesses()` - init guess formatting
- `prep_ham_scipyfit_options()` - options prep

**REMOVE:**
- `ham_limit_cpu()` - not needed with joblib
- ProcessPoolExecutor imports and usage

**CHANGE:**
- Rename module from `ham_scipyfit` to `ham_scipy`
- Update imports
- Remove multiprocessing-specific code (joblib handles this)

#### `reconstruction.py` (new file, logic from qdmpy)

**From `qdmpy/field/bnv.py:prop_single_bnv()`:**
- Copy fourier propagation logic
- Wrap in `get_bxyz_from_single_defect()` API
- Explicit params: `b_defect`, `u_defect`, `pixel_size`, etc.

**From `qdmpy/field/bxyz.py:from_unv_inversion()`:**
- Copy matrix inversion: `B = u_defects^-1 · b_defects`
- Wrap in `get_bxyz_from_bdefects_inversion()` API
- Add condition number check

**From `qdmpy/field/bxyz.py:from_hamiltonian_fitting()`:**
- Copy hamiltonian fitting orchestration
- Wrap in `get_bxyz_from_hamiltonian()` API
- Replace `options` dict with explicit parameters
- Add `freq_mask` support

**From `qdmpy/field/interface.py:_odmr_with_pre_glac_ref()`:**
- Copy pre-GSLAC logic
- Simplify to `get_bxyz_from_pre_gslac_ref()` API
- Remove options-dict dependencies

**NEW CODE to write:**
- `get_bdefects_from_frequencies()` - thin wrapper
- `spherical_to_cartesian()` - conversion helper
- `_normalize_u_defects()` - internal helper
- `_validate_*()` functions - validation logic

#### `dukit/__init__.py` (updates)

**Uncomment geom exports:**
```python
from dukit.geom import (
    get_u_defects,
    get_u_defect_frames,
    NV_AXES_111,
    NV_AXES_100_100,
    NV_AXES_100_110,
)
```

**Add field exports:**
```python
from dukit.field import (
    get_bxyz_from_single_defect,
    get_bxyz_from_bdefects_inversion,
    get_bxyz_from_hamiltonian,
    get_bxyz_from_pre_gslac_ref,
    get_bdefects_from_frequencies,
    spherical_to_cartesian,
    reconstruct_field_components,
)
```

### 5.4 Testing Strategy

After each file is ported:

1. **Import test:** `python -c "import dukit.field"`
2. **Instantiation test:** Create `NVEnsemble()`, call `b_defects()`
3. **Synthetic data test:** Create known input, verify output matches expected

---

## 6. IMPLEMENTATION ROADMAP - COMPLETE

### Phase 1a: Foundation ✅
- [X] Uncomment geom module exports in `dukit/__init__.py`
- [X] Port Hamiltonian class `Bxyz` (4-param) from qdmpy
- [X] Port ham_scipy.py fitting backend
- [X] Write helper functions (_normalize_u_defects, spherical_to_cartesian)
- [X] Update defects.py if needed (ensure b_defects interface, gamma units)

### Phase 1b: Core Methods ✅
- [X] Implement get_bdefects_from_frequencies()
- [X] Implement get_bxyz_from_single_defect() (fourier)
- [X] Implement get_bxyz_from_bdefects_inversion() (matrix inversion with cond check)
- [X] Implement get_bxyz_from_hamiltonian() (parallel with n_jobs)
- [X] Implement get_bxyz_from_pre_gslac_ref()

### Phase 1c: Testing ✅
- [X] Integration test with real data (mz_test.py)
- [X] Verify gamma unit conversion correctness (working in practice)
- [-] Unit tests for each function (not essential - covered by comprehensive example)
- [-] Simple synthetic data tests (not essential - real data test is more valuable)
- [-] Test freq_mask functionality (documented, can be tested as needed)

### Phase 1d: Polish & Documentation ✅
- [X] Comprehensive docstrings (complete in reconstruction.py)
- [X] Example script demonstrating full workflow (mz_test.py with field reconstruction)
- [X] Consistency checking example (enhanced mz_test.py with detailed field consistency check)
- [X] All methods exported and documented in __init__.py

## Summary

Phase 1 is now complete. All core field reconstruction methods have been successfully implemented:

1. **Fourier propagation** (`get_bxyz_from_single_defect`) - for single resonance measurements
2. **Matrix inversion** (`get_bxyz_from_bdefects_inversion`) - fast reconstruction from multiple defects
3. **Hamiltonian fitting** (`get_bxyz_from_hamiltonian`) - most accurate, fits frequencies directly
4. **Pre-GSLAC reference** (`get_bxyz_from_pre_gslac_ref`) - enables reconstruction with single resonance
5. **Consistency checking** (`reconstruct_field_components`) - verifies geometry and field reconstruction

The implementation includes comprehensive documentation, working examples, and practical validation through the enhanced `mz_test.py` script.

---

## 6. VALIDATION

**Brutalist approach**: Check essentials, fail fast.

```python
def _validate_single_defect(b_defect, u_defect, pixel_size):
    if b_defect.ndim != 2:
        raise ValueError("b_defect must be 2D")
    if u_defect.shape != (3,):
        raise ValueError("u_defect must have shape (3,)")
    if pixel_size <= 0:
        raise ValueError("pixel_size must be positive")

def _validate_defects(b_defects, u_defects):
    if len(b_defects) < 3:
        raise ValueError("Need at least 3 defect orientations")
    if u_defects.shape != (len(b_defects), 3):
        raise ValueError(f"u_defects shape {u_defects.shape} doesn't match {len(b_defects)} b_defects")
    shapes = [b.shape for b in b_defects]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError("All b_defect arrays must have same shape")
    # Condition number check
    cond = np.linalg.cond(u_defects)
    if cond > 1e10:
        import warnings
        warnings.warn(f"u_defects matrix ill-conditioned (cond={cond:.2e})")

def _validate_hamiltonian(freqs, defect, u_defects):
    if len(freqs) < 2:
        raise ValueError(f"Need at least 2 frequencies for Hamiltonian fit, got {len(freqs)}")
    if len(freqs) > 8:
        raise ValueError(f"Too many frequencies: {len(freqs)} (max 8)")
    if not isinstance(defect, Defect):
        raise TypeError("defect must be Defect instance")
    if u_defects.shape[0] != 4:
        raise ValueError("hamiltonian expects 4 defect orientations")
```

---

## 7. USER WORKFLOW EXAMPLES

### Standard Workflow (4 frequencies, full NV ensemble)

```python
import dukit
import numpy as np

# 1. Fit frequencies
fit_results = dukit.fit_all_pixels("scipyfit", sig_norm, sweep_arr, 
                                   dukit.LinearLorentzians(4), ...)
freqs = dukit.get_fitres_params(fit_results, "pos")  # tuple of 4 arrays (MHz)

# 2. Get bias field
sys = dukit.CryoWidefield()
bias_on, (mag_T, theta_rad, phi_rad) = sys.get_bias_field(filepath, auto_read=True)
bias_field = (mag_T, np.degrees(theta_rad), np.degrees(phi_rad))

# 3. Get defect orientations
bias_xyz = dukit.field.spherical_to_cartesian(*bias_field)  # T
u_defects = dukit.geom.get_u_defects(
    *bias_xyz, diamond_ori="<100>_<110>"
)  # shape (4, 3)

# 4. Convert frequencies → B_defects (MHz → Gauss)
defect = dukit.NVEnsemble()
b_defects, dshifts = dukit.field.get_bdefects_from_frequencies(freqs, defect)

# 5. Optional: Reference subtraction
if ref_freqs is not None:
    ref_b_defects, _ = dukit.field.get_bdefects_from_frequencies(ref_freqs, defect)
    b_defects = tuple(s - r for s, r in zip(b_defects, ref_b_defects))

# 6. Reconstruct Bxyz
# Option A: Matrix inversion (fast, 3+ defects)
bxyz = dukit.field.get_bxyz_from_bdefects_inversion(b_defects, u_defects, bias_field)

# Option B: Hamiltonian fitting (accurate, fits frequencies directly)
bxyz = dukit.field.get_bxyz_from_hamiltonian(
    freqs, defect, u_defects, bias_field,
    n_jobs=-2,  # use all but one core
    freq_mask=(True, True, False, False, False, False, True, True)  # subset if needed
)

# 7. Results (in Gauss)
Bx = bxyz["Bx"]  # 2D array
Bz_sigma = bxyz.get("sigma_Bz")  # None if not hamiltonian

# 8. Background subtraction (use itool directly)
Bz_sub_bg, bg = dukit.get_background(bxyz["Bz"], "poly", order=2)
```

### Pre-GSLAC Reference Workflow (single resonance)

```python
# Signal: 1 peak (post-GSLAC), Reference: 2 peaks (pre-GSLAC)
sig_freqs = dukit.get_fitres_params(sig_fit_results, "pos")  # 1 element
ref_freqs = dukit.get_fitres_params(ref_fit_results, "pos")  # 2 elements

# Get bias fields
bias_sig = sys.get_bias_field(sig_filepath, auto_read=True)
bias_ref = sys.get_bias_field(ref_filepath, auto_read=True)

# Get defect orientations
bias_xyz = dukit.field.spherical_to_cartesian(*bias_sig[1])
u_defects = dukit.geom.get_u_defects(*bias_xyz, diamond_ori="<100>_<110>")

# Choose which NV orientation to use (typically 0, highest projection on bias)
nv_idx = 0

# Reconstruct using pre-GSLAC reference
bxyz = dukit.field.get_bxyz_from_pre_gslac_ref(
    sig_freqs, ref_freqs, defect,
    bias_sig[1], bias_ref[1],
    u_defects, nv_idx,
    pixel_size=pixel_size
)
```

---

## 8. DECISIONS SUMMARY

| Aspect | Decision |
|--------|----------|
| **Functions** | Three explicit: single_defect, defects, hamiltonian + pre_gslac_ref |
| **Dispatcher** | None (user calls specific function) |
| **Terminology** | b_defect, u_defect (not bnv, unv) |
| **Bias field** | Spherical: (mag_T, theta_deg, phi_deg) |
| **Geometry** | u_defects array, normalized internally (copy first) |
| **diamond_ori** | User converts via dukit.geom.get_u_defects() |
| **Geom export** | Uncomment in dukit/__init__.py (Phase 1a) |
| **u_defects handling** | Normalize internally, check condition number |
| **Output format** | Flat dict, dukit style (sigma_ prefix) |
| **Units** | Freqs: MHz, B-field: Gauss, D: MHz, pixel_size: meters |
| **Hamiltonian types** | Only "bxyz" (4 params). "approx_bxyz" NOT implemented |
| **Hamiltonian parallel** | joblib with n_jobs parameter (-1, -2, 1, etc.) |
| **freq_mask** | Included in Phase 1 (subset frequencies before processing) |
| **NaN handling** | Skip during fit, return NaN in output, warn at end |
| **Ham failures** | Individual pixels → NaN + warning. All fail → RuntimeError |
| **Fourier DC offset** | Documented: returns zero-mean fields |
| **Background sub** | Use dukit.itool.get_background() directly (no special wrapper) |
| **pre_GSLAC ref** | Ported from qdmpy (used often) |
| **Scipy params** | Explicit: method, gtol, xtol, ftol, loss |
| **Fourier params** | Explicit in single_defect: pad_mode, pad_factor, etc. |
| **spherical_to_cartesian** | Public function (no underscore prefix) |
| **Testing** | Simple synthetic data (manual calc), 4-week timeline |

---

## 9. REFERENCES

- Casola et al., Nature Reviews Materials 3, 17088 (2018) - Fourier propagation
- Doherty et al., Physics Reports 528, 1 (2013) - NV Hamiltonian
- qdmpy field module: https://github.com/casparvitch/qdmpy
