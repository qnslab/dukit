# -*- coding: utf-8 -*-
"""
Field reconstruction functions for defect microscopy.

This module provides vector magnetic field reconstruction methods for defect-based
widefield microscopy. Functions convert between different representations of magnetic
field data and perform vector reconstruction from single or multiple defect orientations.

Functions
---------
 - `dukit.field.reconstruction.spherical_to_cartesian`
 - `dukit.field.reconstruction.get_bdefects_from_frequencies`
 - `dukit.field.reconstruction.get_bxyz_from_single_defect`
 - `dukit.field.reconstruction.get_bxyz_from_bdefects_inversion`
 - `dukit.field.reconstruction.get_bxyz_from_hamiltonian`
 - `dukit.field.reconstruction.get_bxyz_from_pre_gslac_ref`
 - `dukit.field.reconstruction._normalize_u_defects`
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.field.reconstruction.spherical_to_cartesian": True,
    "dukit.field.reconstruction.get_bdefects_from_frequencies": True,
    "dukit.field.reconstruction.get_bxyz_from_single_defect": True,
    "dukit.field.reconstruction.get_bxyz_from_bdefects_inversion": True,
    "dukit.field.reconstruction.get_bxyz_from_hamiltonian": True,
    "dukit.field.reconstruction.get_bxyz_from_pre_gslac_ref": True,
    "dukit.field.reconstruction.reconstruct_field_components": True,
    "dukit.field.reconstruction._normalize_u_defects": True,
}

import warnings
from datetime import timezone
from timeit import default_timer as timer
from typing import Tuple

import numpy as np
import numpy.typing as npt
from pyfftw.interfaces import numpy_fft

import dukit.fourier
from dukit.field import ham_scipy, hamiltonian
from dukit.field.defects import Defect

# ============================================================================


def spherical_to_cartesian(
    mag_T: float, theta_deg: float, phi_deg: float
) -> npt.NDArray[np.float64]:
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
    theta_rad = np.radians(theta_deg)
    phi_rad = np.radians(phi_deg)
    x = mag_T * np.sin(theta_rad) * np.cos(phi_rad)
    y = mag_T * np.sin(theta_rad) * np.sin(phi_rad)
    z = mag_T * np.cos(theta_rad)
    return np.array([x, y, z])


# ============================================================================


def _normalize_u_defects(u_defects: npt.NDArray) -> npt.NDArray:
    """
    Normalize defect orientation unit vectors internally (copy first).

    Parameters
    ----------
    u_defects : npt.NDArray
        Defect orientation unit vectors, shape (n_defects, 3).

    Returns
    -------
    u_defects_normalized : npt.NDArray
        Normalized orientation vectors, same shape as input.
        Input is NOT modified.
    """
    u_defects_normalized = u_defects.copy()
    norms = np.linalg.norm(u_defects_normalized, axis=1, keepdims=True)
    u_defects_normalized /= norms
    return u_defects_normalized


# ============================================================================


def get_bdefects_from_frequencies(
    freqs: Tuple[npt.NDArray, ...],
    defect: Defect,
    past_gslac: bool = False,
) -> Tuple[Tuple[npt.NDArray, ...], Tuple[npt.NDArray, ...]]:
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
    Input frequencies in MHz → output B_defects in Gauss (via defect.GAMMA in MHz/T).
    """
    freqs_sorted = tuple(sorted(freqs, key=lambda f: np.nanmean(f)))
    b_defects_tesla = defect.b_defects(freqs_sorted, past_gslac=past_gslac)
    dshifts = defect.dshift_defects(freqs_sorted)

    # Convert Tesla to Gauss (1 T = 1e4 G)
    b_defects = tuple(b * 1e4 for b in b_defects_tesla)

    return b_defects, dshifts


# ============================================================================


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
    from datetime import datetime

    b_defect = np.array(b_defect).copy()

    padded_bdefect, padder = dukit.fourier.pad_image(b_defect, pad_mode, pad_factor)

    fft_bdefect = numpy_fft.fftshift(numpy_fft.fft2(padded_bdefect))
    ky, kx, k = dukit.fourier.define_k_vectors(
        fft_bdefect.shape, pixel_size, k_vector_epsilon=k_vector_epsilon
    )

    u_defect_normalized = u_defect / np.linalg.norm(u_defect)
    u_defect_copy = (
        u_defect_normalized.copy()
        if nv_above_sample
        else np.array([-u_defect_normalized[0], -u_defect_normalized[1], u_defect_normalized[2]])
    )

    kappa = [-1j * kx / k, -1j * ky / k, 1]
    u_dot_kappa = (
        u_defect_copy[0] * kappa[0] + u_defect_copy[1] * kappa[1] + u_defect_copy[2] * kappa[2]
    )

    bdefect2bx = kappa[0] / u_dot_kappa
    bdefect2by = kappa[1] / u_dot_kappa
    bdefect2bz = kappa[2] / u_dot_kappa

    bdefect2bx = dukit.fourier.set_naninf_to_zero(bdefect2bx)
    bdefect2by = dukit.fourier.set_naninf_to_zero(bdefect2by)
    bdefect2bz = dukit.fourier.set_naninf_to_zero(bdefect2bz)

    fft_bx = fft_bdefect * bdefect2bx
    fft_by = fft_bdefect * bdefect2by
    fft_bz = fft_bdefect * bdefect2bz

    bx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_bx)).real
    by = numpy_fft.ifft2(numpy_fft.ifftshift(fft_by)).real
    bz = numpy_fft.ifft2(numpy_fft.ifftshift(fft_bz)).real

    bx_reg = dukit.fourier.unpad_image(bx, padder)
    by_reg = dukit.fourier.unpad_image(by, padder)
    bz_reg = dukit.fourier.unpad_image(bz, padder)

    result = {
        "Bx": bx_reg,
        "By": by_reg,
        "Bz": bz_reg,
        "residual_field": np.zeros_like(bx_reg),
        "_metadata": {
            "method": "single_defect",
            "bias_field": bias_field,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_shape": bx_reg.shape,
            "pixel_size": pixel_size,
            "pad_mode": pad_mode,
            "pad_factor": pad_factor,
            "k_vector_epsilon": k_vector_epsilon,
            "nv_above_sample": nv_above_sample,
            "u_defect": u_defect.tolist(),
        },
    }

    return result


# ============================================================================


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

    # Validate inputs
    if len(sig_freqs) != 1:
        raise ValueError(f"sig_freqs must have exactly 1 frequency, got {len(sig_freqs)}")
    if len(ref_freqs) != 2:
        raise ValueError(f"ref_freqs must have exactly 2 frequencies, got {len(ref_freqs)}")

    # Convert frequencies to B_defects
    # Signal: post-GSLAC, single resonance
    b_defects_sig, _ = get_bdefects_from_frequencies(sig_freqs, defect, past_gslac=True)

    # Reference: pre-GSLAC, two resonances
    b_defects_ref, _ = get_bdefects_from_frequencies(ref_freqs, defect, past_gslac=False)

    # We now have:
    # b_defects_sig: tuple of 1 element (single B_defect from post-GSLAC measurement)
    # b_defects_ref: tuple of 1 element (single B_defect from pre-GSLAC, averaged from 2 resonances)

    # For the pre-GSLAC ref method, we compute the difference
    # This gives us the field change between reference and signal
    b_defect_diff = b_defects_sig[0] - b_defects_ref[0]

    # Normalize u_defects to get the specific orientation
    u_defects_normalized = _normalize_u_defects(u_defects)
    u_defect = u_defects_normalized[u_defect_idx]

    # Use single defect reconstruction on the difference
    result = get_bxyz_from_single_defect(
        b_defect_diff,
        u_defect,
        bias_field_sig,  # Use signal bias field for metadata
        pixel_size,
        pad_mode=pad_mode,
        pad_factor=pad_factor,
        k_vector_epsilon=k_vector_epsilon,
        nv_above_sample=nv_above_sample,
    )

    # Update metadata to indicate this is pre_gslac_ref method
    result["_metadata"]["method"] = "pre_gslac_ref"
    result["_metadata"]["bias_field_sig"] = bias_field_sig
    result["_metadata"]["bias_field_ref"] = bias_field_ref
    result["_metadata"]["u_defect_idx"] = u_defect_idx

    return result


# ============================================================================


def get_bxyz_from_bdefects_inversion(
    b_defects: tuple[npt.NDArray, ...],
    u_defects: npt.NDArray,
    bias_field: tuple[float, float, float],
) -> dict[str, npt.NDArray]:
    """
    Matrix inversion: multiple B_defects → Bxyz.

    Inverts B_defect = u_defects · B to get B = u_defects^-1 · b_defects.
    Uses diagonal approximation (only field along defect axis considered).

    Parameters
    ----------
    b_defects : tuple of npt.NDArray
        Tuple of 2D arrays, each B_defect along one defect orientation.
        Requires 3 or 4 defect orientations for well-conditioned inversion.

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
    from datetime import datetime

    # Validation: need at least 3 defect orientations
    if len(b_defects) < 3:
        raise ValueError(
            f"Need at least 3 defect orientations for matrix inversion, got {len(b_defects)}"
        )

    # Validation: shapes match
    n_defects = len(b_defects)
    if u_defects.shape[0] != n_defects:
        raise ValueError(
            f"u_defects has {u_defects.shape[0]} orientations but {n_defects} b_defects provided"
        )

    # Validation: all b_defects have same shape
    shapes = [b.shape for b in b_defects]
    if not all(s == shapes[0] for s in shapes):
        raise ValueError(f"All b_defects must have same shape, got shapes: {shapes}")

    # Normalize u_defects internally
    u_defects_normalized = _normalize_u_defects(u_defects)

    # For 4 orientations, pick the best-conditioned 3x3 subset
    # For exactly 3, use all of them
    if n_defects == 3:
        u_inv_matrix = u_defects_normalized
        b_defects_to_use = b_defects
    else:
        # For 4 defects, find the best-conditioned 3x3 subset
        cond_numbers = []
        indices_list = []
        for i in range(n_defects):
            for j in range(i + 1, n_defects):
                for k in range(j + 1, n_defects):
                    subset = np.vstack(
                        [u_defects_normalized[i], u_defects_normalized[j], u_defects_normalized[k]]
                    )
                    try:
                        cond_numbers.append(np.linalg.cond(subset))
                    except np.linalg.LinAlgError:
                        # If SVD fails, use large condition number
                        cond_numbers.append(1e15)
                    indices_list.append((i, j, k))

        # Pick the subset with smallest condition number
        best_idx = np.argmin(cond_numbers)
        best_indices = indices_list[best_idx]

        u_inv_matrix = np.vstack([u_defects_normalized[i] for i in best_indices])
        b_defects_to_use = [b_defects[i] for i in best_indices]

    try:
        cond = np.linalg.cond(u_inv_matrix)
    except np.linalg.LinAlgError:
        cond = np.inf

    if cond > 1e10:
        warnings.warn(
            f"u_defects matrix ill-conditioned (cond={cond:.2e}), results may be unstable"
        )

    # Compute inverse
    u_inv = np.linalg.inv(u_inv_matrix)

    # Select corresponding b_defects
    b_defects_stacked = np.stack(b_defects_to_use, axis=-1)

    # Multiply inverse matrix by b_defects for each pixel using einsum
    bxyz = np.einsum("ij,yxj->yxi", u_inv, b_defects_stacked)

    return {
        "Bx": bxyz[:, :, 0],
        "By": bxyz[:, :, 1],
        "Bz": bxyz[:, :, 2],
        "residual_field": np.zeros_like(bxyz[:, :, 0]),
        "_metadata": {
            "method": "bdefects_inversion",
            "bias_field": bias_field,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_shape": bxyz[:, :, 0].shape,
            "n_defects_used": len(b_defects_to_use),
            "condition_number": float(cond),
        },
    }


# ============================================================================
def _construct_u_defect_frames(u_defects: npt.NDArray) -> npt.NDArray:
    """
    Construct full 3x3 rotation frames from u_defects (z-axes).

    For each defect orientation, construct orthonormal [X, Y, Z] frame where
    Z is the defect axis (u_defect). X and Y are chosen to complete the basis.

    Parameters
    ----------
    u_defects : npt.NDArray
        Defect orientation unit vectors, shape (n_defects, 3).

    Returns
    -------
    u_defect_frames : npt.NDArray
        Full rotation matrices, shape (n_defects, 3, 3).
        Each frame is [uX, uY, uZ] as rows.
    """
    n_defects = u_defects.shape[0]
    u_defect_frames = np.zeros((n_defects, 3, 3))

    for i in range(n_defects):
        u_z = u_defects[i] / np.linalg.norm(u_defects[i])

        # Pick a reference vector not parallel to u_z
        ref_vec = np.array([1.0, 0.0, 0.0])

        # If u_z is too close to [1,0,0], use [0,1,0]
        if np.abs(np.dot(u_z, ref_vec)) > 0.99:
            ref_vec = np.array([0.0, 1.0, 0.0])

        # Construct orthonormal basis via Gram-Schmidt
        u_y = np.cross(u_z, ref_vec)
        u_y_norm = np.linalg.norm(u_y)
        if u_y_norm > 1e-10:
            u_y = u_y / u_y_norm
        else:
            # Fallback if cross product failed
            u_y = np.array([0.0, 0.0, 1.0])

        u_x = np.cross(u_y, u_z)
        u_x = u_x / np.linalg.norm(u_x)

        u_defect_frames[i] = np.array([u_x, u_y, u_z])

    return u_defect_frames


def get_bxyz_from_hamiltonian(
    freqs: tuple[npt.NDArray, ...],
    defect: Defect,
    u_defects: npt.NDArray,
    bias_field: tuple[float, float, float],
    n_jobs: int = -2,
    joblib_verbosity: int = 0,
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

    joblib_verbosity : int, default=0
        Verbosity level for joblib parallelization.

    method : str, default="trf"
        scipy.optimize.least_squares method: "trf", "dogbox", or "lm".

    gtol, xtol, ftol : float, default=1e-12
        Convergence tolerances for scipy least_squares.

    loss : str, default="linear"
        Loss function for robust fitting. "linear", "huber", "soft_l1", etc.

    guesses : dict, optional
        Initial parameter guesses. Keys: {"D", "Bx", "By", "Bz"}.
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
    """
    from datetime import datetime

    # Validation
    if len(freqs) < 2:
        raise ValueError(
            f"Need at least 2 frequencies for Hamiltonian fit (4 params: D, Bx, By, Bz), "
            f"got {len(freqs)}"
        )
    if len(freqs) > 8:
        raise ValueError(f"Too many frequencies: {len(freqs)} (max 8)")
    if not isinstance(defect, Defect):
        raise TypeError(f"defect must be Defect instance, got {type(defect)}")

    # Auto-sort frequencies by mean value
    freqs_sorted = tuple(sorted(freqs, key=lambda f: np.nanmean(f)))

    # Apply freq_mask if provided
    if freq_mask is not None:
        if len(freq_mask) != len(freqs_sorted):
            raise ValueError(
                f"freq_mask length {len(freq_mask)} must match freqs length {len(freqs_sorted)}"
            )
        freqs_sorted = tuple(f for f, use in zip(freqs_sorted, freq_mask) if use)
        if len(freqs_sorted) < 2:
            raise ValueError(
                f"freq_mask left only {len(freqs_sorted)} frequencies, need at least 2"
            )

    # Convert to 3D array [idx, y, x]
    freq_data = np.array(freqs_sorted)
    if freq_data.ndim != 3:
        raise ValueError(f"freqs must be tuple of 2D arrays, got shape {freq_data.shape}")

    # Get image shape
    _, ny, nx = freq_data.shape

    # Normalize u_defects
    u_defects_normalized = _normalize_u_defects(u_defects)

    # Construct full u_defect_frames for Hamiltonian
    u_defect_frames = _construct_u_defect_frames(u_defects_normalized)

    # Create chooser (selects which frequencies to use)
    if freq_mask is not None:
        chooser_ar = np.array([use for use in freq_mask if use])
    else:
        chooser_ar = np.ones(len(freqs_sorted), dtype=bool)
    chooser_obj = hamiltonian.Chooser(chooser_ar)

    # Generate initial guesses
    if guesses is None:
        # Convert bias field from spherical to cartesian (Tesla)
        bias_cartesian = spherical_to_cartesian(*bias_field)  # T
        # Convert to Gauss for initial guesses
        bias_gauss = tuple(b * 1e4 for b in bias_cartesian)
        init_guesses, init_bounds = hamiltonian.ham_gen_init_guesses(
            "bxyz",
            bias_gauss,
            zero_field_splitting=defect.zero_field_splitting,
            auto_guess=True,
        )
    else:
        # User provided guesses
        ham_class = hamiltonian.AVAILABLE_HAMILTONIANS["bxyz"]
        init_guesses = guesses.copy()
        init_bounds = {}
        for param_key in ham_class.param_defn:
            if param_key not in init_guesses:
                if param_key == "D":
                    init_guesses[param_key] = defect.zero_field_splitting
                else:
                    init_guesses[param_key] = 0.0
            # Set bounds based on parameter type
            if param_key == "D":
                guess = init_guesses[param_key]
                init_bounds[param_key] = np.array([guess - 50.0, guess + 50.0])
            else:  # Bx, By, Bz
                guess = init_guesses[param_key]
                init_bounds[param_key] = np.array([guess - 100.0, guess + 100.0])

    # Create Hamiltonian
    ham_obj = hamiltonian.Bxyz(chooser_obj, u_defect_frames)

    # Run fitting
    t0 = timer()
    fit_results, sigmas = ham_scipy.fit_hamiltonian_scipyfit(
        data=freq_data,
        hamiltonian_obj=ham_obj,
        init_guesses=init_guesses,
        init_bounds=init_bounds,
        freq_mask=None,
        n_jobs=n_jobs,
        joblib_verbosity=joblib_verbosity,
        method=method,
        gtol=gtol,
        xtol=xtol,
        ftol=ftol,
        loss=loss,
        use_analytic_jac=False,
        scale_x=False,
        shuffle_pixels=True,
        progress_bar=progress_bar,
    )
    t1 = timer()

    # Check for failed pixels
    n_failed = np.sum(np.isnan(fit_results["D"]))
    if n_failed > 0:
        warnings.warn(f"{n_failed} of {ny * nx} pixels failed to converge")
    if n_failed == ny * nx:
        raise RuntimeError("All pixels failed to converge")

    # Build result dictionary with sigma_ prefix
    result = {
        "Bx": fit_results["Bx"],
        "By": fit_results["By"],
        "Bz": fit_results["Bz"],
        "sigma_Bx": sigmas["Bx"],
        "sigma_By": sigmas["By"],
        "sigma_Bz": sigmas["Bz"],
        "D": fit_results["D"],
        "sigma_D": sigmas["D"],
        "residual_field": fit_results["residual_field"],
        "_metadata": {
            "method": "hamiltonian",
            "bias_field": bias_field,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_shape": (ny, nx),
            "n_freqs": len(freqs_sorted),
            "n_failed_pixels": int(n_failed),
            "fit_time_s": t1 - t0,
            "hamiltonian_type": "bxyz",
            "scipy_method": method,
            "gtol": gtol,
            "xtol": xtol,
            "ftol": ftol,
            "loss": loss,
        },
    }

    return result


# ============================================================================


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

    Users can plot the difference between measured and reconstructed fields
    to check if their geometry assumptions are correct.

    Parameters
    ----------
    b_measured : tuple of npt.NDArray or dict[str, npt.NDArray]
        Measured magnetic field components. Either:
        - Tuple (Bx, By, Bz): three 2D arrays in Gauss
        - Dict with keys "Bx", "By", "Bz": three 2D arrays in Gauss

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
    >>> # Plot difference
    >>> fig, ax = plt.subplots()
    >>> im = ax.imshow(bxyz["Bx"] - recon["Bx_from_Bz"])
    >>> plt.colorbar(im)
    >>> ax.set_title("Difference (should be small if geometry is correct)")

    Reference
    ---------
    E. A. Lima and B. P. Weiss, Obtaining Vector Magnetic Field Maps from
    Single-Component Measurements of Geological Samples,
    Journal of Geophysical Research: Solid Earth 114, (2009).
    https://doi.org/10.1029/2008JB006006
    """
    from datetime import datetime

    # Parse input
    if isinstance(b_measured, dict):
        if "Bx" not in b_measured or "By" not in b_measured or "Bz" not in b_measured:
            raise KeyError("b_measured dict must contain 'Bx', 'By', and 'Bz' keys")
        bx, by, bz = b_measured["Bx"], b_measured["By"], b_measured["Bz"]
    elif isinstance(b_measured, (tuple, list)) and len(b_measured) == 3:
        bx, by, bz = b_measured
    else:
        raise ValueError(
            "b_measured must be a tuple (Bx, By, Bz) or dict with 'Bx', 'By', 'Bz' keys"
        )

    # Make copies
    bx = np.array(bx).copy()
    by = np.array(by).copy()
    bz = np.array(bz).copy()

    # Pad each component
    bx_pad, padder = dukit.fourier.pad_image(bx, pad_mode, pad_factor)
    by_pad, _ = dukit.fourier.pad_image(by, pad_mode, pad_factor)
    bz_pad, _ = dukit.fourier.pad_image(bz, pad_mode, pad_factor)

    # FFT
    fft_bx = numpy_fft.fftshift(numpy_fft.fft2(bx_pad))
    fft_by = numpy_fft.fftshift(numpy_fft.fft2(by_pad))
    fft_bz = numpy_fft.fftshift(numpy_fft.fft2(bz_pad))

    fft_bx = dukit.fourier.set_naninf_to_zero(fft_bx)
    fft_by = dukit.fourier.set_naninf_to_zero(fft_by)
    fft_bz = dukit.fourier.set_naninf_to_zero(fft_bz)

    # Get k vectors
    ky, kx, k = dukit.fourier.define_k_vectors(
        fft_bx.shape, pixel_size, k_vector_epsilon=k_vector_epsilon
    )

    # Sign convention
    sign = 1 if nv_above_sample else -1

    # Reconstruction coefficients
    # From Bz → Bx, By
    bz2bx = -sign * (1j / k) * kx
    bz2by = -sign * (1j / k) * ky

    # From Bx, By → Bz
    bx2bz = sign * (1j / k) * kx
    by2bz = sign * (1j / k) * ky

    # Handle NaN/inf
    bz2bx = dukit.fourier.set_naninf_to_zero(bz2bx)
    bz2by = dukit.fourier.set_naninf_to_zero(bz2by)
    bx2bz = dukit.fourier.set_naninf_to_zero(bx2bz)
    by2bz = dukit.fourier.set_naninf_to_zero(by2bz)

    # Reconstruct in Fourier space
    fft_bx_from_bz = fft_bz * bz2bx
    fft_by_from_bz = fft_bz * bz2by
    fft_bz_from_xy = fft_bx * bx2bz + fft_by * by2bz

    # Inverse FFT
    bx_from_bz = numpy_fft.ifft2(numpy_fft.ifftshift(fft_bx_from_bz)).real
    by_from_bz = numpy_fft.ifft2(numpy_fft.ifftshift(fft_by_from_bz)).real
    bz_from_xy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_bz_from_xy)).real

    # Unpad
    bx_from_bz_reg = dukit.fourier.unpad_image(bx_from_bz, padder)
    by_from_bz_reg = dukit.fourier.unpad_image(by_from_bz, padder)
    bz_from_xy_reg = dukit.fourier.unpad_image(bz_from_xy, padder)

    result = {
        "Bx_from_Bz": bx_from_bz_reg,
        "By_from_Bz": by_from_bz_reg,
        "Bz_from_xy": bz_from_xy_reg,
        "_metadata": {
            "method": "reconstruct_field_components",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_shape": bx_from_bz_reg.shape,
            "pixel_size": pixel_size,
            "pad_mode": pad_mode,
            "pad_factor": pad_factor,
            "k_vector_epsilon": k_vector_epsilon,
            "nv_above_sample": nv_above_sample,
        },
    }

    return result


# ============================================================================
