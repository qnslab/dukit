# -*- coding: utf-8 -*-
"""
Implement inversion of magnetic field to current density.

Functions
---------
 - `dukit.source.current.get_divperp_j`
 - `dukit.source.current.get_current_from_bxyz`
 - `dukit.source.current.get_current_from_bdefect`
 - `dukit.source.current.get_current_without_ft`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.source.current.get_divperp_j": True,
    "dukit.source.current.get_current_from_bxyz": True,
    "dukit.source.current.get_current_from_bdefect": True,
    "dukit.source.current.get_current_without_ft": True,
}

# ============================================================================

from datetime import datetime
from typing import Literal

import numpy as np
import numpy.typing as npt
from pyfftw.interfaces import numpy_fft

# ============================================================================
import dukit.fourier
from dukit.fourier import MU_0, define_current_transform

# ============================================================================


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

    Parameters
    ----------
    Jx, Jy : npt.NDArray
        Current density components (A/m), 2D arrays.
    pixel_size : float
        Effective pixel size in meters.
    pad_mode : str or None, default="edge"
        Padding mode for fourier transform. Passed to numpy.pad.
        Set to None for no padding.
    pad_factor : int, default=2
        Padding factor on each side.
    k_vector_epsilon : float, default=1e-6
        Small epsilon added to k-vectors to avoid division by zero.

    Returns
    -------
    divperp_j : npt.NDArray
        Normalized perpendicular divergence (2D array).
    """
    jx = np.copy(Jx)
    jy = np.copy(Jy)

    # first pad each comp
    jx_pad, padder = dukit.fourier.pad_image(jx, pad_mode, pad_factor)
    jy_pad, _ = dukit.fourier.pad_image(jy, pad_mode, pad_factor)

    fft_jx = numpy_fft.fftshift(numpy_fft.fft2(jx_pad))
    fft_jy = numpy_fft.fftshift(numpy_fft.fft2(jy_pad))
    fft_jx = dukit.fourier.set_naninf_to_zero(fft_jx)
    fft_jy = dukit.fourier.set_naninf_to_zero(fft_jy)

    ky, kx, k = dukit.fourier.define_k_vectors(fft_jx.shape, pixel_size, k_vector_epsilon)

    # Note: there was a typo in qdmpy - missing '*' operator for ky term. Fixed here.
    fft_divperp_j = -1j * kx * fft_jx + -1j * ky * fft_jy

    divperp_j = numpy_fft.ifft2(numpy_fft.ifftshift(fft_divperp_j)).real

    # only return non-padded region
    divperp_j_reg = dukit.fourier.unpad_image(divperp_j, padder)
    mx = max(np.abs([np.nanmax(divperp_j_reg), np.nanmin(divperp_j_reg)]))
    return divperp_j_reg / mx


# ============================================================================


def get_current_from_bxyz(
    Bx: npt.NDArray,
    By: npt.NDArray,
    Bz: npt.NDArray | None = None,
    pixel_size: float = 1e-6,
    standoff: float | None = None,
    nv_layer_thickness: float | None = None,
    nv_above_sample: bool = True,
    use_components: Literal["xy", "z", "auto"] = "auto",
    pad_mode: str | None = "edge",
    pad_factor: int = 2,
    k_vector_epsilon: float = 1e-6,
    do_hanning_filter: bool = False,
    hanning_low_cutoff: float | None = None,
    hanning_high_cutoff: float | None = None,
) -> dict[str, npt.NDArray]:
    """
    Reconstruct current density Jx, Jy from magnetic field components.

    Uses Fourier-space inversion to calculate current density Jx, Jy
    from measured B-field components. Supports multiple reconstruction
    methods: from Bx/By (most common), from Bz only, or weighted
    combination of all three.

    Parameters
    ----------
    Bx, By : npt.NDArray
        Magnetic field components in Gauss (2D arrays).
    Bz : npt.NDArray | None, optional
        Bz component. Required if use_components="z" or "xyz".
    pixel_size : float, default=1e-6
        Effective pixel size in meters.
    standoff : float | None
        Distance from NV layer to sample in meters.
    nv_layer_thickness : float | None
        Thickness of NV layer in meters.
    nv_above_sample : bool, default=True
        True if NV layer is above sample (higher z).
    use_components : {"xy", "z", "auto"}, default="auto"
        Which field components to use:
        - "xy": Use Bx, By only (most common)
        - "z": Use Bz only
        - "auto": Use "xy" if Bx, By provided, else "z"
    pad_mode : str or None, default="edge"
        Padding mode for fourier transform. Set to None for no padding.
    pad_factor : int, default=2
        Padding factor on each side.
    k_vector_epsilon : float, default=1e-6
        Small epsilon added to k-vectors to avoid division by zero.
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
        - "_metadata": Dict with parameters used

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
    # copy and convert Gauss -> Tesla
    bx = np.copy(Bx) * 1e-4
    by = np.copy(By) * 1e-4

    # Determine which components to use
    if use_components == "auto":
        use = "xy"
        if Bz is not None:
            # Use xy by default if available
            use = "xy"
    else:
        use = use_components

    if use == "xy":
        # first pad each comp
        bx_pad, padder = dukit.fourier.pad_image(bx, pad_mode, pad_factor)
        by_pad, _ = dukit.fourier.pad_image(by, pad_mode, pad_factor)

        fft_bx = numpy_fft.fftshift(numpy_fft.fft2(bx_pad))
        fft_by = numpy_fft.fftshift(numpy_fft.fft2(by_pad))
        fft_bx = dukit.fourier.set_naninf_to_zero(fft_bx)
        fft_by = dukit.fourier.set_naninf_to_zero(fft_by)

        ky, kx, k = dukit.fourier.define_k_vectors(fft_bx.shape, pixel_size, k_vector_epsilon)

        sign = 1 if nv_above_sample else -1

        # define transform
        _, bx_to_jy = define_current_transform(
            [sign, 0, 0], ky, kx, k, standoff, nv_layer_thickness
        )
        by_to_jx, _ = define_current_transform(
            [0, sign, 0], ky, kx, k, standoff, nv_layer_thickness
        )

        hanning_filt = dukit.fourier.hanning_filter_kspace(
            k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
        )

        bx_to_jy = dukit.fourier.set_naninf_to_zero(hanning_filt * bx_to_jy)
        by_to_jx = dukit.fourier.set_naninf_to_zero(hanning_filt * by_to_jx)

        fft_jx = fft_by * by_to_jx
        fft_jy = fft_bx * bx_to_jy

        # fourier transform back into real space
        jx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jx)).real
        jy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jy)).real

        # only return non-padded region
        jx_reg = dukit.fourier.unpad_image(jx, padder)
        jy_reg = dukit.fourier.unpad_image(jy, padder)

    else:  # use == "z"
        if Bz is None:
            raise ValueError("Bz must be provided when use_components='z'")
        bz = np.copy(Bz) * 1e-4
        bz_pad, padder = dukit.fourier.pad_image(bz, pad_mode, pad_factor)

        fft_bz = numpy_fft.fftshift(numpy_fft.fft2(bz_pad))
        fft_bz = dukit.fourier.set_naninf_to_zero(fft_bz)

        ky, kx, k = dukit.fourier.define_k_vectors(fft_bz.shape, pixel_size, k_vector_epsilon)

        sign = 1 if nv_above_sample else -1

        # define transformation
        bz_to_jx, bz_to_jy = define_current_transform(
            [0, 0, sign], ky, kx, k, standoff, nv_layer_thickness
        )

        hanning_filt = dukit.fourier.hanning_filter_kspace(
            k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
        )

        bz_to_jx = dukit.fourier.set_naninf_to_zero(hanning_filt * bz_to_jx)
        bz_to_jy = dukit.fourier.set_naninf_to_zero(hanning_filt * bz_to_jy)

        fft_jx = bz_to_jx * fft_bz
        fft_jy = bz_to_jy * fft_bz

        # fourier transform back into real space
        jx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jx)).real
        jy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jy)).real

        # only return non-padded region
        jx_reg = dukit.fourier.unpad_image(jx, padder)
        jy_reg = dukit.fourier.unpad_image(jy, padder)

    return {
        "Jx": jx_reg,
        "Jy": jy_reg,
        "Jnorm": np.sqrt(jx_reg**2 + jy_reg**2),
        "_metadata": {
            "method": "get_current_from_bxyz",
            "use_components": use,
            "nv_above_sample": nv_above_sample,
            "standoff": standoff,
            "nv_layer_thickness": nv_layer_thickness,
            "timestamp": datetime.utcnow().isoformat(),
            "image_shape": jx_reg.shape,
            "pixel_size": pixel_size,
        },
    }


# ============================================================================


# ============================================================================


def get_current_without_ft(Bx: npt.NDArray, By: npt.NDArray) -> dict[str, npt.NDArray]:
    """
    Approximate current density without Fourier propagation.

    Simple rescaling: J ≈ (2/μ₀) × B × (perpendicular conversion).
    Fast but less accurate - useful for quick checks.

    Parameters
    ----------
    Bx, By : npt.NDArray
        Magnetic field components in Gauss.

    Returns
    -------
    dict with keys "Jx", "Jy", "Jnorm" in A/m.
    """
    scale = (1e-4 * 2) / MU_0
    jx = -By * scale
    jy = Bx * scale
    return {
        "Jx": jx,
        "Jy": jy,
        "Jnorm": np.sqrt(jx**2 + jy**2),
    }


# ============================================================================


def get_current_from_bdefect(
    b_defect: npt.NDArray,
    u_defect: npt.NDArray,
    pixel_size: float = 1e-6,
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
    pixel_size : float, default=1e-6
        Effective pixel size in meters.
    standoff : float | None
        Distance from NV layer to sample in meters.
    nv_layer_thickness : float | None
        Thickness of NV layer in meters.
    nv_above_sample : bool, default=True
        True if NV layer is above sample (higher z).
    pad_mode : str or None, default="edge"
        Padding mode for fourier transform. Set to None for no padding.
    pad_factor : int, default=2
        Padding factor on each side.
    k_vector_epsilon : float, default=1e-6
        Small epsilon added to k-vectors to avoid division by zero.
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
        - "_metadata": Dict with parameters used

    Reference
    ---------
    D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner,
    N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg,
    Improved Current Density and Magnetization Reconstruction Through
    Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020).
    https://doi.org/10.1103/PhysRevApplied.14.024076
    """
    b = np.copy(b_defect) * 1e-4  # copy and convert Gauss to Tesla
    bnv_pad, padder = dukit.fourier.pad_image(b, pad_mode, pad_factor)

    fft_bnv = numpy_fft.fftshift(numpy_fft.fft2(bnv_pad))

    ky, kx, k = dukit.fourier.define_k_vectors(fft_bnv.shape, pixel_size, k_vector_epsilon)

    if nv_above_sample:
        unv_cpy = np.copy(u_defect)
    else:
        unv_cpy = np.array([-u_defect[0], -u_defect[1], u_defect[2]])

    # define transform
    bnv_to_jx, bnv_to_jy = define_current_transform(
        unv_cpy, ky, kx, k, standoff, nv_layer_thickness
    )

    hanning_filt = dukit.fourier.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    bnv_to_jx = dukit.fourier.set_naninf_to_zero(hanning_filt * bnv_to_jx)
    bnv_to_jy = dukit.fourier.set_naninf_to_zero(hanning_filt * bnv_to_jy)

    fft_jx = bnv_to_jx * fft_bnv
    fft_jy = bnv_to_jy * fft_bnv

    # fourier transform back into real space
    jx = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jx)).real
    jy = numpy_fft.ifft2(numpy_fft.ifftshift(fft_jy)).real

    # only return non-padded region
    jx_reg = dukit.fourier.unpad_image(jx, padder)
    jy_reg = dukit.fourier.unpad_image(jy, padder)

    return {
        "Jx": jx_reg,
        "Jy": jy_reg,
        "Jnorm": np.sqrt(jx_reg**2 + jy_reg**2),
        "_metadata": {
            "method": "get_current_from_bdefect",
            "nv_above_sample": nv_above_sample,
            "standoff": standoff,
            "nv_layer_thickness": nv_layer_thickness,
            "timestamp": datetime.utcnow().isoformat(),
            "image_shape": jx_reg.shape,
            "pixel_size": pixel_size,
        },
    }


# ============================================================================
