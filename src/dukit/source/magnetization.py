# -*- coding: utf-8 -*-
"""Magnetization reconstruction - invert B-field to out-of-plane magnetization Mz"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.source.magnetization": True,
}

# ============================================================================

import numpy as np
from pyfftw.interfaces import numpy_fft

# ============================================================================
import dukit.fourier
from dukit.fourier import (
    MAG_UNIT_CONV,
    define_magnetization_transformation,
)

# ============================================================================


def get_magnetization_from_bxyz(
    Bx: np.ndarray,
    By: np.ndarray,
    Bz: np.ndarray | None = None,
    pixel_size: float = 1.0,
    standoff: float | None = None,
    nv_layer_thickness: float | None = None,
    magnetization_angle: float | None = None,
    use_components: str = "auto",
    pad_mode: str | None = "edge",
    pad_factor: int = 2,
    k_vector_epsilon: float = 1e-6,
    do_hanning_filter: bool = False,
    hanning_low_cutoff: float | None = None,
    hanning_high_cutoff: float | None = None,
) -> dict[str, np.ndarray]:
    r"""Reconstruct magnetization from magnetic field components.

    Can reconstruct either out-of-plane magnetization (Mz) or in-plane
    magnetization (Mpsi) at a specified angle.

    Parameters
    ----------
    Bx, By : np.ndarray
        Magnetic field components in Gauss (2D arrays).
    Bz : np.ndarray | None, optional
        Bz component. Required if use_components="z" or "xyz".
    pixel_size : float, default=1.0
        Effective pixel size in meters.
    standoff : float | None
        Distance from NV layer to sample in meters.
    nv_layer_thickness : float | None
        Thickness of NV layer in meters.
    magnetization_angle : float | None, default=None
        In-plane magnetization angle in degrees from +x towards +y.
        None = out-of-plane (Mz).
        Any value = in-plane at that angle (Mpsi).
    use_components : {"xy", "z", "auto"}, default="auto"
        Which field components to use:
        - "xy": Use Bx, By only (most common)
        - "z": Use Bz only
        - "auto": Use "xy" if Bx, By provided, else "z"
    pad_mode : str | None, default="edge"
        Mode for Fourier padding. See np.pad for options.
    pad_factor : int, default=2
        Factor to pad image on all sides.
    k_vector_epsilon : float, default=1e-6
        Epsilon for k-vectors to avoid division by zero.
    do_hanning_filter : bool, default=False
        Apply Hanning filter to reduce noise amplification.
    hanning_low_cutoff, hanning_high_cutoff : float | None
        Cutoff wavelengths in meters for Hanning filter.

    Returns
    -------
    dict with keys:
        - "Mz" or "Mpsi": Magnetization (μB/nm²)
          - "Mz" if magnetization_angle is None (out-of-plane)
          - "Mpsi" if magnetization_angle is given (in-plane at angle)
        - "_metadata": Dict with parameters used

    Reference
    ---------
    D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner,
    N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg,
    Improved Current Density and Magnetization Reconstruction Through
    Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020).
    https://doi.org/10.1103/PhysRevApplied.14.024076
    """

    if use_components == "auto":
        if Bx is not None and By is not None:
            use_components = "xy"
        elif Bz is not None:
            use_components = "z"
        else:
            raise ValueError("Must provide Bx, By or Bz")

    # Determine output key based on magnetization_angle
    if magnetization_angle is None:
        result_key = "Mz"
    else:
        result_key = "Mpsi"

    meta = {
        "pixel_size": pixel_size,
        "standoff": standoff,
        "nv_layer_thickness": nv_layer_thickness,
        "magnetization_angle": magnetization_angle,
        "use_components": use_components,
        "pad_mode": pad_mode,
        "pad_factor": pad_factor,
        "do_hanning_filter": do_hanning_filter,
        "hanning_low_cutoff": hanning_low_cutoff,
        "hanning_high_cutoff": hanning_high_cutoff,
    }

    if use_components == "xy":
        bx = np.copy(Bx) * 1e-4
        by = np.copy(By) * 1e-4

        bx_pad, padder = dukit.fourier.pad_image(bx, pad_mode, pad_factor)
        by_pad, _ = dukit.fourier.pad_image(by, pad_mode, pad_factor)

        fft_bx = numpy_fft.fftshift(numpy_fft.fft2(bx_pad))
        fft_by = numpy_fft.fftshift(numpy_fft.fft2(by_pad))
        fft_bx = dukit.fourier.set_naninf_to_zero(fft_bx)
        fft_by = dukit.fourier.set_naninf_to_zero(fft_by)

        ky, kx, k = dukit.fourier.define_k_vectors(fft_bx.shape, pixel_size, k_vector_epsilon)

        d_matrix = define_magnetization_transformation(ky, kx, k, standoff, nv_layer_thickness)

        if magnetization_angle is None:
            m_to_bx = d_matrix[2, 0, ::]  # z magnetized
            m_to_by = d_matrix[2, 1, ::]
        else:
            psi = np.deg2rad(magnetization_angle)
            m_to_bx = np.cos(psi) * d_matrix[0, 0, ::] + np.sin(psi) * d_matrix[1, 0, ::]
            m_to_by = np.cos(psi) * d_matrix[0, 1, ::] + np.sin(psi) * d_matrix[1, 1, ::]

        hanning_filt = dukit.fourier.hanning_filter_kspace(
            k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
        )

        with np.errstate(all="ignore"):
            fft_m_bx = fft_bx * hanning_filt / m_to_bx
            fft_m_by = fft_by * hanning_filt / m_to_by
            fft_m = (fft_m_bx + fft_m_by) / 2

        fft_m = dukit.fourier.set_naninf_to_zero(fft_m)

        m = numpy_fft.ifft2(numpy_fft.ifftshift(fft_m)).real

        m_reg = dukit.fourier.unpad_image(m, padder)

        return {
            result_key: m_reg * MAG_UNIT_CONV,
            "_metadata": meta,
        }

    elif use_components == "z":
        if Bz is None:
            raise ValueError("Bz must be provided when use_components='z'")

        bz = np.copy(Bz) * 1e-4
        bz_pad, padder = dukit.fourier.pad_image(bz, pad_mode, pad_factor)

        fft_bz = numpy_fft.fftshift(numpy_fft.fft2(bz_pad))
        fft_bz = dukit.fourier.set_naninf_to_zero(fft_bz)

        ky, kx, k = dukit.fourier.define_k_vectors(fft_bz.shape, pixel_size, k_vector_epsilon)

        d_matrix = define_magnetization_transformation(ky, kx, k, standoff, nv_layer_thickness)

        if magnetization_angle is None:
            m_to_bz = d_matrix[2, 2, ::]  # z magnetized
        else:
            psi = np.deg2rad(magnetization_angle)
            m_to_bz = np.cos(psi) * d_matrix[0, 2, ::] + np.sin(psi) * d_matrix[1, 2, ::]

        hanning_filt = dukit.fourier.hanning_filter_kspace(
            k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
        )

        with np.errstate(all="ignore"):
            fft_m = fft_bz * hanning_filt / m_to_bz

        fft_m = dukit.fourier.set_naninf_to_zero(fft_m)

        m = numpy_fft.ifft2(numpy_fft.ifftshift(fft_m)).real

        m_reg = dukit.fourier.unpad_image(m, padder)

        return {
            result_key: m_reg * MAG_UNIT_CONV,
            "_metadata": meta,
        }

    else:
        raise ValueError(f"Invalid use_components: {use_components}")


# ============================================================================


def get_magnetization_from_bdefect(
    b_defect: np.ndarray,
    u_defect: np.ndarray,
    pixel_size: float = 1.0,
    standoff: float | None = None,
    nv_layer_thickness: float | None = None,
    magnetization_angle: float | None = None,
    nv_above_sample: bool = True,
    pad_mode: str | None = "edge",
    pad_factor: int = 2,
    k_vector_epsilon: float = 1e-6,
    do_hanning_filter: bool = False,
    hanning_low_cutoff: float | None = None,
    hanning_high_cutoff: float | None = None,
) -> dict[str, np.ndarray]:
    r"""Reconstruct magnetization from single B_defect measurement.

    Can reconstruct either out-of-plane magnetization (Mz) or in-plane
    magnetization (Mpsi) at a specified angle.

    Parameters
    ----------
    b_defect : np.ndarray
        B-field along defect axis in Gauss (2D array).
    u_defect : np.ndarray
        Defect orientation unit vector, shape (3,). Normalized internally.
    pixel_size : float, default=1.0
        Effective pixel size in meters.
    standoff : float | None
        Distance from NV layer to sample in meters.
    nv_layer_thickness : float | None
        Thickness of NV layer in meters.
    magnetization_angle : float | None, default=None
        In-plane magnetization angle in degrees from +x towards +y.
        None = out-of-plane (Mz).
        Any value = in-plane at that angle (Mpsi).
    nv_above_sample : bool, default=True
        True if NV layer is above sample (higher z).
    pad_mode : str | None, default="edge"
        Mode for Fourier padding. See np.pad for options.
    pad_factor : int, default=2
        Factor to pad image on all sides.
    k_vector_epsilon : float, default=1e-6
        Epsilon for k-vectors to avoid division by zero.
    do_hanning_filter : bool, default=False
        Apply Hanning filter to reduce noise amplification.
    hanning_low_cutoff, hanning_high_cutoff : float | None
        Cutoff wavelengths in meters for Hanning filter.

    Returns
    -------
    dict with keys:
        - "Mz" or "Mpsi": Magnetization (μB/nm²)
          - "Mz" if magnetization_angle is None (out-of-plane)
          - "Mpsi" if magnetization_angle is given (in-plane at angle)
        - "_metadata": Dict with parameters used

    Reference
    ---------
    D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner,
    N. Dontschuk, P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg,
    Improved Current Density and Magnetization Reconstruction Through
    Vector Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020).
    https://doi.org/10.1103/PhysRevApplied.14.024076
    """

    # Determine output key based on magnetization_angle
    if magnetization_angle is None:
        result_key = "Mz"
    else:
        result_key = "Mpsi"

    meta = {
        "pixel_size": pixel_size,
        "standoff": standoff,
        "nv_layer_thickness": nv_layer_thickness,
        "magnetization_angle": magnetization_angle,
        "nv_above_sample": nv_above_sample,
        "pad_mode": pad_mode,
        "pad_factor": pad_factor,
        "do_hanning_filter": do_hanning_filter,
        "hanning_low_cutoff": hanning_low_cutoff,
        "hanning_high_cutoff": hanning_high_cutoff,
    }

    b = np.copy(b_defect) * 1e-4
    b_pad, padder = dukit.fourier.pad_image(b, pad_mode, pad_factor)

    fft_b = numpy_fft.fftshift(numpy_fft.fft2(b_pad))

    ky, kx, k = dukit.fourier.define_k_vectors(fft_b.shape, pixel_size, k_vector_epsilon)

    d_matrix = define_magnetization_transformation(ky, kx, k, standoff, nv_layer_thickness)

    if nv_above_sample:
        unv_cpy = np.copy(u_defect)
    else:
        unv_cpy = np.array([-u_defect[0], -u_defect[1], u_defect[2]])

    if magnetization_angle is None:
        # z magnetized
        m_to_b = (
            unv_cpy[0] * d_matrix[2, 0, ::]
            + unv_cpy[1] * d_matrix[2, 1, ::]
            + unv_cpy[2] * d_matrix[2, 2, ::]
        )
    else:
        # in-plane magnetization
        psi = np.deg2rad(magnetization_angle)
        b_axis = np.nonzero(unv_cpy)[0]
        m_to_b = None
        for idx in b_axis:
            new = unv_cpy[int(idx)] * (
                np.cos(psi) * d_matrix[0, int(idx), ::] + np.sin(psi) * d_matrix[1, int(idx), ::]
            )
            m_to_b = new if m_to_b is None else m_to_b + new

    hanning_filt = dukit.fourier.hanning_filter_kspace(
        k, do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff, standoff
    )

    with np.errstate(all="ignore"):
        fft_m = fft_b * hanning_filt / m_to_b

    fft_m = dukit.fourier.set_naninf_to_zero(fft_m)

    m = numpy_fft.ifft2(numpy_fft.ifftshift(fft_m)).real

    m_reg = dukit.fourier.unpad_image(m, padder)

    return {
        result_key: m_reg * MAG_UNIT_CONV,
        "_metadata": meta,
    }


# ============================================================================


def normalize_in_plane_mag(
    magnetization: np.ndarray,
    angle_deg: float,
    edge_pixels: int = 10,
) -> np.ndarray:
    """Normalize in-plane magnetization by subtracting line artifacts.

    For each line parallel to the magnetization direction, subtract
    the average of the edge pixels. This removes artifacts from
    reconstruction.

    Parameters
    ----------
    magnetization : np.ndarray
        2D magnetization array.
    angle_deg : float
        In-plane magnetization angle in degrees from +x towards +y.
    edge_pixels : int, default=10
        Number of pixels at each edge to use for averaging.

    Returns
    -------
    Normalized magnetization array.

    Notes
    -----
    Adapted from D. Broadway's original implementation in qdmpy.
    """
    psi = np.deg2rad(angle_deg)
    new_im = magnetization.copy()

    if psi < 0:
        magnetization = np.flip(magnetization, 1)
        new_im = np.flip(new_im, 1)
        psi = np.abs(psi)
        flip_back = True
    else:
        flip_back = False

    height, width = magnetization.shape
    max_y_idx = height - 1

    # Get indices of a line from origin at bottom left at psi (from +x to +y)
    origin_line_y = max_y_idx - np.tan(psi) * np.arange(width)
    origin_line_y_ints = [round(y) for y in origin_line_y.tolist()]

    offset = 0  # Look for parallel lines, at +- 'offset' in y
    for _ in range(height):
        above_y_idxs = []
        above_x_idxs = []
        below_y_idxs = []
        below_x_idxs = []

        for x_idx, y_idx in enumerate(origin_line_y_ints):
            if y_idx - offset >= 0 and y_idx - offset <= max_y_idx:
                above_y_idxs.append(y_idx - offset)
                above_x_idxs.append(x_idx)

            if y_idx + offset >= 0 and y_idx + offset <= max_y_idx:
                below_y_idxs.append(y_idx + offset)
                below_x_idxs.append(x_idx)

        for coords in [
            (above_y_idxs, above_x_idxs),
            (below_y_idxs, below_x_idxs),
        ]:
            im_cut = magnetization[coords]
            if len(im_cut) > 0:
                edge_avg = (np.mean(im_cut[:edge_pixels]) + np.mean(im_cut[-edge_pixels:])) / 2
                new_im[coords] = im_cut - edge_avg

        offset += 1

    if flip_back:
        new_im = np.flip(new_im, 1)

    return new_im


# ============================================================================
