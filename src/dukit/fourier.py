# -*- coding: utf-8 -*-
"""
Shared FFTW tooling.

Functions
---------
 - `dukit.fourier.unpad_image`
 - `dukit.fourier.pad_image`
 - `dukit.fourier.define_k_vectors`
 - `dukit.fourier.set_naninf_to_zero`
 - `dukit.fourier.hanning_filter_kspace`
 - `dukit.fourier.define_magnetization_transformation`
 - `dukit.fourier.define_current_transform`

Constants
---------
 - `dukit.fourier.MAG_UNIT_CONV`
 - `dukit.fourier.MU_0`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.fourier.unpad_image": True,
    "dukit.fourier.pad_image": True,
    "dukit.fourier.define_k_vectors": True,
    "dukit.fourier.set_naninf_to_zero": True,
    "dukit.fourier.hanning_filter_kspace": True,
    "dukit.fourier.MAG_UNIT_CONV": True,
    "dukit.fourier.MU_0": True,
    "dukit.fourier.define_magnetization_transformation": True,
    "dukit.fourier.define_current_transform": True,
}

# ============================================================================

import numpy as np
import numpy.typing as npt
from pyfftw.interfaces import numpy_fft

# ============================================================================

MAG_UNIT_CONV: float = 1e-18 / 9.274010e-24
"""
Convert unit for magnetization to something more helpful.

SI unit measured: Amps: A [for 2D magnetization, A/m for 3D]

More useful: Bohr magnetons per nanometre squared: mu_B nm^-2

```
mu_B -> 9.274 010 e-24 A m^+2 or J/T

m^2 -> 1e+18 nm^2


Measure x amps = x A

 def  mu_B  =  9.2_      in units of A m^2

 => x A = x (1 / 9.2_)   in units of mu_B/m^2

 => x A = x (1e-18/9.2_) in units of mu_B/nm^2
```
"""


MU_0: float = 1.25663706212 * 1e-6
"""
Vacuum permeability
"""


# ============================================================================


def unpad_image(x: npt.NDArray, padder: tuple[tuple[int, int], tuple[int, int]]) -> npt.NDArray:
    """undo a padding defined by `dukit.fourier.pad_image` (it returns
    the padder list)"""
    slices = []
    for c in padder:
        e = None if c[1] == 0 else -c[1]
        slices.append(slice(c[0], e))
    return x[tuple(slices)]


# ============================================================================


def pad_image(
    image: npt.NDArray, pad_mode: str, pad_factor: int
) -> tuple[npt.NDArray, tuple[tuple[int, int], tuple[int, int]]]:
    """
    pad_mode -> see np.pad
    pad_factor -> either side of image
    """

    if len(np.shape(image)) != 2:
        raise ValueError("image passed to pad_image was not 2D.")

    image = np.array(image)

    if pad_mode is None:
        return image, ((0, 0), (0, 0))

    size_y, size_x = image.shape

    y_pad = int(pad_factor * size_y)
    x_pad = int(pad_factor * size_x)
    padder = ((y_pad, y_pad), (x_pad, x_pad))
    padded_image = np.pad(image, mode=pad_mode, pad_width=padder)

    return padded_image, padder


# ============================================================================


def define_k_vectors(
    shape: tuple[int, int],
    raw_pixel_size: float,
    applied_binning: tuple[int, int] | int = 1,
    k_vector_epsilon: float = 1e-6,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Get scaled k vectors (as meshgrid) for fft.

    Arguments
    ----------
    shape : list
        Shape of fft array to get k vectors for.
    raw_pixel_size : float
        I.e. camera pixel size
    applied_binning : 2-tuple of ints or int
        Binning that has been applied. 2-tuple for asymmetric binning
        NOTE sorry untested in fourier currently
    k_vector_epsilon : float
        Add an epsilon value to the k-vectors to avoid some issues with 1/0.

    Returns
    -------
    ky, kx, k : np array
        Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 )
    """
    # scaling for the k vectors so they are in the right units
    # (allow for asymmetric binning)
    # get the fft frequencies and shift the ordering and forces type to be float64
    if isinstance(applied_binning, tuple):
        bin_x, bin_y = applied_binning
        scaling_y = np.float64(2 * np.pi / (raw_pixel_size * bin_y))
        scaling_x = np.float64(2 * np.pi / (raw_pixel_size * bin_x))
        ky_vec = scaling_y * numpy_fft.fftshift(numpy_fft.fftfreq(shape[0]))
        kx_vec = scaling_x * numpy_fft.fftshift(numpy_fft.fftfreq(shape[1]))
    else:
        scl = raw_pixel_size * applied_binning if applied_binning else raw_pixel_size
        scaling = np.float64(2 * np.pi / scl)
        ky_vec = scaling * numpy_fft.fftshift(numpy_fft.fftfreq(shape[0]))
        kx_vec = scaling * numpy_fft.fftshift(numpy_fft.fftfreq(shape[1]))

    # Include a small factor in the k vectors to remove division by zero issues (min_k)
    # Make a meshgrid to pass back
    if k_vector_epsilon:
        ky, kx = np.meshgrid(ky_vec - k_vector_epsilon, kx_vec + k_vector_epsilon, indexing="ij")
    else:
        ky, kx = np.meshgrid(ky_vec, kx_vec, indexing="ij")

    k = np.sqrt(ky**2 + kx**2)
    return -ky, kx, k  # negative here to maintain correct image orientation


# ============================================================================


def set_naninf_to_zero(array: npt.NDArray) -> npt.NDArray:
    """replaces NaNs and infs with zero"""
    idxs = np.logical_or(np.isnan(array), np.isinf(array))
    array[idxs] = 0
    return array


# ============================================================================


def hanning_filter_kspace(
    k: npt.NDArray,
    do_filt: bool,
    hanning_low_cutoff: float | None,
    hanning_high_cutoff: float | None,
    standoff: float | None,
) -> npt.NDArray | int:
    """Computes a hanning image filter with both low and high pass filters.

    Arguments
    ---------
    k : np array
        Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 )
    do_filt : bool
        Do a hanning filter?
    hanning_high_cutoff : float
        Set highpass cutoff k values. Give as a distance/wavelength, e.g. k_high will be
        set via k_high = 2pi/high_cutoff. Should be _smaller_ number than low_cutoff.
    hanning_low_cutoff : float
        Set lowpass cutoff k values. Give as a distance/wavelength, e.g. k_low will be
        set via k_low = 2pi/low_cutoff. Should be _larger_ number than high_cutoff.
    standoff : float
        Distance NV layer <-> Sample.

    Returns
    -------
    img_filter : (2d array, float)
        bandpass filter to remove artifacts in the FFT process.
    """
    # Define Hanning filter to prevent noise amplification at frequencies higher than the
    # spatial resolution

    if do_filt and standoff and standoff > 1e-10:  # standoff greater than an angstrom...
        hy = np.hanning(k.shape[0])
        hx = np.hanning(k.shape[1])
        img_filt = np.sqrt(np.outer(hy, hx))
        # apply cutoffs
        if hanning_high_cutoff is not None:
            k_cut_high = (2 * np.pi) / hanning_high_cutoff
            img_filt[k > k_cut_high] = 0
        else:
            k_cut_high = (2 * np.pi) / standoff
            img_filt[k > k_cut_high] = 0
        if hanning_low_cutoff is not None:
            k_cut_low = (2 * np.pi) / hanning_low_cutoff
            img_filt[k < k_cut_low] = 0
    else:
        img_filt = 1
    return img_filt


# ============================================================================


def define_magnetization_transformation(
    ky: npt.NDArray,
    kx: npt.NDArray,
    k: npt.NDArray,
    standoff: float | None = None,
    nv_layer_thickness: float | None = None,
) -> npt.NDArray:
    """M => b fourier-space transformation.

    Parameters
    ----------
    ky, kx, k : np array
        Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 )
    standoff : float
        Distance NV layer <-> Sample
    nv_layer_thickness : float or None, default : None
        Thickness of NV layer (in metres)

    Returns
    -------
    d_matrix : np array
        Transformation such that B = d_matrix * m. E.g. for z magnetized sample:
        m_to_bnv = (
            unv[0] * d_matrix[2, 0, ::] + unv[1] * d_matrix[2, 1, ::] +
            unv[2] * d_matrix[2, 2, ::]
        )
        -> First index '2' is for z magnetization
        (see m_from_bxy for in-plane mag process), the
        second index is for the (bnv etc.) bfield axis (0:x, 1:y, 2:z),
        and the last index iterates through the k values/vectors.


    See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk,
        P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg,
        Improved Current Density and Magnetization Reconstruction Through Vector
        Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020).
        https://doi.org/10.1103/PhysRevApplied.14.024076
        https://arxiv.org/abs/2005.06788
    """

    if standoff:
        exp_factor = np.exp(k * standoff)
        if nv_layer_thickness:
            # average exp factor exp(-k z) across
            # z = [standoff - nv_thickness / 2, standoff + nv_thickness / 2]
            # get exp(-k z) * sinh(k nv_thickness / 2) / (k nv_thickness / 2)
            # (inverted here)
            arg = k * nv_layer_thickness / 2
            exp_factor *= 1 / (arg / np.sinh(arg))
    else:
        exp_factor = 1

    alpha = 2 * exp_factor / MU_0

    return (1 / alpha) * np.array(
        [
            [-(kx**2) / k, -(kx * ky) / k, -1j * kx],
            [-(kx * ky) / k, -(ky**2) / k, -1j * ky],
            [-1j * kx, -1j * ky, k],
        ]
    )


# ============================================================================


def define_current_transform(
    u_proj: tuple[float, float, float],
    ky: npt.NDArray,
    kx: npt.NDArray,
    k: npt.NDArray,
    standoff: float | None = None,
    nv_layer_thickness: float | None = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """b => J fourier-space transformation.

    Arguments
    ---------
    u_proj : 3-tuple
        Shape: 3, the direction the magnetic field was measured in (projected onto).
    ky, kx, k : np arrays
        Wavenumber meshgrids, k = sqrt( kx^2 + ky^2 )
    standoff : float or None, default : None
        Distance NV layer <-> sample
    nv_layer_thickness : float or None, default : None
        Thickness of NV layer (in metres)

    Returns
    -------
    b_to_jx, b_to_jy : np arrays (2D)

    See D. A. Broadway, S. E. Lillie, S. C. Scholten, D. Rohner, N. Dontschuk,
        P. Maletinsky, J.-P. Tetienne, and L. C. L. Hollenberg,
        Improved Current Density and Magnetization Reconstruction Through Vector
        Magnetic Field Measurements, Phys. Rev. Applied 14, 024076 (2020).
        https://doi.org/10.1103/PhysRevApplied.14.024076
        https://arxiv.org/abs/2005.06788
    """
    if standoff:
        exp_factor = np.exp(k * standoff)
        if nv_layer_thickness:
            # average exp factor exp(-k z) across
            # z = [standoff - nv_thickness / 2, standoff + nv_thickness / 2]
            # get exp(-k z) * sinh(k nv_thickness / 2) / (k nv_thickness / 2)
            # (inverted here)
            arg = k * nv_layer_thickness / 2
            exp_factor *= arg / np.sinh(arg)
    else:
        exp_factor = 1

    alpha = 2 * exp_factor / MU_0

    prefac = alpha / (u_proj[0] * kx + u_proj[1] * ky + 1j * u_proj[2] * k)
    b_to_jx = prefac * -ky
    b_to_jy = prefac * kx

    return b_to_jx, b_to_jy


# ============================================================================
