# -*- coding: utf-8 -*-
"""High-level source reconstruction interface."""

from typing import Literal

import numpy as np

import dukit.fourier
import dukit.itool
from dukit.source.current import (
    get_current_from_bxyz,
)
from dukit.source.magnetization import (
    get_magnetization_from_bxyz,
)


def reconstruct_source(
    bxyz: dict[str, np.ndarray],
    pixel_size: float,
    source_type: Literal["current", "magnetization"] = "current",
    standoff: float | None = None,
    nv_layer_thickness: float | None = None,
    nv_above_sample: bool = True,
    magnetization_angle: float | None = None,
    use_components: Literal["xy", "z", "xyz", "auto"] = "auto",
    subtract_background: bool = False,
    background_method: str | None = None,
    background_params: dict | None = None,
    **fourier_kwargs,
) -> dict[str, np.ndarray]:
    """High-level source reconstruction from Bxyz field.

    Parameters
    ----------
    bxyz : dict
        Dictionary with keys "Bx", "By", "Bz" (2D arrays in Gauss).
        Can also include "u_defects" for defect-based reconstruction.
    pixel_size : float
        Effective pixel size in meters.
    source_type : {"current", "magnetization"}, default="current"
        Type of source to reconstruct.
    standoff : float | None
        Distance from NV layer to sample in meters.
    nv_layer_thickness : float | None
        Thickness of NV layer in meters.
    nv_above_sample : bool, default=True
        True if NV layer is above sample (higher z).
    magnetization_angle : float | None, default=None
        For magnetization only: None for out-of-plane (Mz),
        or angle in degrees for in-plane (Mpsi).
    use_components : {"xy", "z", "xyz", "auto"}, default="auto"
        Which field components to use for reconstruction.
    subtract_background : bool, default=False
        Subtract background from reconstruction results.
    background_method : str | None
        Method for background subtraction (e.g., "poly", "gaussian").
        See dukit.itool.get_background for options.
    background_params : dict | None
        Parameters for background subtraction.
    **fourier_kwargs
        Additional arguments passed to reconstruction functions:
        - pad_mode, pad_factor, k_vector_epsilon
        - do_hanning_filter, hanning_low_cutoff, hanning_high_cutoff

    Returns
    -------
    dict with reconstruction results in the same format as the
    underlying reconstruction function (get_current_from_bxyz or
    get_magnetization_from_bxyz).

    Example
    -------
    >>> import dukit
    >>> import numpy as np
    >>>
    >>> # First get Bxyz from field reconstruction
    >>> bxyz = {
    ...     "Bx": np.random.randn(100, 100) * 10,
    ...     "By": np.random.randn(100, 100) * 10,
    ...     "Bz": np.random.randn(100, 100) * 10,
    ... }
    >>>
    >>> # Reconstruct current density
    >>> result = dukit.source.reconstruct_source(
    ...     bxyz,
    ...     pixel_size=1.5e-6,
    ...     source_type="current",
    ...     standoff=150e-9,
    ...     use_components="xy",
    ... )
    >>> Jx = result["Jx"]
    >>> Jy = result["Jy"]
    >>>
    >>> # Or reconstruct magnetization
    >>> mag = dukit.source.reconstruct_source(
    ...     bxyz,
    ...     pixel_size=1.5e-6,
    ...     source_type="magnetization",
    ...     standoff=150e-9,
    ...     magnetization_angle=None,  # Out-of-plane
    ... )
    >>> Mz = mag["Mz"]
    """

    # Check what reconstruction inputs we have
    has_bxyz = all(key in bxyz for key in ["Bx", "By"])

    if source_type == "current":
        if has_bxyz:
            result = get_current_from_bxyz(
                bxyz["Bx"],
                bxyz["By"],
                Bz=bxyz.get("Bz"),
                pixel_size=pixel_size,
                standoff=standoff,
                nv_layer_thickness=nv_layer_thickness,
                use_components=use_components,
                **fourier_kwargs,
            )
        else:
            raise ValueError("For current reconstruction, bxyz must contain 'Bx' and 'By'")

    elif source_type == "magnetization":
        if has_bxyz:
            result = get_magnetization_from_bxyz(
                bxyz["Bx"],
                bxyz["By"],
                Bz=bxyz.get("Bz"),
                pixel_size=pixel_size,
                standoff=standoff,
                nv_layer_thickness=nv_layer_thickness,
                use_components=use_components,
                **fourier_kwargs,
            )
        else:
            raise ValueError(
                "For magnetization reconstruction, bxyz must contain 'Bx' and 'By' or 'Bz'"
            )

    else:
        raise ValueError(f"Invalid source_type: {source_type}")

    # Optional background subtraction
    if subtract_background:
        if background_method is None:
            background_method = "poly"
        if background_params is None:
            background_params = {}

        result = _subtract_background_from_result(result, background_method, background_params)

    return result


def _subtract_background_from_result(
    result: dict[str, np.ndarray],
    background_method: str,
    background_params: dict,
) -> dict[str, np.ndarray]:
    """Subtract background from reconstruction result arrays.

    Parameters
    ----------
    result : dict
        Reconstruction result dict.
    background_method : str
        Background subtraction method.
    background_params : dict
        Parameters for background subtraction.

    Returns
    -------
    Updated result dict with background subtracted.
    """
    result_copy = result.copy()

    for key, value in result.items():
        if key.startswith("_"):
            continue

        bg, _ = dukit.itool.get_background(value, background_method, **background_params)
        result_copy[key] = value - bg

    return result_copy
