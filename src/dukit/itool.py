# -*- coding: utf-8 -*-
"""
This module holds misc image tooling.

Functions
---------
 - `dukit.itool.mpl_set_run_config`
 - `dukit.itool.mask_polygons`
 - `dukit.itool.get_im_filtered`
 - `dukit.itool.get_background`
 - `dukit.itool.mu_sigma_inside_polygons`
 - `dukit.itool.plot_image`
 - `dukit.itool.plot_image_on_ax`
 - `dukit.itool.get_colormap_range`
 - `dukit.itool.crop_roi`
 - `dukit.itool.crop_sweep`
 - `dukit.itool.rebin_image_stack`
 - `dukit.itool.smooth_image_stack`
 - `dukit.itool.sum_spatially`
 - `dukit.itool.get_aois`
 - `dukit.itool._iterslice`
 - `dukit.itool._iterframe`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.itool.mpl_set_run_config": True,
    "dukit.itool.mask_polygons": True,
    "dukit.itool.get_im_filtered": True,
    "dukit.itool.get_background": True,
    "dukit.itool.mu_sigma_inside_polygons": True,
    "dukit.itool.plot_image": True,
    "dukit.itool.plot_image_on_ax": True,
    "dukit.itool.get_colormap_range": True,
    "dukit.itool.crop_roi": True,
    "dukit.itool.crop_sweep": True,
    "dukit.itool.rebin_image_stack": True,
    "dukit.itool.smooth_image_stack": True,
    "dukit.itool.sum_spatially": True,
    "dukit.itool.get_aois": True,
    "dukit.itool._iterslice": True,
    "dukit.itool._iterframe": True,
}

# ============================================================================

from itertools import product
from typing import Callable

import matplotlib as mpl
import matplotlib.colorbar
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import axes_size, make_axes_locatable
from numpy import typing as npt
from numpy.polynomial.polynomial import polyval2d
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.optimize import least_squares

# ============================================================================
import dukit.polygon
import dukit.rebin
from dukit.warn import warn

# ============================================================================

DEFAULT_RCPARAMS: dict = {
    "figure.constrained_layout.use": False,
    "figure.figsize": [6.4, 4.8],
    "figure.dpi": 80,
    "figure.max_open_warning": 30,
    "lines.linewidth": 0.8,
    "lines.markersize": 3,
    "xtick.labelsize": 10,
    "xtick.major.size": 4,
    "xtick.direction": "in",
    "ytick.labelsize": 10,
    "ytick.direction": "in",
    "ytick.major.size": 4,
    "legend.fontsize": "small",
    "legend.loc": "lower left",
    "scalebar.location": "lower right",
    "scalebar.width_fraction": 0.015,
    "scalebar.box_alpha": 0.5,
    "scalebar.scale_loc": "top",
    "scalebar.sep": 1,
    "axes.formatter.useoffset": False,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-2, 6),
    "errorbar.capsize": 3.0,
}
"""Set of rcparams we've casually decided are reasonable."""


def mpl_set_run_config(default: bool = True, **kwargs):
    """
    Set the matplotlib runtime configuration (rcparams).

    Parameters
    ----------
    default : bool
        Use dukit default rcparams?
    kwargs
        kwargs to pass to rcparams, see
        https://matplotlib.org/stable/users/explain/customizing.html
    """
    if default:
        mpl.rcParams.update(DEFAULT_RCPARAMS)
    else:
        mpl.rcParams.update(**kwargs)


# ============================================================================


def mask_polygons(
    image: npt.NDArray, polygons: list | None = None, invert_mask: bool = False
) -> npt.NDArray:
    """Mask image for the given polygon regions.

    Arguments
    ---------
    image : 2D array-like
        Image array to mask.
    polygons : list, optional
        List of `qdmpy.shared.polygon.Polygon` objects.
        (the default is None, where image is returned with no mask)
    invert_mask : bool, optional
        Invert mask such that background is masked, not polygons (i.e. polygons
        will be operated on if array is passed to np.mean instead of background).
        (the default is False)

    Returns
    -------
    masked_im : np.ma.MaskedArray
        image, now masked
    """
    image = np.array(image)
    if polygons is None:
        return np.ma.masked_array(image, mask=np.zeros(image.shape))
    if len(image.shape) != 2:
        raise ValueError("image is not a 2D array")

    if not isinstance(polygons, (list, tuple)) or not isinstance(
        polygons[0], dukit.polygon.Polygon
    ):
        raise TypeError("polygons were not None, a list or a list of Polygon objects")

    ylen, xlen = image.shape
    masked_area = np.full(image.shape, True)  # all masked to start with

    grid_y, grid_x = np.meshgrid(range(ylen), range(xlen), indexing="ij")

    for p in polygons:
        in_or_out = p.is_inside(grid_y, grid_x)
        # mask all vals that are not background
        m = np.ma.masked_greater_equal(in_or_out, 0).mask
        masked_area = np.logical_and(masked_area, ~m)

    msk = np.logical_not(masked_area) if not invert_mask else masked_area

    return np.ma.masked_array(image, mask=msk)


def get_background(
    image: npt.NDArray,
    method: str,
    polygon_nodes: list | None = None,
    **method_params_dict,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Returns a background for given image, via chosen method.

    Methods available:

    - "fix_zero"
        - Fix background to be a constant offset (z value)
        - params required in method_params_dict:
            "zero" an int/float, defining the constant offset of the background
    - "three_point"
        - Calculate plane background with linear algebra from three [x,y]
          lateral positions given
        - params required in method_params_dict:
            - "points" a len-3 iterable containing [x, y] points
    - "mean"
        - background calculated from mean of image
        - no params required
    - "poly"
        - background calculated from polynomial fit to image.
        - params required in method_params_dict:
            - "order": an int, the 'order' polynomial to fit. (e.g. 1 = plane).
    - "gaussian"
        - background calculated from _gaussian fit to image (with rotation)
        - no params required
    - "lorentzian"
        - as above, but a lorentzian lineshape (with rotation)
    - "interpolate"
        - Background defined by the dataset smoothed via a sigma-_gaussian
            filtering, and method-interpolation over masked (polygon) regions.
        - params required in method_params_dict:
            - "interp_method": nearest, linear, cubic.
            - "sigma": sigma passed to _gaussian filter
                (see scipy.ndimage._gaussian_filter) which is utilized on the
                background before interpolating
    - "gaussian_filter"
        - background calculated from image filtered with a _gaussian filter.
        - params required in method_params_dict:
            - "sigma": sigma passed to _gaussian filter (see
                    scipy.ndimage._gaussian_filter)
    - "gaussian_then_poly"
        - runs gaussian then poly subtraction
        - params required in method_params_dict:
            - "order": an int, the 'order' polynomial to fit. (e.g. 1 = plane).

    polygon utilization:
        - if method is not interpolate, the image is masked where the polygons are
          and the background is calculated without these regions
        - if the method is interpolate, these regions are interpolated over
            (and the rest of the image, _gaussian smoothed, is 'background').

    Arguments
    ---------
    image : 2D array-like
        image to get backgrond of
    method : str
        Method to use, available options above
    **method_params_dict
        Key-value pairs passed onto each background backend. Required params
        given above.
    polygon_nodes : list | None = None
        Optionally provide polygon nodes.

    Returns
    -------
    im_bground : ndarray
        2D numpy array, representing the 'background' of image.
    mask : ndarray
        Mask (True pixels were not used to calculate background).
    """
    # Discuss masking -> if polygons provided, background is calculated but with
    # polygon regions masked -> background calculated without these.
    # also used for interpolation method.
    method_required_settings: dict[str, list] = {
        "fix_zero": ["zero"],
        "three_point": ["points"],
        "mean": [],
        "poly": ["order"],
        "gaussian": [],
        "lorentzian": [],
        "interpolate": ["interp_method", "sigma"],
        "gaussian_filter": ["sigma"],
        "gaussian_then_poly": ["order"],
    }
    method_fns: dict[str, Callable] = {
        "fix_zero": _zero_background,
        "three_point": _three_point_background,
        "mean": _mean_background,
        "poly": _poly_background,
        "gaussian": _gaussian_background,
        "lorentzian": _lorentzian_background,
        "interpolate": _interpolated_background,
        "gaussian_filter": _filtered_background,
        "gaussian_then_poly": _gaussian_then_poly,
    }
    image = np.ma.array(image, mask=np.zeros(np.shape(image)))
    if len(image.shape) != 2:
        raise ValueError("image is not a 2D array")
    if not isinstance(method, str):
        raise TypeError("method must be a string.")
    if not isinstance(method_params_dict, dict):
        raise TypeError("method_params_dict must be a dict.")
    if method not in method_required_settings:
        raise ValueError(
            "'method' argument to get_background not in implemented_methods: "
            + f"{method_required_settings.keys()}"
        )
    for setting in method_required_settings[method]:
        if setting not in method_params_dict:
            raise ValueError(f"{setting} key missing from method_params_dict for method: {method}")

    if polygon_nodes:
        polygons = [dukit.polygon.Polygon(nodes[:, 0], nodes[:, 1]) for nodes in polygon_nodes]
    else:
        polygons = None

    if method != "interpolate":
        # can't mask it for interpolate as we need that info!
        image = mask_polygons(image, polygons)

    if method == "gaussian_filter":
        method_params_dict["filter_type"] = "gaussian"

    if method == "interpolate":
        method_params_dict["polygons"] = polygons

    if method == "three_point" and "sample_size" not in method_params_dict:
        method_params_dict["sample_size"] = 0
    return (
        method_fns[method](image, **method_params_dict),
        np.ma.getmaskarray(image).astype(int),
    )


# ============================================================================


def mu_sigma_inside_polygons(
    image: npt.NDArray, polygons: list | None = None
) -> tuple[float, float]:
    """returns (mean, standard_deviation) for image, only _within_ polygon areas."""
    image = mask_polygons(image, polygons, invert_mask=True)
    return np.mean(image), np.std(image)


# ============================================================================


# hands off to other filters
def get_im_filtered(image: npt.NDArray, filter_type: str, **kwargs) -> npt.NDArray:
    """Wrapped over other filters.
    Current filters defined:
        - filter_type = gaussian, `qdmpy.shared.itool._get_im_filtered_gaussian`
    """
    if not isinstance(image, np.ma.core.MaskedArray):
        image = np.ma.masked_array(image)
    filter_fns = {
        "gaussian": _get_im_filtered_gaussian,
    }
    # filter not so great with nans (and mask doesn't work w filters) -> set to nanmean
    if "upper_threshold" in kwargs and kwargs["upper_threshold"]:
        image[image > kwargs["upper_threshold"]] = np.nan
    if "lower_threshold" in kwargs and kwargs["lower_threshold"]:
        image[image < kwargs["lower_threshold"]] = np.nan
    image[np.logical_or(np.isnan(image), image.mask)] = np.nanmean(image)

    return filter_fns[filter_type](image, **kwargs)


# ============================================================================


def _get_im_filtered_gaussian(image: npt.NDArray, sigma: float) -> npt.NDArray:
    """Returns image filtered through scipy.ndimage.gaussian_filter with
    parameter 'sigma'."""
    return scipy.ndimage.gaussian_filter(
        image,
        sigma=sigma,
    )


# ============================================================================


def _zero_background(image: npt.NDArray, zero: float) -> npt.NDArray:
    """Background defined by z level of 'zero'"""
    if not isinstance(zero, (int, float)):
        raise TypeError("'zero' must be an int or a float.")
    mn = zero
    bg = np.empty(image.shape)
    bg[:] = mn
    return bg


# ============================================================================


def _equation_plane(
    params: list | tuple | npt.NDArray, y: npt.NDArray, x: npt.NDArray
) -> npt.NDArray:
    """params: [a, b, c, d] s.t. d = a*y + b*x + c*z
    so z = (1/c) * (d - (ay + bx)) -> return this."""
    return (1 / params[2]) * (params[3] - params[0] * y - params[1] * x)


def _points_to_params(points: list | tuple | npt.NDArray) -> tuple:
    """
    http://pi.math.cornell.edz/~froh/231f08e1a.pdf
    points: iterable of 3 iterables: [x, y, z]
    returns a,b,c,d parameters (see _equation_plane)
    """
    rearranged_points = [[p[1], p[0], p[2]] for p in points]  # change to [y, x, z]
    pts = np.array(rearranged_points)
    vec1_in_plane = pts[1] - pts[0]
    vec2_in_plane = pts[2] - pts[0]
    a_normal = np.cross(vec1_in_plane, vec2_in_plane)

    d = np.dot(pts[2], a_normal)
    return *a_normal, d


def _three_point_background(
    image: npt.NDArray, points: list | tuple | npt.NDArray, sample_size: int
) -> npt.NDArray:
    """points: len 3 iterable of len 2 iterables: [[x1, y1], [x2, y2], [x3, y3]]
    sample_size: integer
    https://stackoverflow.com/questions/20699821/find-and-draw-regression-plane-to-a-set-of-points
    https://www.geeksforgeeks.org/program-to-find-equation-of-a-plane-passing-through-3-points/
    """

    if len(points) != 3:
        raise ValueError("points needs to be len 3 of format: [x, y] (int or floats).")
    if not isinstance(sample_size, int) or sample_size < 0:
        raise TypeError("sample_size must be an integer >= 0")
    for p in points:
        if len(p) != 2:
            raise ValueError("points needs to be len 3 of format: [x, y] (int or floats).")
        for c in p:
            if not isinstance(c, (int, float)):
                raise ValueError("points needs to be len 3 of format: [x, y] (int or floats).")
        if image.mask[p[1], p[0]]:
            warn(
                "One of the input points was masked (inside a polygon?), "
                + "falling back on polyfit, order 1"
            )
            return _poly_background(image, order=1)

    def _mean_sample(image: npt.NDArray, sample_size: int, yx: tuple) -> float:
        """Mean of samples of 'image' centered on 'yx' out to pixels 'sample_size'
        in both x & y."""

        def _sample_generator(image, sample_size, yx):
            """sample 'image' centered on 'yx' out to pixels 'sample_size'
            in both x & y."""
            for y in range(yx[0] - sample_size, yx[0] + sample_size + 1):
                for x in range(yx[1] - sample_size, yx[1] + sample_size + 1):
                    try:
                        yield image[y, x]
                    except IndexError:
                        continue

        return np.mean(list(_sample_generator(image, sample_size, yx)))

    points = np.array(
        [np.append(p, _mean_sample(image, sample_size, (p[1], p[0]))) for p in points]
    )
    Y, X = np.indices(image.shape)  # noqa: N806
    return _equation_plane(_points_to_params(points), Y, X)


# ============================================================================


def _mean_background(image: npt.NDArray) -> npt.NDArray:
    """Background defined by mean of image."""
    bg = np.empty(image.shape)
    mn = np.nanmean(image)
    bg[:] = mn
    return bg


# ============================================================================


def _residual_poly(
    params: list | tuple | npt.NDArray,
    y: npt.NDArray,
    x: npt.NDArray,
    z: npt.NDArray,
    order: int,
) -> npt.NDArray:
    """
    z = image data, order = highest polynomial order to go to
    y, x: index meshgrids
    """
    # get params to matrix form (as expected by polyval)
    params = np.append(
        params, 0
    )  # add on c[-1, -1] term we don't want (i.e. cross term of next order)
    c = params.reshape((order + 1, order + 1))
    return polyval2d(y, x, c) - z  # note z is flattened, as is polyval


def _poly_background(image: npt.NDArray, order: int) -> npt.NDArray:
    """Background defined by a polynomial fit up to order 'order'."""
    if order == 0:
        return _mean_background(image)

    init_params = np.zeros((order + 1, order + 1))
    init_params[0, 0] = np.nanmean(image)  # set zeroth term to be mean to get it started
    Y, X = np.indices(image.shape)  # noqa: N806
    good_vals = np.logical_and(~np.isnan(image), ~image.mask)
    y = Y[good_vals]
    x = X[good_vals]

    data = image[good_vals]  # flattened
    best_c = least_squares(_residual_poly, init_params.flatten()[:-1], args=(y, x, data, order)).x
    best_c = np.append(best_c, 0)
    c = best_c.reshape((order + 1, order + 1))
    return polyval2d(Y.flatten(), X.flatten(), c).reshape(image.shape)  # eval over full image


# ============================================================================


def _gaussian(p: list | tuple | npt.NDArray, y: npt.NDArray, x: npt.NDArray) -> npt.NDArray:
    """Simple Gaussian function, height, center_y, center_x, width_y, width_x, rot = p."""
    height, center_y, center_x, width_y, width_x, rot, offset = p
    return offset + height * np.exp(
        -(
            (((y - center_y) * np.cos(rot) + (x - center_x) * np.sin(rot)) / width_y) ** 2
            + (((x - center_x) * np.cos(rot) - (y - center_y) * np.sin(rot)) / width_x) ** 2
        )
        / 2
    )


def _moments(image: npt.NDArray) -> tuple:
    """Calculate _moments of image (get initial guesses for _gaussian and _lorentzian
    function), rot=0.0 assumed"""
    offset = np.nanmin(image)
    total = np.nansum(image)
    Y, X = np.indices(image.shape)  # noqa: N806

    center_y = np.nansum(Y * image) / total
    center_x = np.nansum(X * image) / total
    if center_y > np.max(Y) or center_y < 0:
        center_y = np.max(Y) / 2
    if center_x > np.max(X) or center_x < 0:
        center_x = np.max(X) / 2

    col = image[int(center_y), :]
    row = image[:, int(center_x)]
    width_x = np.nansum(np.sqrt(abs((np.arange(col.size) - center_y) ** 2 * col)) / np.nansum(col))
    width_y = np.nansum(np.sqrt(abs((np.arange(row.size) - center_x) ** 2 * row)) / np.nansum(row))
    height = np.nanmax(image)
    return height, center_y, center_x, width_y, width_x, 0.0, offset


def _residual_gaussian(
    p: list | tuple | npt.NDArray,
    y: npt.NDArray,
    x: npt.NDArray,
    data: npt.NDArray,
) -> npt.NDArray:
    """Residual of data with a _gaussian model."""
    return _gaussian(p, y, x) - data


def _gaussian_background(image: npt.NDArray) -> npt.NDArray:
    """Background defined by a Gaussian function."""
    params = _moments(image)
    Y, X = np.indices(image.shape)  # noqa: N806
    good_vals = np.logical_and(~np.isnan(image), ~image.mask)
    y = Y[good_vals]
    x = X[good_vals]
    data = image[good_vals]
    p = least_squares(
        _residual_gaussian,
        params,
        method="trf",
        bounds=(0, np.inf),
        args=(y, x, data),
    ).x
    return _gaussian(p, Y, X)


def _lorentzian(p: list | tuple | npt.NDArray, y: npt.NDArray, x: npt.NDArray) -> npt.NDArray:
    height, center_y, center_x, width_y, width_x, rot, offset = p
    xp = (x - center_x) * np.cos(rot) - (y - center_y) * np.sin(rot)
    yp = (x - center_x) * np.sin(rot) + (y - center_y) * np.cos(rot)

    R = (xp / width_x) ** 2 + (yp / width_y) ** 2

    return offset + height * (1 / (1 + R))


def _residual_lorentzian(
    p: list | tuple | npt.NDArray,
    y: npt.NDArray,
    x: npt.NDArray,
    data: npt.NDArray,
) -> npt.NDArray:
    return _lorentzian(p, y, x) - data


def _lorentzian_background(
    image: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    params = _moments(image)
    Y, X = np.indices(image.shape)  # noqa: N806
    good_vals = np.logical_and(~np.isnan(image), ~image.mask)
    y = Y[good_vals]
    x = X[good_vals]
    data = image[good_vals]
    p = least_squares(
        _residual_lorentzian,
        params,
        bounds=(0, np.inf),
        method="trf",
        args=(y, x, data),
    ).x

    return p, _lorentzian(p, Y, X)


# ============================================================================


def _interpolated_background(
    image: npt.NDArray,
    interp_method: str,
    polygons: list["dukit.polygon.Polygon"],
    sigma: float,
) -> npt.NDArray:
    """Background defined by the dataset smoothed via a sigma-_gaussian filtering,
    and method-interpolation over masked (polygon) regions.

    method available: nearest, linear, cubic.
    """
    if not isinstance(polygons, list) or not isinstance(polygons[0], dukit.polygon.Polygon):
        raise TypeError("polygons arg was not a list of Polygon objects")

    ylen, xlen = image.shape
    isnt_poly = np.full(image.shape, True)  # all masked to start with

    # coordinate grid for all coordinates
    grid_y, grid_x = np.meshgrid(range(ylen), range(xlen), indexing="ij")

    for p in polygons:
        in_or_out = p.is_inside(grid_y, grid_x)
        # mask all vals that are not background
        is_this_poly = np.ma.masked_greater_equal(in_or_out, 0).mask  # >= 0 => inside/on poly
        isnt_poly = np.logical_and(isnt_poly, ~is_this_poly)  # prev isnt_poly and isnt this poly

    # now we want to send all of the values in indexes that is_bg is True to griddata
    pts = []
    vals = []
    for i in range(ylen):
        for j in range(xlen):
            if isnt_poly[i, j]:
                pts.append([i, j])
                vals.append(image[i, j])

    bg_interp = griddata(pts, vals, (grid_y, grid_x), method=interp_method)

    return get_im_filtered(bg_interp, "gaussian", sigma=sigma)


# ============================================================================


def _filtered_background(image: npt.NDArray, filter_type: str, **kwargs) -> npt.NDArray:
    """Background defined by a filter_type-filtering of the image.
    Passed to `qdmpy.shared.itool.get_background"""
    return get_im_filtered(image, filter_type, **kwargs)


# ============================================================================


def _gaussian_then_poly(image: npt.NDArray, order: int) -> npt.NDArray:
    """Background defines as: fit of a gaussian then a polynomial to image."""
    gauss_back = _gaussian_background(image)
    return _poly_background(image - gauss_back, order) + gauss_back


# ============================================================================


def plot_image(
    image_data: npt.NDArray,
    title: str = "title",
    c_map: str = "viridis",
    c_range: tuple[float, float] | tuple[None, None] = (None, None),
    c_label: str = "label",
    opath: str = "",
    show_scalebar: bool = True,
    raw_pixel_size: float = float("nan"),
    applied_binning: tuple[int, int] | int = 0,
    annotate_polygons: bool = False,
    polygon_nodes: list | None = None,
    show_tick_marks: bool = False,
    polygon_patch_params: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots an image given by image_data. Only saves figure if path given.

    Arguments
    ---------
    image_data : np array, 3D
        Data that is plot.
    title : str
        Title of figure, as well as name for save files
    c_map : str
        Colormap object used to map image_data values to a color.
    c_range : 2-tuple of floats
        Range of values in image_data to map to colors
    c_label : str
        Label for colormap axis
    opath: str  = "",
        If given saves figure here.
    show_scalebar: bool = True
        Show the scalebar(s)?
    raw_pixel_size: float = float("nan")
        Pixel size from hardware
    applied_binning: tuple[int, int] | int = 0,
        Any additional binning that HAS been applied
    annotate_polygons: bool = False,
        Annotate the polygons on image?
    polygon_nodes: list | None = None,
        List of polygon nodes, see TODO for format
    show_tick_marks: bool = False
        Show tick marks on axes
    polygon_patch_params: dict | None, default = None
        Passed to mpl.patches.Polygon.

    Returns
    -------
    fig : matplotlib Figure object
    ax : matplotlib Axis object
    """

    fig, ax = plt.subplots()

    fig, ax = plot_image_on_ax(
        fig,
        ax,
        image_data,
        title,
        c_map,
        c_range,
        c_label,
        show_scalebar=show_scalebar,
        raw_pixel_size=raw_pixel_size,
        applied_binning=applied_binning,
        annotate_polygons=annotate_polygons,
        polygon_nodes=polygon_nodes,
        show_tick_marks=show_tick_marks,
        polygon_patch_params=polygon_patch_params,
    )
    if opath:
        fig.savefig(opath)

    return fig, ax


# ============================================================================


def plot_image_on_ax(
    fig: plt.Figure,
    ax: plt.Axes,
    image_data: npt.NDArray,
    title: str = "title",
    c_map: str = "viridis",
    c_range: tuple[float, float] | tuple[None, None] = (None, None),
    c_label: str = "label",
    show_scalebar: bool = True,
    raw_pixel_size: float = float("nan"),
    applied_binning: tuple[int, int] | int = 0,
    annotate_polygons: bool = False,
    polygon_nodes: list | None = None,
    show_tick_marks: bool = False,
    polygon_patch_params: dict | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots an image given by image_data onto given figure and ax.

    Does not save any data.

    Arguments
    ---------
    fig : matplotlib Figure object
    ax : matplotlib Axis object
    image_data : np array, 3D
        Data that is plot.
    title : str
        Title of figure, as well as name for save files
    c_map : str
        Colormap object used to map image_data values to a color.
    c_range : 2-tuple of floats
        Range of values in image_data to map to colors
    c_label : str
        Label for colormap axis
    show_scalebar: bool = True
        Show the scalebar(s)?
    raw_pixel_size: float = float("nan")
        Pixel size from hardware
    applied_binning: tuple[int, int] | int = 0,
        Any additional binning that HAS been applied
    annotate_polygons: bool = False,
        Annotate the polygons on image?
    polygon_nodes: list | None = None,
        List of polygon nodes, see TODO for format
    show_tick_marks: bool = False
        Show tick marks on axes
    polygon_patch_params: dict | None, default = None
        Passed to mpl.patches.Polygon.

    Returns
    -------
    fig : matplotlib Figure object
    ax : matplotlib Axis object
    """

    imshowed = ax.imshow(image_data, cmap=c_map, vmin=c_range[0], vmax=c_range[1])

    ax.set_title(title)

    cbar = _add_colorbar(imshowed, fig, ax)
    cbar.ax.set_ylabel(c_label, rotation=270)

    if show_scalebar and raw_pixel_size:
        if isinstance(applied_binning, tuple):
            bin_x, bin_y = applied_binning
            pixel_x = raw_pixel_size * bin_x if bin_x else raw_pixel_size
            pixel_y = raw_pixel_size * bin_y if bin_y else raw_pixel_size
            scalebar_x = ScaleBar(pixel_x)
            scalebar_y = ScaleBar(pixel_y, rotation="vertical")
            ax.add_artist(scalebar_x)
            ax.add_artist(scalebar_y)
        else:
            pixel = raw_pixel_size * applied_binning if applied_binning else raw_pixel_size
            scalebar = ScaleBar(pixel)
            ax.add_artist(scalebar)

    if polygon_nodes and annotate_polygons:
        if polygon_patch_params is None:
            polygon_patch_params = {
                "facecolor": None,
                "edgecolor": "k",
                "linestyle": "dashed",
                "fill": False,
            }
        for poly in polygon_nodes:
            # polygons reversed to (x,y) indexing for patch
            ax.add_patch(
                matplotlib.patches.Polygon(
                    np.dstack((poly[:, 1], poly[:, 0]))[0],
                    **polygon_patch_params,
                )
            )

    if not show_tick_marks:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

    return fig, ax


# ============================================================================


def _add_colorbar(
    imshowed: mpl.cm.ScalarMappable,
    fig: plt.Figure,
    ax: plt.Axes,
    aspect: float = 20.0,
    pad_fraction: float = 1.0,
    **kwargs,
) -> matplotlib.colorbar.Colorbar:
    """
    Adds a colorbar to matplotlib axis

    Arguments
    ---------
    imshowed : image as returned by ax.imshow
    fig : matplotlib Figure object
    ax : matplotlib Axis object

    Returns
    -------
    cbar : matplotlib colorbar object

    Optional Arguments
    ------------------
    aspect : int
        Reciprocal of aspect ratio passed to new colorbar axis width. Default: 20.
    pad_fraction : int
        Fraction of new colorbar axis width to pad from image. Default: 1.

    **kwargs : other keyword arguments
        Passed to fig.colorbar.

    """
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1.0 / aspect)
    pad = axes_size.Fraction(pad_fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    cbar = fig.colorbar(imshowed, cax=cax, **kwargs)
    tick_locator = mpl.ticker.MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.linewidth = 0.5  # type: ignore
    return cbar


# ============================================================================


def get_colormap_range(
    c_range_type: str,
    c_range_vals: tuple[float, ...],
    image: npt.NDArray,
    auto_sym_zero: bool = True,
) -> tuple[float, float]:
    """
    Produce a colormap range to plot image from, using the options in c_range_dict.

    Arguments
    ---------
    c_range_type : str
         - "min_max" : map between minimum and maximum values in image.
         - "deviation_from_mean" : requires c_range_vals be a 1-tuple float
           between 0 and 1 'dev'. Maps between (1 - dev) * mean and (1 + dev) * mean.
         - "min_max_symmetric_about_mean" : map symmetrically about mean, capturing all
            values in image (default).
         - "min_max_symmetric_about_zero" : map symmetrically about zero, capturing all
            values in image.
         - "percentile" : requires c_range_vals be a 2-tuple w vals between 0 and 100.
            Maps the range between those percentiles of the data.
         - "percentile_symmetric_about_zero" : requires c_range_vals be a 2-tuple
            w vals between 0 and 100. Maps symmetrically about zero, capturing
            all values between those percentiles in the data (plus perhaps a bit more to
            ensure symmety)
         - "strict_range" : requiresc_range_vals 2-tuple.
            Maps colors directly between the values given.
         - "mean_plus_minus" : mean plus or minus this value.
            c_range_vals must be a 1-tuple float
    c_range_vals : tuple
        See above for allowed options
    auto_sym_zero : bool, default=True
        Try and make symmetric around zero, if logical?

    image : array-like
        2D array (image) that fn returns colormap range for.

    Returns
    -------
    c_range : tuple length 2
        i.e. [min value to map to a color, max value to map to a color]
    """

    # mostly these are just checking that the input values are valid

    warning_messages = {
        "deviation_from_mean": """Invalid c_range_dict['values'] encountered.
        For c_range type 'deviation_from_mean', c_range_dict['values'] must be a float,
        between 0 and 1. Changing to 'min_max_symmetric_about_mean' c_range.""",
        "strict_range": """Invalid c_range_dict['values'] encountered.
        For c_range type 'strict_range', c_range_dict['values'] must be a 2-tup,
        with elements that are floats or ints.
        Changing to 'min_max_symmetric_about_mean' c_range.""",
        "mean_plus_minus": """Invalid c_range_dict['values'] encountered.
        For c_range type 'mean_plus_minus', c_range_dict['values'] must be an int or
        float. Changing to 'min_max_symmetric_about_mean' c_range.""",
        "percentile": """Invalid c_range_dict['values'] encountered.
        For c_range type 'percentile', c_range_dict['values'] must be a 2-tup,
         with elements (preferably ints) in [0, 100].
         Changing to 'min_max_symmetric_about_mean' c_range.""",
        "percentile_symmetric_about_zero": """Invalid c_range_dict['values'] encountered.
        For c_range type 'percentile', c_range_dict['values'] must be a 2-tup,
         with elements (preferably ints) in [0, 100].
         Changing to 'min_max_symmetric_about_mean' c_range.""",
    }
    if not c_range_type:
        c_range_type = "min_max_symmetric_about_mean"
        c_range_vals = ()

    if c_range_type not in ["percentile", "min_max"]:
        auto_sym_zero = False

    if auto_sym_zero and np.any(image < 0) and np.any(image > 0):
        if c_range_type == "min_max":
            c_range_type = "min_max_symmetric_about_zero"
        elif c_range_type == "percentile":
            c_range_type = "percentile_symmetric_about_zero"

    range_calculator_dict = {
        "min_max": _min_max,
        "deviation_from_mean": _deviation_from_mean,
        "min_max_symmetric_about_mean": _min_max_sym_mean,
        "min_max_symmetric_about_zero": _min_max_sym_zero,
        "percentile": _percentile,
        "percentile_symmetric_about_zero": _percentile_sym_zero,
        "strict_range": _strict_range,
        "mean_plus_minus": _mean_plus_minus,
    }

    if c_range_type == "strict_range":
        if (
            not isinstance(c_range_vals, (list, tuple))
            or len(c_range_vals) != 2  # noqa: W503
            or (not isinstance(c_range_vals[0], (float, int)))  # noqa: W503
            or (not isinstance(c_range_vals[1], (float, int)))  # noqa: W503
            or c_range_vals[0] > c_range_vals[1]  # noqa: W503
        ):
            warn(warning_messages[c_range_type])
            return _min_max_sym_mean(image, ())
    elif c_range_type == "mean_plus_minus":
        if not isinstance(c_range_vals[0], (float, int)):
            warn(warning_messages[c_range_type])
            return _min_max_sym_mean(image, ())
    elif c_range_type == "deviation_from_mean":
        if (
            not isinstance(c_range_vals, (float, int))
            or c_range_vals < 0  # noqa: W503
            or c_range_vals > 1  # noqa: W503
        ):
            warn(warning_messages[c_range_type])
            return _min_max_sym_mean(image, ())

    elif c_range_type.startswith("percentile"):
        if (
            not isinstance(c_range_vals, (list, tuple))
            or len(c_range_vals) != 2  # noqa: W503
            or not isinstance(c_range_vals[0], (float, int))
            or not isinstance(c_range_vals[1], (float, int))
            or not 100 >= c_range_vals[0] >= 0
            or not 100 >= c_range_vals[1] >= 0
        ):
            warn(warning_messages[c_range_type])
            return _min_max_sym_mean(image, ())
    return range_calculator_dict[c_range_type](image, c_range_vals)


# ============================


def _min_max(image: npt.NDArray, c_range_values: tuple[float, ...]) -> tuple[float, float]:
    """
    Map between minimum and maximum values in image

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `dukit.plot.get_colormap_range`
    """
    return np.nanmin(image), np.nanmax(image)


def _strict_range(image: npt.NDArray, c_range_values: tuple[float, ...]) -> tuple[float, float]:
    """
    Map between c_range_values

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `dukit.plot.get_colormap_range`
    """
    return c_range_values[0], c_range_values[1]


def _min_max_sym_mean(image: npt.NDArray, c_range_values: tuple[float, ...]) -> tuple[float, float]:
    """
    Map symmetrically about mean, capturing all values in image.

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `dukit.plot.get_colormap_range`
    """
    minimum = np.nanmin(image)
    maximum = np.nanmax(image)
    mean = np.mean(image)
    max_distance_from_mean = np.max([abs(maximum - mean), abs(minimum - mean)])
    return mean - max_distance_from_mean, mean + max_distance_from_mean


def _min_max_sym_zero(image: npt.NDArray, c_range_values: tuple[float, ...]) -> tuple[float, float]:
    """
    Map symmetrically about zero, capturing all values in image.

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `dukit.plot.get_colormap_range`
    """
    min_abs = np.abs(np.nanmin(image))
    max_abs = np.abs(np.nanmax(image))
    larger = np.nanmax([min_abs, max_abs])
    return -larger, larger


def _deviation_from_mean(
    image: npt.NDArray, c_range_values: tuple[float, ...]
) -> tuple[float, float]:
    """
    Map a (decimal) deviation from mean,
    i.e. between (1 - dev) * mean and (1 + dev) * mean

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `dukit.plot.get_colormap_range`
    """
    return (1 - c_range_values[0]) * np.mean(image), (1 + c_range_values[0]) * np.mean(image)


def _percentile(image: npt.NDArray, c_range_values: tuple[float, ...]) -> tuple[float, float]:
    """
    Maps the range between two percentiles of the data.

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `dukit.plot.get_colormap_range`
    """
    return tuple(np.nanpercentile(image, c_range_values))


def _percentile_sym_zero(
    image: npt.NDArray, c_range_values: tuple[float, ...]
) -> tuple[float, float]:
    """
    Maps the range between two percentiles of the data, but ensuring symmetry about zero

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `dukit.plot.get_colormap_range`
    """
    plow, phigh = np.nanpercentile(image, c_range_values)  # e.g. [10, 90]
    val = max(abs(plow), abs(phigh))
    return -val, val


def _mean_plus_minus(image: npt.NDArray, c_range_values: tuple[float, ...]) -> tuple[float, float]:
    """
    Maps the range to mean +- value given in c_range_values

    Arguments
    ---------
    image : np array, 3D
        image data being shown as ax.imshow
    c_range_values : tuple[float, ...]
        See `dukit.plot.get_colormap_range`
    """
    mean = np.mean(image)
    return mean - c_range_values, mean + c_range_values


# ============================================================================


def crop_roi(seq: npt.ArrayLike, roi_coords: tuple[int, int, int, int]) -> npt.NDArray:
    """

    Parameters
    ----------
    seq : array-like
        Image or image-stack you want to crop
    roi_coords: 4-tuple
        start_x: int, start_y: int, end_x: int, end_y: int
        If any are -1, then it sets to the edge of image.

    Returns
    -------
    seq_cropped : np.ndarray
        Image or image-stack cropped.
    """
    seq = np.asarray(seq)
    roi = _define_roi(seq, roi_coords)
    if len(np.shape(seq)) == 2:
        return seq[roi[0], roi[1]].copy()
    return seq[roi[0], roi[1], :]


def _define_roi(
    img: npt.ArrayLike, roi_coords: tuple[int, int, int, int]
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Returns
    -------
    roi : 2-tuple of ndarrays
        Meshgrid that can be used to index into arrays,
        e.g. sig[roi[0], roi[1], sweep_param]
    """
    try:
        size_h, size_w = np.shape(img)[:-1]
    except ValueError:  # not enough values to unpack -> 2d image not 3d
        size_h, size_w = np.shape(img)
    start_x, start_y, end_x, end_y = _check_start_end_rectangle(*roi_coords, size_w, size_h)
    return _define_area_roi(start_x, start_y, end_x, end_y)


# ============================================================================


def _define_area_roi(start_x: int, start_y: int, end_x: int, end_y: int) -> tuple[slice, slice]:
    """
    Returns
    -------
    roi : 2-tuple of ndarrays
        Meshgrid that can be used to index into arrays,
        e.g. sig[sweep_param, roi[0], roi[1]]
    """
    # x: npt.NDArray = np.linspace(start_x, end_x, end_x - start_x + 1, dtype=int)
    # y: npt.NDArray = np.linspace(start_y, end_y, end_y - start_y + 1, dtype=int)
    # xv, yv = np.meshgrid(x, y)
    # return yv, xv
    return slice(start_y, end_y + 1), slice(start_x, end_x + 1)


# ============================================================================


def _check_start_end_rectangle(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    full_size_w: int,
    full_size_h: int,
) -> tuple[int, int, int, int]:
    """Restricts roi params to be within image dimensions."""
    start_x = max(start_x, 0)
    start_y = max(start_y, 0)
    if end_x < 0:
        end_x = full_size_w - 1
    if end_y < 0:
        end_y = full_size_h - 1
    if start_x >= end_x:
        warn(f"Rectangle ends [{end_x}] before it starts [{start_x}] (in x), " + "swapping them")
        start_x, end_x = end_x, start_x
    if start_y >= end_y:
        warn(f"Rectangle ends [{end_y}] before it starts [{start_y}] (in y), " + "swapping them")
        start_y, end_y = end_y, start_y
    if start_x >= full_size_w:
        warn(
            f"Rectangle starts [{start_x}] outside image [{full_size_w}] "
            + "(too large in x), setting to zero."
        )
        start_x = 0

    if start_y >= full_size_h:
        warn(
            f"Rectangle starts outside [{start_y}] image [{full_size_h}] "
            + "(too large in y), setting to zero."
        )
        start_y = 0

    if end_x >= full_size_w:
        warn(f"Rectangle too big in x [{end_x}], cropping to image [{full_size_w}].\n")
        end_x = full_size_w - 1
    if end_y >= full_size_h:
        warn(f"Rectangle too big in y [{end_y}], cropping to image [{full_size_h}].\n")
        end_y = full_size_h - 1

    return start_x, start_y, end_x, end_y


# ============================================================================


def crop_sweep(
    sweep_arr: npt.NDArray,
    sig: npt.NDArray,
    ref: npt.NDArray,
    sig_norm: npt.NDArray,
    rem_start=1,
    rem_end=0,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Crop spectral dimension. Usually used to remove first one/few points of e.g. ODMR.

    Parameters
    ----------
    sweep_arr, sig, ref, sig_norm : npdarray's
        ndarrays to be cropped in spectral dimension. (sweep_arr only has spec dim).
    rem_start, rem_end : int
        How many pts to remove from start & end of spectral dimension.

    Returns
    -------
    sweep_arr, sig, ref, sig_norm : npdarray's
        All as input, put with spectral dimension cropped to arr[rem_start:-rem_end]
    """
    end = -rem_end if rem_end > 0 else None

    if rem_start < 0:
        warn("rem_start must be >=0, setting to zero now.")
        rem_start = 0
    if rem_end < 0:
        warn("rem_end must be >=0, setting to zero now.")
        rem_end = 0

    return (
        sweep_arr[rem_start:end].copy(),
        sig[:, :, rem_start:end].copy(),
        ref[:, :, rem_start:end].copy(),
        sig_norm[:, :, rem_start:end].copy(),
    )


# ============================================================================


def smooth_image_stack(
    stack: npt.NDArray,
    sigma: tuple[float, float] | float,
    truncate: float = 4.0,
) -> npt.NDArray:
    """
    Smooth image stack in spatial dimensions with gaussian.

    Parameters
    ----------
    stack : ndarray
    sigma : 2-tuple of floats or float
        If float, then symmetric smoothing in each dim, otherwise tuple(x, y)
    truncate : float, default=4.0
        Truncate the filter at this many standard deviations.
        See scipy.ndimage.gaussian_filter.

    Returns
    -------
    smooth_stack : ndarray
        Image stack smoothed in spatial dimensions.
    """
    if isinstance(sigma, (list, tuple)):
        return gaussian_filter(stack, sigma=(sigma[1], sigma[0], 0), truncate=truncate)
    return gaussian_filter(stack, sigma=(sigma, sigma, 0), truncate=truncate)


# ============================================================================


def rebin_image_stack(stack: npt.NDArray, additional_bins: tuple[int, int] | int) -> npt.NDArray:
    """
    Rebin image stack in spatial dimensions.

    Parameters
    ----------
    stack : ndarray
    additional_bins : 2-tuple of ints, or int
        Binning in-plane.  Make it a power of 2.
        Binning in x then y if 2-tuple, else symmetric.

    Returns
    -------
    smooth_stack : ndarray
        Image stack smoothed in spatial dimensions.
    """
    if not additional_bins:
        return stack
    if isinstance(additional_bins, (tuple, list)):
        return dukit.rebin.rebin(
            stack,
            factor=(additional_bins[1], additional_bins[0], 1),
            func=np.mean,
        )
    return dukit.rebin.rebin(
        stack,
        factor=(additional_bins, additional_bins, 1),
        func=np.mean,
    )


# ============================================================================
def sum_spatially(seq: npt.ArrayLike) -> npt.NDArray:
    """Sum over 0th (spectral) dim of seq if 3D, else return as is."""
    seq = np.asarray(seq)
    if len(np.shape(seq)) == 3:
        return np.sum(seq, axis=-1)
    return seq


# ============================================================================


def get_aois(
    image_shape: tuple[int, int, int] | tuple[int, int],
    *aoi_coords: tuple[int, int, int, int],
) -> tuple[tuple[slice, slice]]:
    aois: list = [] if not aoi_coords else [_define_area_roi(*aoi) for aoi in aoi_coords]

    if len(image_shape) == 3:
        shp = image_shape[:-1]
    else:
        shp = image_shape
    aois.insert(
        0,
        _define_area_roi(shp[0] // 2, shp[1] // 2, shp[0] // 2 + 1, shp[1] // 2 + 1),
    )
    return tuple(aois)


# ============================================================================


def _iterslice(x: npt.NDArray, axis: int = 0):
    """Iterate through array x in slices along axis 'axis' (defaults to 0).
    E.g. iterslice(shape(y,x,freqs), axis=-1) will give iter. of 1d freq slices."""
    sub = [range(s) for s in x.shape]
    sub[axis] = (slice(None),)
    for p in product(*sub):
        yield x[p]


# ============================================================================


def _iterframe(x_3d: npt.NDArray):
    """iterframe(shape(y,x,freqs)) will give iter. of 2d y,x frames."""
    for frame in range(x_3d.shape[2]):
        yield x_3d[:, :, frame]
