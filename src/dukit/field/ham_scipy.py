# -*- coding: utf-8 -*-
"""
Scipy least_squares backend for hamiltonian fitting.

Functions
---------
 - `dukit.field.ham_scipy.gen_ham_scipyfit_init_guesses`
 - `dukit.field.ham_scipy.prep_ham_scipyfit_options`
 - `dukit.field.ham_scipy.fit_hamiltonian_scipyfit`
 - `dukit.field.ham_scipy.fit_hamiltonian_roi_avg_scipyfit`
 - `dukit.field.ham_scipy.ham_to_squares_wrapper`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.field.ham_scipy.gen_ham_scipyfit_init_guesses": True,
    "dukit.field.ham_scipy.prep_ham_scipyfit_options": True,
    "dukit.field.ham_scipy.fit_hamiltonian_scipyfit": True,
    "dukit.field.ham_scipy.fit_hamiltonian_roi_avg_scipyfit": True,
    "dukit.field.ham_scipy.ham_to_squares_wrapper": True,
}
# ============================================================================

import numpy as np
from scipy.optimize import least_squares
from tqdm.autonotebook import tqdm  # auto detects jupyter
import warnings
from timeit import default_timer as timer
from datetime import timedelta
from joblib import Parallel, delayed

# ============================================================================

from dukit.field import hamiltonian

# ============================================================================


def gen_ham_scipyfit_init_guesses(init_guesses: dict, init_bounds: dict) -> tuple:
    """
    Generate arrays of initial fit guesses and bounds in correct form for scipy least_squares.

    init_guesses and init_bounds are dictionaries up to this point, we now convert to np arrays,
    that scipy will recognise.

    Arguments
    ---------
    init_guesses : dict
        Dict holding guesses for each parameter, e.g. key -> value for the ham.
    init_bounds : dict
        Dict holding bounds for each parameter, e.g. key -> [lower, upper] for the ham.

    Returns
    -------
    fit_param_ar : np array, shape: num_params
        The initial fit parameter guesses.
    fit_param_bound_ar : np array, shape: (num_params, 2)
        Fit parameter bounds.
    """
    param_lst = []
    bound_lst = []

    for key in init_guesses.keys():
        param_lst.append(init_guesses[key])
        bound_lst.append(init_bounds[key])

    fit_param_ar = np.array(param_lst)
    fit_param_bound_ar = np.array(bound_lst)
    return fit_param_ar, fit_param_bound_ar


# ==========================================================================


def prep_ham_scipyfit_options(
    ham,
    init_guesses: dict,
    init_bounds: dict,
    method: str = "trf",
    gtol: float = 1e-12,
    xtol: float = 1e-12,
    ftol: float = 1e-12,
    loss: str = "linear",
    use_analytic_jac: bool = False,
    scale_x: bool = False,
) -> dict:
    """
    Prepare options dict in format that scipy least_squares expects.

    Arguments
    ---------
    ham : `dukit.field.hamiltonian.Hamiltonian`
        Hamiltonian object.
    init_guesses : dict
        Initial parameter guesses.
    init_bounds : dict
        Parameter bounds.
    method : str, default="trf"
        scipy.optimize.least_squares method.
    gtol : float, default=1e-12
        Tolerance for gradient.
    xtol : float, default=1e-12
        Tolerance for parameters.
    ftol : float, default=1e-12
        Tolerance for cost function.
    loss : str, default="linear"
        Loss function for robust fitting.
    use_analytic_jac : bool, default=False
        Use analytic jacobian if available.
    scale_x : bool, default=False
        Scale parameters by jacobian.

    Returns
    -------
    scipy_fit_options : dict
        Dictionary with options that scipy.optimize.least_squares expects.
    """
    _, fit_param_bound_ar = gen_ham_scipyfit_init_guesses(init_guesses, init_bounds)
    fit_bounds = (fit_param_bound_ar[:, 0], fit_param_bound_ar[:, 1])

    # see scipy.optimize.least_squares
    scipyfit_options = {
        "method": method,
        "gtol": gtol,
        "xtol": xtol,
        "ftol": ftol,
        "loss": loss,
    }

    if method != "lm":
        scipyfit_options["bounds"] = fit_bounds

    if scale_x:
        scipyfit_options["x_scale"] = "jac"

    # define jacobian option for least_squares fitting
    if not ham.jacobian_defined() or not use_analytic_jac:
        scipyfit_options["jac"] = "2-point"
    else:
        scipyfit_options["jac"] = ham.jacobian_scipyfit

    return scipyfit_options


# ==========================================================================


def _fit_single_pixel(
    residuals_fn,
    p0,
    pixel_data,
    fit_options,
):
    """Fit a single pixel (helper for parallelization)."""
    y, x, data = pixel_data
    try:
        res = least_squares(residuals_fn, p0, args=(data,), **fit_options)
        return ((y, x), res.x, res.jac)
    except Exception:
        # Return NaN for failed fits
        nan_params = np.full(len(p0), np.nan)
        nan_jac = np.full((len(data), len(p0)), np.nan)
        return ((y, x), nan_params, nan_jac)


# ==========================================================================


def fit_hamiltonian_scipyfit(
    data: np.ndarray,
    hamiltonian_obj,
    init_guesses: dict,
    init_bounds: dict,
    freq_mask: tuple[bool, ...] | None = None,
    n_jobs: int = -2,
    method: str = "trf",
    gtol: float = 1e-12,
    xtol: float = 1e-12,
    ftol: float = 1e-12,
    loss: str = "linear",
    use_analytic_jac: bool = False,
    scale_x: bool = False,
    shuffle_pixels: bool = True,
    progress_bar: bool = True,
    joblib_verbosity: int = 0,
) -> tuple[dict, dict]:
    """
    Fits each pixel ODMR result to hamiltonian and returns dictionary of
    param_name -> param_image.

    Uses joblib for parallelization (matches dukit pixel fitting style).

    Arguments
    ---------
    data : np array, 3D
        Normalised measurement array, shape: [idx, y, x]. E.g. b_defects or freqs
    hamiltonian_obj : `dukit.field.hamiltonian.Hamiltonian`
        Model we're fitting to.
    init_guesses : dict
        Initial parameter guesses.
    init_bounds : dict
        Parameter bounds.
    freq_mask : tuple[bool, ...] | None, default=None
        Boolean mask for which frequencies to use.
    n_jobs : int, default=-2
        Number of parallel jobs. -1 uses all, -2 uses all but one.
    method : str, default="trf"
        scipy.optimize.least_squares method.
    gtol : float, default=1e-12
        Tolerance for gradient.
    xtol : float, default=1e-12
        Tolerance for parameters.
    ftol : float, default=1e-12
        Tolerance for cost function.
    loss : str, default="linear"
        Loss function for robust fitting.
    use_analytic_jac : bool, default=False
        Use analytic jacobian if available.
    scale_x : bool, default=False
        Scale parameters by jacobian.
    shuffle_pixels : bool, default=True
        Randomize pixel order for better ETA estimation.
    progress_bar : bool, default=True
        Show progress bar.
    joblib_verbosity : int, default=0
        Verbosity level for joblib.

    Returns
    -------
    ham_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    sigmas: dict
        As ham_results, but containing standard deviations for each parameter across FOV.
    """
    num_pixels = np.shape(data)[1] * np.shape(data)[2]

    # randomize order of fitting pixels (will un-scramble later) so ETA is more correct
    if shuffle_pixels:
        pixel_data, unshuffler = hamiltonian.ham_shuffle_pixels(data)
    else:
        pixel_data = data
        unshuffler = None

    # Get fit options
    fit_options = prep_ham_scipyfit_options(
        hamiltonian_obj,
        init_guesses,
        init_bounds,
        method=method,
        gtol=gtol,
        xtol=xtol,
        ftol=ftol,
        loss=loss,
        use_analytic_jac=use_analytic_jac,
        scale_x=scale_x,
    )

    # Convert dict guesses to array
    init_param_ar, _ = gen_ham_scipyfit_init_guesses(init_guesses, init_bounds)

    # call into the library (measure time)
    t0 = timer()

    # Use joblib for parallelization (like dukit pixel fitting)
    results = Parallel(n_jobs=n_jobs, verbose=joblib_verbosity)(
        delayed(_fit_single_pixel)(
            hamiltonian_obj.residuals_scipyfit,
            init_param_ar,
            pixel_info,
            fit_options,
        )
        for pixel_info in tqdm(
            hamiltonian.ham_pixel_generator(pixel_data),
            desc="hamiltonian fitting",
            ascii=True,
            mininterval=1,
            total=num_pixels,
            unit=" PX",
            disable=(not progress_bar),
        )
    )

    t1 = timer()
    fit_time = timedelta(seconds=t1 - t0).total_seconds()

    res, sigmas = hamiltonian.ham_get_pixel_fitting_results(
        hamiltonian_obj, results, pixel_data
    )

    if shuffle_pixels and unshuffler is not None:
        res = hamiltonian.ham_unshuffle_fit_results(res, unshuffler)
        sigmas = hamiltonian.ham_unshuffle_fit_results(sigmas, unshuffler)

    # Add metadata
    res["_metadata"] = {
        "fit_time_s": fit_time,
        "method": method,
        "n_pixels": num_pixels,
    }
    sigmas["_metadata"] = res["_metadata"].copy()

    return res, sigmas


# ==========================================================================


def fit_hamiltonian_roi_avg_scipyfit(
    data: np.ndarray,
    hamiltonian_obj,
    init_guesses: dict,
    init_bounds: dict,
    method: str = "trf",
    gtol: float = 1e-12,
    xtol: float = 1e-12,
    ftol: float = 1e-12,
    loss: str = "linear",
    use_analytic_jac: bool = False,
    scale_x: bool = False,
) -> tuple[np.ndarray, dict]:
    """
    Fits ROI average to hamiltonian.

    Arguments
    ---------
    data : np array, 3D
        Normalised measurement array, shape: [idx, y, x]. E.g. b_defects or freqs
    hamiltonian_obj : `dukit.field.hamiltonian.Hamiltonian`
        Model we're fitting to.
    init_guesses : dict
        Initial parameter guesses.
    init_bounds : dict
        Parameter bounds.
    method : str, default="trf"
        scipy.optimize.least_squares method.
    gtol : float, default=1e-12
        Tolerance for gradient.
    xtol : float, default=1e-12
        Tolerance for parameters.
    ftol : float, default=1e-12
        Tolerance for cost function.
    loss : str, default="linear"
        Loss function for robust fitting.
    use_analytic_jac : bool, default=False
        Use analytic jacobian if available.
    scale_x : bool, default=False
        Scale parameters by jacobian.

    Returns
    -------
    best_params : array
        Array of best parameters from ROI average.
    fit_options : dict
        Options dictionary for this fit method.
    """

    # average freqs/b_defects over image -> ignore nanmean of empty slice warning (for nan b_defects etc.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        data_roi = np.nanmean(data, axis=(1, 2))

    fit_options = prep_ham_scipyfit_options(
        hamiltonian_obj,
        init_guesses,
        init_bounds,
        method=method,
        gtol=gtol,
        xtol=xtol,
        ftol=ftol,
        loss=loss,
        use_analytic_jac=use_analytic_jac,
        scale_x=scale_x,
    )

    init_param_ar, _ = gen_ham_scipyfit_init_guesses(init_guesses, init_bounds)

    ham_result = least_squares(
        hamiltonian_obj.residuals_scipyfit,
        init_param_ar,
        args=(data_roi,),
        **fit_options,
    )
    best_params = ham_result.x

    return best_params, fit_options


# ==========================================================================


def ham_to_squares_wrapper(fun, p0, shaped_data, fit_optns):
    """
    Simple wrapper of scipy.optimize.least_squares to allow us to keep track of which
    solution is which (or where).

    Arguments
    ---------
    fun : function
        Function object acting as residual
    p0 : np array
        Initial guess: array of parameters
    shaped_data : list (3 elements)
        array returned by `dukit.field.hamiltonian.ham_pixel_generator`: [y, x, data[:, y, x]]
    fit_optns : dict
        Other options (dict) passed to least_squares

    Returns
    -------
    wrapped_squares : tuple
        (y, x), least_squares(...).x, least_squares(...).jac
        I.e. the position of the fit result, the fit result parameters array and the jacobian
        at the solution.
    """
    # shaped_data: [y, x, pl]
    # output: (y, x), result_params
    res = least_squares(fun, p0, args=(shaped_data[2],), **fit_optns)
    return ((shaped_data[0], shaped_data[1]), res.x, res.jac)
