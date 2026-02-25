# -*- coding: utf-8 -*-
"""
This module holds some functions and classes that are shared between different
fitting backends, but are not a part of the user-facing interface.

Functions
---------
 - `dukit.pl.common.gen_init_guesses`
 - `dukit.pl.common.bounds_from_range`
 - `dukit.pl.common.calc_sigmas`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.pl.common.gen_init_guesses": True,
    "dukit.pl.common.bounds_from_range": True,
    "dukit.pl.common.calc_sigmas": True,
}

# ============================================================================

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy.linalg import svd

if TYPE_CHECKING:
    import dukit.pl.model


# ============================================================================

# ============================================================================


def gen_init_guesses(fit_model: "dukit.pl.model.FitModel", guesses_dict: dict, bounds_dict: dict):
    """
    Generate initial guesses (and bounds) in fit parameters from options dictionary.

    Both are returned as dictionaries, you need to use
    'gen_{scipy/gpufit/...}_init_guesses' to convert to the correct (array) format
    for each specific fitting backend.


    Arguments
    ---------
    fit_model : dukit.pl.model.FitModel
        Fit model you've defined already.
    guesses_dict : dict
        Format: key -> list of guesses for each independent version of that fn_type.
        e.g. 'pos': [.., ..] for each pos fn_type.
    bounds_dict : dict
        Format: key -> bounds for that param type (or use _range).
        e.g. 'pos_bounds': [5., 25.]
        or 'pos_range': 5.0
    Returns
    -------
    init_guesses : dict
        Dict holding guesses for each parameter, e.g. key -> list of guesses for each
        independent version of that fn_type.
    init_bounds : dict
        Dict holding guesses for each parameter, e.g. key -> list of bounds for each
        independent version of that fn_type.
    """

    init_guesses = {}
    init_bounds = {}

    for param_key in fit_model.get_param_defn():
        if param_key not in guesses_dict:
            raise KeyError(
                f"Parameter {param_key} of {fit_model.__class__} not in guesses dictionary."
            )
        guess = guesses_dict[param_key]

        if param_key + "_range" in bounds_dict:
            bounds = bounds_from_range(bounds_dict[param_key + "_range"], guess)
        elif param_key + "_bounds" in bounds_dict:
            bounds = bounds_dict[param_key + "_bounds"]
        else:
            raise RuntimeError(
                f"Parameter {param_key}_bounds nor {param_key}_range "
                "in bounds dictionary for {fit_model.__class__}."
            )

        if guess is not None:
            init_guesses[param_key] = guess
            init_bounds[param_key] = np.array(bounds)
        else:
            raise RuntimeError(f"Not sure why your guess for {param_key} is None?")
    return init_guesses, init_bounds


# ============================================================================


def bounds_from_range(rang: float | npt.ArrayLike, guess: float | npt.ArrayLike) -> tuple:
    """
    Generate parameter bounds when given a range option.

    Arguments
    ---------
    rang : float or npt.ArrayLike
        Range for each parameter with name param_key e.g. 'pos_0, pos_1', OR
        a single value, so each parameter has same range.
    guess : float/int or array
        Guess for param, or list of guesses for a given parameter (as for rang)

    Returns
    -------
    bounds : list of lists
        bounds for each parameter. Dimension depends on dimension of param guess.
    """
    if isinstance(guess, (list, tuple, np.ndarray)) and len(list(guess)) > 1:
        # separate bounds for each fn of this type
        if isinstance(rang, (list, tuple, np.ndarray)) and len(list(rang)) > 1:
            bounds = [
                [each_guess - each_range, each_guess + each_range]
                for each_guess, each_range in zip(guess, rang)
            ]
        # separate guess for each fn of this type, all with same range
        else:
            bounds = [
                [
                    each_guess - rang,
                    each_guess + rang,
                ]
                for each_guess in guess
            ]
    else:
        if isinstance(rang, (list, tuple, np.ndarray)) and len(list(rang)) == 1:
            rang = rang[0]
        if isinstance(guess, (list, tuple, np.ndarray)) and len(list(guess)) == 1:
            guess = guess[0]
        bounds = [
            guess - rang,
            guess + rang,
        ]
    return bounds


# ============================================================================


def calc_sigmas(
    fit_model: "dukit.pl.model.FitModel",
    sweep_arr: npt.NDArray,
    pl_vec: npt.NDArray,
    best_params: npt.NDArray,
) -> npt.NDArray:
    """Calculate fit errors (std. dev.) from jacobian.

    Arguments
    ---------
    fit_model : dukit.pl.model.FitModel
        Fit model you've defined already.
    sweep_arr : npt.NDArray, 1D
        Array of sweep values.
    pl_vec : npt.NDArray, 1D
        Array of measured PL values.
    best_params : npt.NDArray, 1D
        Array of best-fit parameters.

    Returns
    -------
    sigmas : npt.NDArray, 1D
        Array of standard deviations for each parameter.
    """
    jac = fit_model.jacobian_scipyfit(best_params, sweep_arr, pl_vec)
    _, s, vt = svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(jac.shape) * s[0]
    s = s[s > threshold]
    vt = vt[: s.size]
    pcov = np.dot(vt.T / s**2, vt)
    # NOTE using/assuming linear cost fn,
    cost = 2 * 0.5 * np.sum(fit_model(best_params, sweep_arr) ** 2)
    resid = fit_model.residuals_scipyfit(best_params, sweep_arr, pl_vec)
    s_sq = cost / (len(resid) - len(best_params))
    pcov *= s_sq
    return np.sqrt(np.diag(pcov))  # array of standard deviations
