# -*- coding: utf-8 -*-
"""
This module holds tools for fitting raw data via scipy. (scipy backend)

Functions
---------
 - `dukit.pl.scipyfit.fit_roi_avg_pl`
 - `dukit.pl.scipyfit.fit_aois_pl`
 - `qdmpy.pl.scipyfit.fit_all_pixels_pl`
 - `dukit.pl.scipyfit._gen_sf_guesses_bounds`
 - `dukit.pl.scipyfit._spfitter`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.pl.scipyfit.fit_roi_avg_pl": True,
    "dukit.pl.scipyfit.fit_aois_pl": True,
    "dukit.pl.scipyfit.fit_all_pixels_pl": True,
    "dukit.pl.scipyfit._gen_sf_guesses_bounds": True,
    "dukit.pl.scipyfit._spfitter": True,
}

# ==========================================================================

import logging
from datetime import timedelta
from timeit import default_timer as timer

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from scipy.optimize import least_squares

import dukit.itool

# ============================================================================
import dukit.pl.common
import dukit.share
import dukit.warn

# ==========================================================================


def fit_roi_avg_pl(
    sig: npt.NDArray,
    ref: npt.NDArray,
    sweep_arr: npt.NDArray,
    fit_model: "dukit.pl.model.FitModel",
    guess_dict: dict,
    bounds_dict: dict,
    norm: str = "div",
    method: str = "trf",
    verbose: int = 0,
    gtol: float = 1e-12,
    xtol: float = 1e-12,
    ftol: float = 1e-12,
    loss: str = "linear",
    jac: str | None = None,
):
    """
    Fit ROI average

    Arguments
    ---------
    sig : np array, 3D
        Sig measurement array, shape: [y, x, sweep_arr]. CROPPED etc. already.
    ref : np array, 3D
        Ref measurement array, shape: [y, x, sweep_arr]. CROPPED etc. already.
    sweep_arr : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : `dukit.pl.model.FitModel`
        The model we're fitting to.
    guess_dict : dict
        dict holding guesses for each parameter type, e.g.
        {'pos': [.., ..], 'amp': [.., ..], ...}
    bounds_dict : dict
        dict holding bound options for each parameter type, e.g.
        {"pos_range": 5.0, "amp_bounds": [0.0, 1.0], ...}
    norm : str default="div"
        Normalisation method

    Optional parameters passed to scipy least_squares
    -------------------------------------------------
    method: str = "trf"
        Fit method, "trf", "dogbox" or "lm"
    verbose: int =0
        Verbosity of fit -> probably want to keep at 0
    gtol: float = 1e-12
        Tolerance for termination by the change of the independent variables.
    xtol: float = 1e-12
        Tolerance for termination by the norm of the gradient.
    ftol: float = 1e-12
        Tolerance for termination by the change of the cost function.
    loss: str = "linear"
        Determines the loss function. This in non trivial check the scipy documentation.
        Best you don't change this.
    jac: str | None = None
        Jacobian. If None uses algebraic jac, otherwise:
        '2-point' is probably best, then '3-point', 'cs' are other options.

    Returns
    -------
    fit_image_results : dukit.share.RoiAvgFit
    """
    if jac is None:
        jac = fit_model.jacobian_scipyfit

    pguess, pbounds = _gen_sf_guesses_bounds(
        fit_model,
        *dukit.pl.common.gen_init_guesses(fit_model, guess_dict, bounds_dict),
    )
    # NOTE do we want to average over spectra per-pixel, or average over counts?
    # I guess we want the local average, so this is ok.

    if norm == "div":
        sig_norm = sig / ref
    elif norm == "sub":
        sig_norm = 1 + (sig - ref) / (sig + ref)
    elif norm == "true_sub":
        sig_norm = (sig - ref) / np.nanmax(sig - ref)

    avg_sig_norm = np.nanmean(sig_norm, axis=(0, 1))
    avg_sig = np.nanmean(sig, axis=(0, 1))
    avg_ref = np.nanmean(ref, axis=(0, 1))
    result = _spfitter(
        fit_model,
        sweep_arr,
        avg_sig_norm,
        pguess,
        bounds=pbounds,
        method=method,
        verbose=verbose,
        gtol=gtol,
        xtol=xtol,
        ftol=ftol,
        loss=loss,
        jac=jac,
    )

    best_params = result[: len(pguess)]
    best_sigmas = result[len(pguess) : -1]
    best_residual = fit_model(best_params, sweep_arr) - avg_sig_norm
    fit_xvec = np.linspace(
        np.min(sweep_arr),
        np.max(sweep_arr),
        10000,
    )
    fit_yvec = fit_model(best_params, fit_xvec)
    fit_yvec_guess = fit_model(pguess, fit_xvec)
    return dukit.share.RoiAvgFit(
        "scipyfit",
        sweep_arr,
        avg_sig_norm,
        avg_sig,
        avg_ref,
        fit_xvec,
        fit_yvec,
        fit_yvec_guess,
        best_params,
        best_sigmas,
        best_residual,
        pguess,
        pbounds,
    )


# ==========================================================================


def fit_aois_pl(
    sig: npt.NDArray,
    ref: npt.NDArray,
    sweep_arr: npt.NDArray,
    fit_model: "dukit.pl.model.FitModel",
    guess_dict: dict,
    bounds_dict: dict,
    *aoi_coords: tuple[int, int, int, int],
    norm: str = "div",
    method: str = "trf",
    verbose=0,
    gtol: float = 1e-12,
    xtol: float = 1e-12,
    ftol: float = 1e-12,
    loss: str = "linear",
    jac: str | None = None,
) -> dict[str, dict[str, dukit.share.AoiAvgFit]]:
    """
    Fit AOI averages

    Arguments
    ---------
    sig : np array, 3D
        Sig measurement array, shape: [y, x, sweep_arr].
    ref : np array, 3D
        Ref measurement array, shape: [y, x, sweep_arr].
    sweep_arr : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : `dukit.pl.model.FitModel`
        The model we're fitting to.
    guess_dict : dict
        dict holding guesses for each parameter type, e.g.
        {'pos': [.., ..], 'amp': [.., ..], ...}
    bounds_dict : dict
        dict holding bound options for each parameter type, e.g.
        {"pos_range": 5.0, "amp_bounds": [0.0, 1.0], ...}
    *aoi_coords : tuple of 4-tuples
        As elsewhere
    norm : str default="div"
        Normalisation method

    Optional parameters passed to scipy least_squares
    -------------------------------------------------
    method: str = "trf"
        Fit method, "trf", "dogbox" or "lm"
    verbose=0
        Verbosity of fit -> probably want to keep at 0
    gtol: float = 1e-12
        Tolerance for termination by the change of the independent variables.
    xtol: float = 1e-12
        Tolerance for termination by the norm of the gradient.
    ftol: float = 1e-12
        Tolerance for termination by the change of the cost function.
    loss: str = "linear"
        Determines the loss function. This in non trivial check the scipy documentation.
        Best you don't change this.
    jac: str | None = None
        Jacobian. If None uses algebraic jac, otherwise:
        '2-point' is probably best, then '3-point', 'cs' are other options.

    Returns
    -------
    fit_image_results : dict
        Format: {"AOI_n": {"scipyfit": AoiAvgFit}, ...}
    """
    if jac is None:
        jac = fit_model.jacobian_scipyfit

    pguess, pbounds = _gen_sf_guesses_bounds(
        fit_model,
        *dukit.pl.common.gen_init_guesses(fit_model, guess_dict, bounds_dict),
    )
    aois = dukit.itool.get_aois(np.shape(sig), *aoi_coords)
    aoi_pl_vecs = []
    sig_ref_signorms = []

    for a in aois:
        s = sig[a[0], a[1], :]
        r = ref[a[0], a[1], :]
        if norm == "div":
            pl_vec = np.nanmean(s / r, axis=(0, 1))
        elif norm == "sub":
            pl_vec = np.nanmean(1 + (s - r) / (s + r), axis=(0, 1))
        elif norm == "true_sub":
            pl_vec = np.nanmean((s - r) / np.nanmax(s - r), axis=(0, 1))
        aoi_pl_vecs.append(pl_vec)
        sig_ref_signorms.append((np.nanmean(s, axis=(0, 1)), np.nanmean(r, axis=(0, 1)), pl_vec))

    results_lst = [
        _spfitter(
            fit_model,
            sweep_arr,
            pl_vec,
            pguess,
            bounds=pbounds,
            method=method,
            verbose=verbose,
            gtol=gtol,
            xtol=xtol,
            ftol=ftol,
            loss=loss,
            jac=jac,
        )
        for pl_vec in aoi_pl_vecs
    ]

    # add the single pixel check on
    output_aoi_coords = list(aoi_coords)
    shp = np.shape(sig)[:-1]
    output_aoi_coords.insert(0, (shp[0] // 2, shp[1] // 2, shp[0] // 2 + 1, shp[1] // 2 + 1))

    ret = dict()
    for n, ((s_avg, r_avg, pl_vec), result) in enumerate(zip(sig_ref_signorms, results_lst)):
        best_params = result[: len(pguess)]
        best_sigmas = result[len(pguess) : -1]
        best_residual = fit_model(best_params, sweep_arr) - pl_vec
        fit_xvec = np.linspace(
            np.min(sweep_arr),
            np.max(sweep_arr),
            10000,
        )
        fit_yvec = fit_model(best_params, fit_xvec)
        ret[f"AOI_{n}"] = {
            "scipyfit": dukit.share.AoiAvgFit(
                n,
                sweep_arr,
                pl_vec,
                s_avg,
                r_avg,
                "scipyfit",
                fit_xvec,
                fit_yvec,
                best_params,
                best_sigmas,
                best_residual,
                pguess,
                pbounds,
                output_aoi_coords[n],
            )
        }

    return ret


# ==========================================================================


def fit_all_pixels_pl(
    sig_norm: npt.NDArray,
    sweep_arr: npt.NDArray,
    fit_model: "dukit.pl.model.FitModel",
    guess_dict: dict,
    bounds_dict: dict,
    roi_avg_result: dukit.share.RoiAvgFit | None = None,
    n_jobs: int = -2,
    joblib_verbosity=5,
    method: str = "trf",
    verbose=0,
    gtol: float = 1e-12,
    xtol: float = 1e-12,
    ftol: float = 1e-12,
    loss: float = "linear",
    jac: str | None = None,
):
    """
    Fits each pixel and returns dictionary of param_name -> param_image.

    Arguments
    ---------
    sig_norm : np array, 3D
        Normalised measurement array, shape: [y, x, sweep_ar].
    sweep_arr : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : `dukit.pl.model.FitModel`
        The model we're fitting to.
    guess_dict : dict
        dict holding guesses for each parameter type, e.g.
        {'pos': [.., ..], 'amp': [.., ..], ...}
    bounds_dict : dict
        dict holding bound options for each parameter type, e.g.
        {"pos_range": 5.0, "amp_bounds": [0.0, 1.0], ...}
    roi_avg_result : dukit.share.RoiAvgFit | None
        The result of fitting the ROI average.
        If done, directly uses guesses provided.
    n_jobs : int, default=-2
        Number of jobs to run concurrently, see joblib docs.
        -2 === leaving one cpu free, etc. for neg numbers.
    joblib_verbosity:int = 5
        How often to update progress bar.

    Optional parameters passed to scipy least_squares
    -------------------------------------------------
    method: str = "trf"
        Fit method, "trf", "dogbox" or "lm"
    verbose=0
        Verbosity of fit -> probably want to keep at 0
    gtol: float = 1e-12
        Tolerance for termination by the change of the independent variables.
    xtol: float = 1e-12
        Tolerance for termination by the norm of the gradient.
    ftol: float = 1e-12
        Tolerance for termination by the change of the cost function.
    loss: float = "linear"
        Determines the loss function. This in non trivial check the scipy documentation.
        Best you don't change this.
    jac: str | None = None
        Jacobian. If None uses algebraic jac, otherwise:
        '2-point' is probably best, then '3-point', 'cs' are other options.

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual_0' as a key.
        Sigmas (stdev on fit error) are given as e.g. pos_0_sigma
    """
    if jac is None:
        jac = fit_model.jacobian_scipyfit

    init_pguess, pbounds = _gen_sf_guesses_bounds(
        fit_model,
        *dukit.pl.common.gen_init_guesses(fit_model, guess_dict, bounds_dict),
    )
    pguess = roi_avg_result.best_params if roi_avg_result is not None else init_pguess

    # call into the library (measure time)
    t0 = timer()
    results_lst = Parallel(n_jobs=n_jobs, verbose=joblib_verbosity)(
        delayed(_spfitter)(
            fit_model,
            sweep_arr,
            pl_vec,
            pguess,
            bounds=pbounds,
            method=method,
            verbose=verbose,
            gtol=gtol,
            xtol=xtol,
            ftol=ftol,
            loss=loss,
            jac=jac,
        )
        for pl_vec in dukit.itool._iterslice(sig_norm, axis=-1)
    )
    t1 = timer()
    dt = timedelta(seconds=t1 - t0).total_seconds()
    logging.info(f"fit time: {dt:.2f}s")

    results_arr = np.array(results_lst).reshape((*sig_norm.shape[:2], 2 * len(pguess) + 1))
    names = list(fit_model.get_param_odict().keys())
    names.extend([f"sigma_{n}" for n in names])
    names.append("residual_0")
    fit_image_results = {
        name: array for name, array in zip(names, dukit.itool._iterframe(results_arr))
    }

    return fit_image_results


# =======================================================================================


def _spfitter(
    fit_model: "dukit.pl.model.FitModel",
    sweep_arr: npt.NDArray,
    pl_vec: npt.NDArray,
    p0: npt.NDArray,
    **fit_optns,
) -> npt.NDArray:
    """Single pixel fitter.
    ~ Fit inputs -> (params; param_std_dev; sum(resid)) @ solution.
    """
    try:
        fitres = least_squares(
            fit_model.residuals_scipyfit,
            p0,
            args=(sweep_arr, pl_vec),
            **fit_optns,
        )
        perr = dukit.pl.common.calc_sigmas(fit_model, sweep_arr, pl_vec, fitres.x)
        resid = fit_model.residuals_scipyfit(fitres.x, sweep_arr, pl_vec)
    except ValueError:
        return np.hstack((np.full_like(p0, np.nan), np.full_like(p0, np.nan), np.nan))
    return np.hstack(
        (
            fitres.x,
            perr,
            np.sum(np.abs(resid, dtype=np.float64), dtype=np.float64),
        )
    )


def _gen_sf_guesses_bounds(
    fit_model: "dukit.pl.model.FitModel", init_guesses: dict, init_bounds: dict
) -> tuple[npt.NDArray, tuple[npt.NDArray, npt.NDArray]]:
    """
    Generate arrays of initial fit guesses and bounds in correct form for scipy
    least_squares.

    init_guesses and init_bounds are dictionaries up to this point, we now convert to
    np arrays, that scipy will recognise. In particular, we specificy that each of the
    'num' of each 'fn_type' have independent parameters, so must have independent
    init_guesses and init_bounds when plugged into scipy.

    Arguments
    ---------
    fit_model : dukit.pl.model.FitModel
        Model definition.
    init_guesses : dict
        Dict holding guesses for each parameter,
         e.g. key -> list of guesses for each independent version of that fn_type.
    init_bounds : dict
        Dict holding guesses for each parameter,
        e.g. key -> list of bounds for each independent version of that fn_type.

    Returns
    -------
    fit_param_ar : np array, shape: num_params
        The initial fit parameter guesses.
    fit_param_bound_ar : np array, shape: (num_params, 2)
        Fit parameter bounds.
    """
    param_lst = []
    bound_lst = []
    for param_name in fit_model.get_param_odict():
        split_param = param_name.split("_")
        param_num_str = int(split_param[-1])
        param_key = "_".join(split_param[:-1])
        param_num = int(param_num_str)  # i.e. pos_0
        try:
            param_lst.append(init_guesses[param_key][param_num])
        except (TypeError, KeyError, IndexError):
            param_lst.append(init_guesses[param_key])
        if len(np.array(init_bounds[param_key]).shape) == 2:
            bound_lst.append(init_bounds[param_key][param_num])
        else:
            bound_lst.append(init_bounds[param_key])

    bound_ar = np.array(bound_lst)
    pbounds = (bound_ar[:, 0], bound_ar[:, 1])
    return np.array(param_lst), pbounds
