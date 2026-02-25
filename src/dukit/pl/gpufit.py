# -*- coding: utf-8 -*-
"""
This module holds tools for fitting raw data via scipy. (scipy backend)

Functions
---------
 - `dukit.pl.gpufit.fit_roi_avg_pl`
 - `dukit.pl.gpufit.fit_aois_pl`
 - `qdmpy.pl.gpufit.fit_all_pixels_pl`
 - `dukit.pl.gpufit._gen_gf_guesses_bounds`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.pl.gpufit.fit_roi_avg_pl": True,
    "dukit.pl.gpufit.fit_aois_pl": True,
    "dukit.pl.gpufit.fit_all_pixels_pl": True,
    "dukit.pl.gpufit._gen_gf_guesses_bounds": True,
}

# ==========================================================================

import logging

import numpy as np
import numpy.typing as npt
import pygpufit.gpufit as gf

import dukit.itool

# ============================================================================
import dukit.pl.common
import dukit.pl.model
import dukit.share
import dukit.warn

# ==========================================================================


def _get_gpufit_modelID(fit_model: "dukit.pl.model.FitModel"):
    """
    Get gpufit modelID from fit_model.

    Parameters
    ----------
    fit_model : dukit.pl.model.FitModel

    Returns
    -------
    gf.ModelID object
    """
    err = gf.get_last_error()
    if err:
        dukit.warn.warn(f"Cpufit error check:\n{err}")
    if isinstance(fit_model, dukit.pl.model.ConstStretchedExp):
        return gf.ModelID.STRETCHED_EXP
    if isinstance(fit_model, dukit.pl.model.ConstDampedRabi):
        return gf.ModelID.DAMPED_RABI
    if isinstance(fit_model, dukit.pl.model.LinearLorentzians):
        return gf.ModelID.LORENTZ8_LINEAR
    if isinstance(fit_model, dukit.pl.model.ConstLorentzians):
        return gf.ModelID.LORENTZ8_CONST
    raise RuntimeError("Model not recognised")


def fit_roi_avg_pl(
    sig: npt.NDArray,
    ref: npt.NDArray,
    sweep_arr: npt.NDArray,
    fit_model: "dukit.pl.model.FitModel",
    guess_dict: dict,
    bounds_dict: dict,
    norm: str = "div",
    estimator_id: str = "LSE",
    tolerance: float = 1e-12,
    max_iterations: int = 50,
):
    """
    Fit AOI averages

    Arguments
    ---------
    sig : np array, 3D
        Sig measurement array, shape: [y, x, sweep_arr]. CROPPED etc. already.
    ref : np array, 3D
        Ref measurement array, shape: [y, x, sweep_arr]. CROPPED etc. already.
    sweep_arr : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : dukit.pl.model.FitModel
        The model we're fitting to.
    guess_dict : dict
        dict holding guesses for each parameter type, e.g.
        {'pos': [.., ..], 'amp': [.., ..], ...}
    bounds_dict : dict
        dict holding bound options for each parameter type, e.g.
        {"pos_range": 5.0, "amp_bounds": [0.0, 1.0], ...}
    norm : str default="div"
        Normalisation method

    Optional parameters passed to gpufit
    ------------------------------------
    tolerance : float = 1e-12
        Fit tolerance threshold
    max_iterations : int = 50
        Maximum fit iterations permitted.
    estimator_id : str = "LSE"
        Estimator to use, "LSE" or "MLE" (least squares or maximum likelihood).
        MLE for Poisson, assuming all noise in data is purely Poissonian.

    Returns
    -------
    fit_image_results : dukit.share.RoiAvgFit
    """
    pguess, pbounds = _gen_gf_guesses_bounds(
        fit_model,
        *dukit.pl.common.gen_init_guesses(fit_model, guess_dict, bounds_dict),
    )

    if norm == "div":
        sig_norm = sig / ref
    elif norm == "sub":
        sig_norm = 1 + (sig - ref) / (sig + ref)
    elif norm == "true_sub":
        sig_norm = (sig - ref) / np.nanmax(sig - ref)

    avg_sig_norm = np.nanmean(sig_norm, axis=(0, 1))
    avg_sig = np.nanmean(sig, axis=(0, 1))
    avg_ref = np.nanmean(ref, axis=(0, 1))

    # only fit the params we want to :)
    params_to_fit = _get_params_to_fit(fit_model)

    # need to repeat avg_sig_norm twice, as gpufit expects 2D array
    roi_norm_twice = np.repeat([avg_sig_norm], repeats=2, axis=0).astype(np.float32)
    guess = np.repeat([pguess], repeats=2, axis=0).astype(dtype=np.float32)
    constraints = np.repeat([pbounds], repeats=2, axis=0).astype(dtype=np.float32)
    constraint_types = np.array(
        [gf.ConstraintType.LOWER_UPPER for i in range(len(params_to_fit))]
    ).astype(np.int32)

    model_id = _get_gpufit_modelID(fit_model)

    estimator_id = {"LSE": gf.EstimatorID.LSE, "MLE": gf.EstimatorID.MLE}.get(
        estimator_id, gf.EstimatorID.LSE
    )

    best_params_cf, _, _, _, _ = gf.fit_constrained(
        roi_norm_twice,
        None,
        model_id,
        guess,
        constraints=constraints,
        constraint_types=constraint_types,
        user_info=sweep_arr.astype(dtype=np.float32),
        parameters_to_fit=params_to_fit,
        estimator_id=estimator_id,
        tolerance=tolerance,
        max_number_iterations=max_iterations,
    )

    best_params = best_params_cf[0, :]  # just take first fit

    # manually compute sigmas, using analytic jacobian etc.
    best_sigmas = dukit.pl.common.calc_sigmas(fit_model, sweep_arr, avg_sig_norm, best_params)
    best_residual = fit_model(best_params, sweep_arr) - avg_sig_norm

    fit_xvec = np.linspace(
        np.min(sweep_arr),
        np.max(sweep_arr),
        10000,
    )
    fit_yvec = fit_model(best_params, fit_xvec)
    fit_yvec_guess = fit_model(pguess, fit_xvec)
    return dukit.share.RoiAvgFit(
        "gpufit",
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
    estimator_id: str = "LSE",
    tolerance: float = 1e-12,
    max_iterations: int = 50,
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
    fit_model : dukit.pl.model.FitModel
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

    Optional parameters passed to gpufit
    ------------------------------------
    tolerance : float = 1e-12
        Fit tolerance threshold
    max_iterations : int = 50
        Maximum fit iterations permitted.
    estimator_id : str = "LSE"
        Estimator to use, "LSE" or "MLE" (least squares or maximum likelihood).
        MLE for Poisson, assuming all noise in data is purely Poissonian.

    Returns
    -------
    fit_image_results : dict
        Format: {"AOI_n": {"scipyfit": AoiAvgFit}, ...}
    """
    pguess, pbounds = _gen_gf_guesses_bounds(
        fit_model,
        *dukit.pl.common.gen_init_guesses(fit_model, guess_dict, bounds_dict),
    )
    # only fit the params we want to :)
    params_to_fit = _get_params_to_fit(fit_model)
    # need to repeat everything, as gpufit expects 2D array
    guess = np.repeat([pguess], repeats=2, axis=0).astype(dtype=np.float32)
    constraints = np.repeat([pbounds], repeats=2, axis=0).astype(dtype=np.float32)
    constraint_types = np.array(
        [gf.ConstraintType.LOWER_UPPER for i in range(len(params_to_fit))]
    ).astype(np.int32)

    model_id = _get_gpufit_modelID(fit_model)

    estimator_id = {"LSE": gf.EstimatorID.LSE, "MLE": gf.EstimatorID.MLE}.get(
        estimator_id, gf.EstimatorID.LSE
    )

    aois = dukit.itool.get_aois(np.shape(sig), *aoi_coords)

    # add the single pixel check on (just for output)
    output_aoi_coords = list(aoi_coords)
    shp = np.shape(sig)[:-1]
    output_aoi_coords.insert(0, (shp[0] // 2, shp[1] // 2, shp[0] // 2 + 1, shp[1] // 2 + 1))

    ret = {}
    for i, a in enumerate(aois):
        s = sig[a[0], a[1], :]
        r = ref[a[0], a[1], :]
        if norm == "div":
            avg_sig_norm = np.nanmean(s / r, axis=(0, 1))
        elif norm == "sub":
            avg_sig_norm = np.nanmean(1 + (s - r) / (s + r), axis=(0, 1))
        elif norm == "true_sub":
            avg_sig_norm = np.nanmean((s - r) / np.nanmax(s - r), axis=(0, 1))

        avg_sig = np.nanmean(s, axis=(0, 1))
        avg_ref = np.nanmean(r, axis=(0, 1))

        this_aoi_twice = np.repeat([avg_sig_norm], repeats=2, axis=0).astype(np.float32)
        best_params_cf, _, _, _, _ = gf.fit_constrained(
            this_aoi_twice,
            None,
            model_id,
            guess,
            constraints=constraints,
            constraint_types=constraint_types,
            user_info=sweep_arr.astype(dtype=np.float32),
            parameters_to_fit=params_to_fit,
            estimator_id=estimator_id,
            tolerance=tolerance,
            max_number_iterations=max_iterations,
        )
        best_params = best_params_cf[0, :]  # just take first fit
        # manually compute sigmas, using analytic jacobian etc.
        best_sigmas = dukit.pl.common.calc_sigmas(fit_model, sweep_arr, avg_sig_norm, best_params)
        best_residual = fit_model(best_params, sweep_arr) - avg_sig_norm

        fit_xvec = np.linspace(
            np.min(sweep_arr),
            np.max(sweep_arr),
            10000,
        )
        fit_yvec = fit_model(best_params, fit_xvec)
        fit_yvec = fit_model(best_params, fit_xvec)
        ret[f"AOI_{i}"] = {
            "gpufit": dukit.share.AoiAvgFit(
                i,
                sweep_arr,
                avg_sig_norm,
                avg_sig,
                avg_ref,
                "gpufit",
                fit_xvec,
                fit_yvec,
                best_params,
                best_sigmas,
                best_residual,
                pguess,
                pbounds,
                output_aoi_coords[i],
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
    estimator_id: str = "LSE",
    tolerance: float = 1e-12,
    max_iterations: int = 50,
):
    """
    Fits each pixel and returns dictionary of param_name -> param_image.

    Arguments
    ---------
    sig_norm : np array, 3D
        Normalised measurement array, shape: [sweep_arr, y, x].
    sweep_arr : np array, 1D
        Affine parameter list (e.g. tau or freq)
    fit_model : dukit.pl.model.FitModel
        The model we're fitting to.
    guess_dict : dict
        Format: key -> list of guesses for each independent version of that fn_type.
        e.g. 'pos': [.., ..] for each pos fn_type.
    bounds_dict : dict
        Format: key -> bounds for that param type (or use _range).
        e.g. 'pos_bounds': [5., 25.]
        or 'pos_range': 5.0
    roi_avg_result : dukit.share.RoiAvgFit | None
        The result of fitting the ROI average.
        If done, directly uses guesses provided.
    n_jobs : int, default=-2
        Number of jobs to run concurrently, see joblib docs.
        -2 === leaving one cpu free, etc. for neg numbers.
    joblib_verbosity:int = 5
        How often to update progress bar.

    Optional parameters passed to gpufit
    ------------------------------------
    tolerance : float = 1e-12
        Fit tolerance threshold
    max_iterations : int = 50
        Maximum fit iterations permitted.
    estimator_id : str = "LSE"
        Estimator to use, "LSE" or "MLE" (least squares or maximum likelihood).
        MLE for Poisson, assuming all noise in data is purely Poissonian.

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual_0' as a key.
        Sigmas (stdev on fit error) are given as e.g. pos_0_sigma
    """
    num_pixels = np.shape(sig_norm)[0] * np.shape(sig_norm)[1]

    init_pguess, pbounds = _gen_gf_guesses_bounds(
        fit_model,
        *dukit.pl.common.gen_init_guesses(fit_model, guess_dict, bounds_dict),
    )
    pguess = roi_avg_result.best_params if roi_avg_result is not None else init_pguess
    # only fit the params we want to :)
    params_to_fit = _get_params_to_fit(fit_model)

    guess_params = np.array([pguess], dtype=np.float32)
    guess = np.repeat(guess_params, repeats=num_pixels, axis=0)

    model_id = _get_gpufit_modelID(fit_model)
    estimator_id = {"LSE": gf.EstimatorID.LSE, "MLE": gf.EstimatorID.MLE}.get(
        estimator_id, gf.EstimatorID.LSE
    )

    constraint_types = np.array(
        [gf.ConstraintType.LOWER_UPPER for i in range(len(params_to_fit))]
    ).astype(np.int32)

    # constraints needs to be reshaped too
    constraints = np.repeat([pbounds], repeats=num_pixels, axis=0).astype(dtype=np.float32)

    # shape data as wanted by gpufit (num_pixels, num_sweeps)
    sig_norm_shaped = sig_norm.reshape(num_pixels, -1).astype(dtype=np.float32)

    fitting_results, _, _, _, execution_time = gf.fit_constrained(
        sig_norm_shaped,
        None,
        model_id,
        guess,
        constraints=constraints,
        constraint_types=constraint_types,
        user_info=sweep_arr.astype(dtype=np.float32),
        parameters_to_fit=params_to_fit,
        estimator_id=estimator_id,
        tolerance=tolerance,
        max_number_iterations=max_iterations,
    )
    logging.info(f"fit time: {execution_time:.2f}s")

    # unsure if this will work...
    results_arr = np.array(fitting_results).reshape((*sig_norm.shape[:2], len(pguess)))

    names = list(fit_model.get_param_odict().keys())
    fit_image_results = {
        name: array for name, array in zip(names, dukit.itool._iterframe(results_arr))
    }

    # add residual_0 image
    fit_image_results["residual_0"] = np.reshape(
        list(
            map(
                lambda y, p: np.sum(y - fit_model(p, sweep_arr)),
                sig_norm_shaped,
                fitting_results,
            ),
        ),
        sig_norm.shape[:2],
    )

    # calc sigmas & get correct shape - not particularly efficient here
    # be careful with sizes, we only want the parameters actually fit
    sigmas_shaped = np.full((num_pixels, len(names)), np.nan)
    for pl_vec, fitp, sigma_vec in zip(sig_norm_shaped, fitting_results, sigmas_shaped):
        sigma_vec[:] = dukit.pl.common.calc_sigmas(fit_model, sweep_arr, pl_vec, fitp)
    sigmas_result = sigmas_shaped.reshape((*sig_norm.shape[:2], len(names)))

    for i, _ in enumerate(names):
        fit_image_results[f"sigma_{names[i]}"] = sigmas_result[:, :, i]
    return fit_image_results


# =======================================================================================


def _gen_gf_guesses_bounds(
    fit_model: "dukit.pl.model.FitModel", init_guesses: dict, init_bounds: dict
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Generate arrays of initial fit guesses and bounds in correct form for gpufit

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
        except (TypeError, KeyError):
            param_lst.append(init_guesses[param_key])
        if len(np.array(init_bounds[param_key]).shape) == 2:
            bound_lst.append(init_bounds[param_key][param_num][0])
            bound_lst.append(init_bounds[param_key][param_num][1])
        else:
            bound_lst.append(init_bounds[param_key][0])
            bound_lst.append(init_bounds[param_key][1])

    # now need to go through and make sure we have enough if its a lorentz
    # (model expects 8 lorentzians)
    expected_n_params = 0  # default: don't do anything
    if isinstance(fit_model, dukit.pl.model.LinearLorentzians):
        expected_n_params = 2 + 3 * 8
    if isinstance(fit_model, dukit.pl.model.ConstLorentzians):
        expected_n_params = 1 + 3 * 8
    while len(param_lst) < expected_n_params:
        param_lst.append(0)
    while len(bound_lst) < 2 * expected_n_params:
        bound_lst.append(0)
        bound_lst.append(1)

    return np.array(param_lst), np.array(bound_lst)


def _get_params_to_fit(
    fit_model: "dukit.pl.model.FitModel",
) -> npt.NDArray[np.int32]:
    model_id = _get_gpufit_modelID(fit_model)
    if model_id in [
        gf.ModelID.LORENTZ8_CONST,
        gf.ModelID.LORENTZ8_LINEAR,
    ]:
        num_lorentzians = fit_model.n_lorentzians
        if model_id == gf.ModelID.LORENTZ8_CONST:
            params_to_fit = [1 for i in range(3 * num_lorentzians + 1)]  # + 1 for const
            num_params = 25
        elif model_id == gf.ModelID.LORENTZ8_LINEAR:
            params_to_fit = [1 for i in range(3 * num_lorentzians + 2)]  # + 2 for c, m
            num_params = 26
        while len(params_to_fit) < num_params:
            params_to_fit.append(0)
    else:
        params_to_fit = [1 for i in range(len(fit_model.get_param_defn()))]

    return np.array(params_to_fit, dtype=np.int32)
