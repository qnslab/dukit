# -*- coding: utf-8 -*-
"""
This module holds the general interface tools for fitting data, independent of fit
backend (e.g. scipy/gpufit etc.).

All of these functions are automatically loaded into the namespace when the fit
sub-package is imported. (e.g. import dukit.fit).

Functions
---------
 - `dukit.pl.interface.fit_roi`
 - `dukit.pl.interface.fit_aois`
 - `dukit.pl.interface.fit_all_pixels`
 - `dukit.pl.interface.load_fit_results`
 - `dukit.pl.interface.get_fitres_params`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.pl.interface.fit_roi": True,
    "dukit.pl.interface.fit_aois": True,
    "dukit.pl.interface.fit_all_pixels": True,
    "dukit.pl.interface.load_fit_results": True,
}

# ============================================================================

import os
import pathlib
from collections import abc
from collections import defaultdict as dd
from typing import Union

import numpy as np
import numpy.typing as npt

# ============================================================================
import dukit.json2dict
import dukit.pl.model
import dukit.pl.scipyfit
import dukit.share

# ============================================================================

CPUFIT_AVAILABLE: bool = False
try:
    import pycpufit.cpufit as cf  # noqa: F401

    CPUFIT_AVAILABLE = True
except ImportError:
    pass
else:
    import dukit.pl.cpufit

GPUFIT_AVAILABLE: bool = False
try:
    import pygpufit.gpufit as gf  # noqa: F401

    GPUFIT_AVAILABLE = True
except ImportError:
    pass
else:
    import dukit.pl.gpufit

# ============================================================================


def fit_roi(
    sig: npt.NDArray,
    ref: npt.NDArray,
    sweep_arr: npt.NDArray,
    fit_model: "dukit.pl.model.FitModel",
    guess_dict: dict,
    bounds_dict: dict,
    norm: str = "div",
    opath: str = "",
    sf_method: str = "trf",
    sf_verbose: object = 0,
    sf_gtol: float = 1e-12,
    sf_xtol: float = 1e-12,
    sf_ftol: float = 1e-12,
    sf_loss: str = "linear",
    sf_jac: str | None = None,
    gf_tolerance: float = 1e-12,
    gf_max_iterations: int = 50,
    gf_estimator_id: str = "LSE",
) -> dict[str, dukit.share.RoiAvgFit]:
    """
    Fit full ROI (region of interest) with the given fit model, for each
    fit backend that is installed.

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
    opath : str
        Output path for saving fit results (.json)

    Optional parameters passed to scipy least_squares
    -------------------------------------------------
    sf_method: str = "trf"
        Fit method, "trf", "dogbox" or "lm"
    sf_verbose=0
        Verbosity of fit -> probably want to keep at 0
    sf_gtol: float = 1e-12
        Tolerance for termination by the change of the independent variables.
    sf_xtol: float = 1e-12
        Tolerance for termination by the norm of the gradient.
    sf_ftol: float = 1e-12
        Tolerance for termination by the change of the cost function.
    sf_loss: str = "linear"
        Determines the loss function. This in non trivial check the scipy documentation.
        Best you don't change this.
    sf_jac: str | None = None
        Jacobian. If None uses algebraic jac, otherwise:
        '2-point' is probably best, then '3-point', 'cs' are other options.

    Optional parameters passed to cpufit/gpufit
    -------------------------------------------
    gf_tolerance : float = 1e-12
        Fit tolerance threshold
    gf_max_iterations : int = 50
        Maximum fit iterations permitted.
    gf_estimator_id : str = "LSE"
        Estimator to use, "LSE" or "MLE" (least squares or maximum likelihood).
        MLE for Poisson, assuming all noise in data is purely Poissonian.

    Returns
    -------
    result : dict
        Dict: fit_backend str => dukit.share.RoiAvgResult
    """
    result = {}
    result["scipyfit"] = dukit.pl.scipyfit.fit_roi_avg_pl(
        sig,
        ref,
        sweep_arr,
        fit_model,
        guess_dict,
        bounds_dict,
        norm,
        method=sf_method,
        verbose=sf_verbose,
        gtol=sf_gtol,
        xtol=sf_xtol,
        ftol=sf_ftol,
        loss=sf_loss,
        jac=sf_jac,
    )
    if CPUFIT_AVAILABLE:
        try:
            result["cpufit"] = dukit.pl.cpufit.fit_roi_avg_pl(
                sig,
                ref,
                sweep_arr,
                fit_model,
                guess_dict,
                bounds_dict,
                norm=norm,
                tolerance=gf_tolerance,
                estimator_id=gf_estimator_id,
                max_iterations=gf_max_iterations,
            )
        except RuntimeError:
            pass  # modelID not found, that's fine
    if GPUFIT_AVAILABLE:
        try:
            result["gpufit"] = dukit.pl.gpufit.fit_roi_avg_pl(
                sig,
                ref,
                sweep_arr,
                fit_model,
                guess_dict,
                bounds_dict,
                norm=norm,
                tolerance=gf_tolerance,
                estimator_id=gf_estimator_id,
                max_iterations=gf_max_iterations,
            )
        except RuntimeError:
            pass  # modelID not found, that's fine

    if opath:
        res_dict = {}
        for backend in result.keys():
            res_dict[backend] = result[backend].to_dict()
        dukit.json2dict.dict_to_json(res_dict, opath)
    return result


# ============================================================================


def fit_aois(
    sig: npt.NDArray,
    ref: npt.NDArray,
    sweep_arr: npt.NDArray,
    fit_model: "dukit.pl.model.FitModel",
    guess_dict: dict,
    bounds_dict: dict,
    *aoi_coords: tuple[int, int, int, int],
    norm: str = "div",
    opath: str = "",
    sf_method: str = "trf",
    sf_verbose=0,
    sf_gtol: float = 1e-12,
    sf_xtol: float = 1e-12,
    sf_ftol: float = 1e-12,
    sf_loss: str = "linear",
    sf_jac: str | None = None,
    gf_tolerance: float = 1e-12,
    gf_max_iterations: int = 50,
    gf_estimator_id: str = "LSE",
) -> dict[str, dict[str, dukit.share.AoiAvgFit]]:
    """
    Fit AOIs and single pixel with all available fit backends.

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
    aoi_coords : tuple of 4 ints
        Coordinates of the AOI to fit, in the form (x0, y0, x1, y1).
    norm : str default="div"
        Normalisation method
    opath : str
        Output path for saving fit results (.json)

    Optional parameters passed to scipy least_squares
    -------------------------------------------------
    sf_method: str = "trf"
        Fit method, "trf", "dogbox" or "lm"
    sf_verbose=0
        Verbosity of fit -> probably want to keep at 0
    sf_gtol: float = 1e-12
        Tolerance for termination by the change of the independent variables.
    sf_xtol: float = 1e-12
        Tolerance for termination by the norm of the gradient.
    sf_ftol: float = 1e-12
        Tolerance for termination by the change of the cost function.
    sf_loss: str = "linear"
        Determines the loss function. This in non trivial check the scipy documentation.
        Best you don't change this.
    sf_jac: str | None = None
        Jacobian. If None uses algebraic jac, otherwise:
        '2-point' is probably best, then '3-point', 'cs' are other options.

    Optional parameters passed to cpufit/gpufit
    -------------------------------------------
    gf_tolerance : float = 1e-12
        Fit tolerance threshold
    gf_max_iterations : int = 50
        Maximum fit iterations permitted.
    gf_estimator_id : str = "LSE"
        Estimator to use, "LSE" or "MLE" (least squares or maximum likelihood).
        MLE for Poisson, assuming all noise in data is purely Poissonian.

    Returns
    -------
    result : dict
        Dict: aoi_n str => fit_backend => dukit.share.AoiAvgResult
    """

    def _recursive_dict_update(to_be_updated_dict, updating_dict):
        """
        Recursively updates to_be_updated_dict with values from updating_dict
        (to all dict depths).
        """
        if not isinstance(to_be_updated_dict, abc.Mapping):
            return updating_dict
        for key, val in updating_dict.items():
            if isinstance(val, abc.Mapping):
                # avoids KeyError by returning {}
                to_be_updated_dict[key] = _recursive_dict_update(
                    to_be_updated_dict.get(key, {}), val
                )
            else:
                to_be_updated_dict[key] = val
        return to_be_updated_dict

    result = dukit.pl.scipyfit.fit_aois_pl(
        sig,
        ref,
        sweep_arr,
        fit_model,
        guess_dict,
        bounds_dict,
        *aoi_coords,
        norm=norm,
        method=sf_method,
        verbose=sf_verbose,
        gtol=sf_gtol,
        xtol=sf_xtol,
        ftol=sf_ftol,
        loss=sf_loss,
        jac=sf_jac,
    )
    if CPUFIT_AVAILABLE:
        try:
            cf_res = dukit.pl.cpufit.fit_aois_pl(
                sig,
                ref,
                sweep_arr,
                fit_model,
                guess_dict,
                bounds_dict,
                *aoi_coords,
                norm=norm,
                tolerance=gf_tolerance,
                max_iterations=gf_max_iterations,
                estimator_id=gf_estimator_id,
            )
            _recursive_dict_update(result, cf_res)
        except RuntimeError:
            pass  # modelID not found, that's fine
    if GPUFIT_AVAILABLE:
        try:
            gf_res = dukit.pl.gpufit.fit_aois_pl(
                sig,
                ref,
                sweep_arr,
                fit_model,
                guess_dict,
                bounds_dict,
                *aoi_coords,
                norm=norm,
                tolerance=gf_tolerance,
                max_iterations=gf_max_iterations,
                estimator_id=gf_estimator_id,
            )
            _recursive_dict_update(result, gf_res)
        except RuntimeError:
            pass  # modelID not found, that's fine

    if opath:
        res_dict = dd(dict)
        for aoi in result.keys():
            for fit_backend in result[aoi].keys():
                res_dict[aoi][fit_backend] = result[aoi][fit_backend].to_dict()
        dukit.json2dict.dict_to_json(res_dict, opath)
    return result


# ============================================================================


def fit_all_pixels(
    fit_backend: str,
    sig_norm: npt.NDArray,
    sweep_arr: npt.NDArray,
    fit_model: "dukit.pl.model.FitModel",
    guess_dict: dict,
    bounds_dict: dict,
    roi_avg_result: Union[None, "dukit.share.RoiAvgFit"],
    odir: str = "",
    sf_n_jobs: int = -2,
    sf_joblib_verbosity: int = 5,
    sf_method: str = "trf",
    sf_verbose: int = 0,
    sf_gtol: float = 1e-12,
    sf_xtol: float = 1e-12,
    sf_ftol: float = 1e-12,
    sf_loss: str = "linear",
    sf_jac: str | None = None,
    gf_tolerance: float = 1e-12,
    gf_max_iterations: int = 50,
    gf_estimator_id: str = "LSE",
) -> dict[str, npt.NDArray]:
    """
    Fit all pixels in a given image with the given fit model, for each
    fit backend that is installed.

    Arguments
    ---------
    fit_backend : str
        Name of the fit backend to use, e.g. "scipyfit", "cpufit", "gpufit".
    sig_norm : np array, 3D
        Sig measurement array, shape: [y, x, sweep_arr].
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
    roi_avg_result : dukit.share.RoiAvgFit | None
        The ROI average fit result, used to get the initial guess for each pixel.
        If none, uses pguesses as given.
    odir : str
        Output *directory* (created if doesn't exist) to put .txt files for params.


    Optional parameters passed to scipy least_squares
    -------------------------------------------------
    sf_n_jobs : int = -2
        Number of jobs to run in parallel. -1 means use all processors, -2 means all
        but one, etc.
    sf_joblib_verbosity : int = 5
        Verbosity of joblib parallelisation.
    sf_method: str = "trf"
        Fit method, "trf", "dogbox" or "lm"
    sf_verbose : int = 0
        Verbosity of fit -> probably want to keep at 0
    sf_gtol: float = 1e-12
        Tolerance for termination by the change of the independent variables.
    sf_xtol: float = 1e-12
        Tolerance for termination by the norm of the gradient.
    sf_ftol: float = 1e-12
        Tolerance for termination by the change of the cost function.
    sf_loss: str = "linear"
        Determines the loss function. This in non trivial check the scipy documentation.
        Best you don't change this.
    sf_jac: str | None = None
        Jacobian. If None uses algebraic jac, otherwise:
        '2-point' is probably best, then '3-point', 'cs' are other options.

    Optional parameters passed to cpufit/gpufit
    -------------------------------------------
    gf_tolerance : float = 1e-12
        Fit tolerance threshold
    gf_max_iterations : int = 50
        Maximum fit iterations permitted.
    gf_estimator_id : str = "LSE"
        Estimator to use, "LSE" or "MLE" (least squares or maximum likelihood).
        MLE for Poisson, assuming all noise in data is purely Poissonian.


    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual_0' as a key.
        Sigmas (stdev on fit error) are given as e.g. pos_0_sigma
    """
    if fit_backend == "scipyfit":
        fit_image_results = dukit.pl.scipyfit.fit_all_pixels_pl(
            sig_norm,
            sweep_arr,
            fit_model,
            guess_dict,
            bounds_dict,
            roi_avg_result,
            sf_n_jobs,
            sf_joblib_verbosity,
            sf_method,
            sf_verbose,
            sf_gtol,
            sf_xtol,
            sf_ftol,
            sf_loss,
            sf_jac,
        )
    if fit_backend == "cpufit":
        fit_image_results = dukit.pl.cpufit.fit_all_pixels_pl(
            sig_norm,
            sweep_arr,
            fit_model,
            guess_dict,
            bounds_dict,
            roi_avg_result,
            tolerance=gf_tolerance,
            max_iterations=gf_max_iterations,
            estimator_id=gf_estimator_id,
        )
    if fit_backend == "gpufit":
        fit_image_results = dukit.pl.gpufit.fit_all_pixels_pl(
            sig_norm,
            sweep_arr,
            fit_model,
            guess_dict,
            bounds_dict,
            roi_avg_result,
            tolerance=gf_tolerance,
            max_iterations=gf_max_iterations,
            estimator_id=gf_estimator_id,
        )

    if odir:
        pathlib.Path(odir).mkdir(parents=True, exist_ok=True)
        for param_key in fit_image_results.keys():
            np.savetxt(
                odir + f"{param_key}.txt",
                fit_image_results[param_key],
                fmt="%.18e",
                delimiter=",",
            )

    return fit_image_results


# ============================================================================
def load_fit_results(idir: str, fit_model: "dukit.pl.model.FitModel") -> dict:
    """
    Load fit results from a json file.

    Arguments
    ---------
    idir : str
        Path to directory containing .txt files.
    fit_model : dukit.pl.model.FitModel
        The model we're fitting to.

    Returns
    -------
    fit_results : dict
        Dict of fit results.
    """
    if not os.path.exists(idir):
        raise FileNotFoundError(f"Directory {idir} does not exist?")
    fit_results = {}
    names = list(fit_model.get_param_odict().keys())
    names.extend([f"sigma_{n}" for n in names])
    names.append("residual_0")
    for name in names:
        fit_results[name] = np.loadtxt(idir + f"{name}.txt", delimiter=",")
    return fit_results


# ============================================================================


def get_fitres_params(fit_results: dict, res_pos_name: str = "pos") -> tuple:
    """
    Get the fit results parameters from a dictionary of fit results & res param name.

    Helper function.

    Arguments
    ---------
    fit_results : dict
        Dictionary of fit results.
    res_pos_name : str = "pos"
        Name of the resonance position parameter, e.g. 'pos' for Lorentzians.

    Returns
    -------
    resonances : tuple
        Tuple of resonance positions, each a 2D array.
    """
    resonances_lst = []
    for key in sorted(list(fit_results.keys())):
        if key.startswith(res_pos_name):
            resonances_lst.append(fit_results[key])
    return tuple(resonances_lst)
