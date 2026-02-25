# -*- coding: utf-8 -*-
"""
Data structures/types shared between other modules.

Classes
-------
 - `dukit.share.AoiAvgFit`
 - `dukit.share.RoiAvgFit`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.share.AoiAvgFit": True,
    "dukit.share.RoiAvgFit": True,
}

# ============================================================================

import numpy.typing as npt

# ============================================================================


# ============================================================================
class AoiAvgFit:
    """Holds result from an AOI fit. Only a method to save.

    Attributes
    ----------
    aoi_num : int
        Number associated with this AOI
    sweep_arr : ndarray, 1D
        Freqs/taus
    avg_sig_norm : ndarray, 1D
        sig_norm (trace/spectrum) averaged over this aoi
    avg_sig : ndarray, 1D
        sig (trace/spectrum) averaged over this aoi
    avg_ref : ndarray, 1D
        ref (trace/spectrum) averaged over this aoi
    fit_backend : str
        Name of the fit backend (e.g. scipy, gpufit, cpufit, etc.) used.
    fit_xvec : ndarray, 1D
        Values where we have calculated the fit function
    fit_yvec : ndarray, 1D
        Values of fit function at solution, for fit_xvec values
    best_params : ndarray, 1D
        Best (optimal) fit/model parameters
    best_sigmas : ndarray, 1D
        Sigmas (stdev of fit error) at solution
    best_residual : float
        Residual vec at solution
    pguess: tuple of floats
        Guesses for fit parameters
    pbounds : 2-tuple, of tuples
        Bounds for fit, first tuple are the lower bounds, second the upper bounds.
    fit_options : dict
        Other options passed to fit(s).

    Methods
    -------
    save_json(filepath)
        Save into given filepath, ending in '.json'.
    """

    def __init__(
        self,
        aoi_num: int,
        sweep_arr: npt.NDArray,
        avg_sig_norm: npt.NDArray,
        avg_sig: npt.NDArray,
        avg_ref: npt.NDArray,
        fit_backend: str,
        fit_xvec: npt.NDArray,
        fit_yvec: npt.NDArray,
        best_params: npt.NDArray,
        best_sigmas: npt.NDArray,
        best_residual: float,
        pguess: npt.ArrayLike,
        pbounds: tuple[npt.ArrayLike, npt.ArrayLike],
        aoi_coords: tuple[int, int, int, int],
    ):
        """
        Arguments
        ---------
        aoi_num : int
            Number associated with this AOI
        sweep_arr : ndarray, 1D
            Freqs/taus
        avg_sig_norm : ndarray, 1D
            sig_norm (trace/spectrum) averaged over this aoi
        avg_sig : ndarray, 1D
            sig (trace/spectrum) averaged over this aoi
        avg_ref : ndarray, 1D
            ref (trace/spectrum) averaged over this aoi
        fit_backend : str
            Name of the fit backend (e.g. scipy, gpufit, cpufit, etc.) used.
        fit_xvec : ndarray, 1D
            Values where we have calculated the fit function
        fit_yvec : ndarray, 1D
            Values of fit function at solution, for fit_xvec values
        best_params : ndarray, 1D
            Best (optimal) fit/model parameters
        best_sigmas : ndarray, 1D
            Sigmas (stdev of fit error) at solution
        best_residual : float
            Residual vec at solution
        pguess: tuple of floats
            Guesses for fit parameters
        pbounds : 2-tuple, of tuples
            Bounds for fit, first tuple are the lower bounds, second the upper bounds.
        aoi_coords : 4-tuple of ints
            start_x, start_y, end_x, end_y
        """
        self.aoi_num = aoi_num
        self.sweep_arr = sweep_arr
        self.avg_sig_norm = avg_sig_norm
        self.avg_sig = avg_sig
        self.avg_ref = avg_ref
        self.fit_backend = fit_backend
        self.fit_xvec = fit_xvec
        self.fit_yvec = fit_yvec
        self.best_params = best_params
        self.best_sigmas = best_sigmas
        self.best_residual = best_residual
        self.pguess = pguess
        self.pbounds = pbounds
        self.aoi_coords = aoi_coords

    def to_dict(self):
        """
        Return a dict of params
        """
        return {
            "aoi_num": self.aoi_num,
            "sweep_arr": self.sweep_arr,
            "avg_sig_norm": self.avg_sig_norm,
            "avg_sig": self.avg_sig,
            "avg_ref": self.avg_ref,
            "fit_backend": self.fit_backend,
            "fit_xvec": self.fit_xvec,
            "fit_yvec": self.fit_yvec,
            "best_params": self.best_params,
            "best_sigmas": self.best_sigmas,
            "best_residual": self.best_residual,
            "pguess": self.pguess,
            "pbounds": self.pbounds,
            "aoi_coords": self.aoi_coords,
        }


# ============================================================================


class RoiAvgFit:
    """Holds result from an ROI fit. Only a method to save.

    Attributes
    ----------
    fit_backend : str
        Name of the fit backend (e.g. scipy, gpufit, cpufit, etc.) used.
    sweep_arr : ndarray, 1D
        Freqs/taus
    avg_sig_norm : ndarray, 1D
        sig_norm (trace/spectrum) averaged over ROI
    avg_sig : ndarray, 1D
        sig (trace/spectrum) averaged over ROI
    avg_ref : ndarray, 1D
        ref (trace/spectrum) averaged over ROI
    fit_xvec : ndarray, 1D
        Values where we have calculated the fit function
    fit_yvec : ndarray, 1D
        Values of fit function at solution, for fit_xvec values
    fit_yvec_guess : ndarray, 1D
        The fit 'guess' values
    best_params : ndarray, 1D
        Best (optimal) fit/model parameters
    best_sigmas : ndarray, 1D
        Sigmas (stdev of fit error) at solution
    best_residual : float
        Residual vec at solution
    pguess: tuple of floats
        Guesses for fit parameters
    pbounds : 2-tuple, of tuples
        Bounds for fit, first tuple are the lower bounds, second the upper bounds.

    Methods
    -------
    save_json(filepath)
        Save into given filepath, ending in '.json'.
    """

    def __init__(
        self,
        fit_backend: str,
        sweep_arr: npt.NDArray,
        avg_sig_norm: npt.NDArray,
        avg_sig: npt.NDArray,
        avg_ref: npt.NDArray,
        fit_xvec: npt.NDArray,
        fit_yvec: npt.NDArray,
        fit_yvec_guess: npt.NDArray,
        best_params: npt.NDArray,
        best_sigmas: npt.NDArray,
        best_residual: float,
        pguess: npt.ArrayLike,
        pbounds: tuple[npt.ArrayLike, npt.ArrayLike],
    ):
        """
        Arguments
        ---------
        fit_backend : str
            Name of the fit backend (e.g. scipy, gpufit, cpufit, etc.) used.
        sweep_arr : ndarray, 1D
            Freqs/taus
        avg_sig_norm : ndarray, 1D
            sig_norm (trace/spectrum) averaged over ROI
        avg_sig : ndarray, 1D
            sig (trace/spectrum) averaged over ROI
        avg_ref : ndarray, 1D
            ref (trace/spectrum) averaged over ROI
        fit_xvec : ndarray, 1D
            Values where we have calculated the fit function
        fit_yvec : ndarray, 1D
            Values of fit function at solution, for fit_xvec values
        fit_yvec_guess : ndarray, 1D
            The fit 'guess' values
        best_params : ndarray, 1D
            Best (optimal) fit/model parameters
        best_sigmas : ndarray, 1D
            Sigmas (stdev of fit error) at solution
        best_residual : float
            Residual vec at solution
        pguess: tuple of floats
            Guesses for fit parameters
        pbounds : 2-tuple, of tuples
            Bounds for fit, first tuple are the lower bounds, second the upper bounds.
        """
        self.fit_backend = fit_backend
        self.sweep_arr = sweep_arr
        self.avg_sig_norm = avg_sig_norm
        self.avg_sig = avg_sig
        self.avg_ref = avg_ref
        self.fit_xvec = fit_xvec
        self.fit_yvec = fit_yvec
        self.fit_yvec_guess = fit_yvec_guess
        self.best_params = best_params
        self.best_sigmas = best_sigmas
        self.best_residual = best_residual
        self.pguess = pguess
        self.pbounds = pbounds

    def to_dict(self):
        """
        Return a dict of params
        """
        return {
            "fit_backend": self.fit_backend,
            "sweep_arr": self.sweep_arr,
            "avg_sig_norm": self.avg_sig_norm,
            "avg_sig": self.avg_sig,
            "avg_ref": self.avg_ref,
            "fit_xvec": self.fit_xvec,
            "fit_yvec": self.fit_yvec,
            "fit_yvec_guess": self.fit_yvec_guess,
            "best_params": self.best_params,
            "best_sigmas": self.best_sigmas,
            "best_residual": self.best_residual,
            "init_pguess": self.pguess,
            "init_pbounds": self.pbounds,
        }
