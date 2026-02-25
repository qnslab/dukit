# -*- coding: utf-8 -*-
"""
Hamiltonian fitting for defect spin systems.

Functions
---------
 - `dukit.field.hamiltonian.fit_hamiltonian_pixels`
 - `dukit.field.hamiltonian.ham_gen_init_guesses`
 - `dukit.field.hamiltonian.ham_bounds_from_range`
 - `dukit.field.hamiltonian.ham_pixel_generator`
 - `dukit.field.hamiltonian.ham_shuffle_pixels`
 - `dukit.field.hamiltonian.ham_unshuffle_pixels`
 - `dukit.field.hamiltonian.ham_unshuffle_fit_results`
 - `dukit.field.hamiltonian.ham_get_pixel_fitting_results`

Classes
-------
 - `dukit.field.hamiltonian.Hamiltonian`
 - `dukit.field.hamiltonian.Bxyz`
"""

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.field.hamiltonian.fit_hamiltonian_pixels": True,
    "dukit.field.hamiltonian.Hamiltonian": True,
    "dukit.field.hamiltonian.Bxyz": True,
    "dukit.field.hamiltonian.ham_gen_init_guesses": True,
    "dukit.field.hamiltonian.ham_bounds_from_range": True,
    "dukit.field.hamiltonian.ham_pixel_generator": True,
    "dukit.field.hamiltonian.ham_shuffle_pixels": True,
    "dukit.field.hamiltonian.ham_unshuffle_pixels": True,
    "dukit.field.hamiltonian.ham_unshuffle_fit_results": True,
    "dukit.field.hamiltonian.ham_get_pixel_fitting_results": True,
}
# ============================================================================
from typing import Dict, List

import numpy as np
import numpy.linalg as LA  # noqa: N812
from collections import OrderedDict
from scipy.linalg import svd
import copy

# ============================================================================

# ============================================================================


S_MAT_X = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2)
r"""Spin-1 operator: S_{\rm X}"""
S_MAT_Y = np.array([[0, -1j, 0], [1j, 0, 1j], [0, 1j, 0]]) / np.sqrt(2)
r"""Spin-1 operator: S_{\rm Y}"""
S_MAT_Z = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
r"""Spin-1 operator: S_{\rm Z}"""


GAMMA = 2.80  # MHz/G
r"""
The Bohr magneton times the Landé g-factor. See [Doherty2013](https://doi.org/10.1016/j.physrep.2013.02.001)
for details of the g-factor anisotropy.

|                                                                  |                                                               |
|------------------------------------------------------------------|---------------------------------------------------------------|
| \( \gamma_{\rm NV} = \mu_{\rm B} g_e  \)                         |                                                               |
| \( \mu_B = 1.39962449361 \times 10^{10}\ {\rm Hz} \rm{T}^{-1} \) |  [NIST](https://physics.nist.gov/cgi-bin/cuu/Value?mubshhz)   |
| \( \mu_B = 1.399...\ {\rm MHz/G} \)                              |                                                               |
| \( g_e \approx 2.0023 \)                                         |  [Doherty2013](https://doi.org/10.1016/j.physrep.2013.02.001) |
| \( \Rightarrow  \gamma_{\rm NV} \approx 2.80 {\rm MHz/G} \)      |                                                               |

"""

# ============================================================================


class Chooser:
    """Chooser class.

    Is fed a boolean 'chooser_ar' on __init__, of length (len(b_defects) or len(freqs)) that
    is used in call to return only the chosen (i.e. True indices in chooser_ar) indices
    of a given array (some_ar in __call__)
    """

    def __init__(self, chooser_ar):
        self.chooser_ar = chooser_ar

    def __call__(self, some_ar):
        return np.array([some_ar[i] for i, do_use in enumerate(self.chooser_ar) if do_use])


# ============================================================================


class Hamiltonian:
    param_defn: List[str] = []
    param_units: Dict[str, str] = {}
    jac_defined = False

    def __init__(self, chooser_obj, u_defect_frames):
        """
        chooser_obj is used on __call__ and measured_data to return an array
        of only the required parts.
        """
        self.chooser_obj = chooser_obj
        self.u_defect_frames = u_defect_frames
        self.u_defects = u_defect_frames[:, 2, :].copy()  # i.e. z axis of each defect ref. frame in lab frame

    # =================================

    def __call__(self, param_ar):
        """
        Evaluates Hamiltonian for given parameter values.

        Arguments
        ---------
        param_ar : np array, 1D
            Array of hamiltonian parameters fed in.
        """
        raise NotImplementedError("You MUST override __call__, check your spelling.")

    # =================================

    def grad_fn(self, param_ar):
        """
        Return jacobian, shape: (len(b_defects/freqs), len(param_ar))
        Each column is a partial derivative, with respect to each param in param_ar
            (i.e. rows, or first index, is indexing though the b_defects/freqs.)
        """
        raise NotImplementedError("No grad_fn defined for this Hamiltonian.")

    # =================================

    def residuals_scipyfit(self, param_ar, measured_data):
        """
        Evaluates residual: fit model - measured_data. Returns a vector!
        Measured data must be a np array (of the same shape that __call__ returns),
        i.e. freqs, or b_defects.
        """
        return self.chooser_obj(self.__call__(param_ar)) - self.chooser_obj(measured_data)

    # =================================

    def jacobian_scipyfit(self, param_ar, measured_data):
        """Evaluates (analytic) jacobian of ham in format expected by scipy least_squares."""

        # need to take out rows (first index) according to chooser_obj.
        keep_rows = self.chooser_obj(list(range(len(measured_data))))
        delete_rows = [r for r in range(len(measured_data)) if r not in keep_rows]
        return np.delete(self.grad_fn(param_ar), delete_rows, axis=0)

    # =================================

    def jacobian_defined(self):
        return self.jac_defined

    # =================================

    def get_param_defn(self):
        return self.param_defn

    # =================================

    def get_param_odict(self):
        """
        get ordered dict of key: param_key (param_name), val: param_unit for all parameters in ham
        """
        return OrderedDict(self.param_units)

    # =================================

    def get_param_unit(self, param_key):
        """
        Get unit for a given param_key
        """
        if param_key == "residual_field":
            return "Error: sum( || residual(params) || ) over b_defects/freqs (a.u.)"
        param_dict = self.get_param_odict()
        return param_dict[param_key]


# ============================================================================


class Bxyz(Hamiltonian):
    r"""
    $$ H_i = D S_{Z_i}^{2} + \gamma_{\rm{NV}} \bf{B} \cdot \bf{S} $$

    where \( {\bf S}_i = (S_{X_i}, S_{Y_i}, S_{Z_i}) \) are the spin-1 operators.
    Here \( (X_i, Y_i, Z_i) \) is the coordinate system of the defect and \( i = 1,2,3,4 \) labels
    each defect orientation with respect to the lab frame.
    """

    param_defn = ["D", "Bx", "By", "Bz"]
    param_units: Dict[str, str] = {
        "D": "Zero field splitting (MHz)",
        "Bx": "Magnetic field, Bx (G)",
        "By": "Magnetic field, By (G)",
        "Bz": "Magnetic field, Bz (G)",
    }
    jac_defined = False

    def __call__(self, param_ar):
        """
        Hamiltonain of the defect spin using only the zero field splitting D and the magnetic field
        bxyz. Takes the fit_params in the order [D, bx, by, bz] and returns the defect frequencies.

        The spin operators need to be rotated to the defect reference frame. This is achieved
        by projecting the magnetic field onto the u_defect frame.

        i.e. we use the spin operatores in the defect frame, the u_defects in the defect frame,
        and thus project the magnetic field into this frame (from the lab frame) to determine
        the defect frequencies (eigenvalues of hamiltonian).
        """
        defect_frequencies = np.zeros(8)
        # D, Bx, By, Bz = param_ar

        Hzero = param_ar[0] * (S_MAT_Z * S_MAT_Z)  # noqa: N806
        for i in range(4):
            bx_proj_onto_u = np.dot(param_ar[1:4], self.u_defect_frames[i, 0, :])
            by_proj_onto_u = np.dot(param_ar[1:4], self.u_defect_frames[i, 1, :])
            bz_proj_onto_u = np.dot(param_ar[1:4], self.u_defect_frames[i, 2, :])

            HB = GAMMA * (  # noqa: N806
                bx_proj_onto_u * S_MAT_X + by_proj_onto_u * S_MAT_Y + bz_proj_onto_u * S_MAT_Z
            )
            freq, _ = LA.eig(Hzero + HB)
            freq = np.sort(np.real(freq))
            # freqs: ms=0, ms=+-1 -> transition freq is delta
            defect_frequencies[i] = np.real(freq[1] - freq[0])
            defect_frequencies[7 - i] = np.real(freq[2] - freq[0])
        return defect_frequencies


# ============================================================================


def ham_gen_init_guesses(
    hamiltonian_type: str,
    bias_field_gauss: tuple[float, float, float],
    zero_field_splitting: float = 2870.0,
    auto_guess: bool = True,
) -> tuple[dict, dict]:
    """
    Generate initial guesses (and bounds) for hamiltonian fitting.

    Arguments
    ---------
    hamiltonian_type : str
        Type of hamiltonian, e.g., "bxyz".
    bias_field_gauss : tuple[float, float, float]
        Bias field in Gauss (Bx, By, Bz).
    zero_field_splitting : float, default=2870.0
        Zero field splitting D in MHz.
    auto_guess : bool, default=True
        If True, use bias_field_gauss as initial guess for Bx, By, Bz.

    Returns
    -------
    init_guesses : dict
        Dict holding guesses for each parameter.
    init_bounds : dict
        Dict holding bounds for each parameter.
    """
    init_guesses = {}
    init_bounds = {}

    if auto_guess:
        bias_x, bias_y, bias_z = bias_field_gauss
        override_guesses = {"Bx": bias_x, "By": bias_y, "Bz": bias_z}
    else:
        override_guesses = {}

    ham_class = AVAILABLE_HAMILTONIANS[hamiltonian_type]
    for param_key in ham_class.param_defn:
        if param_key in override_guesses:
            guess = override_guesses[param_key]
        elif param_key == "D":
            guess = zero_field_splitting
        else:
            guess = 0.0

        # Set bounds based on parameter type
        if param_key == "D":
            bounds = [guess - 50.0, guess + 50.0]
        else:  # Bx, By, Bz
            bounds = [guess - 100.0, guess + 100.0]

        init_guesses[param_key] = guess
        init_bounds[param_key] = np.array(bounds)

    return init_guesses, init_bounds


# ============================================================================


def ham_bounds_from_range(guess, rang):
    """Generate parameter bounds (list, len 2) when given a range option."""
    if isinstance(guess, (list, tuple)) and len(guess) > 1:
        # separate bounds for each fn of this type
        if isinstance(rang, (list, tuple)) and len(rang) > 1:
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
        if isinstance(rang, (list, tuple)):
            if len(rang) == 1:
                rang = rang[0]
            else:
                raise RuntimeError("param range len should match guess len")
        # param guess and range are just single vals (easy!)
        else:
            bounds = [
                guess - rang,
                guess + rang,
            ]
    return bounds


# ============================================================================


def ham_pixel_generator(our_array):
    """
    Simple generator to shape data as expected by to_squares_wrapper in scipy concurrent method.

    Also allows us to track *where* (i.e. which pixel location) each result corresponds to.
    See also: `dukit.field.ham_scipy.ham_to_squares_wrapper`.

    Arguments
    ---------
    our_array : np array, 3D
        Shape: [idx, y, x] (idx for each b_defect, freq etc.)

    Returns
    -------
    generator : list
        [y, x, our_array[:, y, x]] generator (yielded)
    """
    _, len_y, len_x = np.shape(our_array)
    for y in range(len_y):
        for x in range(len_x):
            yield [y, x, our_array[:, y, x]]


# ============================================================================


def ham_shuffle_pixels(data_3d):
    """
    Simple shuffler

    Arguments
    ---------
    data_3d : np array, 3D
        i.e. freqs/b_defect data, [idx, y, x].

    Returns
    -------
    shuffled_in_yx : np array, 3D
        data_3d shuffled in 2nd, 3rd axis.
    unshuffler : (y_unshuf, x_unshuf)
        Both np array. Can be used to unshuffle shuffled_in_yx, i.e. through
        `dukit.field.hamiltonian.ham_unshuffle_pixels`.
    """

    rng = np.random.default_rng()

    y_shuf = rng.permutation(data_3d.shape[1])
    y_unshuf = np.argsort(y_shuf)
    x_shuf = rng.permutation(data_3d.shape[2])
    x_unshuf = np.argsort(x_shuf)

    shuffled_in_y = data_3d[:, y_shuf, :]
    shuffled_in_yx = shuffled_in_y[:, :, x_shuf]

    # return shuffled pixels, and arrays to unshuffle
    return shuffled_in_yx.copy(), (y_unshuf, x_unshuf)


# =================================


def ham_unshuffle_pixels(data_2d, unshuffler):
    """
    Simple shuffler

    Arguments
    ---------
    data_2d : np array, 2D
        i.e. 'image' of a single fit parameter, all shuffled up!
    unshuffler : (y_unshuf, x_unshuf)
        Two arrays returned by `dukit.field.hamiltonian.ham_shuffle_pixels`
        that allow unshuffling of data_2d.

    Returns
    -------
    unshuffled_in_yx: np array, 2D
        data_2d but the inverse operation of `dukit.field.hamiltonian.ham_shuffle_pixels`
        has been applied.
    """
    y_unshuf, x_unshuf = unshuffler
    unshuffled_in_y = data_2d[y_unshuf, :]
    unshuffled_in_yx = unshuffled_in_y[:, x_unshuf]
    return unshuffled_in_yx.copy()


# =================================


def ham_unshuffle_fit_results(fit_result_dict, unshuffler):
    """
    Simple shuffler

    Arguments
    ---------
    fit_result_dict : dict
        Dictionary, key: param_names, val: image (2D) of param values across FOV. Each image
        requires reshuffling (which this function achieves).
        Also has 'residual' as a key.
    unshuffler : (y_unshuf, x_unshuf)
        Two arrays returned by `dukit.field.hamiltonian.ham_shuffle_pixels` that allow
        unshuffling of data_2d.

    Returns
    -------
    fit_result_dict : dict
        Same as input, but each fit parameter has been unshuffled.
    """
    for key, array in fit_result_dict.items():
        fit_result_dict[key] = ham_unshuffle_pixels(array, unshuffler)
    return fit_result_dict


# ============================================================================


def ham_get_pixel_fitting_results(hamiltonian, fit_results, pixel_data):
    """
    Take the fit result data from scipyfit and back it down to a dictionary of arrays.

    Each array is 2D, representing the values for each parameter (specified by the dict key).


    Arguments
    ---------
    hamiltonian : `dukit.field.hamiltonian.Hamiltonian`
        Model we're fitting to.
    fit_results : list of [(y, x), result, jac] objects
        (see `dukit.field.ham_scipy.ham_to_squares_wrapper`)
        A list of each pixel's parameter array, as well as position in image denoted by (y, x).
    pixel_data : np array, 3D
        Normalised measurement array, shape: [idx, y, x]. i.e. b_defects.
        May or may not already be shuffled (i.e. matches fit_results).

    Returns
    -------
    fit_image_results : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
        Also has 'residual' as a key.
    sigmas: dict
        As fit_image_results, but containing parameters errors (standard deviations) across FOV.
    """

    roi_shape = np.shape(pixel_data)[1:]

    # initialise dictionary with key: val = param_name: param_units
    fit_image_results = hamiltonian.get_param_odict()
    sigmas = copy.copy(fit_image_results)

    # override with correct size empty arrays using np.zeros
    for key in fit_image_results.keys():
        fit_image_results[key] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan
        sigmas[key] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan

    fit_image_results["residual_field"] = np.zeros((roi_shape[0], roi_shape[1])) * np.nan

    # Fill the arrays element-wise from the results function, which returns a
    # 1D array of flattened best-fit parameters.
    for (y, x), result, jac in fit_results:
        resid = hamiltonian.residuals_scipyfit(result, pixel_data[:, y, x])
        fit_image_results["residual_field"][y, x] = np.sum(
            np.abs(resid, dtype=np.float64), dtype=np.float64
        )
        # uncertainty (covariance matrix), copied from scipy.optimize.curve_fit
        _, s, vt = svd(jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(jac.shape) * s[0]
        s = s[s > threshold]
        vt = vt[: s.size]
        pcov = np.dot(vt.T / s**2, vt)
        perr = np.sqrt(np.diag(pcov))  # array of standard deviations

        for param_num, param_name in enumerate(hamiltonian.param_defn):
            fit_image_results[param_name][y, x] = result[param_num]
            sigmas[param_name][y, x] = perr[param_num]

    return fit_image_results, sigmas


# ============================================================================
# ============================================================================


AVAILABLE_HAMILTONIANS = {
    "bxyz": Bxyz,
}
"""Dictionary that defines hamiltonians available for use.

Add any classes you define here so you can use them.

You do not need to avoid overlapping parameter names as hamiltonian
classes can not be used in combination.
"""
