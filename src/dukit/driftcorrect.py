# -*- coding: utf-8 -*-
"""
This module holds tools for measurement (lateral) drift correction.

Requires a `dukit.systems.System` for reading files from disk.
Most of the functions are not documented, but the API is only 2 funcs:

Functions
---------
 - `dukit.driftcorrect.drift_correct_test`
 - `dukit.driftcorrect.drift_correct_measurement`

"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.driftcorrect.drift_correct_test": True,
    "dukit.driftcorrect.drift_correct_measurement": True,
}

# ============================================================================

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from skimage.registration import phase_cross_correlation as cross_correl
from skimage.transform import EuclideanTransform, warp
from tqdm.autonotebook import tqdm

import dukit.itool

# ============================================================================
import dukit.json2dict
import dukit.systems

# ============================================================================


def _drift_correct_single(
    refr_pl: npt.NDArray, move_pl: npt.NDArray, target_pl: npt.NDArray
) -> tuple[npt.NDArray, tuple[int, int]]:
    # refr_pl & move_pl should probably be cropped to some feature
    # then the shift is applied to target_pl
    with np.errstate(all="ignore"):
        shift_calc = cross_correl(refr_pl, move_pl, normalization=None)[0]
    tform = EuclideanTransform(translation=[-shift_calc[1], -shift_calc[0]])
    return warp(target_pl, tform, mode="edge"), tuple(shift_calc)


def _drift_correct_stack(
    refr_pl: npt.NDArray, move_pl: npt.NDArray, move_sig_norm: npt.NDArray
) -> tuple[npt.NDArray, tuple[int, int]]:
    # calc's shift in 'mov_pl' to match 'refr_pl' (image regulation) via
    # cross-corr method then applies that to move_sig_norm.
    with np.errstate(all="ignore"):
        shift_calc = cross_correl(refr_pl, move_pl, normalization=None)[0]
    tform = EuclideanTransform(translation=[-shift_calc[1], -shift_calc[0]])

    reg_sig_norm = np.empty(move_sig_norm.shape)
    for i in range(move_sig_norm.shape[-1]):
        reg_sig_norm[:, :, i] = warp(move_sig_norm[:, :, i], tform, mode="edge")
    return reg_sig_norm, tuple(shift_calc)


# ============================================================================


def read_and_drift_correct(
    base_path: str,
    stub: Callable[[int], str],
    image_seq: list | tuple | npt.NDArray,
    system: dukit.systems.System,
    roi_coords: tuple[int, int, int, int],
    ignore_ref: bool = False,
    mask: npt.NDArray[np.bool_] | None = None,  # True where(i) you want to incl im in accum
) -> tuple[npt.NDArray, npt.NDArray]:
    first = True
    for i in tqdm(image_seq):
        sig, ref, _ = system.read_image(base_path + stub(i), ignore_ref, "div")
        if first:
            first = False
            refr_pl = np.sum(sig, axis=-1)
            accum_sig = sig.copy()
            accum_ref = ref.copy()
            prev_sig = sig.copy()
            prev_ref = ref.copy()
            continue

        this_sig = sig - prev_sig
        this_ref = ref - prev_ref
        this_pl = np.sum(this_sig, axis=-1)
        prev_sig = sig
        prev_ref = ref

        this_sig, _ = _drift_correct_stack(
            dukit.itool.crop_roi(refr_pl, roi_coords),
            dukit.itool.crop_roi(this_pl, roi_coords),
            this_sig,
        )
        this_ref, _ = _drift_correct_stack(
            dukit.itool.crop_roi(refr_pl, roi_coords),
            dukit.itool.crop_roi(this_pl, roi_coords),
            this_ref,
        )

        if mask is None or mask[i - image_seq[0]]:
            accum_sig += this_sig
            accum_ref += this_ref

    return accum_sig, accum_ref


# ============================================================================


def drift_correct_measurement(
    directory: str,
    start_num: int,
    end_num: int,
    stub: Callable[[int], str],
    system: dukit.systems.System,
    output_file: str,
    feature_roi_coords: tuple[int, int, int, int],
    image_nums_mask=None,
):
    """

    Arguments
    ---------
    directory : str
        Path to directory that contains all the measurements.
    start_num : int
        Image/measurement to start accumulating from.
    end_num : int
        Image/measurement to end accumulation.
    stub : Callable[[int], str]
        Function that takes image num and returns path to that measurement
        from directory. I.e. directory + stub(X) for filepath ass. with 'X'
    output_file : str
        Output will be stored in directory + output file
    system : dukit.systems.System object
        Used for reading in files
    output_file : str
        Where to save the drift-corrected binary
        Format is based on 'system'.
    feature_roi_coords : tuple[int, int, int, int]
        Define roi, here a feature in PL that is used from cross-correlation.
        start_x, start_y, end_x, end_y. -1 to use edge of image on that side.
        Here the ROI is only used as a PL 'feature'/'window' for
        cross-correlation drift correction, not for cropping output etc.
    image_nums_mask : 1D ndarray
        Same shape as list(range(start_num, end_num + 1)).
        Where false, don't include that image in accumulation.
    """

    sig, ref = read_and_drift_correct(
        directory,
        stub,
        list(range(start_num, end_num + 1)),
        system,
        feature_roi_coords,
        mask=image_nums_mask,
    )

    if isinstance(system, dukit.systems.LVControl):
        # below don't seem necessary now??
        # s = sig.transpose([0, 2, 1])
        # r = ref.transpose([0, 2, 1])
        s, r = sig, ref

        output = []
        for f in range(s.shape[-1]):
            output.append(s[:, :, f])
            output.append(r[:, :, f])
        output_array = np.array(output).flatten()

        with open(output_file, "wb") as fid:
            np.array([0, 0]).astype(np.float32).tofile(fid)
            output_array.astype(np.float32).tofile(fid)

        lines = []
        with open(directory + stub(start_num) + "_metaSpool.txt", "r") as fid:
            for line in fid:
                lines.append(line)

        with open(output_file + "_metaSpool.txt", "w") as fid:
            for line in lines:
                fid.write(line)
    else:
        raise ValueError("Don't know how to write that type of binary yet.")


# ============================================================================


def drift_correct_test(
    directory: str,
    start_num: int,
    end_num: int,
    comparison_nums: list | tuple | npt.NDArray,
    stub: Callable[[int], str],
    system: dukit.systems.System,
    feature_roi_coords: tuple[int, int, int, int],
    ignore_ref: bool = False,
):
    """
    Test the drift correction on a subset (comparison_nums) of the measurements.

    Arguments
    ---------
    directory : str
        Path to directory that contains all the measurements.
    start_num : int
        Image/measurement to start accumulating from.
        (Also the reference frame)
    end_num : int
        Image/measurement to end accumulation.
    comparison_nums : list of ints
        List of image/measurment nums to compare drift calc on.
    stub : function
        Function that takes image num and returns path to that measurement
        from directory. I.e. directory + stub(X) for filepath ass. with 'X'
    system : dukit.systems.System object
        Used for reading in files
    ignore_ref : bool = False
    feature_roi_coords : tuple[int, int, int, int]
        Define roi, here a feature in PL that is used from cross-correlation.
        start_x, start_y, end_x, end_y. -1 to use edge of image on that side.
        Here the ROI is only used as a PL 'feature'/'window' for
        cross-correlation drift correction, not for cropping output etc.

    Returns
    -------
    crop_fig : plt.Figure
        For further editing/saving etc.
    crop_axs : plt.Axes
        For further editing/saving etc.
    """

    # prep fig
    nrows = len(comparison_nums)
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=(10, 4 * nrows))

    # read in all frames, only pull out (non-accum) pl frames we want to compare
    raw_comp_frames = []
    image_seq = list(range(start_num, end_num + 1))
    first = True
    for i in image_seq:
        sig, _, _ = system.read_image(directory + stub(i), ignore_ref, "div")
        pl = np.sum(sig, axis=-1)
        accum_pl = dukit.itool.crop_roi(pl, (-1, -1, -1, -1))  # don't crop here
        if first:
            first = False
            prev_accum_pl = accum_pl.copy()
            if i in comparison_nums:
                raw_comp_frames.append(accum_pl)
            continue
        this_pl = accum_pl - prev_accum_pl
        prev_accum_pl = accum_pl
        if i in comparison_nums:
            raw_comp_frames.append(this_pl)
        if i > max(comparison_nums):
            break

    # plot cropped sig frames in left column
    for i, frame, ax in zip(comparison_nums, raw_comp_frames, axs[:, 0]):
        dukit.itool.plot_image_on_ax(
            fig,
            ax,
            dukit.itool.crop_roi(frame, feature_roi_coords),
            title=f"raw   {i}",
            c_range=(None, None),
            c_label="Counts",
            c_map="gray",
            show_tick_marks=True,
        )

    # do cross-corr on comparison frames
    refr_frame = raw_comp_frames[0]
    corrected_frames = [
        refr_frame,
    ]
    shift_calcs: list[tuple[int, int]] = [
        (0, 0),
    ]
    for frame in raw_comp_frames[1:]:
        corrected_frame, shift_calc = _drift_correct_single(
            dukit.itool.crop_roi(refr_frame, feature_roi_coords),
            dukit.itool.crop_roi(frame, feature_roi_coords),
            frame,
        )
        corrected_frames.append(corrected_frame)
        shift_calcs.append(shift_calc)

    # plot cropped corrected frames in right column
    for i, frame, shift_calc, ax in zip(comparison_nums, corrected_frames, shift_calcs, axs[:, 1]):
        dukit.itool.plot_image_on_ax(
            fig,
            ax,
            dukit.itool.crop_roi(frame, feature_roi_coords),
            title=f"shftd {i}: {shift_calc}",
            c_range=(None, None),
            c_label="Counts",
            show_tick_marks=True,
            c_map="gray",
        )

    return fig, axs
