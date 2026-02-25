# -*- coding: utf-8 -*-
"""
This module holds tools for determining the geometry of the defect-bias field
system etc., required for retrieving/reconstructing vector fields.

Currently only written for NVs.

& Currently not utilized anywhere.

NOTE probably shouldn't be exposed as API
TODO come back and check this is sufficient for ham fit / dc_odmr etc.
Probably want to add these details to `dukit.field.defects.Defect` objects.

Functions
---------
 - `dukit.geom.get_unvs`
 - `dukit.geom.get_unv_frames`

Constants
---------
 - `dukit.geom.NV_AXES_100_110`
 - `dukit.geom.NV_AXES_100_100`
 - `dukit.geom.NV_AXES_111`
"""


# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.geom.get_unvs": True,
    "dukit.geom.get_unv_frames": True,
    "dukit.geom.get_u_defects": True,
    "dukit.geom.get_u_defect_frames": True,
    "dukit.geom.NV_AXES_100_110": True,
    "dukit.geom.NV_AXES_100_100": True,
    "dukit.geom.NV_AXES_111": True,
}

# ============================================================================

import numpy as np
import numpy.linalg as LA  # noqa: N812
import numpy.typing as npt

# ============================================================================


# NOTE for other NV orientations, pass in unvs -> not possible to determine in full
#   generality the orientations for <111> etc.

NV_AXES_100_110: list[dict] = [
    {"nv_number": 0, "ori": (np.sqrt(2 / 3), 0, np.sqrt(1 / 3))},
    {"nv_number": 1, "ori": (-np.sqrt(2 / 3), 0, np.sqrt(1 / 3))},
    {"nv_number": 2, "ori": (0, np.sqrt(2 / 3), -np.sqrt(1 / 3))},
    {"nv_number": 3, "ori": (0, -np.sqrt(2 / 3), -np.sqrt(1 / 3))},
]
"""
<100> top face oriented, <110> edge face oriented diamond (CVD).

NV orientations (unit vectors) relative to lab frame.

Assuming diamond is square to lab frame:

first 3 numbers: orientation of top face of diamond, e.g. <100>

second 3 numbers: orientation of edges of diamond, e.g. <110>

CVD Diamonds are usually <100>, <110>. HPHT usually <100>, <100>.

![](https://i.imgur.com/Rudnzyo.png)

Purple plane corresponds to top (or bottom) face of diamond, orange planes correspond to
edge faces.
"""

NV_AXES_100_100: list[dict] = [
    {"nv_number": 0, "ori": (np.sqrt(1 / 3), np.sqrt(1 / 3), np.sqrt(1 / 3))},
    {
        "nv_number": 1,
        "ori": (-np.sqrt(1 / 3), -np.sqrt(1 / 3), np.sqrt(1 / 3)),
    },
    {
        "nv_number": 2,
        "ori": (np.sqrt(1 / 3), -np.sqrt(1 / 3), -np.sqrt(1 / 3)),
    },
    {
        "nv_number": 3,
        "ori": (-np.sqrt(1 / 3), np.sqrt(1 / 3), -np.sqrt(1 / 3)),
    },
]
"""
<100> top face oriented, <100> edge face oriented diamond (HPHT).

NV orientations (unit vectors) relative to lab frame.

Assuming diamond is square to lab frame:

first 3 numbers: orientation of top face of diamond, e.g. <100>

second 3 numbers: orientation of edges of diamond, e.g. <110>

CVD Diamonds are usually <100>, <110>. HPHT usually <100>, <100>.

![](https://i.imgur.com/cpErjAH.png)

Purple plane: top face of diamond, orange plane: edge faces.
"""

NV_AXES_111: list[dict] = [
    {"nv_number": 0, "ori": (0, 0, 1)},
    {"nv_number": 1, "ori": (np.nan, np.nan, np.nan)},
    {"nv_number": 2, "ori": (np.nan, np.nan, np.nan)},
    {"nv_number": 3, "ori": (np.nan, np.nan, np.nan)},
]
"""
<111> top face oriented.

NV orientations (unit vectors) relative to lab frame.

Only the first nv can be oriented in general. This constant
is defined as a convenience for single-bnv <111> measurements.

<111> diamonds have an NV family oriented in z, i.e. perpindicular
to the diamond surface.
"""


# ============================================================================


def get_unvs(
    bias_x: float,
    bias_y: float,
    bias_z: float,
    unvs: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    | None = None,
    auto_order_unvs: bool = True,
    diamond_ori: str | None = None,
) -> npt.NDArray:
    """
    Returns orientation (relative to lab frame) of NVs. Shape: (4,3) regardless of sample.

    Arguments
    ---------
    TODO

    Returns
    -------
    unvs : np array
        Shape: (4,3). Equivalent to uNV_Z for each NV. (Sorted largest to smallest Bnv)
    """

    if unvs:
        unv_arr = np.array(unvs)
        if unv_arr.shape != (4, 3):
            raise ValueError("Incorrect unvs format passed. Expected shape: (4,3).")
        if auto_order_unvs:
            nv_axes = [{"nv_number": i, "ori": ori.copy()} for i, ori in enumerate(unv_arr)]
            for family in nv_axes:
                projection = np.dot(family["ori"], [bias_x, bias_y, bias_z])
                family["mag"] = np.abs(projection)
                family["sign"] = np.sign(projection)
                # Fix: when projection is zero, sign is 0 which zeroes out the orientation
                if family["sign"] == 0:
                    family["sign"] = 1
            sorted_dict = sorted(nv_axes, key=lambda x: x["mag"], reverse=True)

            for idx in range(len(sorted_dict)):
                unv_arr[idx, :] = np.array(sorted_dict[idx]["ori"]) * sorted_dict[idx]["sign"]
    else:
        unv_arr = np.zeros((4, 3))  # z unit vectors of unv frame (in lab frame)
        if diamond_ori == "<100>_<100>":
            nv_axes = NV_AXES_100_100
        elif diamond_ori == "<100>_<110>":
            nv_axes = NV_AXES_100_110
        elif diamond_ori == "<111>":
            nv_axes = NV_AXES_111
        else:
            raise ValueError("diamond_ori not recognised.")

        for family in nv_axes:
            projection = (
                family["ori"][0] * bias_x + family["ori"][1] * bias_y + family["ori"][2] * bias_z
            )
            family["mag"] = np.abs(projection)
            family["sign"] = np.sign(projection)
            # Fix: when projection is zero, sign is 0 which zeroes out the orientation
            if family["sign"] == 0:
                family["sign"] = 1

        srtd = sorted(nv_axes, key=lambda x: x["mag"], reverse=True)

        for idx, family in enumerate(srtd):
            unv_arr[idx, :] = np.array(family["ori"]) * family["sign"]

    return unv_arr


# ============================================================================


def get_unv_frames(
    bias_x: float,
    bias_y: float,
    bias_z: float,
    unvs: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    | None = None,
    auto_order_unvs: bool = True,
    diamond_ori: str | None = None,
) -> npt.NDArray:
    """
    Returns array representing each NV reference frame.
    I.e. each index is a 2D array: [uNV_X, uNV_Y, uNV_Z] representing the unit vectors
    for that NV reference frame, in the lab frame.

    Arguments
    ---------
    TODO

    Returns
    -------
    unv_frames : np array
        [ [uNV1_X, uNV1_Y, uNV1_Z], [uNV2_X, uNV2_Y, uNV2_Z], ...]
    """
    nv_signed_ori = get_unvs(bias_x, bias_y, bias_z, unvs, auto_order_unvs, diamond_ori)
    unv_frames = np.zeros((4, 3, 3))
    for i in range(4):
        # calculate uNV frame in the lab frame
        unv_z = nv_signed_ori[i]
        # We have full freedom to pick x/y as long as xyz are all orthogonal
        # we can ensure this by picking Y to be orthog. to both the NV axis
        # and another NV axis, then get X to be the cross between those two.
        unv_y = np.cross(unv_z, nv_signed_ori[-i - 1])
        unv_y = unv_y / LA.norm(unv_y)
        unv_x = np.cross(unv_y, unv_z)
        unv_frames[i, ::] = [unv_x, unv_y, unv_z]

    return unv_frames


# ============================================================================

# Aliases for new naming convention (u_defect instead of unv)
get_u_defects = get_unvs
get_u_defect_frames = get_unv_frames

