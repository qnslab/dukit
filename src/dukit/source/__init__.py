# -*- coding: utf-8 -*-
"""Source reconstruction module - invert B-field to current/magnetization."""

__pdoc__ = {
    "dukit.source.current": True,
    "dukit.source.magnetization": True,
}

from dukit.source.current import (
    get_current_from_bdefect,
    get_current_from_bxyz,
    get_current_without_ft,
    get_divperp_j,
)
from dukit.source.magnetization import (
    get_magnetization_from_bdefect,
    get_magnetization_from_bxyz,
    normalize_in_plane_mag,
)
from dukit.source.reconstruct import (
    reconstruct_source,
)

__all__ = [
    "get_current_from_bxyz",
    "get_current_from_bdefect",
    "get_current_without_ft",
    "get_divperp_j",
    "get_magnetization_from_bxyz",
    "get_magnetization_from_bdefect",
    "normalize_in_plane_mag",
    "reconstruct_source",
]
