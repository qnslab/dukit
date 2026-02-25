# -*- coding: utf-8 -*-
"""
Warnings for DUKIT.

Always lowest in import heirarchy

Functions
---------
 - `duki.warn.warn`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.warn.warn": True,
}

# ============================================================================

import warnings

# ============================================================================


def warn(msg: str):
    """Throw a custom DUKITWarning with message 'msg'."""
    warnings.warn(msg, DUKITWarning)


class DUKITWarning(Warning):
    """allows us to separate dukit warnings from those in other packages."""


# ============================================================================
