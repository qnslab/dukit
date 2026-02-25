# -*- coding: utf-8 -*-
"""
Calculating the magnetic field, from ODMR data.

NB: requires fit model to have 'pos' named parameter(s).

Have not implemented vector reconstruction etc. yet.

Classes
-------
- `dukit.field.defects.Defect`
- `dukit.field.defects.SpinOne`
- `dukit.field.defects.NVEnsemble`
- `dukit.field.defects.VBEnsemble`
- `dukit.field.defects.SpinPair`
- `dukit.field.defects.CPairEnsemble`

"""

from dukit.field.defects import (
    CPairEnsemble,
    Defect,
    NVEnsemble,
    SpinOne,
    SpinPair,
    VBEnsemble,
)
