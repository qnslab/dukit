# -*- coding: utf-8 -*-
"""
Calculating the magnetic field, from ODMR data.

NB: requires fit model to have 'pos' named parameter(s).

Classes (Defects)
-----------------
- `dukit.field.defects.Defect`
- `dukit.field.defects.SpinOne`
- `dukit.field.defects.NVEnsemble`
- `dukit.field.defects.VBEnsemble`
- `dukit.field.defects.SpinPair`
- `dukit.field.defects.CPairEnsemble`

Classes (Hamiltonian)
---------------------
- `dukit.field.hamiltonian.Hamiltonian`
- `dukit.field.hamiltonian.Bxyz`

Functions (Field Reconstruction)
--------------------------------
- `dukit.field.get_bxyz_from_single_defect`
- `dukit.field.get_bxyz_from_defects`
- `dukit.field.get_bxyz_from_hamiltonian`
- `dukit.field.get_bxyz_from_pre_gslac_ref`
- `dukit.field.get_bdefects_from_frequencies`
- `dukit.field.spherical_to_cartesian`
- `dukit.field.reconstruct_field_components`

"""

from dukit.field.defects import (
    CPairEnsemble,
    Defect,
    NVEnsemble,
    SpinOne,
    SpinPair,
    VBEnsemble,
)
from dukit.field.ham_scipy import (
    fit_hamiltonian_roi_avg_scipyfit,
    fit_hamiltonian_scipyfit,
)
from dukit.field.hamiltonian import (
    AVAILABLE_HAMILTONIANS,
    Bxyz,
    Hamiltonian,
)

# Reconstruction functions
from dukit.field.reconstruction import (
    get_bdefects_from_frequencies,
    get_bxyz_from_bdefects_inversion,
    get_bxyz_from_hamiltonian,
    get_bxyz_from_pre_gslac_ref,
    get_bxyz_from_single_defect,
    reconstruct_field_components,
    spherical_to_cartesian,
)

# Convenience alias (backwards compatibility)
get_bxyz_from_defects = get_bxyz_from_bdefects_inversion
