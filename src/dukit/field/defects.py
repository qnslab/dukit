# -*- coding: utf-8 -*-
"""
This module holds Defect objects, which are used to represent spin defects
and their properties, and for extracting 'useful' info from ODMR data.

Classes
-------
 - `dukit.field.defects.Defect`
 - `dukit.field.defects.SpinOne`
 - `dukit.field.defects.NVEnsemble`
 - `dukit.field.defects.VBEnsemble`
 - `dukit.field.defects.SpinPair`
 - `dukit.field.defects.CPairEnsemble`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.field.defects.Defect": True,
    "dukit.field.defects.SpinOne": True,
    "dukit.field.defects.NVEnsemble": True,
    "dukit.field.defects.VBEnsemble": True,
    "dukit.field.defects.SpinPair": True,
    "dukit.field.defects.CPairEnsemble": True,
}

# TODO add vector mag methods.

# ==========================================================================

import numpy as np
import numpy.typing as npt

# ============================================================================


# ==========================================================================


class Defect:
    def b_defects(
        self,
        res_freqs: tuple[npt.ArrayLike] | tuple[float, ...],
        past_gslac: bool = False,
    ) -> tuple[npt.ArrayLike] | tuple[float, ...]:
        raise NotImplementedError()

    def dshift_defects(
        self,
        res_freqs: tuple[npt.ArrayLike] | tuple[float],
    ) -> tuple[npt.ArrayLike] | tuple[float, ...]:
        raise NotImplementedError()


class SpinOne(Defect):
    temp_coeff: float
    zero_field_splitting: float
    gslac: float
    gamma: float

    def b_defects(
        self,
        res_freqs: tuple[npt.ArrayLike] | tuple[float, ...],
        past_gslac: bool = False,
    ) -> tuple[npt.ArrayLike] | tuple[float, ...]:
        """Calculate magnetic field(s) [in defect frame(s)]
        in Tesla from resonance frequencies in MHz.
        """
        raise NotImplementedError()

    def dshift_defects(
        self,
        res_freqs: tuple[npt.ArrayLike] | tuple[float],
    ) -> tuple[npt.ArrayLike] | tuple[float, ...]:
        """Calculate d-shifts in MHz from resonance frequencies in MHz."""
        res_freqs = list(res_freqs)
        res_freqs.sort(key=np.nanmean)  # sort into correct order
        num_res = len(res_freqs)
        if num_res == 1:
            dshifts = (res_freqs[0] - self.zero_field_splitting,)
        elif num_res == 2:
            dshifts = ((res_freqs[1] + res_freqs[0]) / 2,)
        else:
            dshifts = []
            for i in range(num_res // 2):
                dshifts.append((res_freqs[-i - 1] + res_freqs[i]) / 2)
            if ((num_res // 2) * 2) + 1 == num_res:
                dshifts.append(np.full_like(res_freqs[0], np.nan))
        return dshifts

    def d_to_T(self, dshift_mhz: npt.ArrayLike):
        dshift_mhz_arr = np.asarray(dshift_mhz)
        return self.temp_coeff * (dshift_mhz_arr - self.zero_field_splitting)


class NVEnsemble(SpinOne):
    temp_coeff: float = 13.51  # K/MHz = -0.074 MHz/K => 10.1038/s41467-021-24725-1
    zero_field_splitting: float = 2.87e3  # MHz
    gslac: float = 0.1024  # Teslas
    gamma: float = 28.0e3  # MHz/T

    def b_defects(
        self,
        res_freqs: tuple[npt.ArrayLike] | tuple[float, ...],
        past_gslac: bool = False,
    ) -> tuple[npt.ArrayLike] | tuple[float, ...]:
        res_freqs = list(res_freqs)
        res_freqs.sort(key=np.nanmean)  # sort into correct order
        num_res = len(res_freqs)
        if num_res == 1:
            if past_gslac:
                b_defs = (res_freqs[0] / self.gamma + self.gslac,)
            elif np.mean(res_freqs[0]) < self.zero_field_splitting:
                b_defs = ((self.zero_field_splitting - res_freqs[0]) / self.gamma,)
            else:
                b_defs = ((res_freqs[0] - self.zero_field_splitting) / self.gamma,)
        elif num_res == 2:
            b_defs = (np.abs(res_freqs[1] - res_freqs[0]) / (2 * self.gamma),)
        else:
            b_defs = []
            for i in range(num_res // 2):
                b_defs.append(np.abs(res_freqs[-i - 1] - res_freqs[i]) / (2 * self.gamma))
            if ((num_res // 2) * 2) + 1 == num_res:
                peak = res_freqs[num_res // 2 + 1]
                sign = -1 if np.mean(peak) < self.zero_field_splitting else 1
                middle_b_def = sign * peak / self.gamma
                b_defs.append(middle_b_def)
        return b_defs


class VBEnsemble(SpinOne):
    # could add full D->T curve later?
    temp_coeff: float = -1.605  # K/MHz = -0.623 MHz/K => 10.1038/s41467-021-24725-1
    zero_field_splitting: float = 3.68e3  # MHz
    gslac: float = 0.1300  # Teslas
    gamma: float = 28.0e3  # MHz/T

    def b_defects(
        self,
        res_freqs: tuple[npt.ArrayLike] | tuple[float, ...],
        past_gslac: bool = False,
    ) -> tuple[npt.ArrayLike] | tuple[float, ...]:
        res_freqs = list(res_freqs)
        res_freqs.sort(key=np.nanmean)  # sort into correct order
        num_res = len(res_freqs)
        if num_res == 1:
            if past_gslac:
                b_defs = (res_freqs[0] / self.gamma + self.gslac,)
            elif np.mean(res_freqs[0]) < self.zero_field_splitting:
                b_defs = ((self.zero_field_splitting - res_freqs[0]) / self.gamma,)
            else:
                b_defs = ((res_freqs[0] - self.zero_field_splitting) / self.gamma,)
        elif num_res == 2:
            b_defs = (np.abs(res_freqs[1] - res_freqs[0]) / (2 * self.gamma),)
        else:
            raise ValueError("VBEnsemble has no b_defects for >2 resonances")
        return b_defs


class SpinPair(Defect):
    gamma: float

    def b_defects(
        self, res_freqs: tuple[npt.ArrayLike] | tuple[float], past_gslac=False
    ) -> tuple[npt.ArrayLike] | tuple[float, ...]:
        return tuple([res_freqs[0] / self.gamma])


class CPairEnsemble(SpinPair):
    gamma = 28.0e3  # MHz/T
