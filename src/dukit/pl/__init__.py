# -*- coding: utf-8 -*-
"""
Module for fitting PL data.

Cpufit/Gpufit are imported if available.

Classes
-------
- `dukit.pl.model.FitModel`
- `dukit.pl.model.ConstStretchedExp`
- `dukit.pl.model.ConstDampedRabi`
- `dukit.pl.model.LinearLorentzians`
- `dukit.pl.model.LinearN14Lorentzians`
- `dukit.pl.model.LinearN15Lorentzians`
- `dukit.pl.model.ConstLorentzians`
- `dukit.pl.model.SkewedLorentzians`

Functions
---------
- `dukit.pl.interface.fit_all_pixels`
- `dukit.pl.interface.fit_roi`
- `dukit.pl.interface.fit_aois`
- `dukit.pl.interface.load_fit_results`
"""

from dukit.pl.model import (
    FitModel,
    ConstStretchedExp,
    ConstBiExponential,
    ConstDampedRabi,
    LinearLorentzians,
    LinearN14Lorentzians,
    LinearN15Lorentzians,
    ConstLorentzians,
    SkewedLorentzians,
)

from dukit.pl.interface import (
    fit_all_pixels,
    fit_roi,
    fit_aois,
    load_fit_results,
    get_fitres_params,
)
