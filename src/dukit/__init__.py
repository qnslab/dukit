# -*- coding: utf-8 -*-
"""
Package for the analysis of (widefield) defect microscopy image data, from the Tetienne
lab at RMIT/UniMelb. For the lab control software see
[DavidBroadway/qdm-control](https://github.com/DavidBroadway/qdm-control)
(may not be public, talk to David Broadway).

On this page we will document the API, for other information see the [README.md](https://github.com/casparvitch/dukit/blob/main/README.md),
[INSTALL.md](https://github.com/casparvitch/dukit/blob/main/INSTALL.md) and [DEVDOCS.md](https://github.com/casparvitch/dukit/blob/main/DEVDOCS.md), or click through the sub-package links below.

See the [examples](https://github.com/casparvitch/dukit/tree/main/examples) folder for nice examples of usage.

[Repository here](https://github.com/casparvitch/dukit)

# DUKIT Public API

### Systems (`dukit.systems`)

Objects that define the hardware and software configuration of a microscope, and hold
the system-specific methods for reading data from the disk. The `System` class is
abstract, and the other classes inherit from it. The `System` class is not intended to
be instantiated directly, but rather to be subclassed. If you are from another group,
the only thing you should need to do (in the whole package) is write your own
System sub-class.

- `dukit.systems.System`
- `dukit.systems.MelbSystem`
- `dukit.systems.LVControl`
- `dukit.systems.PyControl`
- `dukit.systems.Zyla`
- `dukit.systems.CryoWidefield`
- `dukit.systems.LegacyCryoWidefield`
- `dukit.systems.Argus`
- `dukit.systems.LegacyArgus`
- `dukit.systems.PyCryoWidefield`

#### System methods

- `dukit.systems.System.read_image`
- `dukit.systems.System.get_hardware_binning`
- `dukit.systems.System.read_sweep_arr`
- `dukit.systems.System.get_raw_pixel_size`
- `dukit.systems.System.get_bias_field`
- `dukit.systems.System.norm`

### Working with images (`dukit.itool`)

- `dukit.itool.mpl_set_run_config`
    - Set matplotlib rcParams to standardise style of plots.
- `dukit.itool.crop_sweep`
    - Crop in frequency/tau direction.
- `dukit.itool.crop_roi`
    - Crop to region of interest.
- `dukit.itool.smooth_image_stack`
    - Smooth image stack in-frame, doesn't change number of pixels. Can be asymmetric.
- `dukit.itool.rebin_image_stack`
    - Rebin image stack in-frame, does change number of pixels. Can be asymmetric.
- `dukit.itool.sum_spatially`
    - Sum an image spatially
- `dukit.itool.get_im_filtered`
    - Filter an image - currently only gaussian filtering is implemented.
- `dukit.itool.get_background`
    - Get background of an image - many different kinds, see function docs.
- `dukit.itool.get_colormap_range`
    - Calculate a colormap range for an image. Many kinds, see function docs.
- `dukit.itool.plot_image`
    - Base image plotter.
- `dukit.itool.plot_image_on_ax`
    - As above, but onto axis already created.
- `dukit.itool.mask_polygons`
    - Mask region of image by polygons (see `dukit.polygon` module)
- `dukit.itool.mu_sigma_inside_polygons`
    - Calculate mean and standard deviation inside polygons.
- `dukit.itool.get_aois`
    - Get areas of interest from AOI coords tuple(s).

### PL fitting (`dukit.pl`)

- `dukit.pl.model.FitModel`
    - Base class for PL fitting models. Subclass this to make your own.
- `dukit.pl.model.ConstStretchedExp`
    - T1 model.
- `dukit.pl.model.ConstBiExponential`
    - T1 model with two exponential decays.
- `dukit.pl.model.ConstDampedRabi`
    - Rabi model, constant + damped oscillation.
- `dukit.pl.model.LinearLorentzians`
    - Linear combination of lorentzians, in init pass number of lorentzians.
- `dukit.pl.model.LinearN15Lorentzians`
    - Linear combination of lorentzians, in init pass number of lorentzians.
      This model assumes N15 diamond with fixed HF splitting, 2pk w equal A/w.
- `dukit.pl.model.LinearN14Lorentzians`
    - Linear combination of lorentzians, in init pass number of lorentzians.
      This model assumes N14 diamond with fixed HF splitting, 3pk w equal A/w. each.
- `dukit.pl.model.ConstLorentzians`
    - As above, but with constant background.
- `dukit.pl.interface.fit_all_pixels`
    - Fit all pixels in an image stack.
- `dukit.pl.interface.fit_roi`
    - Fit ROI average spectrum.
- `dukit.pl.interface.fit_aois`
    - Fit AOI average spectra.
- `dukit.pl.interface.load_fit_results`
    - Load fit results from file.

### Field calculations (`dukit.field`)

Currently only calculating local field(s) (not vector field)

- `dukit.field.defects.Defect`
    - Base class for defects.
- `dukit.field.defects.SpinOne`
    - Spin-1 defects general class.
- `dukit.field.defects.NVEnsemble`
    - NV ensemble.
- `dukit.field.defects.VBEnsemble`
    - VB ensemble.
- `dukit.field.defects.SpinPair`
    - Spin pair general class.
- `dukit.field.defects.CPairEnsemble`
    - C_? spin pair ensemble. I.e. visible emitter.

#### Defect methods:

- `dukit.field.defects.Defect.b_defects`
- `dukit.field.defects.Defect.dshift_defects`

### Drift correction (`dukit.driftcorrect`)

- `dukit.driftcorrect.drift_correct_test`
    - Test drift correction params.
- `dukit.driftcorrect.drift_correct_measurement`
    - Apply drift correction to image_stack & save output binary.

### Magnetic sample simulation (`dukit.magsim`)

- `dukit.magsim.SandboxMagSim`
    - Play in an empty sandbox.
- `dukit.magsim.ComparisonMagSim`
    - For comparison with a real magnetic field image, to match sim <-> exp.

### Plotting (`dukit.plot`)

Need to be imported via `dukit.plot.<method>` to work properly.

- `dukit.plot.roi_pl_image`
- `dukit.plot.aoi_pl_image`
- `dukit.plot.roi_avg_fits`
- `dukit.plot.aoi_spectra`
- `dukit.plot.aoi_spectra_fit`
- `dukit.plot.pl_param_image`
- `dukit.plot.pl_param_images`

### Some matplotlib interactive widgets (`dukit.widget`)

- `dukit.widget.LineSelector`
    - Use via `dukit.widget.LinecutSelectionWidget` below.
- `dukit.widget.PolygonSelector`
    - Use via `dukit.polygon.PolygonSelectionWidget` below.
- `dukit.widget.LinecutSelectionWidget`
    - Simple linecut selection tool.
- `dukit.widget.BulkLinecutWidget`
    - Linecut selection tool for multiple images, but in same slice.

### Polygon regions of image, very useful! (`dukit.polygon`)

- `dukit.polygon.Polygon`
    - Polygon object, with various useful methods.
- `dukit.polygon.PolygonSelectionWidget`
    - Widget allowing you to select polygons on an image.
- `dukit.polygon.polygon_selector`
    - GUI interface.
- `dukit.polygon.load_polygon_nodes`
    - Load polygon nodes from file.

### Fourier tooling (`dukit.fourier`)

Currently unused I think, will be useful for vector mag, source recon etc. so leaving
in place.

- `dukit.fourier.define_k_vectors`
    - Allows asymmetric shape, but haven't tested that yet.
- `dukit.fourier.hanning_filter_kspace`
    - Smooth fourier operations below resolvable dimensions, to not amplify noise.
- `dukit.fourier.define_current_transform`
- `dukit.fourier.define_magnetization_transformation`
- `dukit.fourier.pad_image`
- `dukit.fourier.unpad_image`
- `dukit.fourier.MAG_UNIT_CONV`
    - Convert between units of magnetization.
- `dukit.fourier.MU_0`
    - Vacuum permeability.

### Other (`dukit.share`)

- `dukit.share.RoiAvgFit`
    - Result from ROI average fit(s).
- `dukit.share.AoiAvgFit`
    - Result from AOI average fit(s).
"""

from ._version import __version__

from dukit.systems import (
    System,
    MelbSystem,
    LVControl,
    PyControl,
    Zyla,
    CryoWidefield,
    LegacyCryoWidefield,
    Argus,
    LegacyArgus,
    PyCryoWidefield,
)
from dukit.itool import (
    mpl_set_run_config,
    crop_sweep,
    crop_roi,
    smooth_image_stack,
    rebin_image_stack,
    sum_spatially,
    get_im_filtered,
    get_background,
    get_colormap_range,
    plot_image,
    plot_image_on_ax,
    mask_polygons,
    get_background,
    mu_sigma_inside_polygons,
    get_aois,
)

from dukit.widget import (
    LineSelector,
    PolygonSelector,
    LinecutSelectionWidget,
    BulkLinecutWidget,
)
from dukit.polygon import (
    Polygon,
    PolygonSelectionWidget,
    polygon_selector,
    load_polygon_nodes,
)

# Not useful yet, until vector field stuff is implemented
# from dukit.geom import (
#     get_unvs,
#     get_unv_frames,
#     NV_AXES_111,
#     NV_AXES_100_100,
#     NV_AXES_100_110,
# )

from dukit.fourier import (
    define_k_vectors,
    hanning_filter_kspace,
    define_current_transform,
    define_magnetization_transformation,
    pad_image,
    unpad_image,
    MAG_UNIT_CONV,
    MU_0,
)

from dukit.driftcorrect import drift_correct_test, drift_correct_measurement

from dukit.magsim import SandboxMagSim, ComparisonMagSim

from dukit.share import RoiAvgFit, AoiAvgFit

import dukit.plot

from dukit.pl import (
    FitModel,
    ConstStretchedExp,
    ConstBiExponential,
    ConstDampedRabi,
    LinearLorentzians,
    LinearN15Lorentzians,
    LinearN14Lorentzians,
    ConstLorentzians,
    SkewedLorentzians,
    fit_all_pixels,
    fit_roi,
    fit_aois,
    load_fit_results,
    get_fitres_params,
)

from dukit.field import (
    Defect,
    SpinOne,
    NVEnsemble,
    VBEnsemble,
    SpinPair,
    CPairEnsemble,
)
