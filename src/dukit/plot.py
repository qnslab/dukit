# -*- coding: utf-8 -*-
"""
This module holds functions for plotting.

Functions
---------
 - `dukit.plot.roi_pl_image`
 - `dukit.plot.aoi_pl_image`
 - `dukit.plot.roi_avg_fits`
 - `dukit.plot.aoi_spectra`
 - `dukit.plot.aoi_spectra_fit`
 - `dukit.plot.pl_param_image`
 - `dukit.plot.pl_param_images`
 - `dukit.plot._add_patch_rect`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.plot.roi_pl_image": True,
    "dukit.plot.aoi_pl_image": True,
    "dukit.plot.roi_avg_fits": True,
    "dukit.plot.aoi_spectra": True,
    "dukit.plot.aoi_spectra_fit": True,
    "dukit.plot.pl_param_image": True,
    "dukit.plot.pl_param_images": True,
    "dukit.plot._add_patch_rect": True,
}

# ============================================================================

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import patches
from matplotlib.lines import Line2D

import dukit.itool

# ============================================================================
import dukit.json2dict
import dukit.pl
import dukit.share
import dukit.warn

# ===========================================================================

AOI_COLORS: list[str] = [
    "blue",
    "tab:brown",
    "purple",
    "darkslategrey",
    "magenta",
    "olive",
    "cyan",
]

FIT_BACKEND_COLORS: dict[str, dict[str, str]] = {
    "scipyfit": {
        "ROIfit_linecolor": "mediumblue",
        "residual_linecolor": "black",
        "AOI_ROI_fit_linecolor": "indigo",
        "AOI_best_fit_linecolor": "crimson",
    },
    "gpufit": {
        "ROIfit_linecolor": "cornflowerblue",
        "residual_linecolor": "dimgrey",
        "AOI_ROI_fit_linecolor": "mediumpurple",
        "AOI_best_fit_linecolor": "coral",
    },
    "cpufit": {
        "ROIfit_linecolor": "lightskyblue",
        "residual_linecolor": "lightgrey",
        "AOI_ROI_fit_linecolor": "plum",
        "AOI_best_fit_linecolor": "goldenrod",
    },
}


# ===========================================================================


def roi_pl_image(
    pl_image: npt.NDArray,
    roi_coords: tuple[int, int, int, int],
    c_range_type: str = "",
    c_range_values: tuple[float, float] = (),
    opath: str = "",
    **kwargs,
):
    """
    Plots full pl image with ROI region annotated.

    Arguments
    ---------
    pl_image : np array, 2D
        UNCCROPPED pl image, but binned
    roi_coords : tuple[int, int, int, int]
        ROI coordinates
    c_range_type : str = ""
        Type of colormap range to use. See `dukit.itool.get_colormap_range`
    c_range_values : tuple[float, float] = ()
        Values to use for colormap range. See `dukit.itool.get_colormap_range`
    opath : str = ""
        If supplied, saves figure here.
    **kwargs
        Plotting options passed to `dukit.itool.plot_image_on_ax`
    Returns
    -------
    fig : plt.Figure

    ax : plt.Axes
    """

    fig, ax = plt.subplots()
    if "c_range" not in kwargs and c_range_type and c_range_values:
        c_range = dukit.itool.get_colormap_range(c_range_type, c_range_values, pl_image)
    else:
        c_range = dukit.itool.get_colormap_range("min_max", (), pl_image)

    if "c_map" in kwargs:
        c_map = kwargs["c_map"]
        del kwargs["c_map"]
    else:
        c_map = "grey"

    fig, ax = dukit.itool.plot_image_on_ax(
        fig,
        ax,
        pl_image,
        title="PL - Full & Rebinned",
        c_range=c_range,
        c_label="Counts",
        c_map=c_map,
        **kwargs,
    )

    _add_patch_rect(ax, roi_coords, label="ROI", edgecolor="r")
    if opath:
        fig.savefig(opath)


# ============================================================================


def aoi_pl_image(
    pl_image: npt.NDArray,
    *aoi_coords: tuple[int, int, int, int],
    c_range_type: str = "",
    c_range_values: tuple[float, float] = (),
    opath: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots pl image cut down to ROI, with annotated AOI regions.

    Arguments
    ---------
    pl_image : np array, 2D
        pl image AFTER cropping.
    *aoi_coords : tuple[int, int, int, int]
        AOI coordinates
    c_range_type : str = ""
        Type of colormap range to use. See `dukit.itool.get_colormap_range`
    c_range_values : tuple[float, float] = ()
        Values to use for colormap range. See `dukit.itool.get_colormap_range`
    opath : str
        If supplied, saves figure here.
    **kwargs
        Plotting options passed to `dukit.itool.plot_image_on_ax`

    Returns
    -------
    fig : plt.Figure

    ax : plt.Axes

    """
    fig, ax = plt.subplots()
    if "c_range" not in kwargs and c_range_type and c_range_values:
        c_range = dukit.itool.get_colormap_range(c_range_type, c_range_values, pl_image)
    else:
        c_range = dukit.itool.get_colormap_range("min_max", (), pl_image)

    if "c_map" in kwargs:
        c_map = kwargs["c_map"]
        del kwargs["c_map"]
    else:
        c_map = "grey"

    fig, ax = dukit.itool.plot_image_on_ax(
        fig,
        ax,
        pl_image,
        title="PL - ROI & Rebinned",
        c_range=c_range,
        c_label="Counts",
        c_map=c_map,
        **kwargs,
    )

    # add single pixel aoi first
    shp = pl_image.shape
    _add_patch_rect(
        ax,
        (shp[0] // 2, shp[1] // 2, shp[0] // 2 + 1, shp[1] // 2 + 1),
        label="AOI 0",
        edgecolor=AOI_COLORS[0],
    )
    for i, aoi in enumerate(aoi_coords):
        _add_patch_rect(ax, aoi, label="AOI " + str(i + 1), edgecolor=AOI_COLORS[i + 1])

    if opath:
        fig.savefig(opath)

    return fig, ax


# ============================================================================


def roi_avg_fits(roi_results: dict[str, dukit.share.RoiAvgFit], opath: str = ""):
    """
    Plots fit of spectrum averaged across ROI, as well as corresponding residual values.

    Arguments
    ---------
    roi_results : dict[str, dukit.share.RoiAvgFit
        dict["fit_backend"] => RoiAvgFit
    opath : str
        If given, save figure here.
    Returns
    -------
    fig : matplotlib Figure object
    """
    figsize = mpl.rcParams["figure.figsize"].copy()
    figsize[0] *= 2
    figsize[1] *= 1.5

    fig = plt.figure(figsize=figsize)
    # xstart, ystart, xend, yend
    # [units are fraction of the image frame, from bottom left corner]
    spectrum_frame = fig.add_axes((0.1, 0.3, 0.8, 0.6))

    rr = next(iter(roi_results.values()))  # any result, just to plot raw data
    lspec_names = []
    lspec_lines = []

    spectrum_frame.plot(
        rr.sweep_arr,
        rr.avg_sig_norm,
        label="raw data",
        ls=" ",
        marker="o",
        mfc="w",
        mec="firebrick",
    )
    lspec_names.append("raw data")
    lspec_lines.append(Line2D([0], [0], ls=" ", marker="o", mfc="w", mec="firebrick"))

    arb_result = next(iter(roi_results.values()))

    spectrum_frame.plot(
        arb_result.fit_xvec,
        arb_result.fit_yvec_guess,
        linestyle=(0, (1, 1)),
        label="init guess",
        c="darkgreen",
    )
    lspec_names.append("init guess")
    lspec_lines.append(Line2D([0], [0], linestyle=(0, (1, 1)), c="darkgreen"))

    plt.setp(spectrum_frame.get_xticklabels(), visible=False)

    spectrum_frame.grid()
    spectrum_frame.set_ylabel("PL (a.u.)")

    # residual plot
    residual_frame = fig.add_axes((0.1, 0.1, 0.8, 0.2), sharex=spectrum_frame)

    lresid_names = []
    lresid_lines = []

    residual_frame.grid()

    residual_frame.set_xlabel("Sweep parameter")
    residual_frame.set_ylabel("Fit - data (a.u.)")

    for _, res in roi_results.items():
        # ODMR spectrum_frame
        spectrum_frame.plot(
            res.fit_xvec,
            res.fit_yvec,
            linestyle="--",
            label=f"{res.fit_backend} best fit",
            c=FIT_BACKEND_COLORS[res.fit_backend]["ROIfit_linecolor"],
        )
        lspec_names.append(f"{res.fit_backend} best fit")
        lspec_lines.append(
            Line2D(
                [0],
                [0],
                linestyle="--",
                c=FIT_BACKEND_COLORS[res.fit_backend]["ROIfit_linecolor"],
            )
        )

        residual_frame.plot(
            res.sweep_arr,
            res.best_residual,
            label=f"{res.fit_backend} residual",
            ls="dashed",
            c=FIT_BACKEND_COLORS[res.fit_backend]["residual_linecolor"],
            marker="o",
            mfc="w",
            mec=FIT_BACKEND_COLORS[res.fit_backend]["residual_linecolor"],
        )
        lresid_names.append(f"{res.fit_backend} residual")
        lresid_lines.append(
            Line2D(
                [0],
                [0],
                ls="dashed",
                c=FIT_BACKEND_COLORS[res.fit_backend]["residual_linecolor"],
                marker="o",
                mfc="w",
                mec=FIT_BACKEND_COLORS[res.fit_backend]["residual_linecolor"],
            )
        )

    # https://jdhao.github.io/2018/01/23/matplotlib-legend-outside-of-axes/
    # https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/linestyles.html
    legend_names = lspec_names.copy()
    legend_names.extend(lresid_names)

    legend_lines = lspec_lines.copy()
    legend_lines.extend(lresid_lines)

    spectrum_frame.legend(
        legend_lines,
        legend_names,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.01),
        fontsize="medium",
        ncol=len(legend_names),
        borderaxespad=0,
        frameon=False,
    )
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # spectrum_frame.legend()

    if opath:
        fig.savefig(opath)

    return fig


# ============================================================================


def aoi_spectra(
    sig: npt.NDArray,
    ref: npt.NDArray,
    sweep_arr: npt.NDArray,
    *aoi_coords: tuple[int, int, int, int],
    specpath: str = "",
    opath: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots spectra from each AOI, as well as subtraction and division norms.

    Arguments
    ---------
    sig : np array, 3D
        Signal component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, y, x]
    ref : np array, 3D
        Reference component of raw data, reshaped and rebinned. Unwanted sweeps removed.
        Cut down to ROI.
        Format: [sweep_vals, y, x]
    sweep_arr : ndarray
        List of sweep parameter values (with removed unwanted sweeps at start/end)
    specpath : str
        Path (preferably to json) to save spectra in.
        Note you *probably* want to use TODO(fit_aois) to output instead.
    opath : str
        Path to save figure in.

    Returns
    -------
    fig : matplotlib Figure object
    """

    aois = dukit.itool.get_aois(sig.shape, *aoi_coords)  # type: ignore

    # pre-process data to plot
    sig_avgs = []
    ref_avgs = []
    sub_norms = []
    div_norms = []
    true_sub_norms = []
    for i, aoi in enumerate(aois):
        sig_aoi = sig[aoi[0], aoi[1], :]
        ref_aoi = ref[aoi[0], aoi[1], :]
        sig_avg = np.nanmean(sig_aoi, axis=(0, 1))
        ref_avg = np.nanmean(ref_aoi, axis=(0, 1))
        sub_norm = np.nanmean(1 + (sig_aoi - ref_aoi) / (sig_aoi + ref_aoi), axis=(0, 1))
        div_norm = np.nanmean(sig_aoi / ref_aoi, axis=(0, 1))
        true_sub_norm = np.nanmean(
            (sig_aoi - ref_aoi)
            / np.nanmax(sig_aoi - ref_aoi, axis=-1).reshape(sig_aoi.shape[:-1] + (1,)),
            axis=(0, 1),
        )
        sig_avgs.append(sig_avg)
        ref_avgs.append(ref_avg)
        sub_norms.append(sub_norm)
        div_norms.append(div_norm)
        true_sub_norms.append(true_sub_norm)

    figsize = mpl.rcParams["figure.figsize"].copy()
    num_wide = 3 if len(aois) < 3 else len(aois)
    figsize[0] *= 0.6 * num_wide
    figsize[1] *= 1.75
    fig, axs = plt.subplots(2, num_wide, figsize=figsize, sharex=True, sharey=False)

    for i, aoi in enumerate(aois):
        # plot sig
        axs[0, i].plot(
            sweep_arr,
            sig_avgs[i],
            label="sig",
            c="blue",
            ls="dashed",
            marker="o",
            mfc="cornflowerblue",
            mec="mediumblue",
        )
        # plot ref
        axs[0, i].plot(
            sweep_arr,
            ref_avgs[i],
            label="ref",
            c="green",
            ls="dashed",
            marker="o",
            mfc="limegreen",
            mec="darkgreen",
        )

        axs[0, i].legend()
        axs[0, i].grid(True)
        axs[0, i].set_title(
            "AOI " + str(i),
            fontdict={"color": AOI_COLORS[i]},
        )
        axs[0, i].set_ylabel("pl (a.u.)")

    linestyles = [
        "--",
        "-.",
        (0, (1, 1)),
        (0, (5, 10)),
        (0, (5, 5)),
        (0, (5, 1)),
        (0, (3, 10, 1, 10)),
        (0, (3, 5, 1, 5)),
        (0, (3, 1, 1, 1)),
        (0, (3, 5, 1, 5, 1, 5)),
        (0, (3, 10, 1, 10, 1, 10)),
        (0, (3, 1, 1, 1, 1, 1)),
    ]

    for i in range(len(aois)):
        # plot subtraction norm
        axs[1, 0].plot(
            sweep_arr,
            sub_norms[i],
            label="AOI " + str(i + 1),
            c=AOI_COLORS[i],
            ls=linestyles[i],
            marker="o",
            mfc="w",
            mec=AOI_COLORS[i],
        )
        axs[1, 0].legend()
        axs[1, 0].grid(True)
        axs[1, 0].set_title("sub: 1 + (sig - ref / sig + ref)")
        axs[1, 0].set_xlabel("Sweep parameter")
        axs[1, 0].set_ylabel("pl (a.u.)")

        # plot division norm
        axs[1, 1].plot(
            sweep_arr,
            div_norms[i],
            label="AOI " + str(i),
            c=AOI_COLORS[i],
            ls=linestyles[i],
            marker="o",
            mfc="w",
            mec=AOI_COLORS[i],
        )
        axs[1, 1].legend()
        axs[1, 1].grid(True)
        axs[1, 1].set_title("div: sig / ref")
        axs[1, 1].set_xlabel("Sweep parameter")
        axs[1, 1].set_ylabel("pl (a.u.)")

        # plot true-sub norm
        axs[1, 2].plot(
            sweep_arr,
            true_sub_norms[i],
            label="AOI " + str(i),
            c=AOI_COLORS[i],
            ls=linestyles[i],
            marker="o",
            mfc="w",
            mec=AOI_COLORS[i],
        )
        axs[1, 2].legend()
        axs[1, 2].grid(True)
        axs[1, 2].set_title("True-sub: (sig - ref) / mx")
        axs[1, 2].set_xlabel("Sweep parameter")
        axs[1, 2].set_ylabel("pl (a.u.)")

    # delete axes that we didn't use
    for i in range(len(aois)):
        if i < 3:  # we used these (3 normalisation types)
            continue
        else:  # we didn't use these
            fig.delaxes(axs[1, i])

    if len(aois) == 2:
        fig.delaxes(axs[0, 2])
    if len(aois) == 1:
        fig.delaxes(axs[0, 1])
        fig.delaxes(axs[0, 2])

    output_dict = {"sweep_arr": sweep_arr}
    for i in range(len(aois)):
        output_dict["AOI_sig_avg" + "_" + str(i)] = sig_avgs[i]
        output_dict["AOI_ref_avg" + "_" + str(i)] = ref_avgs[i]

    if specpath:
        dukit.json2dict.dict_to_json(
            output_dict,
            specpath,
        )

    if opath:
        fig.savefig(opath)

    return fig, axs


# ============================================================================


def aoi_spectra_fit(
    aoi_fit_results: dict,
    roi_fit_results: dict,
    img_shape: tuple[int, int],
    *aoi_coords: tuple[int, int, int, int],
    opath: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots sig and ref spectra, sub and div normalisation and fit for the ROI average,
    a single pixel, and each of the AOIs. Stacked on top of each other for comparison.
    The ROI average fit is plot against the fit of all of the others for comparison.

    Note here and elsewhere the single pixel check is the first element of the AOI array.

    Arguments
    ---------
    aoi_fit_results : dict
        dict["AOI_num"] => dict["fit_backend"] => dukit.share.AoiFit
    roi_fit_results : dict
        dict["fit_backend"] => dukit.share.RoiAvgFit
    aoi_coords : tuple[int, int, int, int]
        AOI coordinates
    img_shape : tuple[int, int]
        Shape of the image, used to get the single pixel AOI.
    opath : str
        If given, save figure here.

    Returns
    -------
    fig : plt.Figure

    ax : plt.Axes

    """

    # rows:
    # ROI avg then each AOI
    # columns:
    # sig & ref, sub & div norm, fit -> compared to ROI {raw, fit, ROI_avg_fit}

    figsize = mpl.rcParams["figure.figsize"].copy()
    figsize[0] *= 3  # number of columns
    figsize[1] *= 1 + len(aoi_fit_results.keys())  # number of rows

    fig, axs = plt.subplots(
        1 + len(aoi_fit_results.keys()),
        3,
        figsize=figsize,
        sharex=True,
        sharey=False,
    )
    axs[-1, 0].set_xlabel("Sweep parameter")
    axs[-1, 1].set_xlabel("Sweep parameter")
    axs[-1, 2].set_xlabel("Sweep parameter")

    fit_results = {"ROI": roi_fit_results}
    fit_results.update(aoi_fit_results)
    for i, name in enumerate(fit_results.keys()):
        fr = next(iter(fit_results[name].values()))  # any fit res, just to plot raw data
        # === plot sig, ref data as first column
        # plot sig
        axs[i, 0].plot(
            fr.sweep_arr,
            fr.avg_sig,
            label="sig",
            c="blue",
            ls="dashed",
            marker="o",
            mfc="cornflowerblue",
            mec="mediumblue",
        )
        # plot ref
        axs[i, 0].plot(
            fr.sweep_arr,
            fr.avg_ref,
            label="ref",
            c="green",
            ls="dashed",
            marker="o",
            mfc="limegreen",
            mec="darkgreen",
        )

        axs[i, 0].legend()
        axs[i, 0].grid(True)
        axs[i, 0].set_title(name, color=(["k"] + AOI_COLORS)[i])
        axs[i, 0].set_ylabel("PL (a.u.)")

        # === plot normalisation as second column
        axs[i, 1].plot(
            fr.sweep_arr,
            1 + (fr.avg_sig - fr.avg_ref) / (fr.avg_sig + fr.avg_ref),
            label="subtraction",
            c="firebrick",
            ls="dashed",
            marker="o",
            mfc="lightcoral",
            mec="maroon",
        )
        axs[i, 1].plot(
            fr.sweep_arr,
            fr.avg_sig / fr.avg_ref,
            label="division",
            c="cadetblue",
            ls="dashed",
            marker="o",
            mfc="powderblue",
            mec="darkslategrey",
        )

        axs[i, 1].legend()
        axs[i, 1].grid(True)
        axs[i, 1].set_title(name + " - norm.", color=(["k"] + AOI_COLORS)[i])
        axs[i, 1].set_ylabel("PL (a.u.)")

        axs[i, 2].plot(
            fr.sweep_arr,
            fr.avg_sig_norm,
            label="raw data",
            ls="",
            marker="o",
            ms=3.5,
            mfc="goldenrod",
            mec="k",
        )
        axs[i, 2].set_title(name + " - fit", color=(["k"] + AOI_COLORS)[i])

        # === now plot fits
        for fit_backend in fit_results[name].keys():
            fit_result = fit_results[name][fit_backend]

            # plot ROI for comparison
            if i and name != "ROI":
                axs[i, 2].plot(
                    fit_results["ROI"][fit_backend].fit_xvec,
                    fit_results["ROI"][fit_backend].fit_yvec,
                    label=f"ROI fit - {fit_backend}",
                    ls="dashed",
                    c=FIT_BACKEND_COLORS[fit_backend]["AOI_ROI_fit_linecolor"],
                )

            axs[i, 2].plot(
                fit_result.fit_xvec,
                fit_result.fit_yvec,
                label=f"{name} fit - {fit_result.fit_backend}",
                ls="dashed",
                c=FIT_BACKEND_COLORS[fit_backend][
                    f"AOI_{'ROI' if name == 'ROI' else 'best'}_fit_linecolor"
                ],
            )
            axs[i, 2].legend()
            axs[i, 2].grid(True)
            axs[i, 2].set_ylabel("PL (a.u.)")

    if opath:
        fig.savefig(opath)

    return fig, axs


# ============================================================================


def pl_param_image(
    fit_model: dukit.pl.FitModel,
    pixel_fit_params: dict,
    param_name: str,
    param_number: int = 0,
    c_range_type: str = "",
    c_range_values: tuple[float, float] = (),
    errorplot: bool = False,
    opath: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots an image corresponding to a single parameter in pixel_fit_params.

    Arguments
    ---------
    options : dict
        Generic options dict holding all the user options.
    fit_model : dukit.pl.model.FitModel
        Model we're fitting to.
    pixel_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    param_name : str
        Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'.

    Optional arguments
    ------------------
    c_range_type : str = ""
        Type of colormap range to use. See `dukit.itool.get_colormap_range`
    c_range_values : tuple[float, float] = ()
        Values to use for colormap range. See `dukit.itool.get_colormap_range`
    param_number : int, default-0
        Which version of the parameter you want. I.e. there might be 8 independent
         parameters in the fit model called 'pos', each labeled 'pos_0', 'pos_1' etc.
    errorplot : bool
        Default: false. Denotes that errors dict has been passed in (e.g. sigmas), so
        ylabel & save names are changed accordingly.
        Can't be True if param_name='residual'.
    opath : str
        If given, tries to save figure here.
    **kwargs
        Plotting options passed to `dukit.itool.plot_image_on_ax`

    Returns
    -------
    fig : matplotlib Figure object
    """

    image = pixel_fit_params[param_name + "_" + str(param_number)]
    if param_name == "residual" and errorplot:
        dukit.warn.warn("residual doesn't have an error, can't plot residual sigma (ret. None).")
        return None

    if errorplot:
        c_label = "SD: " + fit_model.get_param_unit(param_name, param_number)
    else:
        c_label = fit_model.get_param_unit(param_name, param_number)

    if "c_range" not in kwargs and c_range_type and c_range_values:
        c_range = dukit.itool.get_colormap_range(c_range_type, c_range_values, image)
    else:
        c_range = dukit.itool.get_colormap_range("min_max", (), image)

    fig, ax = plt.subplots()
    fig, ax = dukit.itool.plot_image_on_ax(
        fig,
        ax,
        image,
        title=param_name + "_" + str(param_number),
        c_range=c_range,
        c_label=c_label,
        **kwargs,
    )

    if opath:
        fig.savefig(opath)

    return fig, ax


# ============================================================================


def pl_param_images(
    fit_model: dukit.pl.FitModel,
    pixel_fit_params: dict,
    param_name: str,
    c_range_type: str = "",
    c_range_values: tuple[float, float] = (),
    errorplot: bool = False,
    opath: str = "",
    **kwargs,
):
    """
    Plots images for all independent versions of a single parameter type in
    pixel_fit_params.

    Arguments
    ---------
    fit_model : `qdmpy.pl.model.FitModel`
        Model we're fitting to.
    pixel_fit_params : dict
        Dictionary, key: param_keys, val: image (2D) of param values across FOV.
    param_name : str
        Name of parameter you want to plot, e.g. 'fwhm'. Can also be 'residual'.
    c_range_type : str = ""
        Type of colormap range to use. See `dukit.itool.get_colormap_range`
    c_range_values : tuple[float, float] = ()
        Values to use for colormap range. See `dukit.itool.get_colormap_range`
    errorplot : bool
        Default: false. Denotes that errors dict has been passed in (e.g. sigmas), so
        ylabel & save names are changed accordingly.
        Can't be True if param_name='residual'.
    opath : str
        If given, tries to save figure here.
    **kwargs
        Plotting options passed to `dukit.itool.plot_image_on_ax`

    Returns
    -------
    fig : plt.Figure

    ax : plt.Axes
    """

    # if no fit completed
    if pixel_fit_params is None:
        dukit.warn.warn(
            "'pixel_fit_params' arg to function 'pl_param_images' is 'None'.\n"
            + "Probably no pixel fitting completed."  # noqa: W503
        )
        return None

    if param_name == "residual" and errorplot:
        dukit.warn.warn("residual doesn't have an error, can't plot residual sigma (ret. None).")
        return None

    # plot 2 columns wide, as many rows as required

    # first get keys we need
    our_keys = []
    for key in pixel_fit_params:
        if key.startswith(param_name) and not key.endswith("sigma"):
            our_keys.append(key)

    # this is an inner function so no one uses it elsewhere/protect namespace
    def param_sorter(param):
        strings = param.split("_")  # e.g. "amp_exp_2" -> ["amp", "exp", "2"]
        # all we need here is the number at the end actually
        # param_name_split = strings[:-1]  # list of 'words', e.g. ["amp", "exp"]
        num = strings[-1]  # grab the number e.g. "2
        return int(num)

    # sort based on number (just in case)
    our_keys.sort(key=param_sorter)  # i.e. key = lambda x: int(x.split("_")[-1])
    nk = len(our_keys)

    if nk == 1:
        # just one image, so plot normally
        fig, axs = pl_param_image(
            fit_model,
            pixel_fit_params,
            param_name,
            0,
            c_range_type,
            c_range_values,
            errorplot,
        )
    else:
        if nk <= 8:
            num_columns = 4
            num_rows = 2
        else:
            num_columns = 2
            num_rows = math.ceil(nk / 2)

        figsize = mpl.rcParams["figure.figsize"].copy()
        figsize[0] *= num_columns
        figsize[1] *= num_rows

        fig, axs = plt.subplots(
            num_rows,
            num_columns,
            figsize=figsize,
            sharex=False,
            sharey=False,
        )
        # plot 8-lorentzian peaks in a more helpful way
        if nk <= 8 and fit_model in [
            dukit.pl.ConstLorentzians,
            dukit.pl.LinearLorentzians,
        ]:
            param_nums = []  # [0, 1, 2, 3, 7, 6, 5, 4] etc.
            param_nums.extend(list(range(nk // 2)))
            if nk % 2:
                param_nums.append(nk // 2 + 1)
            if len(param_nums) < 4:
                param_nums.extend([-1 for _ in range(4 - len(param_nums))])  # dummies
            param_nums.extend(list(range(nk - 1, (nk - 1) // 2, -1)))  # range(start, stop, step)
            param_nums.extend([-1 for _ in range(8 - len(param_nums))])  # add on dummies
            param_axis_iterator = zip(param_nums, axs.flatten())
        # otherwise plot in a more conventional order
        else:
            param_axis_iterator = enumerate(axs.flatten())  # type: ignore

        for param_number, ax in param_axis_iterator:
            param_key = param_name + "_" + str(param_number)
            try:
                image_data = pixel_fit_params[param_key]
            except KeyError:
                # we have too many axes (i.e. 7 params, 8 subplots), delete the axs
                fig.delaxes(ax)
                continue

            if errorplot:
                c_label = "SD: " + fit_model.get_param_unit(param_name, param_number)
            else:
                c_label = fit_model.get_param_unit(param_name, param_number)

            if "c_range" not in kwargs and c_range_type and c_range_values:
                c_range = dukit.itool.get_colormap_range(c_range_type, c_range_values, image_data)
            else:
                c_range = dukit.itool.get_colormap_range("min_max", (), image_data)

            fig, ax = dukit.itool.plot_image_on_ax(
                fig,
                ax,
                image_data,
                title=param_key,
                c_range=c_range,
                c_label=c_label,
                **kwargs,
            )

        if opath:
            fig.savefig(opath)

    return fig, axs


# ============================================================================


def _add_patch_rect(
    ax: plt.Axes,
    aoi_coord: tuple[int, int, int, int],
    label: str | None = None,
    edgecolor: str = "b",
) -> None:
    """
    Adds a rectangular annotation onto ax.

    Arguments
    ---------
    ax : matplotlib Axis object

    aoi_coord: tuple[int, int, int, int]
        start_x, start_y, end_x, end_y for aoi/roi

    Optional arguments
    ------------------
    label : str
        Text to label annotated square with. Color is defined by edgecolor. Default: None.
    edgecolor : str
        Color of label and edge of annotation. Default: "b".
    """
    start_x, start_y, end_x, end_y = aoi_coord
    start_x = 0 if start_x < 0 else start_x
    start_y = 0 if start_y < 0 else start_y
    end_x = ax.get_xlim()[1] if end_x < 0 else end_x
    end_y = ax.get_ylim()[1] if end_y < 0 else end_y
    rect = patches.Rectangle(
        (start_x, start_y),
        int(end_x - start_x),
        int(end_y - start_y),
        linewidth=1,
        edgecolor=edgecolor,
        facecolor="none",
    )
    ax.add_patch(rect)
    if label:
        ax.text(
            start_x + 0.95 * int(end_x - start_x),  # label posn.: top right
            start_y,
            label,
            {
                "color": edgecolor,
                "fontsize": 10,
                "ha": "center",
                "va": "bottom",
            },
        )


# ============================================================================


def b_defects(
    b_defects: tuple[npt.ArrayLike],
    name: str = "",
    c_range_type: str = "",
    c_range_values: tuple[float, float] = (),
    opath: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the b_defects.

    Parameters
    ----------
    b_defects : tuple[npt.ArrayLike]
        b_defect images or values.
    name : str = ""
        title etc.
    c_range_type : str = ""
        Type of colormap range to use. See `dukit.itool.get_colormap_range`
    c_range_values : tuple[float, float] = ()
        Values to use for colormap range. See `dukit.itool.get_colormap_range`
    opath : str = ""
        If given, save figure here.
    **kwargs
        Plotting options passed to `dukit.itool.plot_image_on_ax`

    Returns
    -------
    fig : plt.Figure

    ax : plt.Axes
    """

    figsize = mpl.rcParams["figure.figsize"].copy()
    width = len(b_defects)
    figsize[0] *= width  # number of columns

    fig, axs = plt.subplots(ncols=width, figsize=figsize)

    if "c_map" in kwargs:
        c_map = kwargs["c_map"]
        del kwargs["c_map"]
    else:
        c_map = "RdBu_r"

    # axs index: axs[row, col]
    for i, bd in enumerate(b_defects):
        if "c_range" not in kwargs and c_range_type and c_range_values:
            c_range = dukit.itool.get_colormap_range(c_range_type, c_range_values, bd)
        else:
            c_range = dukit.itool.get_colormap_range("min_max", (), bd)

        if width == 1:
            ax = axs
        else:
            ax = axs[i]
        dukit.itool.plot_image_on_ax(fig, ax, bd, name, c_map, c_range, "B (T)", **kwargs)

    if opath:
        fig.savefig(opath)
    return fig, axs


# ============================================================================


def dshifts(
    dshifts: tuple[npt.ArrayLike],
    name: str = "",
    c_range_type: str = "",
    c_range_values: tuple[float, float] = (),
    opath: str = "",
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the b_defects.

    Parameters
    ----------
    dshifts : tuple[npt.ArrayLike]
        b_defect images or values.
    name : str = ""
        title etc.
    c_range_type : str = ""
        Type of colormap range to use. See `dukit.itool.get_colormap_range`
    c_range_values : tuple[float, float] = ()
        Values to use for colormap range. See `dukit.itool.get_colormap_range`
    opath : str = ""
        If given, save figure here.
    **kwargs
        Plotting options passed to `dukit.itool.plot_image_on_ax`

    Returns
    -------
    fig : plt.Figure

    ax : plt.Axes
    """

    figsize = mpl.rcParams["figure.figsize"].copy()
    width = len(dshifts)
    figsize[0] *= width  # number of columns

    fig, axs = plt.subplots(ncols=width, figsize=figsize)

    if "c_map" in kwargs:
        c_map = kwargs["c_map"]
        del kwargs["c_map"]
    else:
        c_map = "PRGn"

    # axs index: axs[row, col]
    for i, dshift in enumerate(dshifts):
        if "c_range" not in kwargs and c_range_type and c_range_values:
            c_range = dukit.itool.get_colormap_range(c_range_type, c_range_values, dshift)
        else:
            c_range = dukit.itool.get_colormap_range("min_max", (), dshift)

        if width == 1:
            ax = axs
        else:
            ax = axs[i]
        dukit.itool.plot_image_on_ax(fig, ax, dshift, name, c_map, c_range, "D (MHz)", **kwargs)

    if opath:
        fig.savefig(opath)
    return fig, axs
