import dukit
import os
import matplotlib.pyplot as plt
import numpy as np
import logging

# this script assumes bunch of things in the default kwargs,
# e.g. that you want a div normalisation.

with open(
    os.path.dirname(str(__file__)) + "/../TEST_DATA_PATH.py", encoding="utf-8"
) as fid:
    exec(fid.read())  # reads in TEST_DATA_PATH string

# === SET PARAMS

DIR = TEST_DATA_PATH + "mz_test/"  # type: ignore
FILEPATH = DIR + "ODMR - Pulsed_10"
FIG_FORMAT = "png"

# ADDITIONAL_BINS = (4, 2) # asymmetric: (x, y)
ADDITIONAL_BINS = 4
ADDITIONAL_SMOOTH = 0  # can be asymmetric, but best not.

ROI_COORDS = (65, 65, 190, 190)  # start_x, start_y, ... for 4x binning
# ROI_COORDS = (-1, -1, -1, -1) # full ROI
AOI_COORDS = ((30, 40, 40, 45), (20, 20, 24, 24), (50, 50, 51, 51))
# AOI_COORDS = ((30, 40 , 40, 45),)

# to see how to create these see mz_draw_polys.py
POLY_PATH = FILEPATH + "_oldpolys.json"  # for 4x binning & ROI crop above
polygon_nodes = dukit.load_polygon_nodes(POLY_PATH)
ANNOTATE_POLYS = True

FIT_BACKEND = "scipyfit"
FIT_MODEL = dukit.LinearLorentzians(2)
GUESSES = {"pos": [2720, 3020], "amp": -0.004, "fwhm": 20, "c": 1.0, "m": 0.0}
BOUNDS = {
    "pos_range": 30,
    "amp_range": 0.1,
    "fwhm_range": 50,
    "c_range": 0.1,
    "m_range": 0.01,
}

# === CREATE OUTPUT DIR & set mpl rcparams
OUTPUT_DIR = f"{FILEPATH}_output/"
try:
    os.mkdir(OUTPUT_DIR)
except FileExistsError:
    pass
dukit.mpl_set_run_config()
logging.info(dukit.__version__)
with open(OUTPUT_DIR + "dukit_version.txt", "w") as fid:
    fid.write(dukit.__version__)

FIT_RES_DIR = OUTPUT_DIR + "/data/"
# set below to "" or None or False to *not* load prev fit
# PREV_FIT = ""  # FIT_RES_DIR[:]
PREV_FIT = FIT_RES_DIR[:]

# === START SCRIPT

# === READ IN DATA
sys = dukit.CryoWidefield()
sweep_arr = sys.read_sweep_arr(FILEPATH)
sig, ref, sig_norm = sys.read_image(FILEPATH)
raw_pixel_size = sys.get_raw_pixel_size(FILEPATH)

# === SMOOTH
if ADDITIONAL_SMOOTH:
    sig = dukit.smooth_image_stack(sig, ADDITIONAL_SMOOTH)
    ref = dukit.smooth_image_stack(ref, ADDITIONAL_SMOOTH)
    sig_norm = sys.norm(sig, ref)

# === REBIN & CROP
sig_rebinned = dukit.rebin_image_stack(sig, ADDITIONAL_BINS)
pl_img = dukit.sum_spatially(sig_rebinned)
sig = dukit.crop_roi(sig_rebinned, ROI_COORDS)
del sig_rebinned  # hacky but I want to keep the memory clear
ref = dukit.crop_roi(dukit.rebin_image_stack(ref, ADDITIONAL_BINS), ROI_COORDS)
sig_norm = dukit.crop_roi(
    dukit.rebin_image_stack(sig_norm, ADDITIONAL_BINS), ROI_COORDS
)
pl_img_crop = dukit.sum_spatially(sig)

# === PLOT PL INFO
_ = dukit.plot.roi_pl_image(
    pl_img,
    ROI_COORDS,
    opath=OUTPUT_DIR + f"pl_full.{FIG_FORMAT}",
    show_tick_marks=True,
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
)
_ = dukit.plot.aoi_pl_image(
    pl_img_crop,
    *AOI_COORDS,
    opath=OUTPUT_DIR + f"pl_full.{FIG_FORMAT}",
    show_tick_marks=True,
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
)
_ = dukit.plot.aoi_spectra(
    sig,
    ref,
    sweep_arr,
    specpath=OUTPUT_DIR + "aoi_specta.json",
    opath=OUTPUT_DIR + f"aoi_spectra.{FIG_FORMAT}",
    *AOI_COORDS,
)

# fit_roi,  roi_avg_fits
roi_fit_results = dukit.fit_roi(
    sig,
    ref,
    sweep_arr,
    FIT_MODEL,
    GUESSES,
    BOUNDS,
    opath=OUTPUT_DIR + f"roi_avg_fit.json",
)
_ = dukit.plot.roi_avg_fits(
    roi_fit_results, opath=OUTPUT_DIR + f"roi_avg_fits.{FIG_FORMAT}"
)

# fit_aois, aoi_avg_fits
aoi_fit_results = dukit.fit_aois(
    sig,
    ref,
    sweep_arr,
    FIT_MODEL,
    GUESSES,
    BOUNDS,
    *AOI_COORDS,
    opath=OUTPUT_DIR + f"aoi_fits.json",
)
_ = dukit.plot.aoi_spectra_fit(
    aoi_fit_results,
    roi_fit_results,
    sig.shape[:-1],
    *AOI_COORDS,
    opath=OUTPUT_DIR + f"aoi_spectra_fit.{FIG_FORMAT}",
)

# fit_all_pixels_pl
if PREV_FIT:
    fit_image_results = dukit.load_fit_results(PREV_FIT, FIT_MODEL)
else:
    fit_image_results = dukit.fit_all_pixels(
        FIT_BACKEND,
        sig_norm,
        sweep_arr,
        FIT_MODEL,
        GUESSES,
        BOUNDS,
        roi_fit_results[FIT_BACKEND],
        odir=FIT_RES_DIR,
    )

# pl_param_images
_ = dukit.plot.pl_param_images(
    FIT_MODEL,
    fit_image_results,
    "pos",
    opath=OUTPUT_DIR + f"pl_pos.{FIG_FORMAT}",
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
)

_ = dukit.plot.pl_param_images(
    FIT_MODEL,
    fit_image_results,
    "sigma_pos",
    opath=OUTPUT_DIR + f"sigma_pl_pos.{FIG_FORMAT}",
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
    errorplot=True,
)

defect = dukit.NVEnsemble()
# resonances = (fit_image_results["pos_0"], fit_image_results["pos_1"])
resonances = dukit.get_fitres_params(fit_image_results, "pos")
b_nvs = defect.b_defects(resonances)
dshifts = defect.dshift_defects(resonances)
# calc b_nv_bg, b_nv_sbg
# b_nv_bgs = tuple([dukit.get_background(bnv, "poly", order=2)[0] for bnv in b_nvs])
b_nv_bgs = tuple(
    [
        dukit.get_background(
            bnv,
            "interpolate",
            polygon_nodes=polygon_nodes,
            interp_method="linear",
            sigma=0.6,
        )[0]
        for bnv in b_nvs
    ]
)
b_nv_sbg = tuple([bnv - bnv_bg for bnv, bnv_bg in zip(b_nvs, b_nv_bgs)])
for i, b in enumerate(b_nv_sbg):
    np.savetxt(OUTPUT_DIR + f"/data/b_nv_sbg_{i}.txt", b)


_ = dukit.plot.b_defects(
    b_nvs,
    name="raw b_nvs",
    opath=OUTPUT_DIR + f"raw_b_nvs.{FIG_FORMAT}",
    c_range_type="percentile",
    c_range_values=(2, 98),
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
)
_ = dukit.plot.b_defects(
    b_nv_sbg,
    name="sub_bg b_nvs",
    opath=OUTPUT_DIR + f"b_nvs-sub_bg.{FIG_FORMAT}",
    # c_range_type="percentile",
    c_range_type="strict_range",
    # c_range_values=(2, 98),
    c_range_values=(-1.3e-6, 1.3e-6),
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
    annotate_polygons=ANNOTATE_POLYS,
    polygon_nodes=polygon_nodes,
)

_ = dukit.plot.dshifts(
    dshifts,
    name="dshifts",
    opath=OUTPUT_DIR + f"dshifts.{FIG_FORMAT}",
    # c_range_type="percentile",
    c_range_type="strict_range",
    # c_range_values=(2, 98),
    c_range_values=(2876, 2880),
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
)

# === FIELD RECONSTRUCTION SECTION ===
# Get bias field information
bias_on, (mag_T, theta_rad, phi_rad) = sys.get_bias_field(FILEPATH, auto_read=True)
bias_field = (mag_T, np.degrees(theta_rad), np.degrees(phi_rad))

# Get defect orientations (4 NV axes) for the given bias field
bias_xyz = dukit.field.spherical_to_cartesian(*bias_field)
u_defects = dukit.geom.get_u_defects(*bias_xyz, diamond_ori="<100>_<110>")

# Reconstruct vector field using Hamiltonian method (most accurate)
print("\n=== Reconstructing vector field using Hamiltonian method ===")
bxyz_ham = dukit.field.get_bxyz_from_hamiltonian(
    resonances,
    defect,
    u_defects,
    bias_field,
    n_jobs=-2,  # Use all but one CPU core
    method="trf",
    gtol=1e-12,
    xtol=1e-12,
    ftol=1e-12,
    loss="linear",
    progress_bar=True
)

# Also reconstruct using matrix inversion for comparison
print("\n=== Reconstructing vector field using matrix inversion ===")
bxyz_inv = dukit.field.get_bxyz_from_bdefects_inversion(
    b_nv_sbg,  # Use background-subtracted B_defects
    u_defects,
    bias_field
)

# Save reconstructed field data
print("\n=== Saving reconstructed field data ===")
np.savetxt(OUTPUT_DIR + "/data/Bx_hamiltonian.txt", bxyz_ham["Bx"])
np.savetxt(OUTPUT_DIR + "/data/By_hamiltonian.txt", bxyz_ham["By"])
np.savetxt(OUTPUT_DIR + "/data/Bz_hamiltonian.txt", bxyz_ham["Bz"])
np.savetxt(OUTPUT_DIR + "/data/D_parameter.txt", bxyz_ham["D"])
np.savetxt(OUTPUT_DIR + "/data/sigma_Bx.txt", bxyz_ham["sigma_Bx"])
np.savetxt(OUTPUT_DIR + "/data/sigma_By.txt", bxyz_ham["sigma_By"])
np.savetxt(OUTPUT_DIR + "/data/sigma_Bz.txt", bxyz_ham["sigma_Bz"])

np.savetxt(OUTPUT_DIR + "/data/Bx_inversion.txt", bxyz_inv["Bx"])
np.savetxt(OUTPUT_DIR + "/data/By_inversion.txt", bxyz_inv["By"])
np.savetxt(OUTPUT_DIR + "/data/Bz_inversion.txt", bxyz_inv["Bz"])

# Save metadata
import json
with open(OUTPUT_DIR + "/data/reconstruction_metadata.json", "w") as f:
    metadata = {
        "bias_field_T": bias_field[0],
        "bias_field_theta_deg": bias_field[1],
        "bias_field_phi_deg": bias_field[2],
        "bias_field_xyz_T": bias_xyz.tolist(),
        "u_defects": u_defects.tolist(),
        "hamiltonian_metadata": bxyz_ham["_metadata"],
        "inversion_metadata": bxyz_inv["_metadata"],
    }
    json.dump(metadata, f, indent=2)

# === FIELD CONSISTENCY CHECK ===
print("\n=== Performing field consistency check ===")

# Reconstruct components from Hamiltonian result to check consistency
# This verifies that the field satisfies div(B) = 0 and that geometry/u_defects are correct
print("\nReconstructing field components using div(B) = 0 constraint...")
field_check = dukit.field.reconstruct_field_components(
    (bxyz_ham["Bx"], bxyz_ham["By"], bxyz_ham["Bz"]),
    pixel_size=raw_pixel_size * ADDITIONAL_BINS * 1e-9,  # Convert nm to m
    pad_mode="edge",
    pad_factor=2,
    k_vector_epsilon=1e-6,
    nv_above_sample=True,
)

# Plot consistency check results - reconstructed components
_ = dukit.plot.plot_field_images(
    {
        "Bx_from_Bz": field_check["Bx_from_Bz"],
        "By_from_Bz": field_check["By_from_Bz"],
        "Bz_from_xy": field_check["Bz_from_xy"],
    },
    name="Field Consistency Check - Reconstructed Components",
    c_range_type="percentile",
    c_range_values=(2, 98),
    c_label="Reconstructed Field (G)",
    c_map="RdBu_r",
    opath=OUTPUT_DIR + f"field_consistency_check.{FIG_FORMAT}",
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
)

# Create detailed comparison plots: measured vs reconstructed
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Field Consistency Check: Measured vs Reconstructed Components", fontsize=14)

# Use same color scale for fair comparison
vmin = min(np.nanmin(bxyz_ham["Bx"]), np.nanmin(bxyz_ham["By"]), np.nanmin(bxyz_ham["Bz"]),
            np.nanmin(field_check["Bx_from_Bz"]), np.nanmin(field_check["By_from_Bz"]), np.nanmin(field_check["Bz_from_xy"]))
vmax = max(np.nanmax(bxyz_ham["Bx"]), np.nanmax(bxyz_ham["By"]), np.nanmax(bxyz_ham["Bz"]),
            np.nanmax(field_check["Bx_from_Bz"]), np.nanmax(field_check["By_from_Bz"]), np.nanmax(field_check["Bz_from_xy"]))

# First row: Measured fields
im1 = axes[0, 0].imshow(bxyz_ham["Bx"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[0, 0].set_title("Measured Bx")
plt.colorbar(im1, ax=axes[0, 0])

im2 = axes[0, 1].imshow(bxyz_ham["By"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[0, 1].set_title("Measured By")
plt.colorbar(im2, ax=axes[0, 1])

im3 = axes[0, 2].imshow(bxyz_ham["Bz"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[0, 2].set_title("Measured Bz")
plt.colorbar(im3, ax=axes[0, 2])

# Second row: Reconstructed fields
im4 = axes[1, 0].imshow(field_check["Bx_from_Bz"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[1, 0].set_title("Bx reconstructed from Bz\n(using div(B)=0)")
plt.colorbar(im4, ax=axes[1, 0])

im5 = axes[1, 1].imshow(field_check["By_from_Bz"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[1, 1].set_title("By reconstructed from Bz\n(using div(B)=0)")
plt.colorbar(im5, ax=axes[1, 1])

im6 = axes[1, 2].imshow(field_check["Bz_from_xy"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[1, 2].set_title("Bz reconstructed from Bx,By\n(using div(B)=0)")
plt.colorbar(im6, ax=axes[1, 2])

plt.tight_layout()
plt.savefig(OUTPUT_DIR + f"field_consistency_comparison.{FIG_FORMAT}", dpi=150, bbox_inches="tight")
plt.close()

# Calculate and plot differences
diff_Bx = bxyz_ham["Bx"] - field_check["Bx_from_Bz"]
diff_By = bxyz_ham["By"] - field_check["By_from_Bz"]
diff_Bz = bxyz_ham["Bz"] - field_check["Bz_from_xy"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Field Consistency Check: Differences (Measured - Reconstructed)", fontsize=14)

diff_max = max(np.abs(diff_Bx).max(), np.abs(diff_By).max(), np.abs(diff_Bz).max())

im1 = axes[0].imshow(diff_Bx, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
axes[0].set_title(f"ΔBx = Bx - Bx_from_Bz")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(diff_By, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
axes[1].set_title(f"ΔBy = By - By_from_Bz")
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(diff_Bz, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
axes[2].set_title(f"ΔBz = Bz - Bz_from_xy")
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(OUTPUT_DIR + f"field_consistency_differences.{FIG_FORMAT}", dpi=150, bbox_inches="tight")
plt.close()

# Save consistency check data
np.savetxt(OUTPUT_DIR + "/data/field_consistency_Bx_diff.txt", diff_Bx)
np.savetxt(OUTPUT_DIR + "/data/field_consistency_By_diff.txt", diff_By)
np.savetxt(OUTPUT_DIR + "/data/field_consistency_Bz_diff.txt", diff_Bz)

# Also save the reconstructed components for reference
np.savetxt(OUTPUT_DIR + "/data/field_consistency_Bx_from_Bz.txt", field_check["Bx_from_Bz"])
np.savetxt(OUTPUT_DIR + "/data/field_consistency_By_from_Bz.txt", field_check["By_from_Bz"])
np.savetxt(OUTPUT_DIR + "/data/field_consistency_Bz_from_xy.txt", field_check["Bz_from_xy"])


# === PLOT RECONSTRUCTED VECTOR FIELDS ===
print("\n=== Plotting reconstructed vector fields ===")

# Plot Hamiltonian reconstruction results (Bx, By, Bz)
_ = dukit.plot.plot_field_images(
    {"Bx": bxyz_ham["Bx"], "By": bxyz_ham["By"], "Bz": bxyz_ham["Bz"]},
    name="Hamiltonian Reconstruction",
    c_range_type="percentile",
    c_range_values=(2, 98),
    c_label="Magnetic Field (G)",
    c_map="RdBu_r",
    opath=OUTPUT_DIR + f"Bxyz_hamiltonian.{FIG_FORMAT}",
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
)

# Plot uncertainties from Hamiltonian fit
_ = dukit.plot.plot_field_images(
    {"sigma_Bx": bxyz_ham["sigma_Bx"], "sigma_By": bxyz_ham["sigma_By"], "sigma_Bz": bxyz_ham["sigma_Bz"]},
    name="Field Uncertainties",
    c_range_type="percentile",
    c_range_values=(50, 98),
    c_label="Uncertainty (G)",
    c_map="viridis",
    opath=OUTPUT_DIR + f"field_uncertainties.{FIG_FORMAT}",
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
)

# Plot D parameter
_ = dukit.plot.plot_field_images(
    {"D": bxyz_ham["D"]},
    name="Zero-Field Splitting",
    c_range_type="percentile",
    c_range_values=(1, 99),
    c_label="D (MHz)",
    c_map="viridis",
    opath=OUTPUT_DIR + f"D_parameter.{FIG_FORMAT}",
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
)

# Plot residual field
_ = dukit.plot.plot_field_images(
    {"residual_field": bxyz_ham["residual_field"]},
    name="Fit Residuals",
    c_range_type="percentile",
    c_range_values=(90, 99.9),
    c_label="Residual (a.u.)",
    c_map="hot",
    opath=OUTPUT_DIR + f"residual_field.{FIG_FORMAT}",
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
)

# Plot matrix inversion results for comparison
_ = dukit.plot.plot_field_images(
    {"Bx": bxyz_inv["Bx"], "By": bxyz_inv["By"], "Bz": bxyz_inv["Bz"]},
    name="Matrix Inversion",
    c_range_type="percentile",
    c_range_values=(2, 98),
    c_label="Magnetic Field (G)",
    c_map="RdBu_r",
    opath=OUTPUT_DIR + f"Bxyz_inversion.{FIG_FORMAT}",
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
)

# Create comparison plots between methods
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Field Reconstruction: Hamiltonian vs Matrix Inversion")

# Use same color scale for fair comparison
vmin = min(np.nanmin(bxyz_ham["Bx"]), np.nanmin(bxyz_inv["Bx"]),
            np.nanmin(bxyz_ham["By"]), np.nanmin(bxyz_inv["By"]),
            np.nanmin(bxyz_ham["Bz"]), np.nanmin(bxyz_inv["Bz"]))
vmax = max(np.nanmax(bxyz_ham["Bx"]), np.nanmax(bxyz_inv["Bx"]),
            np.nanmax(bxyz_ham["By"]), np.nanmax(bxyz_inv["By"]),
            np.nanmax(bxyz_ham["Bz"]), np.nanmax(bxyz_inv["Bz"]))

# Bx comparison
im1 = axes[0, 0].imshow(bxyz_ham["Bx"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[0, 0].set_title("Bx (Hamiltonian)")
plt.colorbar(im1, ax=axes[0, 0])
im2 = axes[1, 0].imshow(bxyz_inv["Bx"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[1, 0].set_title("Bx (Matrix Inversion)")
plt.colorbar(im2, ax=axes[1, 0])

# By comparison
im3 = axes[0, 1].imshow(bxyz_ham["By"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[0, 1].set_title("By (Hamiltonian)")
plt.colorbar(im3, ax=axes[0, 1])
im4 = axes[1, 1].imshow(bxyz_inv["By"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[1, 1].set_title("By (Matrix Inversion)")
plt.colorbar(im4, ax=axes[1, 1])

# Bz comparison
im5 = axes[0, 2].imshow(bxyz_ham["Bz"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[0, 2].set_title("Bz (Hamiltonian)")
plt.colorbar(im5, ax=axes[0, 2])
im6 = axes[1, 2].imshow(bxyz_inv["Bz"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[1, 2].set_title("Bz (Matrix Inversion)")
plt.colorbar(im6, ax=axes[1, 2])

plt.tight_layout()
plt.savefig(OUTPUT_DIR + f"field_reconstruction_comparison.{FIG_FORMAT}", dpi=150, bbox_inches="tight")
plt.close()

# Create difference plots to highlight method differences
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Difference: Hamiltonian - Matrix Inversion")

diff_Bx = bxyz_ham["Bx"] - bxyz_inv["Bx"]
diff_By = bxyz_ham["By"] - bxyz_inv["By"]
diff_Bz = bxyz_ham["Bz"] - bxyz_inv["Bz"]

diff_max = max(np.abs(diff_Bx).max(), np.abs(diff_By).max(), np.abs(diff_Bz).max())

im1 = axes[0].imshow(diff_Bx, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
axes[0].set_title("ΔBx")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(diff_By, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
axes[1].set_title("ΔBy")
plt.colorbar(im2, ax=axes[1])

im3 = axes[2].imshow(diff_Bz, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
axes[2].set_title("ΔBz")
plt.colorbar(im3, ax=axes[2])

plt.tight_layout()
plt.savefig(OUTPUT_DIR + f"field_reconstruction_difference.{FIG_FORMAT}", dpi=150, bbox_inches="tight")
plt.close()


# === SOURCE RECONSTRUCTION SECTION ===
print("\n=== Source Reconstruction (Magnetization) ===")

# Calculate effective pixel size in meters
effective_pixel_size = raw_pixel_size * ADDITIONAL_BINS * 1e-9  # nm -> m
print(f"Effective pixel size: {effective_pixel_size*1e6:.2f} μm")

# Reconstruct out-of-plane magnetization from Hamiltonian Bxyz
# Using Bx and By components (most common and stable method)
print("\nReconstructing Mz from Bx, By (Hamiltonian)...")
mz_result = dukit.source.get_magnetization_from_bxyz(
    bxyz_ham["Bx"],
    bxyz_ham["By"],
    Bz=None,  # Not using Bz for Mz reconstruction
    pixel_size=effective_pixel_size,
    standoff=None,  # Can specify standoff in meters for better accuracy
    use_components="xy",  # Use Bx, By only (most stable for Mz)
    pad_mode="edge",
    pad_factor=2,
)

print(f"Mz reconstruction complete.")
print(f"Mz range: [{np.nanmin(mz_result['Mz']):.2e}, {np.nanmax(mz_result['Mz']):.2e}] μB/nm²")

# Save magnetization data
np.savetxt(OUTPUT_DIR + "/data/Mz.txt", mz_result["Mz"])
with open(OUTPUT_DIR + "/data/magnetization_metadata.json", "w") as f:
    mz_metadata = {
        "Mz_range": [float(np.nanmin(mz_result["Mz"])), float(np.nanmax(mz_result["Mz"]))],
        "pixel_size_m": effective_pixel_size,
        "method": mz_result["_metadata"]["method"],
        "use_components": mz_result["_metadata"]["use_components"],
    }
    json.dump(mz_metadata, f, indent=2)

# Plot magnetization
_ = dukit.plot.plot_magnetization(
    mz_result["Mz"],
    name="Out-of-plane Magnetization (Mz)",
    c_range_type="percentile",
    c_range_values=(2, 98),
    opath=OUTPUT_DIR + f"Mz.{FIG_FORMAT}",
    raw_pixel_size=raw_pixel_size,
    applied_binning=ADDITIONAL_BINS,
    annotate_polygons=ANNOTATE_POLYS,
    polygon_nodes=polygon_nodes,
)

# Also reconstruct from matrix inversion Bxyz for comparison
print("\nReconstructing Mz from Bx, By (Matrix Inversion)...")
mz_inv_result = dukit.source.get_magnetization_from_bxyz(
    bxyz_inv["Bx"],
    bxyz_inv["By"],
    pixel_size=effective_pixel_size,
    use_components="xy",
)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Magnetization Reconstruction Comparison", fontsize=14)

# Use same color scale for comparison
vmin = min(np.nanmin(mz_result["Mz"]), np.nanmin(mz_inv_result["Mz"]))
vmax = max(np.nanmax(mz_result["Mz"]), np.nanmax(mz_inv_result["Mz"]))

im1 = axes[0].imshow(mz_result["Mz"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[0].set_title("Mz (Hamiltonian)")
plt.colorbar(im1, ax=axes[0], label="Mz (μB/nm²)")

im2 = axes[1].imshow(mz_inv_result["Mz"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
axes[1].set_title("Mz (Matrix Inversion)")
plt.colorbar(im2, ax=axes[1], label="Mz (μB/nm²)")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR + f"Mz_comparison.{FIG_FORMAT}", dpi=150, bbox_inches="tight")
plt.close()

plt.show()
