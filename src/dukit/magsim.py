# -*- coding: utf-8 -*-
"""Interface to mag simulations.

TODO needs documenting/cleaning
Basically add some examples HERE of how to use this subpkg.

Apologies for lack of type information, I don't understand everything
sufficiently and it currently *just works*.

Classes
-------
 - `dukit.magsim.MagSim`
 - `dukit.magsim.SandboxMagSim`
 - `dukit.magsim.ComparisonMagSim`
"""
# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.magsim.MagSim": True,
    "dukit.magsim.SandboxMagSim": True,
    "dukit.magsim.ComparisonMagSim": True,
}
# ============================================================================

from collections import defaultdict as dd
from copy import copy

import dill as pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as LA  # noqa: N812
import numpy.typing as npt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyfftw.interfaces import numpy_fft
from scipy.ndimage import gaussian_filter
from tqdm.autonotebook import tqdm

# ============================================================================
import dukit.itool
import dukit.json2dict
import dukit.polygon

# ============================================================================


class MagSim:
    polygon_nodes = None
    mag = None
    template_polygon_nodes = None
    bfield = None
    standoff = None
    magnetizations_lst = None
    unit_vectors_lst = None
    ny: int = 0
    nx: int = 0
    pixel_size: float = float("nan")
    base_image: npt.NDArray  # 2D, needs to be defined by __init__

    def _load_polys(self, polys, check_size=False):
        """polys: either path to json/pickle, or dict containing 'nodes' key."""
        if polys is not None:
            if isinstance(polys, dict):
                if check_size and "image_size" in polys:
                    if self.ny != polys["image_size"][0] or self.nx != polys["image_size"][1]:
                        # TODO massage to match?
                        raise RuntimeError(
                            "Image size polygons were defined on as passed to"
                            " add_polygons does " + "not match this MagSim's mesh."
                        )
                return [np.array(p) for p in polys["nodes"]]
            elif isinstance(polys, str):
                return [np.array(p) for p in self._load_dict(polys)["nodes"]]
            else:
                raise TypeError("polygons argument was not a dict or string?")
        return None

    @staticmethod
    def _load_dict(path):
        if not isinstance(path, str):
            raise ValueError("path was not a str/pathlib.PurePath object.")
        elif path.endswith("json"):
            return dukit.json2dict.json_to_dict(path)
        elif path.endswith("pickle"):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError("path did not end in 'json' or 'pickle'")

    @staticmethod
    def _save_dict(path, dictionary):
        if not isinstance(
            path,
            str,
        ):
            raise ValueError("path was not a str/pathlib.PurePath object.")
        elif path.endswith("json"):
            dukit.json2dict.dict_to_json(dictionary, path)
        elif path.endswith("pickle"):
            with open(path, "wb") as f:
                pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError("path did not end in 'json' or 'pickle'")

    @staticmethod
    def _load_image(image):
        if image is not None:
            if isinstance(image, np.ndarray):
                return image
            elif isinstance(image, str):
                return np.loadtxt(image)
            else:
                raise TypeError("image argument must be an np.ndarray or string?")
        return None

    def _polygon_gui(
        self,
        polygon_nodes=None,
        mean_plus_minus=None,
        base_scale=1.25,
        image=None,
        cmap="bwr",
        prompt="Select polygons",
        **kwargs,
    ):
        fig, ax = plt.subplots()
        image = self.base_image if image is None else image

        if (
            mean_plus_minus is not None
            and isinstance(mean_plus_minus, (float, int))
            and not isinstance(self, SandboxMagSim)
        ):
            mean = np.mean(image)
            vmin, vmax = mean - mean_plus_minus, mean + mean_plus_minus
        else:
            vmin = vmax = None

        img = ax.imshow(image, aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.tick_params(
            axis="x",  # changes apply to the x-axis
            which="both",  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
        )
        ax.tick_params(
            axis="y",  # changes apply to the y-axis
            which="both",  # both major and minor ticks are affected
            left=False,
            right=False,
            labelleft=False,
        )
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax)
        ax.set_title(prompt)

        psw = dukit.polygon.PolygonSelectionWidget(ax, base_scale=base_scale, style=kwargs)

        if polygon_nodes is not None:
            psw.load_nodes(polygon_nodes)
        plt.show(block=True)
        psw.disconnect()

        pgons = psw.get_polygons_lst()
        if len(pgons) < 1:
            raise RuntimeError("You didn't define any polygons")

        pgon_lst = [pgon.get_nodes() for pgon in pgons if np.shape(pgon.get_nodes())[0] > 2]
        output_dict = {"nodes": pgon_lst, "image_shape": (self.ny, self.nx)}

        return output_dict

    def add_polygons(self, polys=None):
        """polygons is dict (polygons directly) or str (path to)"""
        self.polygon_nodes = self._load_polys(polys, check_size=True)

    def select_polygons(
        self,
        polygon_nodes=None,
        output_path=None,
        mean_plus_minus=None,
        **kwargs,
    ):
        """manually select polygons"""
        pgon_dict = self._polygon_gui(
            polygon_nodes=polygon_nodes,
            mean_plus_minus=mean_plus_minus,
            **kwargs,
        )
        if output_path is not None:
            self._save_dict(output_path, pgon_dict)

        self.polygon_nodes = [np.array(p) for p in pgon_dict["nodes"]]

    def save_polygons(self, output_path):
        if output_path is not None:
            self._save_dict(
                output_path,
                {
                    "nodes": self.polygon_nodes,
                    "image_shape": (self.ny, self.nx),
                },
            )

    def define_magnets(self, magnetizations, unit_vectors):
        """
        magnetizations: int/float if the same for all polygons, or an iterable of
            len(polygon_nodes)
            -> in units of mu_b / nm^2 (or mu_b / PX^2 for SandboxMagSim)
        unit_vectors: 3-iterable if the same for all polygons (cartesian coords),
            or an iterable of len(polygon_nodes) each element a 3-iterable
        """
        # todo: do we want to be able to add _noise_ here too? / other imperfections?
        if isinstance(magnetizations, (float, int)):
            self.magnetizations_lst = [magnetizations for m, _ in enumerate(self.polygon_nodes)]
        else:
            if len(magnetizations) != len(self.polygon_nodes):
                raise ValueError(
                    f"Number of magnetizations ({len(magnetizations)}) does not match "
                    + f"number of magnets ({len(self.polygon_nodes)})."
                )
            self.magnetizations_lst = magnetizations

        if isinstance(unit_vectors, (np.ndarray, list, tuple)):
            if len(np.shape(unit_vectors)) == 1:
                if len(unit_vectors) == 3:
                    uv_abs = LA.norm(unit_vectors)
                    self.unit_vectors_lst = [
                        tuple(np.array(unit_vectors) / uv_abs)
                        for m, _ in enumerate(self.polygon_nodes)
                    ]
                else:
                    raise RuntimeError(
                        f"I don't recognise that shape of unit_vectors. ({np.shape(unit_vectors)})"
                    )
            else:
                # ensure unit vectors
                self.unit_vectors_lst = [tuple(np.array(uv) / LA.norm(uv)) for uv in unit_vectors]
        else:
            raise TypeError(
                f"unit_vectors wrong type ({type(unit_vectors)}), not ndarray/list/tuple. :("
            )

        if len(self.magnetizations_lst) != len(self.unit_vectors_lst):
            raise RuntimeError(
                f"magnetizations_lst (len: {len(self.magnetizations_lst)}) "
                + "not the same length as unit_vectors_lst (len:"
                f" {len(self.unit_vectors_lst)}). :("
            )

        # now construct mag
        self.mag = dd(lambda: np.zeros((self.ny, self.nx)))
        grid_y, grid_x = np.meshgrid(range(self.ny), range(self.nx), indexing="ij")

        for i, p in tqdm(
            enumerate(self.polygon_nodes),
            ascii=True,
            mininterval=1,
            unit="polygons",
            desc="defining magnets...",
            total=len(self.polygon_nodes),
        ):
            polygon = dukit.polygon.Polygon(p[:, 0], p[:, 1])
            in_or_out = polygon.is_inside(grid_y, grid_x)
            # 2021-08-04 changed from > 0 -> only defined __inside__ polygon
            self.mag[self.unit_vectors_lst[i]][in_or_out > 0] += self.magnetizations_lst[i]

    def save_magnets(self, output_path):
        output_dict = {
            "mag": self.mag,
            "unit_vectors_lst": self.unit_vectors_lst,
            "magnetizations_lst": self.magnetizations_lst,
        }
        self._save_dict(output_path, output_dict)

    def load_magnets(self, path):
        in_dict = self._load_dict(path)
        self.mag = in_dict["mag"]
        self.unit_vectors_lst = in_dict["unit_vectors_lst"]
        self.magnetizations_lst = in_dict["magnetizations_lst"]

    def run(
        self,
        standoff,
        resolution=None,  # NOTE res is given as a sigma not a fwhm. !should! be fwhm.
        pad_mode="mean",
        pad_factor=2,
        k_vector_epsilon=1e-6,
        nv_layer_thickness=None,
    ):
        """Everything units of metres."""
        self.standoff = standoff
        # in future could be generalised to a range of standoffs
        # e.g. if we wanted to average over an nv-depth distribution that would be easy

        # get shape so we can define kvecs
        shp = dukit.fourier.pad_image(
            np.empty(np.shape(self.mag[self.unit_vectors_lst[0]])),
            pad_mode,
            pad_factor,
        )[0].shape

        ky, kx, k = dukit.fourier.define_k_vectors(
            shp, self.pixel_size, k_vector_epsilon=k_vector_epsilon
        )

        d_matrix = dukit.fourier.define_magnetization_transformation(
            ky, kx, k, standoff=standoff, nv_layer_thickness=nv_layer_thickness
        )

        d_matrix = dukit.fourier.set_naninf_to_zero(d_matrix)

        self.bfield = [
            np.zeros((self.ny, self.nx)),
            np.zeros((self.ny, self.nx)),
            np.zeros((self.ny, self.nx)),
        ]

        # convert to A from mu_b / nm^2 magnetization units
        m_scale = 1 / dukit.fourier.MAG_UNIT_CONV

        unique_uvs = dd(list)
        for i, uv in enumerate(self.unit_vectors_lst):
            unique_uvs[uv].append(i)

        for uv in tqdm(
            unique_uvs,
            ascii=True,
            mininterval=1,
            unit="mag. unit vectors",
            desc="propagating stray field...",
            total=len(unique_uvs.keys()),
        ):
            mx, my, mz = (
                self.mag[uv] * uv[0] * m_scale,
                self.mag[uv] * uv[1] * m_scale,
                self.mag[uv] * uv[2] * m_scale,
            )

            mx_pad, p = dukit.fourier.pad_image(mx, pad_mode, pad_factor)
            my_pad, _ = dukit.fourier.pad_image(my, pad_mode, pad_factor)
            mz_pad, _ = dukit.fourier.pad_image(mz, pad_mode, pad_factor)

            fft_mx = numpy_fft.fftshift(numpy_fft.fft2(mx_pad))
            fft_my = numpy_fft.fftshift(numpy_fft.fft2(my_pad))
            fft_mz = numpy_fft.fftshift(numpy_fft.fft2(mz_pad))

            fft_mag_vec = np.stack((fft_mx, fft_my, fft_mz))

            fft_b_vec = np.einsum(
                "ij...,j...->i...", d_matrix, fft_mag_vec
            )  # matrix mul b = d * m (d and m are stacked in last 2 dimensions)

            if nv_layer_thickness is not None and standoff:
                arg = k * nv_layer_thickness / 2
                nv_thickness_correction = np.sinh(arg) / arg
                for vec in fft_b_vec:
                    vec *= nv_thickness_correction

            # take back to real space, unpad & convert bfield to Gauss (from Tesla)
            self.bfield[0] += (
                dukit.fourier.unpad_image(
                    numpy_fft.ifft2(numpy_fft.ifftshift(fft_b_vec[0])).real, p
                )
                * 1e4
            )
            self.bfield[1] += (
                dukit.fourier.unpad_image(
                    numpy_fft.ifft2(numpy_fft.ifftshift(fft_b_vec[1])).real, p
                )
                * 1e4
            )
            self.bfield[2] += (
                dukit.fourier.unpad_image(
                    numpy_fft.ifft2(numpy_fft.ifftshift(fft_b_vec[2])).real, p
                )
                * 1e4
            )

        # for resolution convolve with width=resolution gaussian?
        # yep but width = sigma = in units of pixels. {so do some maths eh}
        # just do it after the fft I think.
        # add noise too? -> add to magnetisation or what?
        if resolution is not None:
            # would be faster to do while in k-space already above.
            sigma = resolution / self.pixel_size
            # in-place
            gaussian_filter(self.bfield[0], sigma, output=self.bfield[0])
            gaussian_filter(self.bfield[1], sigma, output=self.bfield[1])
            gaussian_filter(self.bfield[2], sigma, output=self.bfield[2])

    # def _scale_for_fft(self, ars):
    #     """Norm arrays so fft doesn't freak out. NB not used anymore"""
    #     mx = np.max(np.abs(ars))
    #     return [ar / mx for ar in ars], mx

    def get_bfield_im(self, projection=(0, 0, 1)):
        if self.bfield is None:
            raise AttributeError("simulation not run yet.")
        # access bfield_sensor_plane, project onto projection

        # reshape bfield to be [bx, by, bz] for each pixel of image
        # (as 1 stacked ndarray)
        bfield_reshaped = np.stack(self.bfield, axis=-1)

        proj_vec = np.array(projection)

        return np.apply_along_axis(lambda bvec: np.dot(proj_vec, bvec), -1, bfield_reshaped)

    def get_magnetization_im(self, unit_vector):
        if self.mag is None:
            raise AttributeError("magnetization not defined yet.")
        return self.mag[unit_vector]

    def plot_magsim_magnetization(
        self, unit_vector, annotate_polygons=True, polygon_patch_params=None
    ):
        fig, ax = plt.subplots()
        mag_image = self.get_magnetization_im(unit_vector)
        mx = np.max(np.abs(mag_image))
        c_range = (-mx, mx)
        if annotate_polygons:
            polys = self.polygon_nodes
        else:
            polys = None
        dukit.itool.plot_image_on_ax(
            fig,
            ax,
            mag_image,
            str(unit_vector),
            "PuOr",
            c_range,
            r"M ($\mu_B$ nm$^{-2}$)",
            polygon_nodes=polys,
            polygon_patch_params=polygon_patch_params,
            raw_pixel_size=self.pixel_size,
        )
        return fig, ax

    def plot_magsim_magnetizations(
        self,
        annotate_polygons=True,
        polygon_patch_params=None,
        cmap="PuOr",
        c_range=None,
    ):
        # use single colorbar, different plots
        # https://matplotlib.org/stable/gallery/subplots_axes_and figures\
        #   /colorbar_placement.html
        # calculate c_range smartly.
        if self.magnetizations_lst is None:
            raise AttributeError("no magnetizations_lst found, define it first aye.")

        unique_uvs = dd(list)
        for i, uv in enumerate(self.unit_vectors_lst):
            unique_uvs[uv].append(i)

        mag_images = [self.get_magnetization_im(uv) for uv in unique_uvs]

        if c_range is None:
            mx = max([np.nanmax(mag) for mag in mag_images])
            c_range = (-mx, mx)

        figsize = mpl.rcParams["figure.figsize"].copy()
        nrows = 1
        ncols = len(unique_uvs) + 1
        figsize[0] *= nrows

        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)

        dukit.itool.plot_image_on_ax(
            fig,
            axs[0] if isinstance(axs, np.ndarray) else axs,
            np.sum(mag_images, axis=0),
            "sum",
            cmap,
            c_range,
            self._get_mag_unit_str(),
            polygon_nodes=self.polygon_nodes if annotate_polygons else None,
            polygon_patch_params=polygon_patch_params,
            raw_pixel_size=self.pixel_size,
        )

        for i, (mag, uvs) in enumerate(zip(mag_images, unique_uvs)):
            if annotate_polygons:
                polys = [self.polygon_nodes[j] for j in unique_uvs[uvs]]
            else:
                polys = None

            title = str(uvs)  # below is just copies of the same unv?
            # title = ", ".join([str(self.unit_vectors_lst[uv])
            #   for uv in unique_uvs[uvs]])
            dukit.itool.plot_image_on_ax(
                fig,
                axs[i + 1] if isinstance(axs, np.ndarray) else axs,
                mag,
                title,
                cmap,
                c_range,
                self._get_mag_unit_str(),
                polygon_nodes=polys,
                polygon_patch_params=polygon_patch_params,
                raw_pixel_size=self.pixel_size,
            )

        return fig, axs

    def plot_magsim_bfield_at_nvs(
        self,
        projection=(0, 0, 1),
        annotate_polygons=True,
        polygon_patch_params=None,
        cmap="bwr",
        strict_range=None,
        c_label=None,
    ):
        if self.bfield is None:
            raise AttributeError("No bfield found: no simulation run.")

        if strict_range is not None:
            c_range = strict_range
        else:
            furthest = np.max(np.abs([np.nanmin(self.bfield), np.nanmax(self.bfield)]))
            c_range = (-furthest, furthest)

        polys = None if annotate_polygons is None else self.polygon_nodes

        fig, ax = plt.subplots()
        proj_name = f"({projection[0]:.2f},{projection[1]:.2f},{projection[2]:.2f})"
        c_label_ = f"B . {proj_name}, (G)" if c_label is None else c_label
        dukit.itool.plot_image_on_ax(
            fig,
            ax,
            self.get_bfield_im(projection),
            f"B . {proj_name} at z = {self.standoff}{self._get_dist_unit_str()}",
            cmap,
            c_range,
            c_label_,
            polygon_nodes=polys,
            polygon_patch_params=polygon_patch_params,
            raw_pixel_size=self.pixel_size,
        )
        return fig, ax

    @staticmethod
    def _get_mag_unit_str():
        return r"M ($\mu_B$ nm$^{-2}$)"

    @staticmethod
    def _get_dist_scaling():
        # nm = 1e-9
        return 1e-9

    @staticmethod
    def _get_dist_unit_str():
        return "m"

    def crop_polygons(self, crop_polygon_nodes):
        crop_polygons = [
            dukit.polygon.Polygon(crop_nodes[:, 0], crop_nodes[:, 1])
            for crop_nodes in crop_polygon_nodes
        ]
        keep_idxs = []
        for idx, p in tqdm(
            enumerate(self.polygon_nodes),
            ascii=True,
            mininterval=1,
            unit="polygons",
            desc="cropping polygons...",
            total=len(self.polygon_nodes),
        ):
            for crop_polygon in crop_polygons:
                if not np.all(crop_polygon.is_inside(p[:, 0], p[:, 1]) > 0):
                    break  # don't append to keep lst
            else:
                keep_idxs.append(idx)  # only executes if loop exits normally (not 'break')
        self.polygon_nodes = [val for idx, val in enumerate(self.polygon_nodes) if idx in keep_idxs]
        if self.magnetizations_lst is not None:
            self.magnetizations_lst = [
                val for idx, val in enumerate(self.magnetizations_lst) if idx in keep_idxs
            ]
            self.unit_vectors_lst = [
                val for idx, val in enumerate(self.unit_vectors_lst) if idx in keep_idxs
            ]

    def crop_polygons_gui(self, show_polygons=True, **kwargs):
        if show_polygons:
            pn = self.polygon_nodes
            n_og_polygons = len(self.polygon_nodes)
        else:
            pn = None
            n_og_polygons = 0
        pgon_dict = self._polygon_gui(polygon_nodes=pn, prompt="Select crop polygon", **kwargs)
        new_pgons = [np.array(p) for p in pgon_dict["nodes"][n_og_polygons:]]
        self.crop_polygons(new_pgons)

    def crop_magnetization(self, crop_polygon_nodes):
        if self.mag is None:
            raise AttributeError("You haven't defined mag yet! (use define_magnets).")
        crop_polygons = [
            dukit.polygon.Polygon(crop_nodes[:, 0], crop_nodes[:, 1])
            for crop_nodes in crop_polygon_nodes
        ]
        grid_y, grid_x = np.meshgrid(range(self.ny), range(self.nx), indexing="ij")

        for polygon in tqdm(
            crop_polygons,
            ascii=True,
            mininterval=1,
            unit="polygons",
            desc="cropping magnetization...",
            total=len(crop_polygons),
        ):
            for key in self.mag:
                self.mag[key][polygon.is_inside(grid_y, grid_x) < 0] = 0

    def crop_domains(self, crop_polygon_nodes):
        pass  # overriden in TilingMagSim

    def crop_magnetization_gui(self, **kwargs):
        # crops magnetization, polygons and domain_label_pts
        # (i.e. run after define_magnets)
        unique_uvs = dd(list)
        for i, uv in enumerate(self.unit_vectors_lst):
            unique_uvs[uv].append(i)
        mag_image = np.sum([self.get_magnetization_im(uv) for uv in unique_uvs], axis=0)
        n_og_polygons = len(self.polygon_nodes)
        crop_dict = self._polygon_gui(
            polygon_nodes=self.polygon_nodes,
            image=mag_image,
            cmap="PuOr",
            prompt="Select crop polygon",
            **kwargs,
        )
        crop_nodes = [np.array(p) for p in crop_dict["nodes"][n_og_polygons:]]
        self.crop_magnetization(crop_nodes)
        self.crop_polygons(crop_nodes)
        self.crop_domains(crop_nodes)


class SandboxMagSim(MagSim):
    def __init__(self, mesh_shape, fov_dims):
        """Image conventions: first index is height."""
        self.ny, self.nx = mesh_shape
        self.fov_dims = fov_dims
        self.base_image = np.full(mesh_shape, np.nan)
        pxl_y = fov_dims[0] / self.ny
        pxl_x = fov_dims[1] / self.nx
        if int(np.floor(pxl_y)) != int(np.floor(pxl_x)):
            raise ValueError("fov_dims ratio height:width does not match mesh height:width ratio.")
        self.pixel_size = pxl_y

    def add_template_polygons(self, polygons=None):
        """polygons takes precedence."""
        self.template_polygon_nodes = self._load_polys(polygons, check_size=True)

    def rescale_template(self, factor):
        if self.template_polygon_nodes is None:
            raise RuntimeError("Add/define template_polygon_nodes before rescaling.")

        for polygon in self.template_polygon_nodes:
            for node in polygon:
                node[0] *= factor
                node[1] *= factor

    def adjust_template(self, output_path=None, mean_plus_minus=None, **kwargs):
        if self.template_polygon_nodes is None:
            raise AttributeError("Add template polygons before adjusting.")
        pgon_dict = self._polygon_gui(
            polygon_nodes=self.template_polygon_nodes,
            mean_plus_minus=mean_plus_minus,
            prompt="Adjust template polygons/add new",
            **kwargs,
        )
        if output_path is not None:
            self._save_dict(output_path, pgon_dict)

        self.template_polygon_nodes = [np.array(p) for p in pgon_dict["nodes"]]

    def set_template_as_polygons(self):
        if self.template_polygon_nodes is None:
            raise AttributeError("No template set.")
        self.polygon_nodes = self.template_polygon_nodes


# define FOV size (height, width), standoff height
# resolution: increase size of polygon via this (grab it's image size)
# -> user cannot scroll etc., define to match some other/bnv image
# would be best if image is plane-subtracted
class ComparisonMagSim(MagSim):
    unscaled_polygon_nodes = None

    def __init__(
        self,
        image,  # array-like (image directly) or string (path to)
        fov_dims,  # (the height, width of the image in m)
    ):
        if fov_dims is None:
            raise ValueError("You need to supply fov_dims (the height, width of the image in m).")
        if (
            not isinstance(fov_dims, (tuple, list, np.ndarray))
            or len(fov_dims) != 2
            or not isinstance(fov_dims[0], (int, float))
            or not isinstance(fov_dims[1], (int, float))
        ):
            raise TypeError("fov_dims needs to be length 2 array-like of int/floats")

        # check for path etc. here
        self.base_image = self._load_image(image)

        self.ny, self.nx = self.base_image.shape
        pxl_y = fov_dims[0] / self.ny
        pxl_x = fov_dims[1] / self.nx
        if pxl_y != pxl_x:
            raise ValueError("fov_dims ratio height:width does not match image height:width ratio.")
        self.pixel_size = pxl_y

    def rescale(self, factor):
        if self.polygon_nodes is None:
            raise RuntimeError("Add/define polygon_nodes before rescaling.")

        if self.unscaled_polygon_nodes is None:
            self.unscaled_polygon_nodes = copy(self.polygon_nodes)

        self.ny *= factor
        self.nx *= factor
        self.pixel_size *= 1 / factor
        for polygon in self.polygon_nodes:
            for node in polygon:
                node[0] *= factor
                node[1] *= factor

    def plot_comparison(
        self,
        projection=(0, 0, 1),
        strict_range=None,
        annotate_polygons=True,
        c_label_meas=None,
        c_label_sim=None,
    ):
        # bug: still no annotations on measurement?

        if self.bfield is None:
            raise AttributeError("No bfield found: no simulation run.")

        bfield_proj = self.get_bfield_im(projection)

        if strict_range is not None:
            c_range = strict_range
        else:
            mn = min((np.nanmin(bfield_proj), np.nanmin(self.base_image)))
            mx = max((np.nanmax(bfield_proj), np.nanmax(self.base_image)))
            c_range = (mn, mx)

        c_label_meas_ = "B (G)" if c_label_meas is None else c_label_meas

        proj_name = f"({projection[0]:.2f},{projection[1]:.2f},{projection[2]:.2f})"
        c_label_sim_ = f"B . {proj_name}, (G)" if c_label_sim is None else c_label_sim

        if annotate_polygons is False:
            unscaled_polys = None
            scaled_polys = None
        else:
            unscaled_polys = (
                self.polygon_nodes
                if self.unscaled_polygon_nodes is None
                else self.unscaled_polygon_nodes
            )
            scaled_polys = self.polygon_nodes

        figsize = mpl.rcParams["figure.figsize"].copy()
        nrows = 1
        ncols = 2
        figsize[0] *= nrows

        fig, axs = plt.subplots(figsize=figsize, nrows=nrows, ncols=ncols)

        dukit.itool.plot_image_on_ax(
            fig,
            axs[0],
            self.base_image,
            "measurement",
            "bwr",
            c_range,
            c_label_meas_,
            polygon_nodes=unscaled_polys,
            polygon_patch_params=None,
            raw_pixel_size=self.pixel_size,
        )
        dukit.itool.plot_image_on_ax(
            fig,
            axs[1],
            bfield_proj,
            "simulated",
            "bwr",
            c_range,
            c_label_sim_,
            polygon_nodes=scaled_polys,
            polygon_patch_params=None,
            raw_pixel_size=self.pixel_size,
        )

        return fig, axs


def _is_convex_polygon(polygon_nodes):
    """Return True if the polynomial defined by the sequence of 2D
        points is 'strictly convex': points are valid, side lengths non-
        zero, interior angles are strictly between zero and a straight
        angle, and the polygon does not intersect itself.

        NOTES:  1.  Algorithm: the signed changes of the direction angles
                    from one side to the next side must be all positive or
                    all negative, and their sum must equal plus-or-minus
                    one full turn (2 pi radians). Also check for too few,
                    invalid, or repeated points.
                2.  No check is explicitly done for zero internal angles
                    (180 degree direction-change angle) as this is covered
                    in other ways, including the `n < 3` check.
        SOURCE:
            https://stackoverflow.com/questions/471962/ \
            how-do-i-efficiently-determine-if-a-polygon-is-convex-non-convex-or-complex
        """
    polygon_nodes = polygon_nodes[:-1]  # don't use first/last point twice
    try:  # needed for any bad points or direction changes
        # Check for too few points
        if len(polygon_nodes) < 3:
            return False
        # Get starting information
        old_x, old_y = polygon_nodes[-2]
        new_x, new_y = polygon_nodes[-1]
        new_direction = np.arctan2(new_y - old_y, new_x - old_x)
        angle_sum = 0.0
        # Check each point (the side ending there, its angle) and accum. angles
        for ndx, newpoint in enumerate(polygon_nodes):
            # Update point coordinates and side directions, check side length
            old_x, old_y, old_direction = new_x, new_y, new_direction
            new_x, new_y = newpoint
            new_direction = np.arctan2(new_y - old_y, new_x - old_x)
            if old_x == new_x and old_y == new_y:
                return False  # repeated consecutive points
            # Calculate & check the normalized direction-change angle
            angle = new_direction - old_direction
            if angle <= -np.pi:
                angle += 2 * np.pi  # make it in half-open interval (-Pi, Pi]
            elif angle > np.pi:
                angle -= 2 * np.pi
            if ndx == 0:  # if first time through loop, initialize orientation
                if angle == 0.0:
                    return False
                orientation = 1.0 if angle > 0.0 else -1.0
            else:  # if other time through loop, check orientation is stable
                if orientation * angle <= 0.0:  # not both pos. or both neg.
                    return False
            # Accumulate the direction-change angle
            angle_sum += angle
        # Check that the total number of full turns is plus-or-minus 1
        return abs(round(angle_sum / (2 * np.pi))) == 1
    except (ArithmeticError, TypeError, ValueError):
        return False  # any exception means not a proper convex polygon
