# -*- coding: utf-8 -*-
"""
This module holds the Polygon class.

A class to compute if a point lies inside/outside/on-side of a polygon.
Also defined is a function (polygon_selector) that can be called to select a polygon
region on an image.

For use check examples.

Apologies for lack of typing, I don't understand sufficiently and it
currently *just works*.

Polygon
-------
This is a Python 3 implementation of the Sloan's improved version of the
Nordbeck and Rystedt algorithm, published in the paper:

SLOAN, S.W. (1985): A point-in-polygon program.
    Adv. Eng. Software, Vol 7, No. 1, pp 45-47.

This class has 1 method (is_inside) that returns the minimum distance to the
nearest point of the polygon:

If is_inside < 0 then point is outside the polygon.
If is_inside = 0 then point in on a side of the polygon.
If is_inside > 0 then point is inside the polygon.

Sam Scholten copied from:
http://code.activestate.com/recipes/578381-a-point-in-polygon-program-sw-sloan-algorithm/
-> swapped x & y args order (etc.) for image use.

Classes
-------
 - `dukit.polygon.Polygon`

Functions
---------
 - `dukit.polygon.load_polygon_nodes`
 - `dukit.polygon.polygon_selector`
 - `dukit.polygon.PolygonSelectionWidget`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.polygon.load_polygon_nodes": True,
    "dukit.polygon.polygon_selector": True,
    "dukit.polygon.Polygon": True,
    "dukit.polygon.PolygonSelectionWidget": True,
}

# ============================================================================

import dill as pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numba
from numba import jit
import numpy as np
import numpy.typing as npt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

import dukit.widget
from dukit.fourier import pad_image

# ============================================================================
from dukit.json2dict import dict_to_json, json_to_dict
from dukit.warn import warn

# ============================================================================

CMAP_OPTIONS: list[str] = [
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "Greys",
    "Purples",
    "Blues",
    "Greens",
    "Oranges",
    "Reds",
    "YlOrBr",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
    "binary",
    "gist_yarg",
    "gist_gray",
    "gray",
    "bone",
    "pink",
    "spring",
    "summer",
    "autumn",
    "winter",
    "cool",
    "Wistia",
    "hot",
    "afmhot",
    "gist_heat",
    "copper",
    "PiYG",
    "PRGn",
    "BrBG",
    "PuOr",
    "RdGy",
    "RdBu",
    "RdYlBu",
    "RdYlGn",
    "Spectral",
    "coolwarm",
    "bwr",
    "seismic",
    "twilight",
    "twilight_shifted",
    "hsv",
    "Pastel1",
    "Pastel2",
    "Paired",
    "Accent",
    "Dark2",
    "Set1",
    "Set2",
    "Set3",
    "tab10",
    "tab20",
    "tab20b",
    "tab20c",
    "flag",
    "prism",
    "ocean",
    "gist_earth",
    "terrain",
    "gist_stern",
    "gnuplot",
    "gnuplot2",
    "CMRmap",
    "cubehelix",
    "brg",
    "gist_rainbow",
    "rainbow",
    "jet",
    "nipy_spectral",
    "gist_ncar",
]

# ============================================================================


@jit(nopython=True, cache=True)
def _is_inside_sm(point, polygon):
    # https://github.com/sasamil/PointInPolygon_Py/blob/master/pointInside.py
    # note this fn works in (x,y) coords (but if point/polygon is consistent all is g)
    conv_map = {0: -1, 1: 1}
    length = polygon.shape[0] - 1
    dy2 = point[1] - polygon[0][1]
    intersections = 0
    ii = 0
    jj = 1

    while ii < length:
        dy = dy2
        dy2 = point[1] - polygon[jj][1]

        # consider only lines which are not completely above/bellow/right from the point
        if dy * dy2 <= 0.0 and (point[0] >= polygon[ii][0] or point[0] >= polygon[jj][0]):
            # non-horizontal line
            if dy < 0 or dy2 < 0:
                F = dy * (polygon[jj][0] - polygon[ii][0]) / (dy - dy2) + polygon[ii][0]  # noqa: N806

                if point[0] > F:  # if line is left from the point - the ray moving towards left,
                    #  will intersect it
                    intersections += 1
                elif point[0] == F:  # point on line
                    return 0

            # point on upper peak (dy2=dx2=0) or horizontal line
            # (dy=dy2=0 and dx*dx2<=0)
            elif dy2 == 0 and (
                point[0] == polygon[jj][0]
                or (dy == 0 and (point[0] - polygon[ii][0]) * (point[0] - polygon[jj][0]) <= 0)
            ):
                return 0

        ii = jj
        jj += 1

    # print 'intersections =', intersections
    return conv_map[intersections & 1]


# ============================================================================


@jit(nopython=True, parallel=True, cache=True)
def _is_inside_sm_parallel(points, polygon):
    # https://stackoverflow.com/questions/36399381/ \
    #
    # note this fn works in (x,y) coords (but if point/polygon is consistent all is g.)
    p_ar = np.asarray(points, dtype=np.float64)
    pts_shape = p_ar.shape[:-1]
    p_ar_flat = p_ar.reshape(-1, 2)  # shape: (len_y * len_x, 2), i.e. long list of coords (y, x)
    d = np.zeros(p_ar_flat.shape[0], dtype=numba.int8)
    for i in numba.prange(p_ar_flat.shape[0]):
        d[i] = _is_inside_sm(p_ar_flat[i], polygon)
    d = d.reshape(pts_shape)
    return d


# ============================================================================


class Polygon:
    """
    Polygon object.

    Arguments
    ---------
    y : array-like
        A sequence of nodal y-coords (all unique).

    x : array-like
        A sequence of nodal x-coords (all unique).
    """

    def __init__(self, y, x):
        if len(y) != len(x):
            raise IndexError("y and x must be equally sized.")
        self.y = np.asarray(y, dtype=float)
        self.x = np.asarray(x, dtype=float)
        # Closes the polygon if were open
        y1, x1 = y[0], x[0]
        yn, xn = y[-1], x[-1]
        if x1 != xn or y1 != yn:
            self.y = np.concatenate((self.y, [y1]))
            self.x = np.concatenate((self.x, [x1]))
        # # Anti-clockwise coordinates # irrelevant...?
        # if _tri_2area_det(self.y, self.x) < 0:
        #     self.y = self.y[::-1]
        #     self.x = self.x[::-1]

    # =============================================== #

    def get_nodes(self):
        # get nodes as a list [[y1,x1], [y2,x2]] etc.
        return [[y, x] for y, x in zip(self.y, self.x)]

    # =============================================== #

    def get_yx(self):
        return np.stack((self.y, self.x), axis=-1)

    # =============================================== #

    def is_inside(self, y, x):
        # return value:
        # <0 - the point is outside the polygon
        # =0 - the point is one edge (boundary)
        # >0 - the point is inside the polygon
        xs = np.asarray(x, dtype=float)
        ys = np.asarray(y, dtype=float)
        # Check consistency
        if xs.shape != ys.shape:
            raise IndexError("x and y has different shapes")
        # check if single point (0-d array or scalar)
        if xs.shape == ():
            point = np.array([y, x], dtype=np.float64)
            return _is_inside_sm(point, self.get_yx())
        else:
            return _is_inside_sm_parallel(np.stack((ys, xs), axis=-1), self.get_yx())


# ============================================================================


def polygon_selector(
    array: npt.NDArray | str,
    json_output_path: str | None = None,
    json_input_path: str | None = None,
    mean_plus_minus: float | None = None,
    strict_range: tuple[float, float] | None = None,
    print_help: bool = False,
    pad: int = 0,
    **kwargs,
):
    """
    Generates mpl (qt) gui for selecting a polygon.
    NOTE: you probably just want to use PolygonSelectionWidget.

    Arguments
    ---------
    array : path OR arraylike
        Path to (numpy) .txt file to load as image.
        OR can be an arraylike directly
    json_output_path : str or path-like, default="~/poly.json"
        Path to put output json, defaults to home/poly.json.
    json_input_path : str or path-like, default=None
        Loads previous polygons at this path for editing.
    mean_plus_minus : float, default=None
        Plot image with color scaled to mean +- this number.
    strict_range: length 2 list, default=None
        Plot image with color scaled between these values.
        Precedence over mean_plus_minus.
    print_help : bool, default=False
        View this message.
    pad : bool
        If > 0, pads with zeros by 'pad' fraction times the image size in both dimensions
        The 'padder' (see `qdmpy.sharead.fourier.unpad_image`) is placed in
        the output dict/json.
    **kwargs : dict
        Other keyword arguments to pass to plotters. Currently implemented:
            cmap : string
                Passed to imshow.
            lineprops : dict
                Passed to PolygonSelectionWidget.
            markerprops : dict
                Passed to PolygonSelectionWidget.


    GUI help
    --------
    In the mpl gui, select points to draw polygons.
    Press 'enter' to continue in the program.
    Press the 'esc' key to reset the current polygon
    Hold 'shift' to move all of the vertices (from all polygons)
    Hold 'r' and scroll to resize all of the polygons.
    'ctrl' to move a single vertex in the current polygon
    'alt' to start a new polygon (and finalise the current one)
    'del' to clear all lines from the graphic  (thus deleting all polygons).
    'right click' on a vertex (of a finished polygon) to remove it.
    """
    if print_help:
        print(
            """
        Help
        ====

        Input help
        ----------
        array : path OR arraylike
            Path to (numpy) .txt file to load as image.
            OR can be an arraylike directly
        json_output_path : str or path-like, default="~/poly.json"
            Path to put output json, defaults to home/poly.json.
        json_input_path : str or path-like, default=None
            Loads previous polygons at this path for editing.
        mean_plus_minus : float, default=None
            Plot image with color scaled to mean +- this number.
        strict_range: length 2 list, default=None
            Plot image with color scaled between these values.
            Precedence over mean_plus_minus.
        help : bool, Default=False
            View this message.
        **kwargs : dict
        Other keyword arguments to pass to plotters. Currently implemented:
            cmap : string
                Passed to imshow.
            lineprops : dict
                Passed to PolygonSelectionWidget.
            markerprops : dict
                Passed to PolygonSelectionWidget.

        GUI help
        --------
        In the mpl gui, select points to draw polygons.
        Press 'enter' to continue in the program.
        Press the 'esc' key to reset the current polygon
        Hold 'shift' to move all of the vertices (from all polygons)
        Hold 'r' and scroll to resize all of the polygons.
        'ctrl' to move a single vertex in the current polygon
        'alt' to start a new polygon (and finalise the current one)
        'del' to clear all lines from the graphic  (thus deleting all polygons).
        'right click' on a vertex (of a finished polygon) to remove it.
        """
        )
        return []

    image = np.loadtxt(array) if not isinstance(array, np.ndarray) else array

    if pad > 0:
        image, padder = pad_image(image, "constant", pad)
    else:
        padder = ((0, 0), (0, 0))

    if json_input_path is None:
        polys = None
        polygon_nodes = None
    else:
        polys = json_to_dict(json_input_path)
        polygon_nodes = polys["nodes"]
        if "image_shape" in polys:
            shp = polys["image_shape"]
            if shp[0] != image.shape[0] or shp[1] != image.shape[1]:
                warn("Image shape loaded polygons were defined on does not match current image.")

    fig, ax = plt.subplots()
    minimum = np.nanmin(image)
    maximum = np.nanmax(image)
    if (
        strict_range is not None
        and isinstance(strict_range, (list, np.ndarray, tuple))
        and len(strict_range) == 2
    ):
        vmin, vmax = strict_range
    elif mean_plus_minus is not None and isinstance(mean_plus_minus, (float, int)):
        mean = np.mean(image)
        vmin, vmax = mean - mean_plus_minus, mean + mean_plus_minus
    else:
        vmin, vmax = minimum, maximum
    img = ax.imshow(
        image,
        aspect="equal",
        cmap=kwargs["cmap"] if "cmap" in kwargs else "bwr",
        vmin=vmin,
        vmax=vmax,
    )

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
    ax.set_title("Select polygons")

    psw = PolygonSelectionWidget(ax, style=kwargs)

    if polygon_nodes is not None:
        psw.load_nodes(polygon_nodes)

    plt.tight_layout()
    plt.show(block=True)
    psw.disconnect()

    pgons = psw.get_polygons_lst()
    if len(pgons) < 1:
        raise RuntimeError("You didn't define any polygons")

    # exclude polygons with nodes < 3
    pgon_lst = [pgon.get_nodes() for pgon in pgons if np.shape(pgon.get_nodes())[0] > 2]
    output_dict = {
        "nodes": pgon_lst,
        "image_shape": image.shape,
        "padder": padder,
    }

    dict_to_json(output_dict, json_output_path)

    return output_dict


# =======================================================================================


class PolygonSelectionWidget:
    """
    How to Use
    ----------
    selector = PolygonSelectionWidget(ax, ...)
    plt.show()
    selector.disconnect()
    polygon_lst = selector.get_polygon_lst()

    ```text
    GUI help
    --------
    In the mpl gui, select points to draw polygons.
    Press 'enter' to continue in the program.
    Press the 'esc' key to reset the current polygon
    Hold 'shift' to move all of the vertices (from all polygons)
    Hold 'r' and scroll to resize all of the polygons.
    'ctrl' to move a single vertex in the current polygon
    'alt' to start a new polygon (and finalise the current one)
    'del' to clear all lines from the graphic  (thus deleting all polygons).
    'right click' on a vertex (of a finished polygon) to remove it.
    ```
    """

    def __init__(self, ax, style=None, base_scale=1.5):
        self.canvas = ax.figure.canvas

        dflt_style = {
            "lineprops": {
                "color": "g",
                "linestyle": "-",
                "linewidth": 1.0,
                "alpha": 0.5,
            },
            "markerprops": {
                "marker": "o",
                "markersize": 2.0,
                "mec": "g",
                "mfc": "g",
                "alpha": 0.5,
            },
        }

        self.lp = dflt_style["lineprops"]
        self.mp = dflt_style["markerprops"]
        if style is not None:
            if "lineprops" in style and isinstance(style["lineprops"], dict):
                for key, item in style["lineprops"]:
                    self.lp[key] = item
            if "markerprops" in style and isinstance(style["markerprops"], dict):
                for key, item in style["markerprops"]:
                    self.mp[key] = item

        vsr = 7.5 * self.mp["markersize"]  # linear scaling on what our select radius is
        self.ax = ax
        self.polys = dukit.widget.PolygonSelector(
            ax,
            self.onselect,
            lineprops=self.lp,
            markerprops=self.mp,
            vertex_select_radius=vsr,
            base_scale=base_scale,
        )
        self.pts = []

    def onselect(self, verts):
        # only called when polygon is finished
        self.pts.append(verts)
        self.canvas.draw_idle()

    def disconnect(self):
        self.polys.disconnect_events()
        self.canvas.draw_idle()

    def get_polygons_lst(self):
        lst = []
        for p in self.polys.xy_verts:
            # NOTE opposite indexing convention here (xy_verts -> yx Polygon)
            new_polygon_obj = Polygon(p[1], p[0])
            lst.append(new_polygon_obj)
        return lst

    def load_nodes(self, polygon_nodes):
        # polygon nodes: list of polygons, each polygon is a list of nodes
        for polygon in polygon_nodes:
            nodes_ar = np.array(polygon)

            # x & y convention swapped here
            new_line = Line2D(nodes_ar[:, 1], nodes_ar[:, 0], **self.lp)
            self.ax.add_line(new_line)

            new_line_dict = dict(line_obj=new_line, xs=nodes_ar[:, 1], ys=nodes_ar[:, 0])

            self.polys.artists.append(new_line)
            self.polys.lines.append(new_line_dict)  # list of line dicts

        self.polys.draw_polygon()


# ============================================================================


def load_polygon_nodes(poly_path_or_dict: str | dict | Path) -> list[npt.NDArray]:
    """
    Loads polygon nodes from json file.

    Arguments
    ---------
    poly_path_or_dict : str | dict | Path
        Path to json or pickle/dill file containing polygon nodes, or directly as a dict

    Returns
    -------
    list[npt.NDArray]
        List of polygons, each polygon is an array of nodes.
    """

    def _load_dict(path: str):
        if path.endswith("json"):
            return json_to_dict(path)
        elif path.endswith("pickle"):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError("polygon_nodes path did not end in 'json' or 'pickle'")

    if isinstance(poly_path_or_dict, dict):
        return [np.array(p) for p in poly_path_or_dict["nodes"]]
    elif isinstance(poly_path_or_dict, (str, Path)):
        return [np.array(p) for p in _load_dict(str(poly_path_or_dict))["nodes"]]
    else:
        raise TypeError("polygons argument was not a dict, string or Path?")
