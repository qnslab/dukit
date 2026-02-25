# -*- coding: utf-8 -*-
"""
Various GUI widgets for use in the DUKIT.

Not very well documented. But I don't really understand it so that's fine.
How to use? See polygon.py and linecut.py

Don't import any other dukit modules, ensure this is a leaf in the dep. tree.

Apologies for lack of typing, I don't understand this stuff sufficiently.
But it does currently *just work*.

Classes
-------
 - `dukit.widget.PolygonSelector`
 - `dukit.widget.LineSelector`
 - `dukit.widget.LinecutSelectionWidget`
 - `dukit.widget.BulkLinecutWidget`
"""

# ============================================================================

__author__ = "Sam Scholten"
__pdoc__ = {
    "dukit.widget.PolygonSelector": True,
    "dukit.widget.LineSelector": True,
    "dukit.widget.LinecutSelectionWidget": True,
    "dukit.widget.BulkLinecutWidget": True,
}

# ============================================================================

import copy
from numbers import Integral

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from scipy import integrate

# ============================================================================
import dukit.json2dict

# ======================================================================================
# ======================================================================================
# ======================================================================================
# code from:
# https://matplotlib.org/_modules/matplotlib/widgets.html#PolygonSelector
# vendored, as I wanted to tweak some things - not the best idea I know.
# ======================================================================================
# ======================================================================================
# ======================================================================================


class Widget:
    """
    Abstract base class for GUI neutral widgets
    """

    drawon = True
    eventson = True
    _active = True

    def set_active(self, active):
        """Set whether the widget is active."""
        self._active = active

    def get_active(self):
        """Get whether the widget is active."""
        return self._active

    # set_active is overridden by SelectorWidgets.
    active = property(
        get_active,
        lambda self, active: self.set_active(active),
        doc="Is the widget active?",
    )

    def ignore(self, event):
        """Return True if event should be ignored.

        This method (or a version of it) should be called at the beginning
        of any event callback.
        """
        return not self.active


# =======================================================================================


class AxesWidget(Widget):
    """Widget that is connected to a single
    :class:`~matplotlib.axes.Axes`.

    To guarantee that the widget remains responsive and not garbage-collected,
    a reference to the object should be maintained by the user.

    This is necessary because the callback registry
    maintains only weak-refs to the functions, which are member
    functions of the widget.  If there are no references to the widget
    object it may be garbage collected which will disconnect the
    callbacks.

    Attributes:

    *ax* : :class:`~matplotlib.axes.Axes`
        The parent axes for the widget
    *canvas* : :class:`~matplotlib.backend_bases.FigureCanvasBase` subclass
        The parent figure canvas for the widget.
    *active* : bool
        If False, the widget does not respond to events.
    """

    def __init__(self, ax):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.cids = []

    def connect_event(self, event, callback):
        """Connect callback with an event.

        This should be used in lieu of `figure.canvas.mpl_connect` since this
        function stores callback ids for later clean up.
        """
        cid = self.canvas.mpl_connect(event, callback)
        self.cids.append(cid)

    def disconnect_events(self):
        """Disconnect all events created by this widget."""
        for c in self.cids:
            self.canvas.mpl_disconnect(c)


# =======================================================================================


class _SelectorWidget(AxesWidget):
    def __init__(
        self,
        ax,
        onselect,
        useblit=False,
        button=None,
        state_modifier_keys=None,
    ):
        AxesWidget.__init__(self, ax)

        self.visible = True
        self.onselect = onselect
        self.useblit = useblit and self.canvas.supports_blit
        self.connect_default_events()

        self.state_modifier_keys = dict(move=" ", clear="escape", square="shift", center="control")
        self.state_modifier_keys.update(state_modifier_keys or {})

        self.background = None
        self.artists = []

        if isinstance(button, Integral):
            self.validButtons = [button]
        else:
            self.validButtons = button

        # will save the data (position at mouseclick)
        self.eventpress = None
        # will save the data (pos. at mouserelease)
        self.eventrelease = None
        self._prev_event = None
        self.state = set()

    def set_active(self, active):
        AxesWidget.set_active(self, active)
        if active:
            self.update_background(None)

    def update_background(self, event):
        """force an update of the background"""
        # If you add a call to `ignore` here, you'll want to check edge case:
        # `release` can call a draw event even when `ignore` is True.
        if self.useblit:
            self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def connect_default_events(self):
        """Connect the major canvas events to methods."""
        self.connect_event("motion_notify_event", self.onmove)
        self.connect_event("button_press_event", self.press)
        self.connect_event("button_release_event", self.release)
        self.connect_event("draw_event", self.update_background)
        self.connect_event("key_press_event", self.on_key_press)
        self.connect_event("key_release_event", self.on_key_release)
        self.connect_event("scroll_event", self.on_scroll)

    def ignore(self, event):
        """return *True* if *event* should be ignored"""
        if not self.active or not self.ax.get_visible():
            return True

        # If canvas was locked
        if not self.canvas.widgetlock.available(self):
            return True

        if not hasattr(event, "button"):
            event.button = None

        # Only do rectangle selection if event was triggered
        # with a desired button
        if self.validButtons is not None:
            if event.button not in self.validButtons:
                return True

        # If no button was pressed yet ignore the event if it was out
        # of the axes
        if self.eventpress is None:
            return event.inaxes != self.ax

        # If a button was pressed, check if the release-button is the
        # same.
        if event.button == self.eventpress.button:
            return False

        # If a button was pressed, check if the release-button is the
        # same.
        return event.inaxes != self.ax or event.button != self.eventpress.button

    def update(self):
        """draw using newfangled blit or oldfangled draw depending on
        useblit

        """
        if not self.ax.get_visible():
            return False

        if self.useblit:
            if self.background is not None:
                self.canvas.restore_region(self.background)
            for artist in self.artists:
                self.ax.draw_artist(artist)

            self.canvas.blit(self.ax.bbox)

        else:
            self.canvas.draw_idle()
        return False

    def _get_data(self, event):
        """Get the xdata and ydata for event, with limits"""
        # 2021-03-12 @samsc added 'if event is None or...' here due to error
        if event is None or event.xdata is None:
            return None, None
        x0, x1 = self.ax.get_xbound()
        y0, y1 = self.ax.get_ybound()
        xdata = max(x0, event.xdata)
        xdata = min(x1, xdata)
        ydata = max(y0, event.ydata)
        ydata = min(y1, ydata)
        return xdata, ydata

    def _clean_event(self, event):
        """Clean up an event

        Use prev event if there is no xdata
        Limit the xdata and ydata to the axes limits
        Set the prev event
        """
        if event.xdata is None:
            event = self._prev_event
        else:
            event = copy.copy(event)
        event.xdata, event.ydata = self._get_data(event)

        self._prev_event = event
        return event

    def press(self, event):
        """Button press handler and validator"""
        if not self.ignore(event):
            event = self._clean_event(event)
            self.eventpress = event
            self._prev_event = event
            key = event.key or ""
            key = key.replace("ctrl", "control")
            # move state is locked in on a button press
            if key == self.state_modifier_keys["move"]:
                self.state.add("move")
            self._press(event)
            return True
        return False

    def _press(self, event):
        """Button press handler"""
        pass

    def release(self, event):
        """Button release event handler and validator"""
        if not self.ignore(event) and self.eventpress:
            event = self._clean_event(event)
            self.eventrelease = event
            self._release(event)
            self.eventpress = None
            self.eventrelease = None
            self.state.discard("move")
            return True
        return False

    def _release(self, event):
        """Button release event handler"""
        pass

    def onmove(self, event):
        """Cursor move event handler and validator"""
        if not self.ignore(event) and self.eventpress:
            event = self._clean_event(event)
            self._onmove(event)
            return True
        return False

    def _onmove(self, event):
        """Cursor move event handler"""
        pass

    def on_scroll(self, event):
        """Mouse scroll event handler and validator"""
        if not self.ignore(event):
            self._on_scroll(event)

    def _on_scroll(self, event):
        """Mouse scroll event handler"""
        pass

    def on_key_press(self, event):
        """Key press event handler and validator for all selection widgets"""
        if self.active:
            key = event.key or ""
            key = key.replace("ctrl", "control")
            if key == self.state_modifier_keys["clear"]:
                for artist in self.artists:
                    artist.set_visible(False)
                self.update()
                return
            for state, modifier in self.state_modifier_keys.items():
                # if modifier in key:
                if modifier == key:
                    self.state.add(state)
            self._on_key_press(event)

    def _on_key_press(self, event):
        """Key press event handler - use for widget-specific key press actions."""
        pass

    def on_key_release(self, event):
        """Key release event handler and validator."""
        if self.active:
            key = event.key or ""
            for state, modifier in self.state_modifier_keys.items():
                # if modifier in key:
                if modifier == key:
                    self.state.discard(state)
            self._on_key_release(event)

    def _on_key_release(self, event):
        """Key release event handler."""
        pass  # I added this (Sam)

    def set_visible(self, visible):
        """Set the visibility of our artists."""
        self.visible = visible
        for artist in self.artists:
            artist.set_visible(visible)


# =======================================================================================


class ToolHandles:
    """Control handles for canvas tools.

    Arguments
    ---------
    ax : :class:`matplotlib.axes.Axes`
        Matplotlib axes where tool handles are displayed.
    x, y : 1D arrays
        Coordinates of control handles.
    marker : str
        Shape of marker used to display handle. See `matplotlib.pyplot.plot`.
    marker_props : dict
        Additional marker properties. See :class:`matplotlib.lines.Line2D`.
    """

    def __init__(self, ax, x, y, marker="o", marker_props=None, useblit=True):
        self.ax = ax

        props = dict(
            marker=marker,
            markersize=7,
            mfc="w",
            ls="none",
            alpha=0.5,
            visible=False,
            label="_nolegend_",
        )
        props.update(marker_props if marker_props is not None else {})
        self._markers = Line2D(x, y, animated=useblit, **props)
        self.ax.add_line(self._markers)
        self.artist = self._markers

    @property
    def x(self):
        return self._markers.get_xdata()

    @property
    def y(self):
        return self._markers.get_ydata()

    def set_data(self, pts, y=None):
        """Set x and y positions of handles"""
        if y is not None:
            x = pts
            pts = np.array([x, y])
        self._markers.set_data(pts)

    def set_visible(self, val):
        self._markers.set_visible(val)

    def set_animated(self, val):
        self._markers.set_animated(val)

    def closest(self, x, y):
        """Return index and pixel distance to closest index."""
        pts = np.transpose((self.x, self.y))
        # Transform data coordinates to pixel coordinates.
        pts = self.ax.transData.transform(pts)
        diff = pts - ((x, y))
        if diff.ndim == 2:
            dist = np.sqrt(np.sum(diff**2, axis=1))
            return np.argmin(dist), np.min(dist)
        else:
            return 0, np.sqrt(np.sum(diff**2))


class PolygonSelector(_SelectorWidget):
    """OLD DOCSTRING
    Select a polygon region of an axes.

    Place vertices with each mouse click, and make the selection by completing
    the polygon (clicking on the first vertex). Hold the *ctrl* key and click
    and drag a vertex to reposition it (the *ctrl* key is not necessary if the
    polygon has already been completed). Hold the *shift* key and click and
    drag anywhere in the axes to move all vertices. Press the *esc* key to
    start a new polygon.

    For the selector to remain responsive you must keep a reference to
    it.

    Arguments
    ---------
    ax : :class:`~matplotlib.axes.Axes`
        The parent axes for the widget.
    onselect : function
        When a polygon is completed or modified after completion,
        the `onselect` function is called and passed a list of the vertices as
        ``(xdata, ydata)`` tuples.
    useblit : bool, optional
    lineprops : dict, optional
        The line for the sides of the polygon is drawn with the properties
        given by `lineprops`. The default is ``dict(color='k', linestyle='-',
        linewidth=2, alpha=0.5)``.
    markerprops : dict, optional
        The markers for the vertices of the polygon are drawn with the
        properties given by `markerprops`. The default is ``dict(marker='o',
        markersize=7, mec='k', mfc='k', alpha=0.5)``.
    vertex_select_radius : float, optional
        A vertex is selected (to complete the polygon or to move a vertex)
        if the mouse click is within `vertex_select_radius` pixels of the
        vertex. The default radius is 15 pixels.

    Examples
    --------
    :doc:`/gallery/widgets/polygon_selector_demo`
    """

    def __init__(
        self,
        ax,
        onselect,
        useblit=False,
        lineprops=None,
        markerprops=None,
        vertex_select_radius=15.0,
        base_scale=1.05,
    ):
        # The state modifiers 'move', 'square', and 'center' are expected by
        # _SelectorWidget but are not supported by PolygonSelector
        # Note: could not use the existing 'move' state modifier in-place of
        # 'move_all' because _SelectorWidget automatically discards 'move'
        # from the state on button release.

        state_modifier_keys = dict(
            clear="escape",
            move_vertex="control",
            move_all="shift",
            finished="enter",
            next="alt",
            delete="del",
            move="not-applicable",
            square="not-applicable",
            center="not-applicable",
            rescale_all="r",
        )
        _SelectorWidget.__init__(
            self,
            ax,
            onselect,
            useblit=useblit,
            state_modifier_keys=state_modifier_keys,
        )

        self._xs, self._ys = [0], [0]
        self._polygon_completed = False

        if lineprops is None:
            lineprops = dict(color="k", linestyle="-", linewidth=2, alpha=0.5)
        lineprops["animated"] = self.useblit
        self.lineprops = lineprops
        self.current_line = Line2D(self._xs, self._ys, **self.lineprops)
        self.lines = []  # list of line dicts (see _finalise_polygon)
        self.ax.add_line(self.current_line)

        if markerprops is None:
            self.markerprops = dict(mec="k", mfc=lineprops.get("color", "k"))
        else:
            self.markerprops = markerprops
        self._polygon_handles = ToolHandles(
            self.ax,
            self._xs,
            self._ys,
            useblit=self.useblit,
            marker_props=self.markerprops,
        )

        self._active_handle_idx = -1
        self.vertex_select_radius = vertex_select_radius

        self.artists = [self.current_line, self._polygon_handles.artist]
        self.set_visible(True)

        self.base_scale = base_scale

    @property
    def _nverts(self):
        return len(self._xs)

    def _remove_vertex(self, i):
        """Remove vertex with index i."""
        if self._nverts > 2 and self._polygon_completed and i in (0, self._nverts - 1):
            # If selecting the first or final vertex, remove both first and
            # last vertex as they are the same for a closed polygon
            self._xs.pop(0)
            self._ys.pop(0)
            self._xs.pop(-1)
            self._ys.pop(-1)
            # Close the polygon again by appending the new first vertex to the
            # end
            self._xs.append(self._xs[0])
            self._ys.append(self._ys[0])
        else:
            self._xs.pop(i)
            self._ys.pop(i)
        if self._nverts <= 2:
            # If only one point left, return to incomplete state to let user
            # start drawing again
            self._polygon_completed = False

    def _press(self, event):
        """Button press event handler"""

        # Check for selection of a tool handle on current polygon
        if (self._polygon_completed or "move_vertex" in self.state) and len(self._xs) > 0:
            h_idx, h_dist = self._polygon_handles.closest(event.x, event.y)
            if h_dist < self.vertex_select_radius:
                self._active_handle_idx = h_idx

        # Save the vertex positions at the time of the press event (needed to
        # support the 'move_all' state modifier). Also used for rescale_all
        self._xs_at_press, self._ys_at_press = self._xs[:], self._ys[:]

        for line in self.lines:
            line["xs_at_press"] = line["xs"]
            line["ys_at_press"] = line["ys"]

    def _release(self, event):
        """Button release event handler"""

        # Release active tool handle.
        if self._active_handle_idx >= 0:
            if event.button == 3:
                self._remove_vertex(self._active_handle_idx)
                self.draw_polygon()
            self._active_handle_idx = -1

        # Complete the polygon.
        elif len(self._xs) > 3 and self._xs[-1] == self._xs[0] and self._ys[-1] == self._ys[0]:
            self._polygon_completed = True

        # Place new vertex.
        elif (
            not self._polygon_completed
            and "move_all" not in self.state
            and "move_vertex" not in self.state
            and "rescale_all" not in self.state
        ):
            self._xs.insert(-1, event.xdata)
            self._ys.insert(-1, event.ydata)

        if self._polygon_completed:
            self.onselect(self.verts)

    def onmove(self, event):
        """Cursor move event handler and validator"""
        # Method overrides _SelectorWidget.onmove because the polygon selector
        # needs to process the move callback even if there is no button press.
        # _SelectorWidget.onmove include logic to ignore move event if
        # eventpress is None.
        if not self.ignore(event):
            event = self._clean_event(event)
            self._onmove(event)
            return True
        return False

    def _onmove(self, event):
        """Cursor move event handler"""

        # Move the active vertex (ToolHandle).
        if self._active_handle_idx >= 0:
            idx = self._active_handle_idx
            self._xs[idx], self._ys[idx] = event.xdata, event.ydata
            # Also update the end of the polygon line if the first vertex is
            # the active handle and the polygon is completed.
            if idx == 0 and self._polygon_completed:
                self._xs[-1], self._ys[-1] = event.xdata, event.ydata

        # Move all vertices.
        elif "move_all" in self.state and self.eventpress:
            dx = event.xdata - self.eventpress.xdata
            dy = event.ydata - self.eventpress.ydata
            for k in range(len(self._xs)):
                self._xs[k] = self._xs_at_press[k] + dx
                self._ys[k] = self._ys_at_press[k] + dy
            # also need to update all of the past lines
            for line in self.lines:
                new_xs = []
                new_ys = []
                for k in range(len(line["xs"])):
                    new_xs.append(line["xs_at_press"][k] + dx)
                    new_ys.append(line["ys_at_press"][k] + dy)
                line["xs"] = new_xs
                line["ys"] = new_ys
                # line["line_obj"].set_data(new_xs, new_ys)

        # Do nothing if completed or waiting for a move.
        elif (
            self._polygon_completed
            or "move_vertex" in self.state
            or "move_all" in self.state
            or "rescale_all" in self.state
        ):
            return

        # Position pending vertex.
        else:
            if self._xs:
                # Calculate distance to the start vertex.
                x0, y0 = self.current_line.get_transform().transform((self._xs[0], self._ys[0]))
                v0_dist = np.hypot(x0 - event.x, y0 - event.y)
                # Lock on to the start vertex if near it and ready to complete.
                if len(self._xs) > 3 and v0_dist < self.vertex_select_radius:
                    self._xs[-1], self._ys[-1] = self._xs[0], self._ys[0]
                else:
                    self._xs[-1], self._ys[-1] = event.xdata, event.ydata

        self.draw_polygon()

    def _on_key_press(self, event):
        """Key press event handler"""
        # Remove the pending vertex if entering the 'move_vertex' or
        # 'move_all' mode
        if not self._polygon_completed and (
            "move_vertex" in self.state or "move_all" in self.state or "rescale_all" in self.state
        ):
            self._xs, self._ys = self._xs[:-1], self._ys[:-1]
            self.draw_polygon()
        elif "finished" in self.state:
            # we're finished, move on from everything
            if self._polygon_completed:
                self._finalise_polygon()
            plt.close(self.ax.figure)
        elif "next" in self.state:
            # do exactly the same as clear, but finalise this polygon first
            self._finalise_polygon()
            event = self._clean_event(event)
            self._xs, self._ys = [event.xdata], [event.ydata]
            self._polygon_completed = False
            self.set_visible(True)
            self.update()
        elif "delete" in self.state:
            # clear
            event = self._clean_event(event)
            self._xs, self._ys = [event.xdata], [event.ydata]
            self._polygon_completed = False
            self.set_visible(True)
            # clear all finalised lines
            for line in self.lines:
                line["line_obj"].remove()
            self.lines = []
            self.artists = [self.current_line, self._polygon_handles.artist]
            self.draw_polygon()

    def _on_key_release(self, event):
        """Key release event handler"""
        # Add back the pending vertex if leaving the 'move_vertex' or
        # 'move_all' mode (by checking the released key)
        if not self._polygon_completed and (
            event.key == self.state_modifier_keys.get("move_vertex")
            or event.key == self.state_modifier_keys.get("move_all")
            or event.key == self.state_modifier_keys.get("rescale_all")
        ):
            self._xs.append(event.xdata)
            self._ys.append(event.ydata)
            self.draw_polygon()
        # Reset the polygon if the released key is the 'clear' key.
        elif event.key == self.state_modifier_keys.get("clear"):
            event = self._clean_event(event)
            self._xs, self._ys = [event.xdata], [event.ydata]
            self._polygon_completed = False
            self.set_visible(True)

    def draw_polygon(self):
        """Redraw the polygon(s) based on the new vertex positions."""
        self.current_line.set_data(self._xs, self._ys)

        for line in self.lines:
            line["line_obj"].set_data(line["xs"], line["ys"])
        # Only show one tool handle at the start and end vertex of the polygon
        # if the polygon is completed or the user is locked on to the start
        # vertex.
        if self._polygon_completed or (
            len(self._xs) > 3 and self._xs[-1] == self._xs[0] and self._ys[-1] == self._ys[0]
        ):
            self._polygon_handles.set_data(self._xs[:-1], self._ys[:-1])
        else:
            self._polygon_handles.set_data(self._xs, self._ys)
        self.update()

    def _finalise_polygon(self):
        """Copy the current polygon so we can move on to another one"""
        new_line = copy.copy(self.current_line)

        new_line_dict = dict(line_obj=new_line, xs=self._xs, ys=self._ys)

        self.ax.add_line(new_line)
        self.artists.append(new_line)
        self.lines.append(new_line_dict)  # list of line dicts
        self.update()

    @property
    def verts(self):
        """Get the polygon vertices.

        Returns
        -------
        list
            A list of the vertices of the polygon as ``(xdata, ydata)`` tuples. for each
            polygon (A, B, ...) selected
            ``[ [(Ax1, Ay1), (Ax2, Ay2)], [(Bx1, By1), (Bx2, By2)] ]``
        """
        # return list(zip(self._xs[:-1], self._ys[:-1]))
        pts = []
        for line in self.lines:
            pts.append(list(zip(line["xs"], line["ys"])))
        return pts

    @property
    def xy_verts(self):
        """
        Return list of the vertices for each polygon in the format:
        [ ( [Ax1, Ax2, ...], [Ay1, Ay2, ...] ), ( [Bx1, Bx2, ...], [By1, By2, ...] ) ]
        """
        pts = []
        for line in self.lines:
            pts.append(list(zip(line["xs"], line["ys"])))

        polygons = []
        for p in pts:
            x = []
            y = []
            for c in p:
                x.append(c[0])
                y.append(c[1])
            polygons.append((x, y))
        return polygons

    def _on_scroll(self, event):
        if "rescale_all" in self.state:
            if event.button == "up":
                # zoom in
                scale_factor = self.base_scale
            elif event.button == "down":
                # zoom out
                scale_factor = 1 / self.base_scale
            else:
                scale_factor = 1

            if self._xs and self._ys:
                center_x, center_y = (np.mean(self._xs), np.mean(self._ys))
                for k, _ in enumerate(self._xs):
                    self._xs[k] = (self._xs[k] - center_x) * scale_factor + center_x
                    self._ys[k] = (self._ys[k] - center_y) * scale_factor + center_y

            # update past lines
            for line in self.lines:
                if line["xs"] and line["ys"]:
                    new_xs = []
                    new_ys = []
                    cx_line, cy_line = (
                        np.mean(line["xs"]),
                        np.mean(line["ys"]),
                    )
                    for k, _ in enumerate(line["xs"]):
                        new_xs.append((line["xs"][k] - cx_line) * scale_factor + cx_line)
                        new_ys.append((line["ys"][k] - cy_line) * scale_factor + cy_line)
                    line["xs"] = new_xs
                    line["ys"] = new_ys
                    line["line_obj"].set_data(new_xs, new_ys)
        else:
            return

        self.draw_polygon()


# ======================================================================================
# ======================================================================================
# ======================================================================================
# end old code
# ======================================================================================
# ======================================================================================
# ======================================================================================


class LineSelector(_SelectorWidget):
    """similar to PolygonSelector but an open line."""

    def __init__(
        self,
        ax,
        onselect,
        useblit=False,
        lineprops=None,
        markerprops=None,
        vertex_select_radius=15.0,
        ondraw=lambda x: None,
    ):
        # The state modifiers 'move', 'square', and 'center' are expected by
        # _SelectorWidget but are not supported by LineSelector
        # Note: could not use the existing 'move' state modifier in-place of
        # 'move_all' because _SelectorWidget automatically discards 'move'
        # from the state on button release.

        state_modifier_keys = dict(
            clear="escape",
            move_vertex="control",
            move_all="shift",
            delete="del",
            finished="enter",
            next="alt",
            move="not-applicable",
            square="not-applicable",
            center="not-applicable",
        )
        _SelectorWidget.__init__(
            self,
            ax,
            onselect,
            useblit=useblit,
            state_modifier_keys=state_modifier_keys,
        )
        self._xs, self._ys = [0], [0]
        self._line_completed = False

        if lineprops is None:
            lineprops = dict(color="k", linestyle="-", linewidth=2, alpha=0.5)
        lineprops["animated"] = self.useblit
        self.lineprops = lineprops
        self.current_line = Line2D(self._xs, self._ys, **self.lineprops)
        self.finished_line = None
        self.ax.add_line(self.current_line)

        if markerprops is None:
            self.markerprops = dict(mec="k", mfc=lineprops.get("color", "k"))
        else:
            self.markerprops = markerprops
        self._line_handles = ToolHandles(
            self.ax,
            self._xs,
            self._ys,
            useblit=self.useblit,
            marker_props=self.markerprops,
        )

        self._active_handle_idx = -1
        self.vertex_select_radius = vertex_select_radius

        self.artists = [self.current_line, self._line_handles.artist]
        self.set_visible(True)
        self.ondraw = ondraw  # fn that takes single arg (verts) and does whatever with it.

    @property
    def _nverts(self):
        return len(self._xs)

    def _remove_vertex(self, i):
        """Remove vertex with index i."""
        self._xs.pop(i)
        self._ys.pop(i)

    def _press(self, event):
        """Button press event handler"""

        # Check for selection of a tool handle on current polygon
        if (self._line_completed or "move_vertex" in self.state) and len(self._xs) > 0:
            h_idx, h_dist = self._line_handles.closest(event.x, event.y)
            if h_dist < self.vertex_select_radius:
                self._active_handle_idx = h_idx

        # Save the vertex positions at the time of the press event (needed to
        # support the 'move_all' state modifier). Also used for rescale_all
        self._xs_at_press, self._ys_at_press = self._xs[:], self._ys[:]

        if self.finished_line is not None:
            self.finished_line["xs_at_press"] = self.finished_line["xs"]
            self.finished_line["ys_at_press"] = self.finished_line["ys"]

    def _release(self, event):
        """Button release event handler"""

        # Release active tool handle.
        if self._active_handle_idx >= 0:
            if event.button == 3:
                self._remove_vertex(self._active_handle_idx)
                self.draw_line()
            self._active_handle_idx = -1

        # Place new vertex.
        elif (
            not self._line_completed
            and "move_all" not in self.state
            and "move_vertex" not in self.state
        ):
            self._xs.insert(-1, event.xdata)
            self._ys.insert(-1, event.ydata)

    def onmove(self, event):
        """Cursor move event handler and validator"""
        # Method overrides _SelectorWidget.onmove because the polygon selector
        # needs to process the move callback even if there is no button press.
        # _SelectorWidget.onmove include logic to ignore move event if
        # eventpress is None.
        if not self.ignore(event):
            event = self._clean_event(event)
            self._onmove(event)
            return True
        return False

    def _onmove(self, event):
        """Cursor move event handler"""

        # Move the active vertex (ToolHandle).
        if self._active_handle_idx >= 0:
            idx = self._active_handle_idx
            self._xs[idx], self._ys[idx] = event.xdata, event.ydata
            # Also update the end of the polygon line if the first vertex is
            # the active handle and the polygon is completed.
            if idx == 0 and self._line_completed:
                self._xs[-1], self._ys[-1] = event.xdata, event.ydata

        # Move all vertices.
        elif "move_all" in self.state and self.eventpress:
            dx = event.xdata - self.eventpress.xdata
            dy = event.ydata - self.eventpress.ydata
            for k in range(len(self._xs)):
                self._xs[k] = self._xs_at_press[k] + dx
                self._ys[k] = self._ys_at_press[k] + dy

            if self.finished_line is not None:
                new_xs, new_ys = [], []
                for k in range(len(self.finished_line["xs"])):
                    new_xs.append(self.finished_line["xs_at_press"][k] + dx)
                    new_ys.append(self.finished_line["ys_at_press"][k] + dy)
                self.finished_line["xs"] = new_xs
                self.finished_line["ys"] = new_ys

        # Do nothing if completed or waiting for a move.
        elif self._line_completed or "move_vertex" in self.state or "move_all" in self.state:
            return
        else:
            if self._xs:
                self._xs[-1], self._ys[-1] = event.xdata, event.ydata

        self.draw_line()

    def _on_key_press(self, event):
        """Key press event handler"""
        # Remove the pending vertex if entering the 'move_vertex' or
        # 'move_all' mode
        if not self._line_completed and ("move_vertex" in self.state or "move_all" in self.state):
            self._xs, self._ys = self._xs[:-1], self._ys[:-1]
            self.draw_line()
        elif "finished" in self.state:
            # we're finished, move on from everything
            if self._line_completed:
                self._finalise_line()
            plt.close(self.ax.figure)
        elif "next" in self.state:
            # do exactly the same as clear, but finalise the line first
            # new for lineselector: remove last point (added by 'alt'), and redraw.
            self._xs, self._ys = self._xs[:-1], self._ys[:-1]
            self.draw_line()
            self._finalise_line()
            event = self._clean_event(event)
            self._xs, self._ys = [event.xdata], [event.ydata]
            self._line_completed = True
            self.set_visible(True)
            self.update()
            self.onselect(self.verts)
        elif "delete" in self.state:
            # clear
            event = self._clean_event(event)
            self._xs, self._ys = [event.xdata], [event.ydata]
            self._line_completed = False
            self.set_visible(True)
            # clear finished line
            if self.finished_line is not None:
                self.finished_line["line_obj"].remove()
                self.finished_line = None
            self.artists = [self.current_line, self._line_handles.artist]
            self.draw_line()

    def _on_key_release(self, event):
        """Key release event handler"""
        # Add back the pending vertex if leaving the 'move_vertex' or
        # 'move_all' mode (by checking the released key)
        if not self._line_completed and (
            event.key == self.state_modifier_keys.get("move_vertex")
            or event.key == self.state_modifier_keys.get("move_all")
        ):
            self._xs.append(event.xdata)
            self._ys.append(event.ydata)
            self.draw_line()
        # Reset the polygon if the released key is the 'clear' key.
        elif event.key == self.state_modifier_keys.get("clear"):
            event = self._clean_event(event)
            self._xs, self._ys = [event.xdata], [event.ydata]
            self.set_visible(True)

    def draw_line(self):
        """Redraw the line based on the new vertex positions."""
        self.current_line.set_data(self._xs, self._ys)

        if self.finished_line is not None:
            self.finished_line["line_obj"].set_data(
                self.finished_line["xs"], self.finished_line["ys"]
            )
        # Only show one tool handle at the start and end vertex of the polygon
        # if the line is completed or the user is locked on to the start
        # vertex. --> seems irrelevant here?
        if self._line_completed or (
            len(self._xs) > 3 and self._xs[-1] == self._xs[0] and self._ys[-1] == self._ys[0]
        ):
            self._line_handles.set_data(self._xs[:-1], self._ys[:-1])
        else:
            self._line_handles.set_data(self._xs, self._ys)
        self.update()
        if self.finished_line:
            self.ondraw(self.verts)
        else:
            self.ondraw(self.current_verts)

    def _finalise_line(self):
        """Copy the current line so we can move on to another one"""
        new_line = copy.copy(self.current_line)

        new_line_dict = dict(line_obj=new_line, xs=self._xs, ys=self._ys)

        self.ax.add_line(new_line)
        self.artists.append(new_line)
        self.finished_line = new_line_dict
        self.update()

    @property
    def verts(self):
        """Get the line vertices.

        Returns
        -------
        list
            A list of the vertices of the line as ``(xdata, ydata)`` tuples.
            `[(Ax1, Ay1), (Ax2, Ay2)]`
        """
        # return list(zip(self._xs[:-1], self._ys[:-1]))
        pts = []
        if self.finished_line is not None:
            pts = list(zip(self.finished_line["xs"], self.finished_line["ys"]))
        return pts

    @property
    def current_verts(self):
        ret = list(zip(self.current_line.get_xdata(), self.current_line.get_ydata()))
        return ret

    @property
    def xy_verts(self):
        """
        Return list of the vertices for the line in this format:
        ( [Ax1, Ax2, ...], [Ay1, Ay2, ...] )
        """
        pts = []
        if self.finished_line is not None:
            pts = list(zip(self.finished_line["xs"], self.finished_line["ys"]))

        x, y = [], []
        for p in pts:
            x.append(p[0])
            y.append(p[1])
        return (x, y)


# ======================================================================================


class BulkLinecutWidget:
    """
    How to use
    ----------
    import matplotlib.pyplot as plt
    import numpy as np
    from qdmpy.shared.linecut import BulkLinecutWidget
    # TODO update

    path = "<WHATEVER>"
    times = [0.325, 1, 5, 10, 20, 21, 22, 25, 30, 40]
    paths = [f"{path}/{t}.txt" for t in times]
    images = [np.loadtxt(p) for p in paths]
    selector_image = images[4]

    fig, axs = plt.subplots(ncols=3, figsize=(12, 6))
    axs[0].imshow(selector_image)  # (data can be nans if you want an empty selector)
    selector = BulkLinecutWidget(*axs, images, times)
    plt.show()
    selector.disconnect(path="/home/samsc/share/result.json")
    """

    def __init__(
        self,
        imax: plt.Axes,
        profax: plt.Axes,
        resax: plt.Axes,
        images: list[npt.NDArray] | tuple[npt.NDArray],
        xlabels: list[str] | tuple[str],
        style: dict | None = None,
        useblit: bool = False,
        dointegral: bool = True,
    ):
        # check that input ax has an imshow (else not so useful eh)
        if not any([[isinstance(t, AxesImage) for t in imax.get_children()]]):
            raise ValueError("input axis does not contain an AxesImage (imshow).")
        self.images = images
        self.xlabels = xlabels

        self.pts = []

        dflt_style = {
            "lineprops": {
                "color": "k",
                "linestyle": "-",
                "linewidth": 1.0,
                "alpha": 0.5,
            },
            "markerprops": {
                "marker": "o",
                "markersize": 2.0,
                "mec": "k",
                "mfc": "k",
                "alpha": 0.5,
            },
        }

        self.lp = dflt_style["lineprops"]
        self.mp = dflt_style["markerprops"]

        if style is not None:
            if "lineprops" in style and isinstance(style["lineprops"], dict):
                for key, item in style["lineprops"].items():
                    self.lp[key] = item
            if "markerprops" in style and isinstance(style["markerprops"], dict):
                for key, item in style["markerprops"].items():
                    self.mp[key] = item

        self.line_selector = LineSelector(
            imax,
            self.onselect,
            ondraw=self.ondraw,
            lineprops=self.lp,
            markerprops=self.mp,
            vertex_select_radius=7.5 * 2.0,
            useblit=useblit,
        )
        self.imax = imax
        self.profax = profax
        self.resax = resax
        self.integrals = [0 for i in xlabels]
        self.integral = 0
        self.do_integral = dointegral

        self.canvas = self.imax.figure.canvas

        dummy_x = np.zeros((5, len(xlabels)))
        dummy_y = np.zeros((5, len(xlabels)))
        self.profiles = self.profax.plot(dummy_x, dummy_y, marker="o", ls="-", label=xlabels)
        # handles, _ = self.profax.get_legend_handles_labels()
        # self.profax.legend(handles, self.xlabels, loc="upper left")
        self.profax.legend()

        (self.integrals_plot,) = self.resax.plot(xlabels, self.integrals, "ko-")

    def ondraw(self, verts):
        if len(verts) == 1:
            # change all profiles
            for p, prof in enumerate(self.profiles):
                prof.set_xdata([0])
                prof.set_ydata(self.images[p][int(verts[0][1]), int(verts[0][0])])
        else:
            idxs, jdxs = zip(*verts)
            pxl_ar = [0]
            i_lst = []
            j_lst = []
            for n in range(len(idxs) - 1):
                i0, i1 = idxs[n], idxs[n + 1]
                j0, j1 = jdxs[n], jdxs[n + 1]
                num = int(np.sqrt((i1 - i0) ** 2 + (j1 - j0) ** 2) * 2)  # *2 to be safe
                if not num:
                    continue

                ivec = np.linspace(i0, i1, num).astype(int)
                jvec = np.linspace(j0, j1, num).astype(int)

                new_ij = [
                    (ivec[v], jvec[u])
                    for v, u in zip(range(len(ivec) - 1), range(len(jvec) - 1))
                    if ivec[v + 1] != ivec[v] or jvec[u + 1] != jvec[u]
                ]
                new_ij.append((ivec[-1], jvec[-1]))
                new_is, new_js = zip(*new_ij)
                if not i_lst:
                    i_lst, j_lst = list(new_is), list(new_js)
                    i_ar, j_ar = np.asarray(new_is), np.asarray(new_js)
                else:
                    i_lst.extend(list(new_is[1:]))
                    j_lst.extend(list(new_js[1:]))
                    i_ar, j_ar = np.asarray(new_is[1:]), np.asarray(new_js[1:])
                if not (i_ar.ndim and i_ar.size):
                    continue
                pxl_ar.extend(
                    (pxl_ar[-1] + np.sqrt((i_ar - i_ar[0]) ** 2 + (j_ar - j_ar[0]) ** 2)).tolist()
                )

            for p, prof in enumerate(self.profiles):
                z = self.images[p][j_lst, i_lst]
                if z.ndim and z.size:  # ensure no empty array
                    prof.set_xdata(pxl_ar[1:])  # get rid of initial 0 on pxl_ar (bit hacky)
                    prof.set_ydata(list(z))
                    if self.do_integral:
                        self.integrals[p] = integrate.simpson(z, pxl_ar[1:])
                    else:
                        self.integrals[p] = np.nan
                else:
                    # dummy data for this guy...
                    prof.set_xdata((np.nan,))
                    prof.set_ydata((np.nan,))
            self.integrals_plot.set_ydata(self.integrals)
        self.profax.relim()
        self.profax.autoscale_view()
        self.resax.relim()
        self.resax.autoscale_view()
        self.canvas.draw()
        self.canvas.flush_events()

    def onselect(self, verts):
        # only called when line is finished
        self.pts = verts  # list of vertices `[(Ax1, Ay1), (Ax2, Ay2)]`
        self.canvas.draw_idle()

    def disconnect(self, path=None):
        if path is None:
            print()
            print("xlabels:")
            print(self.xlabels)
            print()
            print("profiles xdata:")
            print(np.transpose([prof.get_xdata() for prof in self.profiles]).tolist())
            print("profiles ydata:")
            print(np.transpose([prof.get_ydata() for prof in self.profiles]).tolist())
            print("integrals:")
            print(self.integrals)
            print()
        else:
            output_dict = {
                "xlabels": self.xlabels,
                "integrals": self.integrals,
                "profile_x": np.transpose([prof.get_xdata() for prof in self.profiles]).tolist(),
                "profile_y": np.transpose([prof.get_ydata() for prof in self.profiles]).tolist(),
            }
            dukit.json2dict.dict_to_json(output_dict, path)
        self.line_selector.disconnect_events()
        self.canvas.draw_idle()


class LinecutSelectionWidget:
    """
    How to Use
    ----------
    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(data) # (data may be nans if you want empty selector)
    selector = LinecutSelectionWidget(axs[0], axs[1], ...)
    plt.show()
    selector.disconnect()

    """

    def __init__(self, imax, lineax, data, style=None, useblit=False):
        # check that input ax has an imshow (else not so useful eh)
        if not any([[isinstance(t, AxesImage) for t in imax.get_children()]]):
            raise ValueError("input axis does not contain an AxesImage (imshow).")

        self.data = data
        self.imax = imax
        self.lineax = lineax
        self.integral = 0

        self.canvas = self.imax.figure.canvas

        dflt_style = {
            "lineprops": {
                "color": "k",
                "linestyle": "-",
                "linewidth": 1.0,
                "alpha": 0.5,
            },
            "markerprops": {
                "marker": "o",
                "markersize": 2.0,
                "mec": "k",
                "mfc": "k",
                "alpha": 0.5,
            },
        }

        self.lp = dflt_style["lineprops"]
        self.mp = dflt_style["markerprops"]
        if style is not None:
            if "lineprops" in style and isinstance(style["lineprops"], dict):
                for key, item in style["lineprops"].items():
                    self.lp[key] = item
            if "markerprops" in style and isinstance(style["markerprops"], dict):
                for key, item in style["markerprops"].items():
                    self.mp[key] = item

        vsr = 7.5 * self.mp["markersize"]  # linear scaling on what our select radius is

        (self.profile,) = self.lineax.plot([1, 2, 3], [1, 2, 3], "ko-")
        self.lineax.title.set_text(f"Integral: {self.integral}")

        self.pts = []
        self.line_selector = LineSelector(
            imax,
            self.onselect,
            ondraw=self.ondraw,
            lineprops=self.lp,
            markerprops=self.mp,
            vertex_select_radius=vsr,
            useblit=useblit,
        )

    def ondraw(self, verts):
        if len(verts) == 1:
            self.profile.set_xdata([0])
            self.profile.set_ydata(self.data[int(verts[0][1]), int(verts[0][0])])
            self.lineax.title.set_text(f"Integral: {self.integral}")
        else:
            idxs, jdxs = zip(*verts)
            t_ar = [0]
            i_lst = []
            j_lst = []
            for n in range(len(idxs) - 1):
                try:
                    i0, i1 = idxs[n], idxs[n + 1]
                    j0, j1 = jdxs[n], jdxs[n + 1]
                    # *2 in there to be safe
                    num = int(np.sqrt((i1 - i0) ** 2 + (j1 - j0) ** 2) * 2)
                except (IndexError, TypeError, ValueError):
                    num = 0
                if not num:
                    continue

                ivec = np.linspace(i0, i1, num).astype(int)
                jvec = np.linspace(j0, j1, num).astype(int)

                new_ij = [
                    (ivec[v], jvec[u])
                    for v, u in zip(range(len(ivec) - 1), range(len(jvec) - 1))
                    if ivec[v + 1] != ivec[v] or jvec[u + 1] != jvec[u]
                ]
                new_ij.append((ivec[-1], jvec[-1]))
                new_is, new_js = zip(*new_ij)
                if not i_lst:
                    i_lst, j_lst = list(new_is), list(new_js)
                    i_ar, j_ar = np.asarray(new_is), np.asarray(new_js)
                else:
                    i_lst.extend(list(new_is[1:]))
                    j_lst.extend(list(new_js[1:]))
                    i_ar, j_ar = np.asarray(new_is[1:]), np.asarray(new_js[1:])
                if not (i_ar.ndim and i_ar.size):
                    continue
                t_ar.extend(
                    (t_ar[-1] + np.sqrt((i_ar - i_ar[0]) ** 2 + (j_ar - j_ar[0]) ** 2)).tolist()
                )

            z = self.data[j_lst, i_lst]
            if z.ndim and z.size:  # ensure no empty array
                self.profile.set_xdata(t_ar[1:])  # get rid of initial 0 on t_ar (bit hacky)
                self.profile.set_ydata(list(z))
                self.integral = integrate.simpson(z, t_ar[1:])
                self.lineax.title.set_text(f"Integral: {self.integral:.6e}")
        self.lineax.relim()
        self.lineax.autoscale_view()
        self.canvas.draw()
        self.canvas.flush_events()

    def onselect(self, verts):
        # only called when line is finished
        self.pts = verts  # list of vertices `[(Ax1, Ay1), (Ax2, Ay2)]`
        self.canvas.draw_idle()

    def disconnect(self, path=None):
        if path is None:
            print()
            print("profile xdata:")
            print(self.profile.get_xdata())
            print("profile ydata:")
            print(self.profile.get_ydata())
            print("integral:")
            print(self.integral)
            print()
        else:
            output_dict = {
                "profile_x": self.profile.get_xdata(),
                "profile_y": self.profile.get_ydata(),
                "integral": self.integral,
            }
            dukit.json2dict.dict_to_json(output_dict, path)
        self.line_selector.disconnect_events()
        self.canvas.draw_idle()
