"""
ergane.visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~
fastplotlib-backed animated visualisation for AthenaK / Athena++ simulations.

Typical usage (via SimulationData):
    >>> viz = sim.visualize(fields=["density", "pressure"])
    >>> viz.show()

Standalone usage:
    >>> from ergane import SimulationData, Visualization
    >>> sim = SimulationData(athinp=..., datafolder=...)
    >>> viz = Visualization(sim, fields=["density", "pressure", "velx", "vely"])
    >>> viz.show()
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from .simulation_data import SimulationData


# ── Default colourmap table ───────────────────────────────────────────────────

_DEFAULT_CMAPS: dict[str, str] = {
    "density":  "inferno",
    "pressure": "inferno",
    "velx":     "seismic",
    "vely":     "seismic",
    "velz":     "seismic",
    "bx":       "bwr",
    "by":       "bwr",
    "bz":       "bwr",
}

# Human-readable subplot titles
_TITLES: dict[str, str] = {
    "density":  "Density",
    "pressure": "Pressure",
    "velx":     "v_x",
    "vely":     "v_y",
    "velz":     "v_z",
    "bx":       "B_x",
    "by":       "B_y",
    "bz":       "B_z",
}


# ── Layout helper ─────────────────────────────────────────────────────────────

def _grid_shape(n: int) -> tuple[int, int]:
    """
    Choose a (rows, cols) layout for *n* subplots that is as square as possible.

    Examples:  1→(1,1)  2→(1,2)  3→(1,3)  4→(2,2)  5→(2,3)  6→(2,3)
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


# ── Visualization class ───────────────────────────────────────────────────────

class Visualization:
    """
    Animated fastplotlib figure for an AthenaK / Athena++ simulation.

    Parameters
    ----------
    sim : SimulationData
        The simulation to visualise.
    fields : list of str, optional
        Which fields to include.  Defaults to ``sim.fields_available``.
    cmaps : dict, optional
        Per-field colourmap overrides, e.g. ``{"density": "plasma"}``.
    size : tuple[int, int], optional
        Window size in pixels ``(width, height)``.  Auto-sized if omitted.

    Attributes
    ----------
    figure : fastplotlib.Figure
        The underlying figure — customise subplots, titles, etc. before
        calling ``.show()``.
    """

    def __init__(
        self,
        sim: SimulationData,
        fields: Optional[List[str]] = None,
        cmaps: Optional[dict[str, str]] = None,
        size: Optional[tuple[int, int]] = None,
    ):
        import fastplotlib as fpl  # imported here so the module is importable without fpl

        self._sim = sim
        self._fields = fields if fields is not None else sim.fields_available
        self._cmaps = dict(_DEFAULT_CMAPS)
        if cmaps:
            self._cmaps.update(cmaps)

        n = len(self._fields)
        rows, cols = _grid_shape(n)

        # Build subplot name grid (pad with empty strings)
        names = []
        for r in range(rows):
            row_names = []
            for c in range(cols):
                idx = r * cols + c
                if idx < n:
                    row_names.append(_TITLES.get(self._fields[idx], self._fields[idx]))
                else:
                    row_names.append("")
            names.append(row_names)

        # Auto window size: ~500 px per column / row, capped sensibly
        if size is None:
            w = min(500 * cols, 1800)
            h = min(500 * rows, 1200)
            size = (w, h)

        self.figure = fpl.Figure(
            shape=(rows, cols),
            size=size,
            names=names,
            controller_ids="sync",
        )

        # ── Load the last frame as the initial display ──────────────────
        frame = self._sim.get_frame(self._sim.frame_numbers[-1])

        self._images: dict[str, object] = {}  # field → fpl ImageGraphic
        self._subplot_coords: dict[str, tuple[int, int]] = {}

        for idx, field_name in enumerate(self._fields):
            r, c = divmod(idx, cols)
            data = getattr(frame, field_name)
            if data is None:
                continue  # field not available — leave subplot blank

            img = self.figure[r, c].add_image(
                data=data,
                name=field_name,
            )
            img.cmap = self._cmaps.get(field_name, "inferno")
            self._images[field_name] = img
            self._subplot_coords[field_name] = (r, c)

        for subplot in self.figure:
            subplot.toolbar = False

        # ── Animation state ─────────────────────────────────────────────
        self._frame_idx = self._sim.n_frames - 1  # we displayed the last frame first

        def _update(figure_instance):
            self._frame_idx = (self._frame_idx + 1) % self._sim.n_frames
            num = self._sim.frame_numbers[self._frame_idx]
            f = self._sim.get_frame(num)
            for field_name, img in self._images.items():
                data = getattr(f, field_name)
                if data is not None:
                    img.data = data

        self.figure.add_animations(_update)

    # ── Public API ────────────────────────────────────────────────────────────

    def show(self) -> None:
        """
        Open the figure window and start the animation loop.

        This call **blocks** until the window is closed (equivalent to the
        ``fpl.loop.run()`` pattern used in the original scripts).
        """
        import fastplotlib as fpl

        self.figure.show()
        fpl.loop.run()

    def __repr__(self) -> str:
        return (
            f"<Visualization  sim='{self._sim.basename}'  "
            f"fields={self._fields}>"
        )
