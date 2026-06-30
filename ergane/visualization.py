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
    "density":  "log10(Density)",
    "pressure": "Pressure",
    "velx":     "$v_x$",
    "vely":     "$v_y$",
    "velz":     "$v_z$",
    "bx":       "$B_x$",
    "by":       "$B_y$",
    "bz":       "$B_z$",
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
    Animated figure for an AthenaK / Athena++ simulation.

    Acts as a factory that returns either a FastplotlibVisualization
    or a MatplotlibVisualization instance depending on the chosen backend.

    Parameters
    ----------
    sim : SimulationData
        The simulation to visualise.
    fields : list of str, optional
        Which fields to include.  Defaults to ``sim.fields_available``.
    cmaps : dict, optional
        Per-field colourmap overrides, e.g. ``{"density": "plasma"}``.
    backend : str, optional
        The visualization backend to use: ``"fastplotlib"`` (default) or ``"matplotlib"``.
    size : tuple[int, int], optional
        Window size in pixels ``(width, height)``.  Auto-sized if omitted.
    **kwargs
        Additional backend-specific arguments.
    """

    def __new__(
        cls,
        sim: SimulationData,
        fields: Optional[List[str]] = None,
        cmaps: Optional[dict[str, str]] = None,
        backend: str = "fastplotlib",
        *args,
        **kwargs,
    ) -> Visualization:
        if cls is Visualization:
            if backend == "matplotlib":
                return object.__new__(MatplotlibVisualization)
            elif backend == "fastplotlib":
                return object.__new__(FastplotlibVisualization)
            else:
                raise ValueError(f"Unknown backend: {backend}")
        return object.__new__(cls)

    def show(self) -> None:
        """
        Open the figure window and start the animation loop.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


# ── Fastplotlib Backend ───────────────────────────────────────────────────────

class FastplotlibVisualization(Visualization):
    """
    Animated fastplotlib figure for an AthenaK / Athena++ simulation.
    """

    def __init__(
        self,
        sim: SimulationData,
        fields: Optional[List[str]] = None,
        cmaps: Optional[dict[str, str]] = None,
        backend: str = "fastplotlib",
        size: Optional[tuple[int, int]] = None,
        **kwargs,
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
        self._histogram_tools: dict[str, object] = {}  # field → fpl HistogramLUTTool

        for idx, field_name in enumerate(self._fields):
            r, c = divmod(idx, cols)
            data = getattr(frame, field_name)
            if data is None:
                continue  # field not available — leave subplot blank

            if field_name == "density":
                data = np.log10(np.maximum(data, 1e-10))

            # Flip vertically to match origin='lower' (standard physical coordinate system)
            data = np.flipud(data)

            img = self.figure[r, c].add_image(
                data=data,
                name=field_name,
            )
            img.cmap = self._cmaps.get(field_name, "inferno")
            self._images[field_name] = img
            self._subplot_coords[field_name] = (r, c)

            # ── Add interactive histogram/colorbar tool for this image ───
            try:
                from fastplotlib.tools import HistogramLUTTool

                hist_tool = HistogramLUTTool(
                    data=data,
                    images=img,
                    name=f"{field_name}_histogram",
                )
                self.figure[r, c].docks["right"].add_graphic(hist_tool)
                self.figure[r, c].docks["right"].size = 80
                self.figure[r, c].docks["right"].auto_scale(maintain_aspect=False)
                self.figure[r, c].docks["right"].controller.enabled = False

                self._histogram_tools[field_name] = hist_tool
            except Exception as e:
                # If histogram creation fails, log warning but continue
                print(f"Warning: Could not create histogram/colorbar tool for {field_name}: {e}")

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
                    if field_name == "density":
                        data = np.log10(np.maximum(data, 1e-10))
                    # Flip vertically to match origin='lower'
                    data = np.flipud(data)
                    img.data = data

        self.figure.add_animations(_update)

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
            f"<FastplotlibVisualization  sim='{self._sim.basename}'  "
            f"fields={self._fields}>"
        )


# ── Matplotlib Backend ────────────────────────────────────────────────────────

class MatplotlibVisualization(Visualization):
    """
    Animated matplotlib figure for an AthenaK / Athena++ simulation.
    """

    def __init__(
        self,
        sim: SimulationData,
        fields: Optional[List[str]] = None,
        cmaps: Optional[dict[str, str]] = None,
        backend: str = "matplotlib",
        size: Optional[tuple[int, int]] = None,
        interval: int = 100,
        **kwargs,
    ):
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        self._sim = sim
        self._fields = fields if fields is not None else sim.fields_available
        self._cmaps = dict(_DEFAULT_CMAPS)
        if cmaps:
            self._cmaps.update(cmaps)

        n = len(self._fields)
        rows, cols = _grid_shape(n)

        # Matplotlib figsize is in inches. Convert from size (pixels) / 100.
        if size is None:
            w = min(5 * cols, 18)
            h = min(5 * rows, 12)
            figsize = (w, h)
        else:
            figsize = (size[0] / 100.0, size[1] / 100.0)

        # Create figure and axes
        self.figure, axes = plt.subplots(
            rows,
            cols,
            figsize=figsize,
            squeeze=False,
        )
        self._axes = axes

        # Hide any unused axes in the grid
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            self.figure.delaxes(self._axes[r, c])

        # Load the last frame first to show it initially
        self._frame_idx = self._sim.n_frames - 1
        frame = self._sim.get_frame(self._sim.frame_numbers[self._frame_idx])

        self._images: dict[str, object] = {}
        self._subplot_coords: dict[str, tuple[int, int]] = {}

        for idx, field_name in enumerate(self._fields):
            r, c = divmod(idx, cols)
            ax = self._axes[r, c]
            data = getattr(frame, field_name)
            if data is None:
                continue

            if field_name == "density":
                data = np.log10(np.maximum(data, 1e-10))

            cmap = self._cmaps.get(field_name, "inferno")
            # We use origin='lower' as standard for simulation output grids
            im = ax.imshow(data, cmap=cmap, origin='lower')
            ax.set_title(_TITLES.get(field_name, field_name))
            self.figure.colorbar(im, ax=ax)
            
            self._images[field_name] = im
            self._subplot_coords[field_name] = (r, c)

        self.figure.suptitle(f"Time: {frame.time:.4f} (Frame {frame.number})")
        self.figure.tight_layout()

        # Set up matplotlib animation
        self.ani = animation.FuncAnimation(
            self.figure,
            self._update,
            frames=self._sim.n_frames,
            interval=interval,
            blit=False,
            cache_frame_data=False,
        )

    def _update(self, frame_idx: int):
        num = self._sim.frame_numbers[frame_idx]
        f = self._sim.get_frame(num)
        for field_name, im in self._images.items():
            data = getattr(f, field_name)
            if data is not None:
                if field_name == "density":
                    data = np.log10(np.maximum(data, 1e-10))
                im.set_data(data)

        self.figure.suptitle(f"Time: {f.time:.4f} (Frame {num})")
        return list(self._images.values())

    def show(self) -> None:
        """
        Open the figure window and start the animation loop.
        """
        import matplotlib.pyplot as plt
        plt.show()

    def __repr__(self) -> str:
        return (
            f"<MatplotlibVisualization  sim='{self._sim.basename}'  "
            f"fields={self._fields}>"
        )