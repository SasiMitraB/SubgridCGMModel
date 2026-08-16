"""
ergane.visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-backend animated visualisation for AthenaK / Athena++ simulations.

Backends
--------
- ``"fastplotlib"`` (default) — GPU-accelerated interactive window.
- ``"matplotlib"``            — Animated matplotlib window (blocking).
- ``"jupyter"``               — Inline ipywidgets display for Jupyter notebooks.
- ``"auto"``                  — Selects ``"jupyter"`` inside a Jupyter kernel,
                                ``"fastplotlib"`` otherwise.

Typical usage (via SimulationData):
    >>> viz = sim.visualize(backend='jupyter')   # Jupyter notebook
    >>> viz.show()

Standalone usage:
    >>> from ergane import SimulationData, Visualization
    >>> sim = SimulationData(athinp=..., datafolder=...)
    >>> viz = Visualization(sim, fields=["density", "pressure"], backend='jupyter')
    >>> viz.show()
"""

from __future__ import annotations

import math
from pathlib import Path
from types import MethodType
from typing import List, Optional

import numpy as np

from .simulation_data import SimulationData


# ── Default colourmap table ───────────────────────────────────────────────────

_DEFAULT_CMAPS: dict[str, str] = {
    "density":     "inferno",
    "pressure":    "inferno",
    "eint":        "inferno",
    "temperature": "inferno",
    "velx":        "seismic",
    "vely":        "seismic",
    "velz":        "seismic",
    "scalar_00":   "viridis",
    "bx":          "bwr",
    "by":          "bwr",
    "bz":          "bwr",
}

# Human-readable subplot titles
_TITLES: dict[str, str] = {
    "density":     "log10(Density)",
    "pressure":    "log10(Pressure)",
    "eint":        "log10(Internal Energy)",
    "temperature": "log10(Temperature)",
    "velx":        "$v_x$ (km/s)",
    "vely":        "$v_y$ (km/s)",
    "velz":        "$v_z$ (km/s)",
    "scalar_00":   "Passive Scalar",
    "bx":          "$B_x$",
    "by":          "$B_y$",
    "bz":          "$B_z$",
}


def _field_title(field_name: str) -> str:
    if field_name.startswith("scalar_"):
        suffix = field_name.split("_", 1)[1]
        return f"Passive Scalar {suffix}"
    return _TITLES.get(field_name, field_name)


def _prepare_field_data(
    sim: SimulationData,
    field_name: str,
    data: np.ndarray,
    *,
    log_scale: bool = True,
) -> np.ndarray:
    """Apply log10 to density, pressure, eint, and temperature fields when log_scale is True."""
    if log_scale:
        if field_name == "density":
            floor = 1e-10 * sim.units.density
            return np.log10(np.maximum(data, floor))
        elif field_name in ("pressure", "eint"):
            floor = 1e-10 * sim.units.pressure
            return np.log10(np.maximum(data, floor))
        elif field_name == "temperature":
            floor = 1e-10
            return np.log10(np.maximum(data, floor))
    return data


# ── Layout helper ─────────────────────────────────────────────────────────────

def _grid_shape(n: int) -> tuple[int, int]:
    """
    Choose a (rows, cols) layout for *n* subplots that is as square as possible.

    Examples:  1→(1,1)  2→(1,2)  3→(1,3)  4→(2,2)  5→(2,3)  6→(2,3)
    """
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _is_jupyter() -> bool:
    """Return True when running inside a Jupyter kernel."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and "IPKernelApp" in shell.config
    except ImportError:
        return False


def _render_frame_worker(args):
    (
        athinp_path,
        datafolder_path,
        units_obj,
        frame_num,
        out_png_path,
        fields,
        cmaps,
        clims,
        figsize,
        dpi,
    ) = args

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim = SimulationData(
        athinp=athinp_path,
        datafolder=datafolder_path,
    )
    if units_obj is not None:
        sim.set_units(units_obj)

    frame = sim.get_frame(frame_num)

    n = len(fields)
    rows, cols = _grid_shape(n)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        fig.delaxes(axes[r, c])

    for idx, field_name in enumerate(fields):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        data = getattr(frame, field_name)
        if data is None:
            continue

        data = _prepare_field_data(sim, field_name, data, log_scale=True)
        cmap = cmaps.get(field_name, "inferno")
        clim = clims.get(field_name) if clims else None

        if clim is not None:
            im = ax.imshow(data, cmap=cmap, origin="lower", vmin=clim[0], vmax=clim[1])
        else:
            im = ax.imshow(data, cmap=cmap, origin="lower")

        ax.set_title(_field_title(field_name))
        fig.colorbar(im, ax=ax)

    fig.suptitle(f"Time: {frame.time:.4f} (Frame {frame.number})")
    fig.tight_layout()
    fig.savefig(out_png_path, dpi=dpi)
    plt.close(fig)


def _enable_parallel_save(viz_instance: MatplotlibVisualization, n_jobs_default: int = 16) -> None:
    animation_obj = viz_instance.ani
    original_save = animation_obj.save
    viz_instance._original_save = original_save

    def save_parallel_wrapper(self, filename, *args, **kwargs):
        self._draw_was_started = True
        n_jobs = kwargs.pop("n_jobs", kwargs.pop("n_cores", n_jobs_default))
        if n_jobs > 1:
            return viz_instance.save(filename, *args, n_jobs=n_jobs, **kwargs)
        else:
            return original_save(filename, *args, **kwargs)

    animation_obj.save = MethodType(save_parallel_wrapper, animation_obj)


# ── Visualization class ───────────────────────────────────────────────────────

class Visualization:
    """
    Animated figure for an AthenaK / Athena++ simulation.

    Acts as a factory that returns either a FastplotlibVisualization,
    MatplotlibVisualization, or JupyterVisualization instance depending on 
    the chosen backend.

    Parameters
    ----------
    sim : SimulationData
        The simulation to visualise.
    fields : list of str, optional
        Which fields to include.  Defaults to ``sim.fields_available``.
    cmaps : dict, optional
        Per-field colourmap overrides, e.g. ``{"density": "plasma"}``.
    backend : str, optional
        The visualization backend to use:
          - ``"fastplotlib"`` (default) – GPU-accelerated interactive window.
          - ``"matplotlib"``            – Animated matplotlib window.
          - ``"jupyter"``               – Inline ipywidgets display (Jupyter only).
          - ``"auto"``                  – Uses ``"jupyter"`` if inside a Jupyter
            kernel, ``"fastplotlib"`` otherwise.
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
            if backend == "auto":
                backend = "jupyter" if _is_jupyter() else "fastplotlib"
            if backend == "matplotlib":
                return object.__new__(MatplotlibVisualization)
            elif backend == "fastplotlib":
                return object.__new__(FastplotlibVisualization)
            elif backend == "jupyter":
                return object.__new__(JupyterVisualization)
            else:
                raise ValueError(
                    f"Unknown backend: {backend!r}. "
                    f"Choose from 'fastplotlib', 'matplotlib', 'jupyter', 'auto'."
                )
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
                    row_names.append(_field_title(self._fields[idx]))
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

            data = _prepare_field_data(self._sim, field_name, data, log_scale=True)

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
                    data = _prepare_field_data(self._sim, field_name, data, log_scale=True)
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

            data = _prepare_field_data(self._sim, field_name, data, log_scale=True)

            cmap = self._cmaps.get(field_name, "inferno")
            # We use origin='lower' as standard for simulation output grids
            im = ax.imshow(data, cmap=cmap, origin='lower')
            ax.set_title(_field_title(field_name))
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
        _enable_parallel_save(self, n_jobs_default=16)

    def _update(self, frame_idx: int):
        num = self._sim.frame_numbers[frame_idx]
        f = self._sim.get_frame(num)
        for field_name, im in self._images.items():
            data = getattr(f, field_name)
            if data is not None:
                data = _prepare_field_data(self._sim, field_name, data, log_scale=True)
                im.set_data(data)

        self.figure.suptitle(f"Time: {f.time:.4f} (Frame {num})")
        return list(self._images.values())

    def save(
        self,
        filename: str | Path,
        n_jobs: int = 16,
        fps: int = 60,
        dpi: int = 150,
        writer: str | None = "ffmpeg",
        crf: int = 10,
        lossless: bool = False,
        progress_bar: bool = True,
        **kwargs,
    ) -> None:
        """
        Save the animation to a video file using multi-core parallel rendering.

        Parameters
        ----------
        filename : str or Path
            Target video filepath (e.g. "output.mp4").
        n_jobs : int, optional
            Number of CPU cores to use for parallel frame rendering. Default is 16.
        fps : int, optional
            Frames per second for the output video. Default is 60.
        dpi : int, optional
            Resolution in dots per inch. Default is 150.
        writer : str, optional
            Video encoder to use (default: "ffmpeg").
        crf : int, optional
            Constant Rate Factor / Quantization Parameter (lower = less compression / higher quality, default: 10).
        lossless : bool, optional
            If True, encode with true lossless compression. Default is False.
        progress_bar : bool, optional
            Whether to display a progress bar. Default is True.
        """
        import concurrent.futures
        import os
        import subprocess
        import tempfile
        from pathlib import Path

        if n_jobs is None or n_jobs <= 1:
            if hasattr(self, "_original_save"):
                return self._original_save(filename, writer=writer, fps=fps, dpi=dpi, **kwargs)
            return self.ani.save(filename, writer=writer, fps=fps, dpi=dpi, **kwargs)

        if hasattr(self, "ani"):
            self.ani._draw_was_started = True

        filename_path = Path(filename).resolve()
        filename_path.parent.mkdir(parents=True, exist_ok=True)

        athinp_path = str(self._sim._athinp_path.resolve()) if self._sim._athinp_path else None
        datafolder_path = str(self._sim._datafolder.resolve())
        units_obj = self._sim.units
        figsize = self.figure.get_size_inches()

        frame_numbers = self._sim.frame_numbers
        total_frames = len(frame_numbers)

        # Compute fixed colorbar limits across frames for consistent video rendering
        clims: dict[str, tuple[float, float]] = {}
        sample_nums = [frame_numbers[0], frame_numbers[-1]]
        for num in sample_nums:
            f = self._sim.get_frame(num)
            for field_name in self._fields:
                d = getattr(f, field_name)
                if d is not None:
                    d_prep = _prepare_field_data(self._sim, field_name, d, log_scale=True)
                    v_min, v_max = float(np.nanmin(d_prep)), float(np.nanmax(d_prep))
                    if field_name not in clims:
                        clims[field_name] = (v_min, v_max)
                    else:
                        clims[field_name] = (
                            min(clims[field_name][0], v_min),
                            max(clims[field_name][1], v_max),
                        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tasks = [
                (
                    athinp_path,
                    datafolder_path,
                    units_obj,
                    num,
                    os.path.join(tmpdir, f"frame_{i:06d}.png"),
                    self._fields,
                    self._cmaps,
                    clims,
                    figsize,
                    dpi,
                )
                for i, num in enumerate(frame_numbers)
            ]

            try:
                from tqdm import tqdm
                have_tqdm = progress_bar
            except ImportError:
                have_tqdm = False

            desc = f"Rendering {total_frames} frames ({n_jobs} cores)"
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
                if have_tqdm:
                    list(tqdm(executor.map(_render_frame_worker, tasks), total=total_frames, desc=desc))
                else:
                    list(executor.map(_render_frame_worker, tasks))

            # Detect available encoders and configure high quality / low compression
            encoders_to_try = []
            try:
                enc_check = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    check=False,
                )
                stdout = enc_check.stdout
            except Exception:
                stdout = ""

            if "h264_nvenc" in stdout:
                if lossless:
                    encoders_to_try.append(["-c:v", "h264_nvenc", "-preset", "p7", "-tune", "lossless"])
                else:
                    encoders_to_try.append(["-c:v", "h264_nvenc", "-preset", "p7", "-rc", "constqp", "-qp", str(crf)])

            if "libx264" in stdout:
                if lossless:
                    encoders_to_try.append(["-c:v", "libx264", "-crf", "0", "-preset", "slow"])
                else:
                    encoders_to_try.append(["-c:v", "libx264", "-crf", str(crf), "-preset", "slow"])

            encoders_to_try.append(["-c:v", "mpeg4", "-qscale:v", "1"])

            encoded = False
            last_err = ""
            for enc_args in encoders_to_try:
                ffmpeg_cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),
                    "-i", os.path.join(tmpdir, "frame_%06d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    *enc_args,
                    "-pix_fmt", "yuv420p",
                    str(filename_path),
                ]
                res = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode == 0:
                    encoded = True
                    break
                last_err = res.stderr

            if not encoded:
                raise RuntimeError(f"FFmpeg encoding failed:\n{last_err}")

            print(f"[{self._sim.basename}] Saved animation ({total_frames} frames using {n_jobs} cores) to: {filename_path}")

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


# ── Jupyter (ipywidgets) Backend ──────────────────────────────────────────────

class JupyterVisualization(Visualization):
    """
    Interactive inline visualisation for Jupyter notebooks.

    Uses ``ipywidgets`` to provide:
    - A **frame slider** to scrub through the simulation in time.
    - A **Play** button for auto-play animation.
    - A **log-scale toggle** for the density field.
    - A **colourmap selector** dropdown per field.

    Requirements: ``ipywidgets``, ``matplotlib``.
    The notebook should have ``%matplotlib widget`` active (or ``inline``) so
    that the figure renders inside the cell output.

    Parameters
    ----------
    sim : SimulationData
        The simulation to visualise.
    fields : list of str, optional
        Which fields to show.  Defaults to ``sim.fields_available``.
    cmaps : dict, optional
        Per-field colourmap overrides.
    figsize_per_panel : tuple[float, float], optional
        Size of each subplot panel in inches ``(width, height)``.
        Default is ``(4.5, 3.5)``.
    interval : int, optional
        Auto-play step interval in milliseconds.  Default is 200.
    """

    def __init__(
        self,
        sim: SimulationData,
        fields: Optional[List[str]] = None,
        cmaps: Optional[dict[str, str]] = None,
        backend: str = "jupyter",
        figsize_per_panel: tuple = (4.5, 3.5),
        interval: int = 200,
        **kwargs,
    ):
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError as exc:
            raise ImportError(
                "JupyterVisualization requires ipywidgets. "
                "Install it with: pip install ipywidgets"
            ) from exc

        import matplotlib
        import matplotlib.pyplot as plt

        self._sim = sim
        self._fields = fields if fields is not None else sim.fields_available
        self._cmaps = dict(_DEFAULT_CMAPS)
        if cmaps:
            self._cmaps.update(cmaps)

        n = len(self._fields)
        rows, cols = _grid_shape(n)
        pw, ph = figsize_per_panel
        figsize = (pw * cols, ph * rows)

        # ── Build the matplotlib figure (non-blocking) ──────────────────
        plt.ioff()
        self.figure, axes_grid = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        plt.ion()

        self._axes: dict[str, object] = {}
        self._images_mpl: dict[str, object] = {}  # field → AxesImage
        self._cbars: dict[str, object] = {}        # field → Colorbar

        # Hide unused axes
        for idx in range(n, rows * cols):
            r, c = divmod(idx, cols)
            self.figure.delaxes(axes_grid[r, c])

        # Load first frame for initial render
        first_num = sim.frame_numbers[0]
        frame0 = sim.get_frame(first_num)

        for idx, field_name in enumerate(self._fields):
            r, c = divmod(idx, cols)
            ax = axes_grid[r, c]
            self._axes[field_name] = ax

            data = getattr(frame0, field_name)
            if data is None:
                ax.set_visible(False)
                continue

            display_data = self._prepare_data(field_name, data, log_scale=True)
            cmap = self._cmaps.get(field_name, "inferno")

            im = ax.imshow(display_data, cmap=cmap, origin="lower", aspect="auto")
            ax.set_title(_field_title(field_name), fontsize=10)
            ax.set_xlabel("x [code]", fontsize=8)
            ax.set_ylabel("y [code]", fontsize=8)
            ax.tick_params(labelsize=7)

            cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=7)

            self._images_mpl[field_name] = im
            self._cbars[field_name] = cbar

        self.figure.suptitle(
            f"{sim.basename}  |  t = {frame0.time:.4f}  (frame {first_num})",
            fontsize=11,
        )
        self.figure.tight_layout()

        # ── Widgets ──────────────────────────────────────────────────────
        n_frames = sim.n_frames

        frame_slider = widgets.IntSlider(
            value=0, min=0, max=n_frames - 1, step=1,
            description="Frame:",
            continuous_update=False,
            layout=widgets.Layout(width="70%"),
            style={"description_width": "60px"},
        )
        play = widgets.Play(
            value=0, min=0, max=n_frames - 1, step=1,
            interval=interval,
            description="Play",
        )
        widgets.jslink((play, "value"), (frame_slider, "value"))

        log_toggle = widgets.ToggleButton(
            value=True,
            description="Log scale",
            button_style="info",
            icon="adjust",
            layout=widgets.Layout(width="150px"),
        )

        available_cmaps = [
            "inferno", "plasma", "viridis", "magma", "hot",
            "seismic", "bwr", "RdBu_r", "coolwarm", "jet",
            "turbo", "cividis", "twilight",
        ]
        cmap_dropdowns = {}
        for field_name in self._fields:
            dd = widgets.Dropdown(
                options=available_cmaps,
                value=self._cmaps.get(field_name, "inferno"),
                description=f"{field_name}:",
                style={"description_width": "80px"},
                layout=widgets.Layout(width="220px"),
            )
            cmap_dropdowns[field_name] = dd

        time_label = widgets.Label(
            value=f"t = {frame0.time:.4f}  |  frame {first_num}  (idx 0 / {n_frames - 1})"
        )

        # ── Update callback ───────────────────────────────────────────────
        def _redraw(frame_pos, log_on):
            num = sim.frame_numbers[frame_pos]
            frame = sim.get_frame(num)
            for field_name, im in self._images_mpl.items():
                data = getattr(frame, field_name)
                if data is None:
                    continue
                cmap_name = cmap_dropdowns[field_name].value
                display_data = self._prepare_data(field_name, data, log_scale=log_on)
                im.set_data(display_data)
                im.set_cmap(cmap_name)
                vmin, vmax = float(np.nanmin(display_data)), float(np.nanmax(display_data))
                im.set_clim(vmin, vmax)
            self.figure.suptitle(
                f"{sim.basename}  |  t = {frame.time:.4f}  (frame {num})",
                fontsize=11,
            )
            self.figure.canvas.draw_idle()
            time_label.value = (
                f"t = {frame.time:.4f}  |  frame {num}  "
                f"(idx {frame_pos} / {n_frames - 1})"
            )

        def _on_frame_change(change):
            _redraw(change["new"], log_toggle.value)

        def _on_log_change(change):
            _redraw(frame_slider.value, change["new"])

        def _make_cmap_observer(fn):
            def _on_cmap_change(change):
                _redraw(frame_slider.value, log_toggle.value)
            return _on_cmap_change

        frame_slider.observe(_on_frame_change, names="value")
        log_toggle.observe(_on_log_change, names="value")
        for fn, dd in cmap_dropdowns.items():
            dd.observe(_make_cmap_observer(fn), names="value")

        # ── Assemble widget layout ────────────────────────────────────────
        cmap_box = widgets.HBox(
            list(cmap_dropdowns.values()),
            layout=widgets.Layout(flex_wrap="wrap"),
        )
        controls = widgets.VBox([
            widgets.HBox([play, frame_slider]),
            time_label,
            log_toggle,
            widgets.HTML("<b>Colourmap overrides:</b>"),
            cmap_box,
        ])

        self._widget = widgets.VBox([self.figure.canvas, controls])
        self._display_fn = display

    def _prepare_data(
        self,
        field_name: str,
        data: np.ndarray,
        *,
        log_scale: bool,
    ) -> np.ndarray:
        """Apply log10 to density, pressure, and temperature fields when log_scale is True."""
        return _prepare_field_data(self._sim, field_name, data, log_scale=log_scale)

    def show(self) -> None:
        """
        Display the interactive widget inline in the current Jupyter cell.

        Call this at the end of a notebook cell.  Use the slider or Play
        button to scrub through frames; toggle Log scale for density display;
        choose per-field colourmaps from the dropdowns.
        """
        self._display_fn(self._widget)

    @property
    def widget(self):
        """The root ``ipywidgets.VBox`` containing the figure and controls."""
        return self._widget

    def __repr__(self) -> str:
        return (
            f"<JupyterVisualization  sim='{self._sim.basename}'  "
            f"fields={self._fields}  n_frames={self._sim.n_frames}>"
        )