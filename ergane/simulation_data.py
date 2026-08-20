"""
ergane.simulation_data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
High-level interface to an AthenaK / Athena++ simulation output directory.

Typical usage
-------------
>>> from ergane import SimulationData
>>> sim = SimulationData(athinp="kh2d/kh2d-sin.athinput",
...                      datafolder="kh2d/outputs")
>>> sim.n_frames          # 301
>>> sim.physics            # "hydro"
>>> sim.nx, sim.ny         # 256, 512
>>> sim.density[300]       # np.ndarray shape (512, 256)  [code units]
>>>
>>> # Physical units
>>> from ergane.units import Units
>>> sim.set_units(Units.cgs(length=3.086e18, density=1.67e-24, velocity=1e5))
>>> sim.density[300]       # array in g/cm³
>>>
>>> # Time-based access
>>> sim.frame_at(t=2.5)             # Frame nearest to t=2.5
>>> sim.frames_between(1.0, 3.0)    # list of Frames
>>> sim.density.at_time(2.5)        # np.ndarray nearest to t=2.5
>>> sim.density.between(1.0, 3.0)   # list of np.ndarray

Indexing convention
-------------------
Integer indexing uses the **file suffix number** (the zero-padded integer
in the filename, e.g. KH.hydro_w.00300.vtk → key 300), NOT a positional
index.  Slice notation (``sim.density[0:10]``) is positional over the sorted
list of frame numbers and returns a list.  Negative integers are also
positional (``sim.density[-1]`` returns the last frame).
"""

from __future__ import annotations

import bisect
import dataclasses
import re
from pathlib import Path
from typing import List, Optional

import numpy as np

from .athinput_parser import parse_athinput
from .units import Units
from .vtk_reader import extract_frame_num, parse_athena_vtk, read_vtk_time
from . import bin_reader


def read_bin_time(path: str | Path) -> float:
    """
    Read *only* the simulation time from a BIN file header.
    """
    with open(path, "rb") as fp:
        code_header = fp.readline().split()
        if len(code_header) < 1 or code_header[0] != b"Athena":
            return 0.0
        pheader_count = int(fp.readline().split(b"=")[-1])
        pheader = {}
        for _ in range(pheader_count - 1):
            key, val = [x.strip() for x in fp.readline().decode("utf-8").split("=")]
            pheader[key] = val
        return float(pheader.get("time", 0.0))


# ── Field name normalisation ──────────────────────────────────────────────────

def _resolve_scalar(raw_fields: dict, *candidates: str) -> Optional[np.ndarray]:
    for name in candidates:
        if name in raw_fields:
            return raw_fields[name]
    return None


def _resolve_all(raw_fields: dict, gamma: float = 5.0 / 3.0) -> dict[str, np.ndarray]:
    """
    Normalise raw field names to the standard library keys:
    density, pressure, eint, velx, vely, velz, bx, by, bz.

    When raw files output internal energy density ('eint'), actual pressure
    is derived via P = (gamma - 1) * eint.

    Missing fields are silently omitted (not every simulation has B-fields).
    """
    out: dict[str, np.ndarray] = {}

    # Density
    arr = _resolve_scalar(raw_fields, "rho", "dens")
    if arr is not None:
        out["density"] = arr

    # Velocities — may come as a single vector field "vel" or as separate scalars
    if "vel" in raw_fields:
        v = raw_fields["vel"]
        out["velx"] = v[..., 0]
        out["vely"] = v[..., 1]
        if v.shape[-1] > 2:
            out["velz"] = v[..., 2]
    else:
        for key, src in [("velx", "velx"), ("vely", "vely"), ("velz", "velz")]:
            if src in raw_fields:
                out[key] = raw_fields[src]

    # Handle velocities from momenta if reading conserved variables (hydro_u)
    if "density" in out:
        for v_key, mom_key in [("velx", "mom1"), ("vely", "mom2"), ("velz", "mom3")]:
            if v_key not in out and mom_key in raw_fields:
                out[v_key] = raw_fields[mom_key] / (out["density"] + 1e-30)

    # Pressure and internal energy
    arr_press = _resolve_scalar(raw_fields, "press", "pressure", "p")
    arr_eint = _resolve_scalar(raw_fields, "eint", "internal_energy")
    arr_ener = _resolve_scalar(raw_fields, "ener", "total_energy")

    if arr_press is not None:
        out["pressure"] = arr_press
        out["eint"] = arr_press / (gamma - 1.0)
    elif arr_eint is not None:
        out["eint"] = arr_eint
        out["pressure"] = arr_eint * (gamma - 1.0)
    elif arr_ener is not None:
        # Total energy E = e_int + 0.5 * rho * v^2
        e_kin = 0.0
        if "density" in out:
            rho = out["density"]
            vsq = sum(out[v] ** 2 for v in ("velx", "vely", "velz") if v in out)
            e_kin = 0.5 * rho * vsq
        e_int = arr_ener - e_kin
        out["eint"] = e_int
        out["pressure"] = e_int * (gamma - 1.0)

    # Passive scalars — Athena output uses s_00, s_01, ... for primitive scalars
    # and r_00, r_01, ... for conserved scalar mass densities.
    for key, arr in raw_fields.items():
        m = re.fullmatch(r"[sr]_(\d+)", key)
        if m:
            out[f"scalar_{int(m.group(1)):02d}"] = arr

    # B-fields — may come as vector "Bcc" or separate scalars
    if "Bcc" in raw_fields:
        b = raw_fields["Bcc"]
        out["bx"] = b[..., 0]
        out["by"] = b[..., 1]
        if b.shape[-1] > 2:
            out["bz"] = b[..., 2]
    else:
        for key, src in [("bx", "bcc1"), ("by", "bcc2"), ("bz", "bcc3")]:
            if src in raw_fields:
                out[key] = raw_fields[src]

    return out


# ── Frame dataclass ───────────────────────────────────────────────────────────

@dataclasses.dataclass
class Frame:
    """
    All physical fields for a single simulation snapshot.

    Field arrays are already scaled to physical units if ``units`` were set on
    the parent ``SimulationData``.  ``time`` and coordinates ``x``, ``y`` are
    also scaled accordingly.
    """

    number:   int
    time:     float
    x:        np.ndarray
    y:        np.ndarray
    units:    Units = dataclasses.field(default_factory=Units.code, repr=False)
    # Physical fields (None when not present in this simulation)
    density:  Optional[np.ndarray] = None
    pressure: Optional[np.ndarray] = None
    eint:     Optional[np.ndarray] = None
    velx:     Optional[np.ndarray] = None
    vely:     Optional[np.ndarray] = None
    velz:     Optional[np.ndarray] = None
    bx:       Optional[np.ndarray] = None
    by:       Optional[np.ndarray] = None
    bz:       Optional[np.ndarray] = None
    scalars:  dict[str, np.ndarray] = dataclasses.field(default_factory=dict, repr=False)

    @property
    def xc(self) -> np.ndarray:
        """Cell center coordinates along X1 (x-axis)."""
        return 0.5 * (self.x[:-1] + self.x[1:])

    @property
    def yc(self) -> np.ndarray:
        """Cell center coordinates along X2 (y-axis)."""
        return 0.5 * (self.y[:-1] + self.y[1:])

    def __getattr__(self, name: str):
        scalars = self.__dict__.get("scalars", {})
        if name in scalars:
            return scalars[name]
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

    @property
    def temperature(self) -> Optional[np.ndarray]:
        """
        Compute the temperature field in Kelvin.
        T = (P / rho) * (mu * m_H / k_B) [CGS]
        When in code units, P_unit = 1.59916e-14 dyne/cm^2 converts P to CGS.
        """
        if self.density is None or self.pressure is None:
            return None
        mu = getattr(self.units, "mu", 0.62)
        m_H = 1.6726e-24
        k_B = 1.3807e-16
        if getattr(self.units, "system", "code") == "code":
            P_unit = 1.59916e-14
            return (self.pressure * P_unit / (self.density + 1e-30)) * (mu / k_B)
        return (self.pressure / (self.density + 1e-30)) * (mu * m_H / k_B)

    def __repr__(self) -> str:
        available = [
            k for k in ("density", "pressure", "eint", "temperature", "velx", "vely", "velz", "bx", "by", "bz")
            if getattr(self, k) is not None
        ]
        available.extend(sorted(self.scalars))
        return (
            f"<Frame #{self.number}  t={self.time:.4f} [{self.units.system}]"
            f"  fields={available}>"
        )


# ── Lazy field accessor ───────────────────────────────────────────────────────

class _FieldAccessor:
    """
    Lazy per-field accessor returned by ``sim.density``, ``sim.pressure``, etc.

    Integer indexing (by frame number)
    ------------------------------------
    ``accessor[42]``    → np.ndarray for frame number 42 (file suffix)
    ``accessor[-1]``    → last frame (negative positional)

    Slice indexing (positional over sorted frame list)
    ---------------------------------------------------
    ``accessor[0:5]``   → list of 5 arrays for the first 5 frames

    Time-based access
    -----------------
    ``accessor.at_time(t)``             → array for frame nearest to *t*
    ``accessor.between(t0, t1)``        → list of arrays in [t0, t1]
    ``accessor.between(t0, t1, times=True)`` → list of (time, array) tuples
    """

    def __init__(self, sim: "SimulationData", field_name: str):
        self._sim   = sim
        self._field = field_name

    # ── Frame-number / positional indexing ────────────────────────────────────

    def __getitem__(self, key) -> "np.ndarray | list[np.ndarray]":
        if isinstance(key, int):
            # Negative indexing → positional into sorted frame list
            if key < 0:
                key = self._sim.frame_numbers[key]
            frame = self._sim.get_frame(key)
            arr = getattr(frame, self._field)
            if arr is None:
                raise KeyError(
                    f"Field '{self._field}' is not available "
                    f"(physics={self._sim.physics}).  "
                    f"Available fields: {self._sim.fields_available}"
                )
            return arr
        elif isinstance(key, slice):
            nums = self._sim.frame_numbers[key]
            return [self[n] for n in nums]
        else:
            raise TypeError(f"Index must be int or slice, got {type(key)}")

    # ── Time-based access ─────────────────────────────────────────────────────

    def at_time(self, t: float, method: str = "nearest") -> np.ndarray:
        """
        Return the field array for the frame whose simulation time is closest
        to *t*.

        Parameters
        ----------
        t : float
            Target simulation time (in the same units as the simulation, i.e.
            code-unit time even if physical units are set on the data).
        method : {'nearest'}
            Only ``'nearest'`` is currently supported.

        Returns
        -------
        np.ndarray
        """
        num = self._sim._frame_num_at_time(t, method=method)
        return self[num]

    def between(
        self,
        t_start: float,
        t_end:   float,
        *,
        include_times: bool = False,
    ) -> "list[np.ndarray] | list[tuple[float, np.ndarray]]":
        """
        Return field arrays for all frames whose simulation time falls within
        ``[t_start, t_end]`` (inclusive).

        Parameters
        ----------
        t_start, t_end : float
            Time range in code-unit time.
        include_times : bool
            If True, return a list of ``(time, array)`` tuples instead of
            a plain list of arrays.

        Returns
        -------
        list of np.ndarray, or list of (float, np.ndarray) tuples
        """
        nums = self._sim._frame_nums_between(t_start, t_end)
        if not nums:
            return []
        if include_times:
            times_arr = self._sim._time_index_array()
            t_lookup  = dict(zip(self._sim.frame_numbers, times_arr))
            return [(t_lookup[n], self[n]) for n in nums]
        return [self[n] for n in nums]

    def __repr__(self) -> str:
        return (
            f"<FieldAccessor  field='{self._field}'  "
            f"n_frames={self._sim.n_frames}  "
            f"units='{self._sim.units.system}'>"
        )


# ── SimulationData ────────────────────────────────────────────────────────────

class SimulationData:
    """
    High-level interface to an AthenaK / Athena++ simulation.

    Parameters
    ----------
    datafolder : str or Path
        Directory containing the VTK output files, or a parent directory that
        has a ``vtk/`` subdirectory (both are tried automatically).
    athinp : str or Path, optional
        Path to the athinput parameter file.  Provides grid metadata and
        problem parameters; highly recommended but not strictly required.

    Examples
    --------
    >>> sim = SimulationData(athinp="kh2d/kh2d-sin.athinput",
    ...                      datafolder="kh2d/outputs")
    >>> sim.density[300]                    # code-unit array
    >>> sim.frame_at(t=2.5).density        # Frame nearest t=2.5
    >>> sim.density.between(1.0, 3.0)      # list of arrays in that time window
    >>>
    >>> from ergane.units import Units
    >>> sim.set_units(Units.cgs(3.086e18, 1.67e-24, 1e5))
    >>> sim.density[300]                    # now in g/cm³
    """

    # ── Construction ──────────────────────────────────────────────────────────

    def __init__(
        self,
        datafolder: str | Path,
        athinp:     Optional[str | Path] = None,
        dtype:      np.dtype = np.float64,
    ):
        self._datafolder  = Path(datafolder)
        self._athinp_path = Path(athinp) if athinp else None
        self._dtype       = dtype

        # Parse athinput (optional)
        self.params: dict[str, dict[str, str]] = {}
        if self._athinp_path and self._athinp_path.exists():
            self.params = parse_athinput(self._athinp_path)

        # Discover output files and physics type
        data_dir, self._file_format = self._locate_data_dir()
        self._hydro_files, self._bcc_files, self._physics = self._discover_files(data_dir, self._file_format)

        # Sorted list of frame numbers (file suffix integers)
        self._frame_numbers: list[int] = sorted(self._hydro_files.keys())

        # Lazy caches
        self._grid_cache:       Optional[dict]       = None
        self._time_index_cache: Optional[np.ndarray] = None  # shape (n_frames,)

        # Units — try to parse from params if available, otherwise default to code units
        self._units: Units = Units.from_params(self.params)

        # Lazy field accessor singletons
        self._accessors: dict[str, _FieldAccessor] = {}
        self._scalar_fields: list[str] = self._infer_scalar_fields()

    # ── Internal: file discovery ──────────────────────────────────────────────

    def _locate_data_dir(self) -> tuple[Path, str]:
        """Find the directory containing output files and return (path, format)."""
        for fmt in ["bin", "vtk"]:
            for d in [self._datafolder, self._datafolder / fmt]:
                if d.is_dir() and any(d.glob(f"*.{fmt}")):
                    return d, fmt
        raise FileNotFoundError(
            f"No .bin or .vtk files found in {self._datafolder} or its subdirectories."
        )

    def _discover_files(
        self, data_dir: Path, fmt: str
    ) -> tuple[dict[int, Path], dict[int, Path], str]:
        """
        Scan *data_dir* for hydro and B-field files of format *fmt* ('vtk' or 'bin').

        Returns ``(hydro_files, bcc_files, physics_type)``.
        """
        # MHD: look for *.mhd_w.*.<fmt> (hydro primitives) + *.mhd_bcc.*.<fmt> (B-field)
        mhd_hydro = sorted(data_dir.glob(f"*.mhd_w.*.{fmt}"),  key=extract_frame_num)
        mhd_bcc   = sorted(data_dir.glob(f"*.mhd_bcc.*.{fmt}"), key=extract_frame_num)

        if mhd_hydro:
            hydro_map = {extract_frame_num(p): p for p in mhd_hydro}
            bcc_map   = {extract_frame_num(p): p for p in mhd_bcc}
            if bcc_map:
                common    = set(hydro_map) & set(bcc_map)
                hydro_map = {k: hydro_map[k] for k in common}
                bcc_map   = {k: bcc_map[k]   for k in common}
            return hydro_map, bcc_map, "mhd"

        # Pure hydro: *.hydro_w.*.<fmt>
        hydro = sorted(data_dir.glob(f"*.hydro_w.*.{fmt}"), key=extract_frame_num)
        if hydro:
            return {extract_frame_num(p): p for p in hydro}, {}, "hydro"

        # Conserved fallback: *.hydro_u.*.<fmt>
        hydro_u = sorted(data_dir.glob(f"*.hydro_u.*.{fmt}"), key=extract_frame_num)
        if hydro_u:
            return {extract_frame_num(p): p for p in hydro_u}, {}, "hydro"

        # Generic fallback
        generic = sorted(data_dir.glob(f"*.{fmt}"), key=extract_frame_num)
        if generic:
            return {extract_frame_num(p): p for p in generic}, {}, "hydro"

        raise FileNotFoundError(f"No recognisable Athena {fmt.upper()} files found in {data_dir}")

    # ── Internal: data loading ────────────────────────────────────────────────

    def _load_raw_frame(self, num: int) -> tuple[dict, float, np.ndarray, np.ndarray]:
        """
        Parse the file(s) for frame *num*.

        Returns ``(normalised_fields, time_code, x_code, y_code)`` — all in
        code units.  Unit scaling is applied in ``get_frame()``.
        """
        if num not in self._hydro_files:
            raise KeyError(
                f"Frame {num} does not exist.  "
                f"Available range: {self._frame_numbers[0]} – {self._frame_numbers[-1]}"
            )

        gamma = self.gamma if self.gamma is not None else 5.0 / 3.0
        if self._file_format == "vtk":
            raw    = parse_athena_vtk(self._hydro_files[num], dtype=self._dtype)
            merged = dict(raw["fields"])

            if num in self._bcc_files:
                bcc_raw = parse_athena_vtk(self._bcc_files[num], dtype=self._dtype)
                merged.update(bcc_raw["fields"])

            return _resolve_all(merged, gamma=gamma), raw["time"], raw["x"], raw["y"]

        elif self._file_format == "bin":
            data_h = bin_reader.read_all_ranks_binary_as_athdf(str(self._hydro_files[num]), dtype=self._dtype)
            merged = {}
            time_val = data_h["Time"]
            x_code = data_h["x1f"]
            y_code = data_h["x2f"]
            
            exclude_keys = {"Time", "NumCycles", "MaxLevel", "x1f", "x1v", "x2f", "x2v", "x3f", "x3v", "Levels"}
            for k, val in data_h.items():
                if k not in exclude_keys and isinstance(val, np.ndarray):
                    if val.shape[0] == 1:
                        merged[k] = val[0]
                    else:
                        merged[k] = val

            if num in self._bcc_files:
                data_b = bin_reader.read_all_ranks_binary_as_athdf(str(self._bcc_files[num]), dtype=self._dtype)
                for k, val in data_b.items():
                    if k not in exclude_keys and isinstance(val, np.ndarray):
                        if val.shape[0] == 1:
                            merged[k] = val[0]
                        else:
                            merged[k] = val

            return _resolve_all(merged, gamma=gamma), time_val, x_code, y_code

    def _infer_scalar_fields(self) -> list[str]:
        """Discover scalar field names from the first available frame."""
        if not self._frame_numbers:
            return []
        first = self._frame_numbers[0]
        raw_fields, _, _, _ = self._load_raw_frame(first)
        return sorted(k for k in raw_fields if k.startswith("scalar_"))

    def _peek_grid(self) -> dict:
        """Cache grid metadata (ncx, ncy, x, y) from the first frame."""
        if self._grid_cache is None:
            first = self._frame_numbers[0]
            if self._file_format == "vtk":
                raw   = parse_athena_vtk(self._hydro_files[first], dtype=self._dtype)
                arr0  = raw["fields"][next(iter(raw["fields"]))]
                ncy, ncx = arr0.shape[:2]
                self._grid_cache = {"x": raw["x"], "y": raw["y"], "ncx": ncx, "ncy": ncy}
            elif self._file_format == "bin":
                data_h = bin_reader.read_all_ranks_binary_as_athdf(str(self._hydro_files[first]), dtype=self._dtype)
                exclude_keys = {"Time", "NumCycles", "MaxLevel", "x1f", "x1v", "x2f", "x2v", "x3f", "x3v", "Levels"}
                arr0 = None
                for k, val in data_h.items():
                    if k not in exclude_keys and isinstance(val, np.ndarray):
                        arr0 = val
                        break
                if arr0 is None:
                    raise KeyError("No data arrays found in binary file.")
                nz, ncy, ncx = arr0.shape
                self._grid_cache = {"x": data_h["x1f"], "y": data_h["x2f"], "ncx": ncx, "ncy": ncy}
        return self._grid_cache

    def _get_accessor(self, field: str) -> _FieldAccessor:
        if field not in self._accessors:
            self._accessors[field] = _FieldAccessor(self, field)
        return self._accessors[field]

    # ── Internal: time index ──────────────────────────────────────────────────

    def _time_index_array(self) -> np.ndarray:
        """
        Build and cache the time index by reading only the header/first lines
        of each file. Very fast: ~1 ms per file, no binary data read.

        Returns a float64 array of shape ``(n_frames,)`` aligned with
        ``self.frame_numbers``.
        """
        if self._time_index_cache is None:
            if self._file_format == "vtk":
                ts = [
                    read_vtk_time(self._hydro_files[num])
                    for num in self._frame_numbers
                ]
            elif self._file_format == "bin":
                ts = [
                    read_bin_time(self._hydro_files[num])
                    for num in self._frame_numbers
                ]
            self._time_index_cache = np.array(ts, dtype=np.float64)
        return self._time_index_cache

    def _frame_num_at_time(self, t: float, method: str = "nearest") -> int:
        """Return the frame number whose time is nearest to *t*."""
        ts   = self._time_index_array()
        idx  = int(np.argmin(np.abs(ts - t)))
        return self._frame_numbers[idx]

    def _frame_nums_between(self, t_start: float, t_end: float) -> list[int]:
        """Return frame numbers whose time falls within [t_start, t_end]."""
        ts   = self._time_index_array()
        mask = (ts >= t_start) & (ts <= t_end)
        return [self._frame_numbers[i] for i in np.where(mask)[0]]

    # ── Units ─────────────────────────────────────────────────────────────────

    @property
    def units(self) -> Units:
        """The currently active unit system (default: code units)."""
        return self._units

    def set_units(self, units: Units) -> None:
        """
        Attach a physical unit system to this simulation.

        After calling this, all field arrays returned by accessors and
        ``get_frame()`` are automatically multiplied by the appropriate scale
        factor.  Simulation coordinates (x, y) and time in ``Frame`` objects
        are also converted.

        Parameters
        ----------
        units : Units
            A ``Units`` instance.  Use ``Units.code()`` to reset to code units.

        Examples
        --------
        >>> from ergane.units import Units
        >>> sim.set_units(Units.cgs(length=3.086e18, density=1.67e-24, velocity=1e5))
        >>> sim.density[0]          # g/cm³
        >>> sim.units.label("velx") # "v_x [CGS]"
        """
        self._units = units
        # Clear accessor singletons so they pick up the new units
        self._accessors = {}

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def frame_numbers(self) -> list[int]:
        """Sorted list of all available frame numbers (file suffix integers)."""
        return self._frame_numbers

    @property
    def n_frames(self) -> int:
        """Total number of available frames."""
        return len(self._frame_numbers)

    @property
    def physics(self) -> str:
        """``"hydro"`` or ``"mhd"``."""
        return self._physics

    @property
    def fields_available(self) -> list[str]:
        """Normalised field names available in this simulation."""
        base = ["density", "pressure", "eint", "velx", "vely"]
        if self._physics == "mhd":
            base += ["bx", "by"]
        if "density" in base and "pressure" in base:
            base.append("temperature")
        base.extend(self._scalar_fields)
        return base

    # Grid — prefer athinput values; fall back to first VTK frame
    @property
    def nx(self) -> int:
        """Number of cells in X1."""
        if "mesh" in self.params:
            return int(self.params["mesh"]["nx1"])
        return self._peek_grid()["ncx"]

    @property
    def ny(self) -> int:
        """Number of cells in X2."""
        if "mesh" in self.params:
            return int(self.params["mesh"]["nx2"])
        return self._peek_grid()["ncy"]

    @property
    def x1min(self) -> float:
        """Domain lower bound in X1 (code units)."""
        if "mesh" in self.params:
            return float(self.params["mesh"]["x1min"])
        return float(self._peek_grid()["x"].min())

    @property
    def x1max(self) -> float:
        """Domain upper bound in X1 (code units)."""
        if "mesh" in self.params:
            return float(self.params["mesh"]["x1max"])
        return float(self._peek_grid()["x"].max())

    @property
    def x2min(self) -> float:
        """Domain lower bound in X2 (code units)."""
        if "mesh" in self.params:
            return float(self.params["mesh"]["x2min"])
        return float(self._peek_grid()["y"].min())

    @property
    def x2max(self) -> float:
        """Domain upper bound in X2 (code units)."""
        if "mesh" in self.params:
            return float(self.params["mesh"]["x2max"])
        return float(self._peek_grid()["y"].max())

    @property
    def gamma(self) -> Optional[float]:
        """Adiabatic index γ, or None if not in athinput."""
        for section in ("hydro", "mhd"):
            if section in self.params and "gamma" in self.params[section]:
                return float(self.params[section]["gamma"])
        return None

    @property
    def basename(self) -> Optional[str]:
        """Job basename from athinput, or inferred from filenames."""
        if "job" in self.params:
            return self.params["job"].get("basename")
        first_path = next(iter(self._hydro_files.values()))
        return first_path.name.split(".")[0]

    @property
    def times(self) -> np.ndarray:
        """
        Simulation time for every frame, in code-unit time.

        Built by reading only the ASCII header of each VTK file (fast).
        The result is cached after the first call.
        """
        return self._time_index_array()

    # ── Field accessors ───────────────────────────────────────────────────────

    @property
    def density(self) -> _FieldAccessor:
        """Accessor for the density field.  Supports ``[frame_num]``, ``[slice]``,
        ``.at_time(t)``, and ``.between(t0, t1)``."""
        return self._get_accessor("density")

    @property
    def pressure(self) -> _FieldAccessor:
        """Accessor for the pressure field."""
        return self._get_accessor("pressure")

    @property
    def eint(self) -> _FieldAccessor:
        """Accessor for the internal energy density field."""
        return self._get_accessor("eint")

    @property
    def temperature(self) -> _FieldAccessor:
        """Accessor for the temperature field (computed from pressure and density)."""
        return self._get_accessor("temperature")

    @property
    def velx(self) -> _FieldAccessor:
        """Accessor for the X-velocity field."""
        return self._get_accessor("velx")

    @property
    def vely(self) -> _FieldAccessor:
        """Accessor for the Y-velocity field."""
        return self._get_accessor("vely")

    @property
    def velz(self) -> _FieldAccessor:
        """Accessor for the Z-velocity field."""
        return self._get_accessor("velz")

    @property
    def bx(self) -> _FieldAccessor:
        """Accessor for B_x (MHD only)."""
        return self._get_accessor("bx")

    @property
    def by(self) -> _FieldAccessor:
        """Accessor for B_y (MHD only)."""
        return self._get_accessor("by")

    @property
    def bz(self) -> _FieldAccessor:
        """Accessor for B_z (MHD only)."""
        return self._get_accessor("bz")

    # ── Frame-level access ────────────────────────────────────────────────────

    def get_frame(self, num: int) -> Frame:
        """
        Load all fields for frame *num* and return a ``Frame`` object.

        Field arrays, coordinates, and time are scaled by the active unit
        system (see ``set_units()``).

        Parameters
        ----------
        num : int
            Frame number (file suffix integer, e.g. 300 for ``*.00300.vtk``).

        Returns
        -------
        Frame
        """
        raw_fields, time_code, x_code, y_code = self._load_raw_frame(num)
        u = self._units

        # Apply unit scaling to field arrays
        scaled_fields = {
            k: (arr * u.scale(k) if u.scale(k) != 1.0 else arr)
            for k, arr in raw_fields.items()
        }

        scalar_fields = {k: v for k, v in scaled_fields.items() if k.startswith("scalar_")}

        return Frame(
            number=num,
            time=time_code * u.time,
            x=x_code * u.length,
            y=y_code * u.length,
            units=u,
            **{k: scaled_fields.get(k)
               for k in ("density", "pressure", "eint", "velx", "vely", "velz", "bx", "by", "bz")},
            scalars=scalar_fields,
        )

    # ── Time-based frame queries ──────────────────────────────────────────────

    def frame_at(self, t: float, method: str = "nearest") -> Frame:
        """
        Return the ``Frame`` whose simulation time is nearest to *t*.

        Time is given in **code units** regardless of whether physical units
        are set (the time index is always stored in code time).

        Parameters
        ----------
        t : float
            Target simulation time.
        method : {'nearest'}
            Selection method.  Only ``'nearest'`` is supported.

        Returns
        -------
        Frame

        Example
        -------
        >>> frame = sim.frame_at(t=2.5)
        >>> print(frame)      # Frame #125  t=2.5000  fields=[...]
        """
        num = self._frame_num_at_time(t, method=method)
        return self.get_frame(num)

    def frames_between(self, t_start: float, t_end: float) -> List[Frame]:
        """
        Return all ``Frame`` objects whose simulation time is in
        ``[t_start, t_end]`` (inclusive), sorted by time.

        Times are in **code units**.

        Parameters
        ----------
        t_start, t_end : float
            Time range.

        Returns
        -------
        list[Frame]

        Example
        -------
        >>> frames = sim.frames_between(1.0, 3.0)
        >>> print(f"Got {len(frames)} frames")
        """
        nums = self._frame_nums_between(t_start, t_end)
        return [self.get_frame(n) for n in nums]

    # ── Visualization ─────────────────────────────────────────────────────────

    def visualize(
        self,
        fields: Optional[List[str]] = None,
        cmaps:  Optional[dict[str, str]] = None,
        backend: str = "fastplotlib",
        **kwargs,
    ) -> "Visualization":
        """
        Create an animated visualisation backed by fastplotlib or matplotlib.

        Parameters
        ----------
        fields : list of str, optional
            Fields to display.  Defaults to all available fields.
        cmaps : dict, optional
            Per-field colourmap overrides, e.g. ``{"density": "plasma"}``.
        backend : str, optional
            The visualization library to use: ``"fastplotlib"`` (default)
            or ``"matplotlib"``.
        **kwargs
            Extra keyword arguments passed to the backend visualization class.

        Returns
        -------
        Visualization
            Call ``.show()`` to open the window.
        """
        from .visualization import Visualization
        return Visualization(self, fields=fields, cmaps=cmaps, backend=backend, **kwargs)

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<SimulationData  basename='{self.basename}'  "
            f"physics='{self.physics}'  "
            f"grid=({self.nx}×{self.ny})  "
            f"n_frames={self.n_frames}  "
            f"units='{self.units.system}'>"
        )
