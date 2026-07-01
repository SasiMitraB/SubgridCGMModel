"""
ergane.vtk_reader
~~~~~~~~~~~~~~~~~~~~~~~
Low-level parser for the binary VTK files written by AthenaK and Athena++.

Supports two layout variants:
  - AthenaK  (STRUCTURED_POINTS  / ORIGIN + SPACING header)
  - Athena++ (RECTILINEAR_GRID   / X_COORDINATES header)

Returns a dict:
    {
        "time":   float,
        "x":      np.ndarray (nx+1 node positions),
        "y":      np.ndarray (ny+1 node positions),
        "fields": {name: np.ndarray, ...}
    }

Scalar fields  → shape (ncy, ncx)          [2-D]  or (ncz, ncy, ncx)  [3-D]
Vector fields  → shape (ncy, ncx, 3)       [2-D]  or (ncz, ncy, ncx, 3) [3-D]
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_newline(f) -> None:
    """Consume a trailing newline if present (VTK binary blocks may or may not have one)."""
    b = f.read(1)
    if b != b"\n":
        f.seek(-1, 1)


# ── Public API ────────────────────────────────────────────────────────────────

def parse_athena_vtk(path: str | Path, dtype: np.dtype = np.float64) -> dict:
    """
    Parse a single Athena/AthenaK binary VTK file.

    Parameters
    ----------
    path : str or Path
        Path to the .vtk file.
    dtype : np.dtype, optional
        Target datatype for the loaded coordinate and field arrays. Default is np.float64.

    Returns
    -------
    dict with keys:
        "time"   – simulation time extracted from the VTK comment line.
        "x"      – 1-D node coordinate array along X.
        "y"      – 1-D node coordinate array along Y.
        "fields" – dict mapping field name → numpy array of type `dtype`.
    """
    fields: dict[str, np.ndarray] = {}

    with open(path, "rb") as f:
        f.readline()  # "# vtk DataFile Version X.X"
        comment = f.readline().decode("ascii", errors="replace").strip()
        f.readline()  # "BINARY"
        f.readline()  # "DATASET STRUCTURED_POINTS" or "DATASET RECTILINEAR_GRID"

        # Grid dimensions (node counts, not cell counts)
        dim_line = f.readline().decode("ascii").strip()
        nx, ny, nz = map(int, dim_line.split()[1:])

        ncx = max(1, nx - 1)
        ncy = max(1, ny - 1)
        ncz = max(1, nz - 1)
        n_cells = ncx * ncy * ncz

        next_line = f.readline().decode("ascii").strip()

        if next_line.startswith("ORIGIN"):
            # ── AthenaK style: STRUCTURED_POINTS ──────────────────────────
            ox, oy, oz = map(float, next_line.split()[1:])
            spacing_line = f.readline().decode("ascii").strip()
            dx, dy, dz = map(float, spacing_line.split()[1:])

            x = np.linspace(ox, ox + dx * ncx, nx, dtype=dtype)
            y = np.linspace(oy, oy + dy * ncy, ny, dtype=dtype)

            # Fast-forward to CELL_DATA block
            while True:
                line = f.readline().decode("ascii", errors="ignore").strip()
                if line.startswith("CELL_DATA"):
                    break

        elif next_line.startswith("X_COORDINATES"):
            # ── Legacy Athena++ style: RECTILINEAR_GRID ───────────────────
            n_x = int(next_line.split()[1])
            x = np.frombuffer(f.read(n_x * 4), dtype=">f4").astype(dtype).copy()
            _read_newline(f)

            y_hdr = f.readline().decode("ascii").strip()
            n_y = int(y_hdr.split()[1])
            y = np.frombuffer(f.read(n_y * 4), dtype=">f4").astype(dtype).copy()
            _read_newline(f)

            z_hdr = f.readline().decode("ascii").strip()
            n_z = int(z_hdr.split()[1])
            # z coordinate read but not returned (only used in 3-D)
            np.frombuffer(f.read(n_z * 4), dtype=">f4")
            _read_newline(f)

            f.readline()  # consume CELL_DATA line

        else:
            raise ValueError(
                f"Unrecognised VTK header after DATASET line: {next_line!r}"
            )

        # ── Parse binary field blocks ──────────────────────────────────────
        while True:
            raw = f.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="ignore").strip()
            if not line:
                continue
            parts = line.split()
            if not parts:
                continue

            if parts[0] == "SCALARS":
                name = parts[1]
                f.readline()  # "LOOKUP_TABLE default"
                arr = (
                    np.frombuffer(f.read(n_cells * 4), dtype=">f4")
                    .astype(dtype)
                    .copy()
                )
                arr = (
                    arr.reshape(ncy, ncx)
                    if ncz == 1
                    else arr.reshape(ncz, ncy, ncx)
                )
                _read_newline(f)
                fields[name] = arr

            elif parts[0] == "VECTORS":
                name = parts[1]
                arr = (
                    np.frombuffer(f.read(n_cells * 3 * 4), dtype=">f4")
                    .astype(dtype)
                    .copy()
                )
                arr = (
                    arr.reshape(ncy, ncx, 3)
                    if ncz == 1
                    else arr.reshape(ncz, ncy, ncx, 3)
                )
                _read_newline(f)
                fields[name] = arr

    # Extract simulation time from the comment line (handles 'time=X' and 'time= X')
    m = re.search(r"time=\s*(\S+)", comment)
    time_val = float(m.group(1)) if m else 0.0

    return {"time": time_val, "x": x, "y": y, "fields": fields}


def extract_frame_num(path: Path) -> int:
    """Return the zero-padded integer suffix from a VTK/BIN filename, e.g. *.00042.vtk or *.00042.bin → 42."""
    m = re.search(r"\.(\d+)\.(vtk|bin)$", path.name)
    return int(m.group(1)) if m else -1


def read_vtk_time(path: str | Path) -> float:
    """
    Read *only* the simulation time from a VTK file header.

    This reads just the first two ASCII lines of the file (< 200 bytes),
    making it ~100× faster than a full ``parse_athena_vtk`` call.  Use this
    to build a time index across hundreds of files without loading any field
    data.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    float
        Simulation time, or 0.0 if no ``time=`` token is found.
    """
    with open(path, "rb") as f:
        f.readline()  # "# vtk DataFile Version X.X"
        comment = f.readline().decode("ascii", errors="replace").strip()
    m = re.search(r"time=\s*(\S+)", comment)
    return float(m.group(1)) if m else 0.0
