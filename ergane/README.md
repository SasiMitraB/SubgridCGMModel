# ergane

A Python library for loading, inspecting, and visualising output data from
[AthenaK](https://github.com/IAS-Astrophysics/athenak) and
[Athena++](https://github.com/PrincetonUniversity/athena) simulations.

The library is designed around a single principle: **do a lot with very little
code**.  You point it at an output directory and an input file, and it takes
care of file discovery, VTK parsing, field normalisation, physical units
conversion, and interactive visualisation — all lazily, so only the data you
actually request is ever read from disk.

> **Name**: *Ergane* (Ἐργάνη) is an epithet of Athena in her role as goddess
> of craft, skill, and tools — a fitting name for a library built to work with
> Athena simulation data.


---

## Installation / Setup

The library lives in the `ergane/` directory at the root of this
repository.  Make sure it is on your Python path:

```python
import sys
sys.path.insert(0, "/path/to/FluidSimsLearning")
```

Or, from inside the `FluidSimsLearning/` directory, just run scripts directly.

**Dependencies**: `numpy`, `fastplotlib` (for visualisation only).

---

## Quick Start

```python
from ergane import SimulationData

# Kelvin-Helmholtz instability (pure hydro)
kh = SimulationData(
    athinp    = "kh2d/kh2d-sin.athinput",
    datafolder= "kh2d/outputs",
)

# Inspect the simulation
print(kh)              # <SimulationData  basename='KH'  physics='hydro'  grid=(256×512)  n_frames=301>
print(kh.nx, kh.ny)   # 256  512
print(kh.gamma)        # 1.666667
print(kh.n_frames)     # 301
print(kh.params["time"]["tlim"])  # '6.0'

# Read a field for a specific frame (lazy — only that file is parsed)
rho = kh.density[300]     # np.ndarray shape (512, 256)
p   = kh.pressure[0]
vx  = kh.velx[-1]         # negative indexing → last frame

# Slice over the sorted frame list (positional)
first_five = kh.density[0:5]   # list of 5 arrays

# Get all fields for a frame at once
frame = kh.get_frame(150)
print(frame)                   # Frame #150  t=3.000  fields=[...]
print(frame.density.shape)     # (512, 256)
print(frame.time)              # 3.000...

# Animate in fastplotlib
kh.visualize().show()
kh.visualize(fields=["density", "pressure"]).show()
```

---

## SimulationData — Full API

### Constructor

```python
SimulationData(datafolder, athinp=None)
```

| Parameter    | Type         | Description |
|---|---|---|
| `datafolder` | `str\|Path`  | Directory containing VTK files, or a parent that has a `vtk/` subdirectory. |
| `athinp`     | `str\|Path`  | Optional path to the Athena input file.  Enables reading of all simulation parameters. |

The physics type (`"hydro"` or `"mhd"`) is detected automatically from the
filename patterns:

| Pattern | Physics |
|---|---|
| `*.hydro_w.*.vtk` | hydro |
| `*.mhd_w.*.vtk` + `*.mhd_bcc.*.vtk` | MHD |
| Any other `*.vtk` | hydro (fallback) |

### Properties

| Property | Type | Description |
|---|---|---|
| `n_frames` | `int` | Total number of available frames |
| `frame_numbers` | `list[int]` | Sorted frame-number list (file suffix integers) |
| `physics` | `str` | `"hydro"` or `"mhd"` |
| `fields_available` | `list[str]` | Normalised field names for this simulation |
| `times` | `np.ndarray` | Simulation time for every frame (code units, cached) |
| `nx`, `ny` | `int` | Number of cells in X1, X2 |
| `x1min`, `x1max` | `float` | Domain bounds in X1 (code units) |
| `x2min`, `x2max` | `float` | Domain bounds in X2 (code units) |
| `gamma` | `float\|None` | Adiabatic index from athinput |
| `basename` | `str\|None` | Job basename (from athinput or inferred) |
| `params` | `dict[str,dict]` | Full nested athinput parameter dict |
| `units` | `Units` | Active unit system (default: code units) |

### Field accessors

These all return `_FieldAccessor` objects, which support multiple indexing modes (see below).

```python
sim.density    sim.pressure
sim.velx       sim.vely      sim.velz
sim.bx         sim.by        sim.bz    # MHD only
```

### Frame-level access

```python
frame = sim.get_frame(num)         # by file-suffix number
frame = sim.frame_at(t=2.5)       # nearest frame to t=2.5 (code time)
frames = sim.frames_between(1.0, 3.0)  # list[Frame], all frames in [t0, t1]
```

### Visualization

```python
viz = sim.visualize()                                   # all available fields
viz = sim.visualize(fields=["density", "pressure"])     # specific fields
viz = sim.visualize(cmaps={"bx": "PuOr"})              # custom colourmaps
viz.show()                                              # opens window + animation loop
viz.figure                                              # raw fastplotlib Figure
```

---

## Field Accessor — Indexing Modes

All field accessors (`sim.density`, `sim.bx`, …) support four ways to access data:

### By frame number (file suffix)

```python
sim.density[300]     # frame with suffix 00300 in the filename
sim.density[-1]      # last frame (negative positional)
sim.density[0:5]     # first 5 frames (positional slice → list)
```

> **Note**: indexing with an integer uses the **filename suffix** (e.g. 300 → `KH.hydro_w.00300.vtk`), not a position in a list.  Slices and negative integers are positional over the sorted frame list.

### By simulation time

```python
# Single frame nearest to t=2.5 (code units)
arr = sim.density.at_time(2.5)

# All frames in [t_start, t_end]
arrays = sim.density.between(1.0, 3.0)                    # list of ndarray
pairs  = sim.density.between(1.0, 3.0, include_times=True) # list of (t, ndarray)
```

---

## Physical Units

By default all field arrays are returned in **code units** (scale factors = 1).
Attach a `Units` object to convert to physical units:

```python
from ergane import Units

# Define a CGS unit system: 1 code length = 1 pc, 1 code density = 1 mp/cc,
# 1 code velocity = 1 km/s
cgs = Units.cgs(
    length   = 3.086e18,    # cm  (1 pc)
    density  = 1.67e-24,    # g/cm³  (1 proton mass / cc)
    velocity = 1.0e5,       # cm/s  (1 km/s)
    labels   = {
        "density":  "g cm⁻³",
        "pressure": "dyn cm⁻²",
        "velx":     "km s⁻¹",
        "vely":     "km s⁻¹",
    },
)

sim.set_units(cgs)

# Now all field arrays are automatically multiplied by the right scale factor
rho = sim.density[0]            # array in g/cm³
p   = sim.pressure[0]           # array in dyn/cm²
print(sim.units.label("density"))  # "g cm⁻³"
print(sim.units.label("bx"))       # "B_x [CGS]"  (auto-generated)

# Frame objects also carry converted time and coordinates
frame = sim.get_frame(100)
print(frame.time)               # seconds (= code_time × units.time)
print(frame.x)                  # array in cm

# Reset to code units
sim.set_units(Units.code())
```

### Derived scales

| Quantity | Derivation |
|---|---|
| `units.time` | `length / velocity` |
| `units.pressure` | `density × velocity²` |
| `units.magnetic` | `√(density × velocity²)` ← Gaussian CGS convention |

Any of these can be overridden by passing them explicitly to the constructor:

```python
u = Units(
    length=..., density=..., velocity=...,
    pressure=1.23e-6,  # override auto-derived pressure
    magnetic=4.5e-7,   # override auto-derived B-field scale
)
```

### Units API

```python
Units(length, density, velocity, time=None, pressure=None, magnetic=None,
      system="custom", labels=None)

Units.code()                       # all scales = 1
Units.cgs(length, density, velocity, **kwargs)
Units.si(length, density, velocity, **kwargs)

u.scale("density")    # float — multiply code array by this
u.label("density")    # str  — unit string for axis labels
```

---

## The `Frame` Object

`get_frame()` returns a `Frame` dataclass with all fields pre-loaded:

```python
frame = sim.get_frame(150)

frame.number      # 150
frame.time        # simulation time (scaled if units are set)
frame.x           # 1-D coordinate array along X (scaled)
frame.y           # 1-D coordinate array along Y (scaled)
frame.units       # the active Units object
frame.density     # np.ndarray or None
frame.pressure    # np.ndarray or None
frame.velx        # np.ndarray or None
frame.vely        # np.ndarray or None
frame.velz        # np.ndarray or None
frame.bx          # np.ndarray or None  (MHD only)
frame.by          # np.ndarray or None  (MHD only)
frame.bz          # np.ndarray or None  (MHD only)
```

---

## MHD Example — Orszag-Tang Vortex

```python
from ergane import SimulationData, Units

ot = SimulationData(
    athinp    = "orszang_tang_vortex/athinput.orszag-tang",
    datafolder= "orszang_tang_vortex/outputs",
)

print(ot)          # physics='mhd'  grid=(400×400)  n_frames=200

# Access B-fields
bx  = ot.bx[50]                    # B_x at frame 50
by  = ot.by.at_time(1.0)           # B_y nearest t=1.0
bx_list = ot.bx.between(0.5, 1.5) # 100 arrays

# MHD animation (6 panels)
ot.visualize().show()

# Pick specific fields
ot.visualize(fields=["density", "pressure", "bx", "by"]).show()
```

---

## Low-level API

These functions are also exported from `ergane` for direct use:

```python
from ergane import parse_athena_vtk, read_vtk_time, parse_athinput

# Parse a single VTK file fully
data = parse_athena_vtk("outputs/vtk/KH.hydro_w.00300.vtk")
# data = {"time": 6.0, "x": ..., "y": ..., "fields": {"rho": ..., ...}}

# Read only the time (reads 2 lines — very fast)
t = read_vtk_time("outputs/vtk/KH.hydro_w.00300.vtk")

# Parse an athinput file
params = parse_athinput("kh2d/kh2d-sin.athinput")
params["mesh"]["nx1"]   # '256'
params["time"]["tlim"]  # '6.0'
```

---

## Package Layout

```
ergane/
  __init__.py          Public API exports
  simulation_data.py   SimulationData, Frame, _FieldAccessor
  visualization.py     Visualization (fastplotlib wrapper)
  units.py             Units — physical unit system
  vtk_reader.py        Binary VTK parser (AthenaK + Athena++ formats)
  athinput_parser.py   Athena input file parser
```

---

## Performance Notes

- **Lazy loading**: only the files you actually index are parsed.  The full
  binary field data for 301 frames is never held in memory simultaneously.
- **Fast time index**: `sim.times` and all time-based queries (`at_time`,
  `between`, `frame_at`) read only the 2-line ASCII header of each VTK file
  (~60 ms for 301 frames, cached after the first call).
- **No preloading**: constructing `SimulationData` is near-instant — no VTK
  files are opened until you request data.
