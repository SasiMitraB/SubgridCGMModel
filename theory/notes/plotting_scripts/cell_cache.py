"""
Per-frame cache of per-cell quantities of the 256x1024x256 radiative KH run, shared by
the plotting scripts.

Each frame is written to its own file as soon as it is computed, so an interrupted run
resumes where it stopped, frames the run gains later are the only ones computed, and any
--every reuses the same files. The next frame is read in a background thread while this
one is computed: the run sits on a spinning disk, so the read is the slow part.

    d, sim = cell_cache.load("momentum_cells", b, every, FIELDS, coarse_cells, version)
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from vis_athenak.simulation_data import SimulationData

RUN_DIR = Path("/data/sasi/simulation_outputs/kh3d_256x1024x256")
ATHINPUT = RUN_DIR / "kh3d_256x1024x256.athinput"
CACHE = Path(__file__).resolve().parent / "cache"


def frame_dir(name, b, version):
    """One directory per quantity set, block size and version: a version bump starts a
    new directory instead of mixing old and new frames."""
    return CACHE / f"{name}_b{b}_v{version}"


def save_frame(path, cells):
    """Write atomically (temporary file, then rename), so a run that is killed midway
    never leaves a truncated frame behind."""
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **cells)
    tmp.replace(path)


def read_frame(frame, fields):
    """The fields of a frame (float64)."""
    frame.load(fields)
    q = {k: frame[k].astype(np.float64) for k in fields}
    frame.unload()
    return frame.time, q


def load(name, b, every, fields, coarse_cells, version, recompute=False):
    """Per-cell arrays of every every-th frame, concatenated over frames, with the time
    and frame number of each cell and the list of frames; and the SimulationData.

    coarse_cells(q, sim, b) -> {key: per-cell array} is called on the fields q of a frame
    for each frame missing from the cache (every frame with recompute)."""
    sim = SimulationData(str(ATHINPUT), str(RUN_DIR / "bin"))
    cache = frame_dir(name, b, version)
    cache.mkdir(parents=True, exist_ok=True)
    frames = list(range(0, len(sim), every))
    path = lambda i: cache / f"frame_{i:05d}.npz"
    todo = [i for i in frames if recompute or not path(i).exists()]

    if todo:
        print(f"computing {len(todo)} of {len(frames)} frames into {cache}/")
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(read_frame, sim[todo[0]], fields)
            for k, i in enumerate(todo):
                time, q = pending.result()
                if k + 1 < len(todo):
                    pending = pool.submit(read_frame, sim[todo[k + 1]], fields)
                cells = coarse_cells(q, sim, b)
                n = next(iter(cells.values())).size
                cells["time"] = np.full(n, time)
                cells["frame"] = np.full(n, i)
                save_frame(path(i), cells)
                print(f"\rframe {i} ({k + 1}/{len(todo)})", end="", flush=True)
        print()

    out = {}
    for i in frames:
        with np.load(path(i)) as f:
            for key in f.files:
                out.setdefault(key, []).append(f[key])
    d = {k: np.concatenate(v) for k, v in out.items()}
    return dict(d, frames=np.array(frames), b=b), sim
