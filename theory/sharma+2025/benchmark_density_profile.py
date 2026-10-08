"""
Density profile along the long axis of a 3D KH run, numpy vs cupy.

For each of the last N snapshots, average ``dens`` over the two short axes
(x1, x3) and take the standard deviation over them, giving <rho>(x2) and
sigma_rho(x2).  The profiles are plotted, one curve per snapshot coloured by
time, and the same reduction is timed with numpy (CPU) and cupy (GPU).

    export CUDA_PATH=/usr/local/cuda-13.3
    python benchmark_density_profile.py kh3d_128x512x128 --last 500

Timings (files already in the page cache, see ``--no-warm``):
    reduction only   mean + std of one snapshot already in memory, numpy vs
                     cupy, plus the host -> GPU copy
    full passes      one pass over every snapshot per pipeline: read only;
                     numpy; cupy moving only dens to the GPU; cupy through
                     SimulationData(device="gpu"), which moves every field of
                     the .bin file
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cupy as cp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vis_athenak.simulation_data import SimulationData  # noqa: E402

DATA = Path("/data/sasi/simulation_outputs")
PLOTS = Path(__file__).resolve().parent / "plots"
SHORT = (0, 2)  # arrays are (nx3, nx2, nx1): average over x3 and x1


def profile(xp, rho):
    """Mean and std of ``rho`` over the short axes, accumulated in float64."""
    mean = rho.mean(axis=SHORT, dtype=xp.float64)
    std = rho.std(axis=SHORT, dtype=xp.float64)
    return mean, std


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", help="run folder under /data/sasi/simulation_outputs")
    ap.add_argument("--last", type=int, default=500, help="number of final snapshots")
    ap.add_argument("--no-warm", action="store_true",
                    help="skip the page-cache warm-up read")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--tmin", type=float, default=5.0,
                    help="start time (code units = Myr) of the averaged profile plot")
    args = ap.parse_args()

    root = DATA / args.run
    sim = SimulationData(root / f"{args.run}.athinput", root / "bin")
    frames = sim[-args.last:]
    n = len(frames)
    print(f"{args.run}: using the last {n} of {len(sim)} snapshots "
          f"(asked for {args.last})")

    if not args.no_warm:
        t = time.perf_counter()
        for p in (f.paths[next(iter(f.paths))] for f in frames):
            with open(p, "rb") as fp:
                while fp.read(1 << 26):
                    pass
        print(f"  page-cache warm-up read: {time.perf_counter() - t:.1f} s")

    # GPU warm-up: kernel compilation and memory pool, not part of the timing.
    profile(cp, cp.ones((4, 4, 4), dtype=cp.float32))
    cp.cuda.Device().synchronize()

    # Each pipeline is a separate full pass over the snapshots.
    def run(device, reduce):
        sim_d = SimulationData(root / f"{args.run}.athinput", root / "bin", device=device)
        out = []
        t = time.perf_counter()
        for frame in sim_d[-args.last:]:
            out.append(reduce(frame))
            frame.unload()
        cp.cuda.Device().synchronize()
        return out, time.perf_counter() - t

    passes = {
        "read only (CPU)": ("cpu", lambda f: f["dens"].shape),
        "numpy: read + reduce": ("cpu", lambda f: profile(np, f["dens"])),
        "cupy: read, dens.to_gpu(), reduce": (
            "cpu", lambda f: profile(cp, f.fields["dens"].to_gpu().data)),
        'cupy: SimulationData(device="gpu")': ("gpu", lambda f: profile(cp, f["dens"])),
    }
    results, totals = {}, {}
    for label, (device, reduce) in passes.items():
        results[label], totals[label] = run(device, reduce)

    means_np = np.array([m for m, _ in results["numpy: read + reduce"]])
    stds_np = np.array([s for _, s in results["numpy: read + reduce"]])
    for label in list(passes)[2:]:
        m = cp.stack([m for m, _ in results[label]]).get()
        s = cp.stack([s for _, s in results[label]]).get()
        assert np.allclose(means_np, m, rtol=1e-10, atol=0), label
        # std relative to the mean: near-uniform slabs have std ~ rounding noise
        assert np.allclose(stds_np / means_np, s / means_np, rtol=0, atol=1e-6), label
    print("  numpy and cupy profiles agree")

    # The reduction alone, on the last snapshot already in memory.
    rho = frames[-1]["dens"]
    rho_gpu = cp.asarray(rho)
    reps = 20

    def per_call(fn, sync=False):
        fn()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        if sync:
            cp.cuda.Device().synchronize()
        return (time.perf_counter() - t) / reps

    t_np = per_call(lambda: profile(np, rho))
    t_cp = per_call(lambda: profile(cp, rho_gpu), sync=True)
    t_h2d = per_call(lambda: cp.asarray(rho), sync=True)

    print(f"\n  dens per snapshot: {rho.shape} float32 = {rho.nbytes / 2**20:.0f} MB\n")
    print("  reduction only (mean + std over x1, x3), one snapshot in memory:")
    print(f"    numpy  {1e3 * t_np:8.2f} ms")
    print(f"    cupy   {1e3 * t_cp:8.2f} ms   ({t_np / t_cp:.1f}x faster)")
    print(f"    host -> GPU copy of dens {1e3 * t_h2d:.2f} ms\n")
    print(f"  full passes over {n} snapshots (files in page cache):")
    print(f"    {'':38s} {'total':>8s} {'per snapshot':>13s}")
    for label, sec in totals.items():
        print(f"    {label:38s} {sec:7.2f}s {1e3 * sec / n:10.1f} ms")
    base = totals["numpy: read + reduce"]
    best = min(totals[k] for k in list(passes)[2:])
    print(f"\n  end to end: cupy is {base / best:.2f}x the speed of numpy")
    del rho_gpu

    if args.no_plot:
        return
    # ── Plot ─────────────────────────────────────────────────────────────
    frame = frames[-1]
    x2 = frame["x2v"]
    times = np.array([f.time for f in frames])
    cmap = plt.get_cmap("viridis")
    norm = matplotlib.colors.Normalize(times.min(), times.max())

    fig, (ax_m, ax_s) = plt.subplots(2, 1, figsize=(7, 7), sharex=True,
                                     constrained_layout=True)
    for t, m, s in zip(times, means_np, stds_np):
        ax_m.plot(x2, m, color=cmap(norm(t)), lw=0.8, alpha=0.8)
        ax_s.plot(x2, s, color=cmap(norm(t)), lw=0.8, alpha=0.8)
    ax_m.set_yscale("log")
    ax_m.set_ylabel(r"$\langle\rho\rangle_{x,z}$  [code]")
    ax_s.set_yscale("log")
    ax_s.set_ylabel(r"$\sigma_\rho$ over $x,z$  [code]")
    ax_s.set_xlabel(r"$y$ ($x_2$)  [code length]")
    ax_m.set_title(f"{args.run}: density profile, last {n} snapshots "
                   f"(t = {times.min():.2f}–{times.max():.2f})")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=(ax_m, ax_s), label="time [code]")

    PLOTS.mkdir(parents=True, exist_ok=True)
    out = PLOTS / f"density_profile_{args.run}.png"
    fig.savefig(out, dpi=150)
    print(f"\n  saved {out}")

    # ── Time-averaged profile for t > tmin, with shaded spread ───────────
    late = times > args.tmin
    if not late.any():
        print(f"  no snapshots after t = {args.tmin}; skipping the averaged plot")
        return
    m_late, s_late = means_np[late], stds_np[late]
    mean = m_late.mean(axis=0)
    std_time = m_late.std(axis=0)                     # of the slab mean, across snapshots
    std_all = np.sqrt((s_late**2).mean(axis=0) + std_time**2)  # over every cell and snapshot
    floor = 0.5 * mean.min()                          # keep the bands on the log axis

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    color = "C0"
    ax.fill_between(x2, np.maximum(mean - std_all, floor), mean + std_all,
                    color=color, alpha=0.2, lw=0,
                    label=r"$\pm1\sigma$ over $x, z$ and $t$ (all cells)")
    ax.fill_between(x2, np.maximum(mean - std_time, floor), mean + std_time,
                    color=color, alpha=0.45, lw=0,
                    label=r"$\pm1\sigma$ of $\langle\rho\rangle_{x,z}$ over $t$")
    ax.plot(x2, mean, color=color, lw=1.5,
            label=r"$\overline{\langle\rho\rangle}_{x,z}$, mean over $t$")
    ax.set_yscale("log")
    ax.set_ylim(bottom=floor)
    ax.set_xlabel(r"$y$ ($x_2$)  [code length]")
    ax.set_ylabel(r"$\rho$  [code]")
    ax.set_title(f"{args.run}: density profile, t > {args.tmin:g} "
                 f"({late.sum()} snapshots, t = {times[late].min():.2f}–{times[late].max():.2f})")
    ax.legend(loc="lower left", fontsize=9)
    out = PLOTS / f"density_profile_mean_std_t{args.tmin:g}_{args.run}.png"
    fig.savefig(out, dpi=150)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
