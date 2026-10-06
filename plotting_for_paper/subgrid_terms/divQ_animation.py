#!/usr/bin/env python3
"""
divQ_animation.py
===============================================================================
Animation of div Q (@eq-final) on the 0.625 pc (32x32 HR cells) and 1.25 pc
(64x64) coarse grids of the 512x1024 HR run, with streamlines of Q on top
(line width grows with log10|Q|). Q_z = 0 in 2D.

Outputs: assets/divQ_streamlines.mp4, assets/divQ_streamlines_t8.png
===============================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import SymLogNorm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ergane
from cooling_cg_comparison import HR_DIR, OUT_DIR
from gate_evaluation import DIV_UNIT, FLUX_UNIT, divergence, subgrid_energy_flux

GRIDS = [32, 64]   # HR cells per coarse cell: 0.625 pc and 1.25 pc
Q_DECADES = 4      # streamline width spans the top Q_DECADES decades of |Q|


def load(sim, frames):
    """Q_x, Q_y [erg cm^-2 s^-1] and div Q [erg cm^-3 s^-1] on each coarse grid, per frame."""
    maps = {b: {} for b in GRIDS}
    for n in frames:
        f = sim.get_frame(n)
        rho = f.density / f.units.density
        P = f.pressure / f.units.pressure
        vx = f.velx / f.units.velocity
        vy = f.vely / f.units.velocity
        for b in GRIDS:
            qx, qy = subgrid_energy_flux(rho, P, vx, vy, b)
            dq = divergence(qx, qy, (sim.x1max - sim.x1min) / qx.shape[1], (sim.x2max - sim.x2min) / qx.shape[0])
            maps[b][n] = (qx * FLUX_UNIT, qy * FLUX_UNIT, dq * DIV_UNIT)
    return maps


def make_figure(sim, maps, frames):
    dx_hr = (sim.x1max - sim.x1min) / sim.nx
    extent = [sim.x1min, sim.x1max, sim.x2min, sim.x2max]
    div_max = max(np.abs(m[2]).max() for b in GRIDS for m in maps[b].values())
    top = 10.0 ** np.floor(np.log10(div_max))
    norm = SymLogNorm(linthresh=top * 1e-3, vmin=-top, vmax=top, base=10)
    q_top = max(np.log10(np.hypot(m[0], m[1]).max()) for b in GRIDS for m in maps[b].values())

    fig, axes = plt.subplots(1, len(GRIDS), figsize=(9.6, 8.2), constrained_layout=True)
    panels = []
    for ax, b in zip(axes, GRIDS):
        qx0 = maps[b][frames[0]][0]
        ny, nx = qx0.shape
        xc = sim.x1min + (np.arange(nx) + 0.5) * (sim.x1max - sim.x1min) / nx
        yc = sim.x2min + (np.arange(ny) + 0.5) * (sim.x2max - sim.x2min) / ny
        im = ax.imshow(maps[b][frames[0]][2], origin="lower", extent=extent, cmap="coolwarm", norm=norm)
        ax.set_title(f"{b * dx_hr:g} pc coarse cells ({b}$\\times${b} HR cells)", fontsize=11)
        ax.set_xlabel("x [pc]")
        panels.append({"b": b, "ax": ax, "im": im, "xc": xc, "yc": yc, "stream": None})
    axes[0].set_ylabel("y [pc]")
    fig.colorbar(panels[-1]["im"], ax=axes, location="right", shrink=0.6, ticks=[-top, 0, top],
                 label=r"$\nabla\cdot\mathbf{Q}$ [erg cm$^{-3}$ s$^{-1}$]  (blue: energy gained)")
    title = fig.suptitle("", fontsize=11)
    legend = (f"streamlines of $\\mathbf{{Q}}$, width $\\propto \\log_{{10}}|\\mathbf{{Q}}|$ over the top "
              f"{Q_DECADES} decades (max $10^{{{q_top:.1f}}}$ erg cm$^{{-2}}$ s$^{{-1}}$)")

    def update(n):
        for p in panels:
            qx, qy, dq = maps[p["b"]][n]
            p["im"].set_data(dq)
            if p["stream"] is not None:
                p["stream"].lines.remove()
                for patch in p["ax"].patches[:]:
                    patch.remove()
            qmag = np.hypot(qx, qy)
            width = 0.3 + 1.7 * np.clip((np.log10(np.maximum(qmag, 1e-300)) - (q_top - Q_DECADES)) / Q_DECADES, 0, 1)
            p["stream"] = p["ax"].streamplot(p["xc"], p["yc"], qx, qy, color="#222222", linewidth=width,
                                             density=0.9, arrowsize=0.8)
            p["ax"].set_xlim(extent[:2])
            p["ax"].set_ylim(extent[2:])
        title.set_text(f"Divergence of the subgrid energy flux, t = {sim.times[sim.frame_numbers.index(n)]:.2f} Myr\n{legend}")
        return []

    return fig, update


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t-range", type=float, nargs=2, default=[5.0, 10.0])
    parser.add_argument("--snapshot-time", type=float, default=8.0)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    sim = ergane.SimulationData(datafolder=str(HR_DIR / "bin"),
                                athinp=str(HR_DIR / "kh_radiative_512x1024.athinput"))
    keep = (sim.times >= args.t_range[0]) & (sim.times <= args.t_range[1])
    frames = [n for n, k in zip(sim.frame_numbers, keep) if k]
    snap = frames[int(np.argmin(np.abs(sim.times[keep] - args.snapshot_time)))]
    maps = load(sim, frames)

    fig, update = make_figure(sim, maps, frames)
    update(snap)
    out = OUT_DIR / "divQ_streamlines_t8.png"
    fig.savefig(out, dpi=150)
    print("Saved", out)

    out = OUT_DIR / "divQ_streamlines.mp4"
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    try:
        writer = FFMpegWriter(fps=args.fps, codec="h264_nvenc", extra_args=["-preset", "p4", "-pix_fmt", "yuv420p"])
        anim.save(str(out), writer=writer, dpi=120)
    except Exception:
        writer = FFMpegWriter(fps=args.fps, codec="mpeg4", extra_args=["-q:v", "2", "-pix_fmt", "yuv420p"])
        anim.save(str(out), writer=writer, dpi=120)
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
