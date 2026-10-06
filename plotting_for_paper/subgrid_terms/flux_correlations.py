#!/usr/bin/env python3
"""
flux_correlations.py
===============================================================================
How the subgrid energy flux Q (@eq-energy-flux) correlates with the resolved
(coarse-grained) fields of the 512x1024 HR run:

  <rho>        volume average
  T_CG         EOS temperature of the coarse cell, mu m_p <P> / (k_B <rho>)
  v~_x, v~_y   mass-weighted velocities <rho v_i> / <rho>

In 2D v_z = 0, so Q_z = 0 and only Q_x, Q_y are shown. Every coarse cell of
every frame in --t-range goes into one joint histogram per (field, component).
Q is signed and spans many decades, so its axis is symlog.

Outputs (in assets/):
  subgrid_Q_vs_temperature.png, subgrid_Q_vs_density.png,
  subgrid_Q_vs_vx.png, subgrid_Q_vs_vy.png
===============================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ergane
from data.coarse_grain_utils import coarse_grain
from cooling_cg_comparison import HR_DIR, LOGT_OFF, LOGT_ON, OUT_DIR
from gate_evaluation import FLUX_UNIT, subgrid_energy_flux

VEL_UNIT_KMS = 3.08568e13 / 3.15576e13  # pc/Myr in km/s

FIELDS = [
    {"key": "T_c", "log": True, "label": r"$\log_{10} T_{CG}$ [K]",
     "name": r"coarse temperature $T_{CG}$", "out": "subgrid_Q_vs_temperature.png"},
    {"key": "rho_c", "log": True, "label": r"$\log_{10}\langle\rho\rangle$ [$m_H$ cm$^{-3}$]",
     "name": r"coarse density $\langle\rho\rangle$", "out": "subgrid_Q_vs_density.png"},
    {"key": "ux_c", "log": False, "label": r"$\tilde v_x$ [km s$^{-1}$]",
     "name": r"coarse velocity $\tilde v_x$", "out": "subgrid_Q_vs_vx.png"},
    {"key": "uy_c", "log": False, "label": r"$\tilde v_y$ [km s$^{-1}$]",
     "name": r"coarse velocity $\tilde v_y$", "out": "subgrid_Q_vs_vy.png"},
]


def collect(sim, frames, b):
    """Coarse fields and Q_x, Q_y [erg cm^-2 s^-1] of every coarse cell, flattened."""
    out = {k: [] for k in ("rho_c", "T_c", "ux_c", "uy_c", "Qx", "Qy")}
    for n in frames:
        f = sim.get_frame(n)
        rho = f.density / f.units.density
        P = f.pressure / f.units.pressure
        vx = f.velx / f.units.velocity
        vy = f.vely / f.units.velocity
        cg = coarse_grain(rho, vx, vy, P, b=b)
        qx, qy = subgrid_energy_flux(rho, P, vx, vy, b)
        for k in ("rho_c", "T_c", "ux_c", "uy_c"):
            out[k].append(cg[k].ravel())
        out["Qx"].append(qx.ravel() * FLUX_UNIT)
        out["Qy"].append(qy.ravel() * FLUX_UNIT)
    d = {k: np.concatenate(v) for k, v in out.items()}
    d["ux_c"] *= VEL_UNIT_KMS
    d["uy_c"] *= VEL_UNIT_KMS
    return d


def symlog_edges(top, linthresh, n_log=30, n_lin=4):
    """Bin edges in data units that are uniform on a symlog axis."""
    pos = np.logspace(np.log10(linthresh), np.log10(top), n_log + 1)
    lin = np.linspace(-linthresh, linthresh, 2 * n_lin + 1)[1:-1]
    return np.concatenate([-pos[::-1], lin, pos])


def plot_field(d, spec, t_range, n_frames, b, dx_pc):
    x = np.log10(d[spec["key"]]) if spec["log"] else d[spec["key"]]
    xbins = np.linspace(np.percentile(x, 0.05), np.percentile(x, 99.95), 71)

    absQ = np.abs(np.concatenate([d["Qx"], d["Qy"]]))
    top = 10.0 ** np.ceil(np.log10(absQ.max()))
    linthresh = 10.0 ** np.floor(np.log10(np.percentile(absQ[absQ > 0], 20)))
    ybins = symlog_edges(top, linthresh)

    hists = {c: np.histogram2d(x, d[c], bins=[xbins, ybins])[0] for c in ("Qx", "Qy")}
    norm = LogNorm(1, max(h.max() for h in hists.values()))

    fig, axes = plt.subplots(2, 1, figsize=(7, 7.5), sharex=True, constrained_layout=True)
    for ax, comp, sub in zip(axes, ("Qx", "Qy"), ("x", "y")):
        q = d[comp]
        im = ax.pcolormesh(xbins, ybins, np.ma.masked_equal(hists[comp].T, 0), norm=norm, cmap="viridis")
        ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
        ax.set_ylim(-top, top)
        ax.axhline(0, color="#333333", lw=0.8)

        if spec["key"] == "T_c":
            for edge in (LOGT_ON, LOGT_OFF):
                ax.axvline(edge, color="#333333", lw=1, ls="--")
        frac_neg = (q < 0).mean()
        ax.text(0.02, 0.96, f"$\\mathsf{{Q}}_{sub} < 0$ in {100 * frac_neg:.0f}% of cells",
                transform=ax.transAxes, va="top", fontsize=9, color="#333333",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=2))
        ax.set_ylabel(rf"$\mathsf{{Q}}_{sub}$ [erg cm$^{{-2}}$ s$^{{-1}}$]")
        fig.colorbar(im, ax=ax, label="coarse cells (all frames)")
    axes[-1].set_xlabel(spec["label"])
    axes[0].set_title(f"Subgrid energy flux against the {spec['name']}, {dx_pc:g} pc coarse cells\n"
                      f"t = {t_range[0]:.0f}–{t_range[1]:.0f} Myr ({n_frames} frames), "
                      f"{b}$\\times${b} HR cells per coarse cell", fontsize=10)

    out = OUT_DIR / spec["out"]
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block", type=int, default=32, help="coarse-graining factor (default 32)")
    parser.add_argument("--t-range", type=float, nargs=2, default=[5.0, 10.0])
    args = parser.parse_args()

    sim = ergane.SimulationData(datafolder=str(HR_DIR / "bin"),
                                athinp=str(HR_DIR / "kh_radiative_512x1024.athinput"))
    keep = (sim.times >= args.t_range[0]) & (sim.times <= args.t_range[1])
    frames = [n for n, k in zip(sim.frame_numbers, keep) if k]
    d = collect(sim, frames, args.block)
    dx_pc = (sim.x1max - sim.x1min) / sim.nx * args.block  # coarse cell size (code length = pc)
    for spec in FIELDS:
        print("Saved", plot_field(d, spec, args.t_range, len(frames), args.block, dx_pc))


if __name__ == "__main__":
    main()
