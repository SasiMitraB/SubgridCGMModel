#!/usr/bin/env python3
"""
cooling_cg_comparison.py
===============================================================================
Subgrid cooling term of the 512x1024 HR run:

  C = <n^2 Lambda(T)> - C_CG,   C_CG = (<rho>/mu m_p)^2 Lambda(T_CG)

where T_CG = mu m_p <P> / (k_B <rho>) is the EOS temperature of the coarse cell
(@eq-cooling). Both terms use the cooling window of the HR run itself
(AthenaK ISMCoolFn is zero for T <= 1.05e4 K and T > 0.95e6 K).

Outputs (in assets/):
  subgrid_cooling_snapshot.png      - HR temperature | T_CG | map of C
  cooling_maps_snapshot.png         - <n^2 Lambda(T)> | C_CG | C
  subgrid_cooling_distribution.png  - distribution of C over --t-range (default 5-10 Myr)
  subgrid_cooling_vs_temperature.png - joint distribution of T_CG and C over --t-range
===============================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "conv_nn"))

import ergane
from data.coarse_grain_utils import coarse_grain, compute_subgrid_cooling
from pdf_cnn import lambda_cool

HR_DIR = PROJECT_ROOT / "simulation_outputs/hr_gpu_512x1024"
OUT_DIR = Path(__file__).resolve().parent / "assets"

# Cooling window of the HR run (athenak/src/srcterms/ismcooling.hpp)
LOGT_ON = 4.1
LOGT_OFF = 5.9
T_NORM = LogNorm(1e3, 1e7)


def lam(T):
    return lambda_cool(T, mask=True, LOGT_ACTIVE_START=LOGT_ON, LOGT_ACTIVE_END=LOGT_OFF)


def load_frame(sim, n, b):
    """HR temperature, T_CG and the subgrid cooling term C [erg cm^-3 s^-1] for frame n."""
    f = sim.get_frame(n)
    rho = f.density / f.units.density
    P = f.pressure / f.units.pressure
    vx = f.velx / f.units.velocity
    vy = f.vely / f.units.velocity
    T = f.temperature

    cg = coarse_grain(rho, vx, vy, P, b=b)
    cool = compute_subgrid_cooling(rho, T, cg["rho_c"], cg["T_c"], lam, b=b)
    return {
        "t": sim.times[sim.frame_numbers.index(n)],  # code time = Myr (Frame.time is mis-scaled in CGS)
        "T": T,
        "T_c": cg["T_c"],
        "true": cool["cool_true"],   # <n^2 Lambda(T)>
        "cg": cool["cool_closure"],  # C_CG = <n>^2 Lambda(T_CG)
        "C": cool["tau_lambda"],     # <n^2 Lambda> - C_CG
    }


def plot_cooling_maps(sim, frame, b):
    """Maps of <n^2 Lambda(T)>, C_CG and C for one snapshot."""
    d = load_frame(sim, frame, b)
    extent = [sim.x1min, sim.x1max, sim.x2min, sim.x2max]

    positive = np.concatenate([d["true"][d["true"] > 0], d["cg"][d["cg"] > 0]])
    cool_norm = LogNorm(vmin=10.0 ** np.floor(np.log10(positive.min())),
                        vmax=10.0 ** np.ceil(np.log10(positive.max())))
    cool_cmap = plt.get_cmap("magma").copy()
    cool_cmap.set_bad("#d9d9d9")  # zero cooling

    nonzero = d["C"][d["C"] != 0]
    vmax = np.abs(nonzero).max()
    top = 10.0 ** np.floor(np.log10(vmax))

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 6.4), sharey=True, constrained_layout=True)
    for ax, key in zip(axes[:2], ["true", "cg"]):
        im = ax.imshow(np.ma.masked_less_equal(d[key], 0), origin="lower", extent=extent,
                       norm=cool_norm, cmap=cool_cmap)
    fig.colorbar(im, ax=axes[:2], location="bottom", shrink=0.8,
                 label=r"cooling rate [erg cm$^{-3}$ s$^{-1}$] (grey: zero)")

    im = axes[2].imshow(d["C"], origin="lower", extent=extent, cmap="coolwarm",
                        norm=SymLogNorm(linthresh=top * 1e-4, vmin=-vmax, vmax=vmax, base=10))
    fig.colorbar(im, ax=axes[2], location="bottom", ticks=[-top, 0, top],
                 label=r"$\mathsf{C}$ [erg cm$^{-3}$ s$^{-1}$]")

    titles = [r"$\langle n^2\Lambda(T)\rangle$",
              r"$\mathcal{C}_{CG} = \langle n\rangle^2\Lambda(T_{CG})$",
              r"$\mathsf{C} = \langle n^2\Lambda\rangle - \mathcal{C}_{CG}$"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [pc]")
    axes[0].set_ylabel("y [pc]")

    fig.suptitle(f"t = {d['t']:.2f} Myr, coarse cells of {b}$\\times${b} HR cells", fontsize=11)
    out = OUT_DIR / "cooling_maps_snapshot.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_snapshot(sim, frame, b):
    d = load_frame(sim, frame, b)
    C = d["C"]
    extent = [sim.x1min, sim.x1max, sim.x2min, sim.x2max]

    nonzero = C[C != 0]
    vmax = np.abs(nonzero).max()
    top = 10.0 ** np.floor(np.log10(vmax))  # largest whole decade inside the data range
    linthresh = top * 1e-4

    fig, (axT, axTc, axC) = plt.subplots(1, 3, figsize=(9.5, 6.4), sharey=True, constrained_layout=True)

    im = axT.imshow(d["T"], origin="lower", extent=extent, norm=T_NORM, cmap="magma")
    axTc.imshow(d["T_c"], origin="lower", extent=extent, norm=T_NORM, cmap="magma")
    fig.colorbar(im, ax=[axT, axTc], location="bottom", label=r"$T$ [K]", shrink=0.8)

    im = axC.imshow(C, origin="lower", extent=extent, cmap="coolwarm",
                    norm=SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=10))
    ticks = [-top, 0, top]
    fig.colorbar(im, ax=axC, location="bottom", ticks=ticks,
                 label=r"$\mathsf{C}$ [erg cm$^{-3}$ s$^{-1}$]")

    titles = [f"HR temperature ({sim.nx}$\\times${sim.ny})",
              f"$T_{{CG}}$ ({sim.nx // b}$\\times${sim.ny // b})",
              r"$\mathsf{C} = \langle n^2\Lambda\rangle - \mathcal{C}_{CG}$"]
    for ax, title in zip([axT, axTc, axC], titles):
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [pc]")
    axT.set_ylabel("y [pc]")

    fig.suptitle(f"t = {d['t']:.2f} Myr, coarse cells of {b}$\\times${b} HR cells", fontsize=11)
    out = OUT_DIR / "subgrid_cooling_snapshot.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def collect_frames(sim, frames, b):
    """C and T_CG of every coarse cell in the given frames, flattened."""
    data = [load_frame(sim, n, b) for n in frames]
    return {
        "C": np.concatenate([d["C"].ravel() for d in data]),
        "T_c": np.concatenate([d["T_c"].ravel() for d in data]),
        "t": np.array([d["t"] for d in data]),
    }


def plot_distribution(stack):
    """Distribution of C over all coarse cells of the given frames, split by sign."""
    C, t = stack["C"], stack["t"]
    nonzero = C[C != 0]

    logabs = np.log10(np.abs(nonzero))
    bins = np.linspace(np.floor(logabs.min()), np.ceil(logabs.max()), 41)
    neg, pos = logabs[nonzero < 0], logabs[nonzero > 0]

    fig, ax = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    ax.hist(neg, bins=bins, histtype="step", lw=2, color="#3b4cc0",
            label=rf"$\mathsf{{C}} < 0$: $\mathcal{{C}}_{{CG}}$ too high ({100 * neg.size / nonzero.size:.0f}%)")
    ax.hist(pos, bins=bins, histtype="step", lw=2, color="#b40426",
            label=rf"$\mathsf{{C}} > 0$: $\mathcal{{C}}_{{CG}}$ too low ({100 * pos.size / nonzero.size:.0f}%)")
    ax.set_xlabel(r"$\log_{10}|\mathsf{C}|$ [erg cm$^{-3}$ s$^{-1}$]")
    ax.set_ylabel("coarse cells (all frames)")
    ax.set_title(f"Distribution of $\\mathsf{{C}}$, t = {t.min():.0f}–{t.max():.0f} Myr "
                 f"({t.size} frames)", fontsize=11)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)  # room for the legend
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.text(0.02, 0.80, f"{C.size - nonzero.size} of {C.size} cells have $\\mathsf{{C}} = 0$ (not shown)",
            transform=ax.transAxes, fontsize=9, color="#333333")

    out = OUT_DIR / "subgrid_cooling_distribution.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_vs_temperature(stack):
    """Joint distribution of T_CG and C: where C_CG undercools (C > 0) and overcools (C < 0)."""
    C, Tc, t = stack["C"], stack["T_c"], stack["t"]
    nz = C != 0
    logT = np.log10(Tc[nz])
    logabs = np.log10(np.abs(C[nz]))
    sign = np.sign(C[nz])

    xbins = np.linspace(np.floor(logT.min() * 4) / 4, np.ceil(logT.max() * 4) / 4, 81)
    ybins = np.linspace(np.floor(logabs.min()), np.ceil(logabs.max()), 61)
    h_pos, _, _ = np.histogram2d(logT[sign > 0], logabs[sign > 0], bins=[xbins, ybins])
    h_neg, _, _ = np.histogram2d(logT[sign < 0], logabs[sign < 0], bins=[xbins, ybins])
    norm = LogNorm(1, max(h_pos.max(), h_neg.max()))

    fig, (ax_pos, ax_neg) = plt.subplots(2, 1, figsize=(7, 7), sharex=True, constrained_layout=True)
    panels = [
        (ax_pos, h_pos, "Reds", rf"$\mathsf{{C}} > 0$: $\mathcal{{C}}_{{CG}}$ undercools ({(sign > 0).sum()} cells)"),
        (ax_neg, h_neg, "Blues", rf"$\mathsf{{C}} < 0$: $\mathcal{{C}}_{{CG}}$ overcools ({(sign < 0).sum()} cells)"),
    ]
    for ax, h, cmap, label in panels:
        im = ax.pcolormesh(xbins, ybins, np.ma.masked_equal(h.T, 0), norm=norm, cmap=cmap)
        fig.colorbar(im, ax=ax, label="coarse cells (all frames)")
        for logT_edge in (LOGT_ON, LOGT_OFF):
            ax.axvline(logT_edge, color="#333333", lw=1, ls="--")
        ax.set_ylabel(r"$\log_{10}|\mathsf{C}|$ [erg cm$^{-3}$ s$^{-1}$]")
        ax.text(0.02, 0.95 if ax is ax_pos else 0.05, label, transform=ax.transAxes,
                va="top" if ax is ax_pos else "bottom", fontsize=9, color="#333333")
    ax_neg.invert_yaxis()  # |C| grows away from the boundary between the panels
    ax_neg.set_xlabel(r"$\log_{10} T_{CG}$ [K]")
    ax_pos.set_title(f"$\\mathsf{{C}}$ against coarse temperature, t = {t.min():.0f}–{t.max():.0f} Myr "
                     f"({t.size} frames)\n(dashed: cooling cutoffs at $\\log_{{10}} T$ = {LOGT_ON:.2f} and {LOGT_OFF:.2f})",
                     fontsize=11)

    out = OUT_DIR / "subgrid_cooling_vs_temperature.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block", type=int, default=32, help="coarse-graining factor (default 32)")
    parser.add_argument("--snapshot-time", type=float, default=8.0, help="time [Myr] of the snapshot")
    parser.add_argument("--t-range", type=float, nargs=2, default=[5.0, 10.0],
                        help="time range [Myr] of the distribution plot")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    sim = ergane.SimulationData(datafolder=str(HR_DIR / "bin"),
                                athinp=str(HR_DIR / "kh_radiative_512x1024.athinput"))
    snap = sim.frame_numbers[int(np.argmin(np.abs(sim.times - args.snapshot_time)))]
    print("Saved", plot_snapshot(sim, snap, args.block))
    print("Saved", plot_cooling_maps(sim, snap, args.block))
    in_range = (sim.times >= args.t_range[0]) & (sim.times <= args.t_range[1])
    frames = [n for n, keep in zip(sim.frame_numbers, in_range) if keep]
    stack = collect_frames(sim, frames, args.block)
    print("Saved", plot_distribution(stack))
    print("Saved", plot_vs_temperature(stack))


if __name__ == "__main__":
    main()
