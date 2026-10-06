#!/usr/bin/env python3
"""
flux_vs_cooling_time.py
===============================================================================
Time series of the subgrid energy flux Q against the total cooling of the
512x1024 HR run, component by component.

Q (erg cm^-2 s^-1) and <n^2 Lambda(T)> (erg cm^-3 s^-1) have different units;
what enters the energy equation (@eq-final) is div Q = d_x Q_x + d_y Q_y, which
has the units of the cooling. Summed over the whole domain, div Q is zero up to
the y boundaries (x is periodic), so the net ratio is taken over the cooling
cells (<n^2 Lambda> > 0), the cells the flux feeds:

  net_i   = - sum_{cooling cells} d_i Q_i / sum <n^2 Lambda>
            (> 0: the subgrid flux delivers energy to the cooling cells)
  gross_i =   sum_{all cells} |d_i Q_i| / sum <n^2 Lambda>

Output: assets/subgrid_Q_vs_cooling_time.png
===============================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import ergane
from data.coarse_grain_utils import coarse_grain, compute_subgrid_cooling
from cooling_cg_comparison import HR_DIR, OUT_DIR, lam
from gate_evaluation import DIV_UNIT, subgrid_energy_flux


def flux_derivatives(qx, qy, dx, dy):
    """d_x Q_x (periodic, centred) and d_y Q_y (centred, one-sided at the edges); as gate_evaluation.divergence."""
    dqx = (np.roll(qx, -1, axis=1) - np.roll(qx, 1, axis=1)) / (2.0 * dx)
    dqy = np.gradient(qy, dy, axis=0)
    return dqx, dqy


def time_series(sim, b):
    rows = []
    for n, t in zip(sim.frame_numbers, sim.times):
        f = sim.get_frame(n)
        rho = f.density / f.units.density
        P = f.pressure / f.units.pressure
        vx = f.velx / f.units.velocity
        vy = f.vely / f.units.velocity
        cg = coarse_grain(rho, vx, vy, P, b=b)
        cool = compute_subgrid_cooling(rho, f.temperature, cg["rho_c"], cg["T_c"], lam, b=b)["cool_true"]
        qx, qy = subgrid_energy_flux(rho, P, vx, vy, b)
        dqx, dqy = flux_derivatives(qx, qy, (sim.x1max - sim.x1min) / qx.shape[1],
                                    (sim.x2max - sim.x2min) / qx.shape[0])
        dqx, dqy = dqx * DIV_UNIT, dqy * DIV_UNIT
        on = cool > 0
        rows.append([t, cool.sum(), -dqx[on].sum(), -dqy[on].sum(), np.abs(dqx).sum(), np.abs(dqy).sum()])
    return np.array(rows)


def plot(ts, b, dx_pc):
    t, cool = ts[:, 0], ts[:, 1]
    ok = cool > 0
    colors = {"x": "#1f77b4", "y": "#d62728"}

    fig, (ax_net, ax_gross) = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True, constrained_layout=True)
    for i, c in enumerate("xy"):
        ax_net.plot(t[ok], ts[ok, 2 + i] / cool[ok], color=colors[c], lw=1.4,
                    label=rf"$-\sum_{{\rm cool}}\partial_{c}\mathsf{{Q}}_{c}\,/\,\sum\langle n^2\Lambda\rangle$")
        ax_gross.plot(t[ok], ts[ok, 4 + i] / cool[ok], color=colors[c], lw=1.4,
                      label=rf"$\sum|\partial_{c}\mathsf{{Q}}_{c}|\,/\,\sum\langle n^2\Lambda\rangle$")
    ax_net.axhline(0, color="#333333", lw=0.8)
    ax_net.set_ylabel("net, over cooling cells")
    ax_net.set_title(f"Subgrid energy flux relative to the total cooling, {dx_pc:g} pc coarse cells "
                     f"({b}$\\times${b} HR cells)\n"
                     r"net $> 0$: the subgrid flux delivers energy to the cells that cool", fontsize=10)
    ax_gross.set_ylim(bottom=0)
    ax_gross.set_ylabel("gross, over all cells")
    ax_gross.set_xlabel("t [Myr]")
    for ax in (ax_net, ax_gross):
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9, edgecolor="none", ncol=2)
        ax.grid(alpha=0.3)

    out = OUT_DIR / "subgrid_Q_vs_cooling_time.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--block", type=int, default=32, help="coarse-graining factor (default 32)")
    args = parser.parse_args()

    sim = ergane.SimulationData(datafolder=str(HR_DIR / "bin"),
                                athinp=str(HR_DIR / "kh_radiative_512x1024.athinput"))
    ts = time_series(sim, args.block)
    dx_pc = (sim.x1max - sim.x1min) / sim.nx * args.block
    late = ts[:, 0] >= 5.0
    for i, c in enumerate("xy"):
        print(f"t >= 5 Myr, {c}: net {np.mean(ts[late, 2 + i] / ts[late, 1]):.3f}, "
              f"gross {np.mean(ts[late, 4 + i] / ts[late, 1]):.3f}")
    print("Saved", plot(ts, args.block, dx_pc))


if __name__ == "__main__":
    main()
