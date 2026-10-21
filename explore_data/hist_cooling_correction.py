"""
2D histogram of the cooling correction C against the resolved temperature T_CG over
every coarse cell of an AthenaK run (theory/notes/Overleaf):
  T_CG = (mu m_u / k_B) <P>/<rho>                                 (eq-temperature)
  C    = <n^2 Lambda(T)> - (<rho>/mu m_u)^2 Lambda(T_CG)         (eq-cooling)
with the cooling function of athenak/src/srcterms/ismcooling.hpp.  The per-cell
values are cached in an .npz next to the figure, so replotting does not reread the run.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm

from vis_athenak import SimulationData
from vis_athenak.utils.ism_cooling import ISMCoolFn
from vis_athenak.utils.units import Units

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.coarse_grain_utils import block_mean

run = "/data/sasi/simulation_outputs/kh3d_256x1024x256"
athinput = f"{run}/kh3d_256x1024x256.athinput"
b = 16  # coarse-graining block size
linthresh = 1e-27  # |C| below this is linear on the symlog axis [erg cm^-3 s^-1]


def cooling_function(T):
    """Lambda(T) [erg cm^3 s^-1] as in ismcooling.hpp: zero outside 1.05e4 < T <= 0.95e6 K,
    Koyama & Inutsuka (2002) for T <= 10^4.2 K, the SPEX table above."""
    koyama = 2.0e-19 * np.exp(-1.184e5 / (T + 1.0e3)) + 2.8e-28 * np.sqrt(T) * np.exp(-92.0 / T)
    Lambda = np.where(T <= 10**4.2, koyama, ISMCoolFn(T))
    return np.where((T > 1.05e4) & (T <= 0.95e6), Lambda, 0.0)


def coarse_cells(frame, units, gamma):
    """T_CG [K] and C [erg cm^-3 s^-1] of every coarse cell of a frame."""
    frame.load(["dens", "eint"])  # only these two variables are read
    rho = frame["dens"].astype(np.float64)
    P = (gamma - 1) * frame["eint"].astype(np.float64)
    frame.unload()
    true_c = block_mean((units.number_density_cgs * rho)**2
                        * cooling_function(units.temperature_cgs * P / rho), b)
    rho_c, P_c = block_mean(rho, b), block_mean(P, b)
    T_cg = units.temperature_cgs * P_c / rho_c
    resolved = (units.number_density_cgs * rho_c)**2 * cooling_function(T_cg)
    return T_cg.ravel(), (true_c - resolved).ravel()


def compute(frames):
    sim = SimulationData(athinput, f"{run}/bin")
    units = Units.from_params(sim.params)
    gamma = float(sim.params["hydro"]["gamma"])
    T, C, t = [], [], []
    for i in tqdm(frames, unit="frame"):
        T_i, C_i = coarse_cells(sim[i], units, gamma)
        T.append(T_i)
        C.append(C_i)
        t.append(np.full(T_i.size, sim[i].time))
    return dict(T_cg=np.concatenate(T), C=np.concatenate(C), time=np.concatenate(t),
                frames=np.array(frames))


def plot(d, out):
    logT = np.log10(d["T_cg"])
    C = d["C"]
    ybins = np.concatenate([-np.logspace(-23, np.log10(linthresh), 61)[:-1],
                            np.linspace(-linthresh, linthresh, 11),
                            np.logspace(np.log10(linthresh), -23, 61)[::-1][1:][::-1]])
    ybins = np.sort(ybins)
    xbins = np.linspace(3.8, 6.2, 121)

    fig, ax = plt.subplots(figsize=(7, 5.5), layout="constrained")
    *_, im = ax.hist2d(logT, np.clip(C, ybins[0], ybins[-1]), bins=[xbins, ybins],
                       norm=LogNorm(), cmap="viridis")
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.axvspan(4.1, 5.9, color="0.5", alpha=0.12, lw=0, label=r"$\log T_{CG}\in[4.1,5.9]$")
    for T_edge in (1.05e4, 0.95e6):
        ax.axvline(np.log10(T_edge), color="k", ls=":", lw=0.8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel=r"$\log_{10} T_{CG}$ [K]", ylabel=r"$C$ [erg cm$^{-3}$ s$^{-1}$]",
           title=f"{len(C)} coarse cells ({b}$^3$), {len(d['frames'])} frames, "
                 f"t = {d['time'].min():.0f}–{d['time'].max():.0f}")
    ax.legend(loc="lower left", fontsize=8)
    fig.colorbar(im, ax=ax, label="coarse cells")
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--every", type=int, default=4, help="use every n-th frame")
    parser.add_argument("--out", default="hist_cooling_correction.png")
    parser.add_argument("--recompute", action="store_true", help="ignore the .npz cache")
    args = parser.parse_args()

    cache = Path(args.out).with_suffix(".npz")
    n = len(SimulationData(athinput, f"{run}/bin"))
    frames = list(range(0, n, args.every))
    if cache.exists() and not args.recompute and list(np.load(cache)["frames"]) == frames:
        d = dict(np.load(cache))
    else:
        d = compute(frames)
        np.savez(cache, **d)
    plot(d, args.out)
