"""
Animate the cooling rate n^2 Lambda(T) in the middle z-slice over every frame of an
AthenaK run, with the cooling function of athenak/src/srcterms/ismcooling.hpp:
  left:   true cooling at full resolution
  middle: true cooling coarse grained, <n^2 Lambda(T)>
  right:  resolved cooling from the coarse variables, (<rho>/mu m_u)^2 Lambda(T_CG),
          T_CG = (mu m_u / k_B) <P>/<rho>  (theory/notes/Overleaf, eq-cooling)
  far right: the cooling correction C = <n^2 Lambda(T)> - resolved
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

from vis_athenak import SimulationData
from vis_athenak.plotting.slices import Plane, draw_slice
from vis_athenak.simulation_data.plane import read_slab
from vis_athenak.utils.ism_cooling import ISMCoolFn
from vis_athenak.utils.units import Units

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.coarse_grain_utils import block_mean

run = "/data/sasi/simulation_outputs/kh3d_256x1024x256"
athinput = f"{run}/kh3d_256x1024x256.athinput"
b = 16  # coarse-graining block size: 64x256x64 -> 16x64x16
vmin, vmax = 1e-27, 2e-23  # cooling rate colour limits [erg cm^-3 s^-1]
fps = 15


def cooling_function(T):
    """Lambda(T) [erg cm^3 s^-1] as in ismcooling.hpp: zero outside 1.05e4 < T <= 0.95e6 K,
    Koyama & Inutsuka (2002) for T <= 10^4.2 K, the SPEX table above."""
    koyama = 2.0e-19 * np.exp(-1.184e5 / (T + 1.0e3)) + 2.8e-28 * np.sqrt(T) * np.exp(-92.0 / T)
    Lambda = np.where(T <= 10**4.2, koyama, ISMCoolFn(T))
    return np.where((T > 1.05e4) & (T <= 0.95e6), Lambda, 0.0)


def cooling(rho, P):
    """n^2 Lambda(T) [erg cm^-3 s^-1] from code-unit density and pressure."""
    n = units.number_density_cgs * rho
    return n**2 * cooling_function(units.temperature_cgs * P / rho)


def slices(frame):
    """Middle z-slices of the true (fine and coarse grained) and resolved cooling.
    Only the z-slab k0:k0+b of the coarse cells holding the middle slice is read."""
    slab = read_slab(frame, ["dens", "eint"], "z", k0, k0 + b)
    rho, P = slab["dens"], (gamma - 1) * slab["eint"]
    true = cooling(rho, P)
    # <rho>, <P> as coarse_grain's rho_c, P_c; T_CG from <P>/<rho>
    resolved = cooling(block_mean(rho, b), block_mean(P, b))[0]
    true_c = block_mean(true, b)[0]
    return true[k - k0], true_c, resolved, true_c - resolved


sim = SimulationData(athinput, f"{run}/bin")
units = Units.from_params(sim.params)
gamma = float(sim.params["hydro"]["gamma"])
grid = Plane(sim[0]).grid  # from the file header only
x, y = grid["x1f"], grid["x2f"]
faces = [(x, y)] + 3 * [(x[::b], y[::b])]

nx3, nx2, nx1 = (len(grid[f"x{d}f"]) - 1 for d in (3, 2, 1))
k = nx3 // 2  # fine middle slice
k0 = k // b * b  # first fine slice of the coarse cell row containing it
coarse = f"{nx1 // b}x{nx2 // b}x{nx3 // b}"
titles = [f"true, {nx1}x{nx2}x{nx3}", f"true, {coarse}", f"resolved, {coarse}", f"correction C, {coarse}"]

fig, axes = plt.subplots(1, 4, figsize=(12, 8), layout="constrained")
panels = slices(sim[0])
images = []
for ax, xy, data, title in zip(axes[:3], faces, panels, titles):
    images.append(draw_slice(ax, *xy, data, cmap="inferno", norm="log", vmin=vmin, vmax=vmax,
                             colorbar=False))
    ax.set_facecolor("lightgray")  # zero cooling (masked in log scale)
    ax.set_title(title)
xc, yc = faces[3]
images.append(axes[3].imshow(panels[3], origin="lower", extent=(xc[0], xc[-1], yc[0], yc[-1]),
                             cmap="RdBu_r", norm=SymLogNorm(linthresh=vmin, vmin=-vmax, vmax=vmax),
                             interpolation="nearest"))
axes[3].set(title=titles[3], xlabel="x")
for ax in axes[1:]:
    ax.set_ylabel("")
fig.colorbar(images[0], ax=axes[:3], label=r"$n^2\Lambda(T)$ [erg cm$^{-3}$ s$^{-1}$]",
             shrink=0.8)
fig.colorbar(images[3], ax=axes[3], label=r"$C$ [erg cm$^{-3}$ s$^{-1}$]", shrink=0.8)
suptitle = fig.suptitle(f"t = {sim[0].time:.2f}")


def update(i):
    frame = sim[i]
    for image, data in zip(images, slices(frame)):
        image.set_data(data)
    suptitle.set_text(f"t = {frame.time:.2f}")


anim = FuncAnimation(fig, update, frames=len(sim))
with tqdm(total=len(sim), unit="frame", desc="rendering") as bar:
    anim.save("cooling_slice.mp4", writer="ffmpeg", fps=fps, dpi=150,
              progress_callback=lambda i, n: bar.update(1))
