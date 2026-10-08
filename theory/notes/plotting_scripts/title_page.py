"""
Title-page band: one snapshot of the 2D radiative KH run at its native
resolution Δ and coarse-grained to cells of 8Δ, 16Δ and 32Δ (Δ = native cell
of the 512x1024 run).

Each panel shows log density with arrows for the coarse momentum <ρv>, and is
saved as its own PDF sized to fill a quarter of the title-page band (5.25 x 9.2 cm),
with no axes; titlepage.tex adds the labels.

Coarse-graining: density is the volume average over b x b blocks, momentum the
volume average of ρv (so the coarse velocity <ρv>/<ρ> is mass-weighted).

Run from anywhere (vis_athenak must be importable, e.g. `pip install -e vis_athenak`):
    python theory/notes/plotting_scripts/title_page.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from vis_athenak.simulation_data import SimulationData

RUN_DIR = Path("/data/sasi/2D simulations/hr_gpu_512x1024")
ATHINPUT = RUN_DIR / "kh_radiative_512x1024.athinput"
FRAME = 750
BLOCKS = (1, 8, 16, 32)

OUT_DIR = Path(__file__).resolve().parents[1] / "Overleaf" / "images" / "title_page"

# Panel geometry: a quarter of the 21 x 9.2 cm band at the top of the title page
CM = 1 / 2.54
PANEL_W, PANEL_H = 5.25, 9.2                # cm
X_RANGE = (-5.0, 5.0)                       # full box width [pc]
Y_CENTER = 0.0                              # mixing layer roughly mid-panel at t = 7.5 Myr
Y_HALF = 0.5 * (X_RANGE[1] - X_RANGE[0]) * PANEL_H / PANEL_W
Y_RANGE = (Y_CENTER - Y_HALF, Y_CENTER + Y_HALF)

# inferno without its palest 15%, so the arrows stay visible over the cold slab
CMAP = ListedColormap(plt.get_cmap("inferno")(np.linspace(0, 0.85, 256)))
LOGRHO_LIM = (-3.05, -0.95)                 # hot 1e-3, cold 1e-1 (code units)

ARROW_STRIDE = 32                            # one arrow per 32x32 native cells (same spots in every panel)
ARROW_COLOR = "#00E5FF"                      # cyan, outlined in ARROW_EDGE: stands out on all of inferno
ARROW_EDGE = "#0B0B12"


def block_mean(f, b):
    ny, nx = f.shape
    return f.reshape(ny // b, b, nx // b, b).mean(axis=(1, 3))


def load_frame():
    sim = SimulationData(ATHINPUT, RUN_DIR / "bin")
    frame = next(f for f in sim.frames if f.number == FRAME)
    frame.load(["dens", "mom1", "mom2"])
    rho = np.asarray(frame["dens"])[0]
    mx = np.asarray(frame["mom1"])[0]
    my = np.asarray(frame["mom2"])[0]
    x1f = np.asarray(frame["x1f"]).ravel()
    x2f = np.asarray(frame["x2f"]).ravel()
    return frame.time, rho, mx, my, x1f, x2f


def draw_panel(rho, mx, my, x1f, x2f, b, mom_scale, path):
    rho_c = block_mean(rho, b)
    mx_c, my_c = block_mean(mx, b), block_mean(my, b)
    xe, ye = x1f[::b], x2f[::b]

    fig = plt.figure(figsize=(PANEL_W * CM, PANEL_H * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.pcolormesh(xe, ye, np.log10(rho_c), cmap=CMAP,
                  vmin=LOGRHO_LIM[0], vmax=LOGRHO_LIM[1],
                  shading="flat", rasterized=True)

    # Arrows at the centres of 32x32 native blocks: the coarse momentum of the
    # b x b cell containing each point, so the panels differ only by b.
    s = ARROW_STRIDE
    iy = np.arange(s // 2, rho.shape[0], s)
    ix = np.arange(s // 2, rho.shape[1], s)
    IY, IX = np.meshgrid(iy, ix, indexing="ij")
    xc = 0.5 * (x1f[:-1] + x1f[1:])
    yc = 0.5 * (x2f[:-1] + x2f[1:])
    u, v = mx_c[IY // b, IX // b], my_c[IY // b, IX // b]
    ax.quiver(xc[IX], yc[IY], u, v, color=ARROW_COLOR,
              angles="xy", scale_units="xy", scale=mom_scale,
              width=0.006, headwidth=3.5, headlength=4, headaxislength=3.5,
              pivot="mid", edgecolor=ARROW_EDGE, linewidth=0.3)

    ax.set_xlim(*X_RANGE)
    ax.set_ylim(*Y_RANGE)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(path, dpi=400)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    time, rho, mx, my, x1f, x2f = load_frame()
    print(f"frame {FRAME}: t = {time:.3f} Myr, grid {rho.shape[1]} x {rho.shape[0]}")

    # One momentum scale for every panel: the 95th percentile of |<ρv>| at the
    # coarsest level spans about one arrow spacing.
    b_max = max(BLOCKS)
    mag = np.hypot(block_mean(mx, b_max), block_mean(my, b_max))
    spacing = ARROW_STRIDE * (x1f[1] - x1f[0])
    mom_scale = np.percentile(mag, 95) / (0.9 * spacing)

    for b in BLOCKS:
        path = OUT_DIR / f"density_{b}dx.pdf"
        draw_panel(rho, mx, my, x1f, x2f, b, mom_scale, path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
