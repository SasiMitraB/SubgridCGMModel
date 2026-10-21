"""
The trace of the subgrid momentum tensor R over every coarse cell of the
256x1024x256 radiative KH run, through the bounded fraction of eq-f:

  rho_c k_sg  = Tr(R)/2 = (<rho v^2> - <rho v>^2/<rho>)/2         (eq-R-split)
  rho_c k_max = <E> - <rho v>^2/2<rho>,   k_max = e_th + k_sg       (eq-kmax)
  f           = k_sg / k_max  in [0, 1)                             (eq-f)

with e_th = <P>/[(gamma-1) <rho>] and <P> = (gamma-1) <e_int>, so k_max is the
whole unresolved specific energy of the cell and f its share in subgrid motion.

and its shape S = R/Tr(R) (eq-R-split), with the five outputs z_k that give
it back through eq-A and eq-S, z_k = <log S - Tr(log S) I/3, E^(k)>_F.

Figures (Overleaf/images/chapter2/):
  ksg_fraction.pdf           2D histogram of f against k_max
  rshape_barycentric.pdf     eigenvalues of S on the barycentric triangle
  rshape_targets.pdf         distributions of the targets z_1..z_5
  rshape_anisotropy.pdf      ||S - I/3||_F against f, and against time
  rshape_orientation.pdf     angle of the principal axis of S in the x-y plane

Per-cell values are cached in an .npz, so replotting does not reread the run.

    python theory/notes/plotting_scripts/momentum_tensor.py [--b 16] [--every 4] [--recompute]
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from vis_athenak.simulation_data import SimulationData
from vis_athenak.utils.units import Units

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data.coarse_grain_utils import block_mean  # noqa: E402

import notes_style as ns  # noqa: E402

RUN_DIR = Path("/data/sasi/simulation_outputs/kh3d_256x1024x256")
ATHINPUT = RUN_DIR / "kh3d_256x1024x256.athinput"
FIELDS = ["dens", "velx", "vely", "velz", "eint"]

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE_VERSION = 2  # bump when the stored per-cell quantities change

KMS2 = 1e10           # erg g^-1 in (km/s)^2
LOGF_EDGES = np.linspace(-18, 0, 121)
T_MARKS = (1e4, 1e6)  # K, temperatures whose e_th is marked on the k_max axis
PAIRS = ["xx", "yy", "zz", "xy", "xz", "yz"]  # independent components of tau
F_MIX = 1e-3          # f above this: mixing-layer cells; below: pure hot or cold phase
S_MIN = 1e-10         # smallest eigenvalue of S below this: degenerate (z_k infinite)
ROOT2, ROOT6 = np.sqrt(2), np.sqrt(6)


# ─── Per-cell quantities ─────────────────────────────────────────────────────

def blocks(f, b):
    """View of a 3D field as (Z, b, Y, b, X, b) blocks."""
    nz, ny, nx = f.shape
    return f.reshape(nz // b, b, ny // b, b, nx // b, b)


def read_frame(frame):
    """The fields of a frame (float64); the disk read is the slow part."""
    frame.load(FIELDS)
    q = {k: frame[k].astype(np.float64) for k in FIELDS}
    frame.unload()
    return frame.time, q


def coarse_cells(q, b):
    """tau_ij = R_ij/rho_c, k_sg, e_th, k_max and f for each coarse cell (code units).

    tau is built from central moments, <rho dv_i dv_j>/rho_c with dv = v minus
    the Favre mean of the block, not from <rho v_i v_j> - <rho v_i><rho v_j>/rho_c:
    the same thing in exact arithmetic, but positive semi-definite in floating
    point, where the difference of two near-equal terms is not.
    """
    rho = q["dens"]
    rho_c = block_mean(rho, b)
    up = (slice(None), None) * 3   # coarse (Z, Y, X) -> broadcast over the blocks
    rho_b = blocks(rho, b)
    dv = {}
    for c, v in zip("xyz", ("velx", "vely", "velz")):
        dv[c] = blocks(q[v], b) - (block_mean(rho * q[v], b) / rho_c)[up]
    cells = {f"tau_{i}{j}": (rho_b * dv[i] * dv[j]).mean(axis=(1, 3, 5)) / rho_c
             for i, j in PAIRS}
    k_sg = 0.5 * (cells["tau_xx"] + cells["tau_yy"] + cells["tau_zz"])
    e_th = block_mean(q["eint"], b) / rho_c
    k_max = e_th + k_sg
    cells.update(k_sg=k_sg, e_th=e_th, k_max=k_max, f=k_sg / k_max)
    return {k: v.ravel() for k, v in cells.items()}


def compute(b, every):
    sim = SimulationData(str(ATHINPUT), str(RUN_DIR / "bin"))
    units = Units.from_params(sim.params)
    gamma = float(sim.params["hydro"]["gamma"])
    frames = list(range(0, len(sim), every))
    out = {}
    # Read the next frame in a background thread while this one is computed:
    # the run sits on a spinning disk, so the read takes ~4x the compute.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(read_frame, sim[frames[0]])
        for k, i in enumerate(frames):
            time, q = pending.result()
            if k + 1 < len(frames):
                pending = pool.submit(read_frame, sim[frames[k + 1]])
            cells = coarse_cells(q, b)
            cells["time"] = np.full(cells["f"].size, time)
            cells["frame"] = np.full(cells["f"].size, i)
            for key, v in cells.items():
                out.setdefault(key, []).append(v)
            print(f"\rframe {i}/{frames[-1]}", end="", flush=True)
    print()
    d = {k: np.concatenate(v) for k, v in out.items()}
    # specific energies to cgs [erg g^-1]
    for key in ("k_sg", "e_th", "k_max", *(f"tau_{p}" for p in PAIRS)):
        d[key] = d[key] * units.velocity_cgs**2
    # e_th [erg g^-1] = T_TO_ETH * T [K]
    t_to_eth = units.velocity_cgs**2 / ((gamma - 1) * units.temperature_cgs)
    return dict(d, frames=np.array(frames), b=b, t_to_eth=t_to_eth, version=CACHE_VERSION)


def binned_median(x, y, edges, min_count=20):
    """Median of y in each x bin holding at least min_count values (else nan)."""
    idx = np.digitize(x, edges) - 1
    out = np.full(edges.size - 1, np.nan)
    for k in range(edges.size - 1):
        g = y[idx == k]
        if g.size >= min_count:
            out[k] = np.median(g)
    return out


def centres(edges):
    return 0.5 * (edges[1:] + edges[:-1])


# ─── f against k_max ─────────────────────────────────────────────────────────

def plot_fraction(d, out):
    logk = np.log10(d["k_max"] / KMS2)
    with np.errstate(divide="ignore"):
        logf = np.log10(d["f"])
    n_low = np.count_nonzero(logf < LOGF_EDGES[0])  # includes f = 0
    lo, hi = np.floor(logk.min() * 4) / 4, np.ceil(logk.max() * 4) / 4
    logk_edges = np.linspace(lo, hi, 121)

    fig, ax = plt.subplots(figsize=ns.figsize(7.4), layout="constrained")
    # cells below the lowest f bin (f = 0 among them) land in the bottom row
    *_, im = ax.hist2d(logk, np.clip(logf, LOGF_EDGES[0], LOGF_EDGES[-1]),
                       bins=[logk_edges, LOGF_EDGES],
                       norm=LogNorm(), cmap=ns.SLATE, rasterized=True)
    ax.plot(centres(logk_edges), binned_median(logk, logf, logk_edges),
            color=ns.ACCENT, label=r"median $f$")

    # f = 0 means k_max = e_th: mark e_th of the cold and hot phases
    for T in T_MARKS:
        x = np.log10(d["t_to_eth"] * T / KMS2)
        ax.axvline(x, color=ns.HAIRLINE, ls=(0, (1, 2)), lw=0.8)
        ax.text(x, 0.5, rf" $\eth(10^{{{np.log10(T):.0f}}}\,\mathrm{{K}})$",
                transform=ax.get_xaxis_transform(), rotation=90,
                ha="left", va="center", fontsize=7, color=ns.MUTED)
    ax.text(0.98, 0.03,
            rf"\itshape bottom row: {n_low / logf.size:.1%} of cells with $f<10^{{{LOGF_EDGES[0]:.0f}}}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=ns.MUTED)

    ax.set(xlabel=r"$\log_{10}\,\kmax$ [km$^2$\,s$^{-2}$]",
           ylabel=r"$\log_{10}\,f = \log_{10}\,\ksg/\kmax$",
           xlim=(logk_edges[0], logk_edges[-1]), ylim=(LOGF_EDGES[0], LOGF_EDGES[-1]))
    ax.legend(loc="center")
    ns.colorbar(fig, im, ax, "coarse cells")
    ns.save(fig, out)


# ─── Shape tensor S and its outputs z_k (sec:R-matrix) ───────────────────────

def shape(d):
    """S = tau/Tr(tau) of every cell with k_sg > 0, and what is derived from it.

    Arrays are over all cells, nan where k_sg = 0 (and z_k nan where S is
    degenerate, smallest eigenvalue < S_MIN). Tensor indices are (x, y, z):
    x the flow, y across the layer.
    """
    n = d["f"].size
    tau = np.empty((n, 3, 3))
    for (i, j), p in zip([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)], PAIRS):
        tau[:, i, j] = tau[:, j, i] = d[f"tau_{p}"]
    tr = np.trace(tau, axis1=1, axis2=2)
    ok = tr > 0
    s, V = np.linalg.eigh(tau[ok] / tr[ok, None, None])   # ascending eigenvalues

    # z_k = <log S - Tr(log S)/3, E^(k)>_F, the inverse of eq-S
    good = s[:, 0] > S_MIN
    logs = np.log(np.where(good[:, None], s, 1.0))
    logs -= logs.mean(axis=1, keepdims=True)
    A = np.einsum("nik,nk,njk->nij", V, logs, V)
    z = np.stack([(A[:, 0, 0] - A[:, 1, 1]) / ROOT2,
                  (A[:, 0, 0] + A[:, 1, 1] - 2 * A[:, 2, 2]) / ROOT6,
                  ROOT2 * A[:, 0, 1], ROOT2 * A[:, 0, 2], ROOT2 * A[:, 1, 2]], axis=1)
    z[~good] = np.nan

    # principal (largest-eigenvalue) axis: angle from x in the x-y plane, sign folded
    e1 = V[:, :, 2]
    theta = np.degrees(np.arctan2(e1[:, 1], e1[:, 0]))
    theta = (theta + 90) % 180 - 90

    def full(a):
        out = np.full((n,) + a.shape[1:], np.nan)
        out[ok] = a
        return out

    return dict(s=full(s), z=full(z), aniso=full(np.sqrt(((s - 1 / 3)**2).sum(axis=1))),
                theta=full(theta), e1z=full(np.abs(e1[:, 2])))


def phases(d, sh):
    """Masks of mixing-layer and pure-phase cells (both with k_sg > 0)."""
    has = np.isfinite(sh["aniso"])
    return {"mixing": has & (d["f"] > F_MIX), "pure": has & (d["f"] <= F_MIX)}


PHASES = [("mixing", rf"mixing layer, $f>10^{{{np.log10(F_MIX):.0f}}}$", ns.SERIES[0]),
          ("pure", rf"pure phase, $f\le10^{{{np.log10(F_MIX):.0f}}}$", ns.SERIES[1])]


# ─── 1. Anisotropy: barycentric map ──────────────────────────────────────────

CORNERS = np.array([[1, 0], [0, 0], [0.5, np.sqrt(3) / 2]])  # 1-, 2-, 3-component


def barycentric(s):
    """Point in the triangle of S's eigenvalues (ascending s; Banerjee et al. 2007)."""
    l3, l2, l1 = s[:, 0], s[:, 1], s[:, 2]
    c = np.stack([l1 - l2, 2 * (l2 - l3), 3 * l3], axis=1)
    return c @ CORNERS


def plot_barycentric(d, sh, out):
    masks = phases(d, sh)
    edges = [np.linspace(0, 1, 101), np.linspace(0, np.sqrt(3) / 2, 88)]
    H = {}
    for key, *_ in PHASES:
        x, y = barycentric(sh["s"][masks[key]]).T
        H[key] = np.histogram2d(x, y, bins=edges)[0]
    norm = LogNorm(1, max(h.max() for h in H.values()))

    fig, axes = plt.subplots(1, 2, figsize=ns.figsize(6.2), layout="constrained")
    for ax, (key, label, _) in zip(axes, PHASES):
        im = ax.pcolormesh(*edges, np.ma.masked_equal(H[key], 0).T, norm=norm,
                           cmap=ns.SLATE, rasterized=True)
        ax.fill(*CORNERS.T, fill=False, edgecolor=ns.MUTED, lw=0.6)
        for (cx, cy), name, ha, va in zip(CORNERS, ["one-\ncomponent", "two-\ncomponent", "isotropic"],
                                          ["left", "right", "center"], ["top", "top", "bottom"]):
            ax.text(cx, cy + (0.03 if va == "bottom" else -0.03), name, ha=ha, va=va,
                    fontsize=7, color=ns.MUTED, linespacing=1)
        ax.set_title(rf"{label}" "\n" rf"{masks[key].sum():,} cells".replace(",", r"\,"))
        ax.set(xlim=(-0.08, 1.08), ylim=(-0.12, 0.98), aspect="equal")
        ax.set_axis_off()
    ns.colorbar(fig, im, axes, "coarse cells", shrink=0.8)
    ns.save(fig, out)


# ─── 2. The network targets z_k ──────────────────────────────────────────────

Z_RANGE = 6           # |z_k| shown; beyond it only pure-phase tails
Z_LABELS = [r"$z_1$: $xx-yy$", r"$z_2$: $xx+yy-2zz$", r"$z_3$: $xy$",
            r"$z_4$: $xz$", r"$z_5$: $yz$", r"$\lVert\mathsf{A}\rVert_F$"]


def plot_targets(d, sh, out):
    masks = phases(d, sh)
    z = np.column_stack([sh["z"], np.sqrt(np.nansum(sh["z"]**2, axis=1))])
    good = np.isfinite(sh["z"][:, 0])
    bins_z = np.linspace(-Z_RANGE, Z_RANGE, 81)
    bins_a = np.linspace(0, 2 * Z_RANGE, 81)

    fig, axes = plt.subplots(2, 3, figsize=ns.figsize(9.0), sharey=True, layout="constrained")
    for k, ax in enumerate(axes.flat):
        bins = bins_a if k == 5 else bins_z
        for key, label, colour in PHASES:
            v = z[masks[key] & good, k]
            w = np.full(v.size, 1 / v.size)
            ax.hist(np.clip(v, bins[0], bins[-1]), bins=bins, weights=w, histtype="step",
                    color=colour, lw=1, label=label)
        if k < 5:
            ax.axvline(0, color=ns.HAIRLINE, lw=0.6)
            med = np.median(z[masks["mixing"] & good, k])
            ax.text(0.97, 0.95, "median\n" rf"${med:+.2f}$", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color=ns.SERIES[0])
        ax.set_yscale("log")
        ax.set_xlabel(Z_LABELS[k])
    for ax in axes[:, 0]:
        ax.set_ylabel("fraction of cells")
    n_deg = np.count_nonzero(np.isfinite(sh["aniso"]) & ~good)
    note = (rf"\itshape {n_deg:,} degenerate cells (an eigenvalue of $\mathsf{{S}}$ "
            rf"$<10^{{{np.log10(S_MIN):.0f}}}$) left out; the outer bins collect the tails")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside upper center", ncols=2,
               title=note.replace(",", r"\,"), title_fontsize=7)
    ns.save(fig, out)


# ─── 3. Anisotropy against f and against time ────────────────────────────────

LOGF_MIN = -10        # f axis of the anisotropy plot starts here
ANISO_MARKS = [(np.sqrt(1 / 6), "two-component"), (np.sqrt(2 / 3), "one-component")]


def plot_anisotropy(d, sh, out):
    masks = phases(d, sh)
    has = masks["mixing"] | masks["pure"]
    logf, a = np.clip(np.log10(d["f"][has]), LOGF_MIN, 0), sh["aniso"][has]
    fedges = np.linspace(LOGF_MIN, 0, 101)
    aedges = np.linspace(0, np.sqrt(2 / 3), 81)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=ns.figsize(6.4), sharey=True,
                                 width_ratios=[1.6, 1], layout="constrained")
    *_, im = ax.hist2d(logf, a, bins=[fedges, aedges], norm=LogNorm(), cmap=ns.SLATE,
                       rasterized=True)
    ax.plot(centres(fedges), binned_median(logf, a, fedges), color=ns.ACCENT, label="median")
    ax.axvline(np.log10(F_MIX), color=ns.MUTED, ls=(0, (3, 2)), lw=0.6)
    ax.set(xlabel=rf"$\log_{{10}}\,f$ \quad (\itshape $f<10^{{{LOGF_MIN}}}$ in the first column)",
           ylabel=r"$\lVert\mathsf{S}-\Id/3\rVert_F$")
    ax.legend(loc="lower left")
    ns.colorbar(fig, im, ax, "coarse cells", location="top")

    times = np.unique(d["time"])
    for key, label, colour in PHASES:
        q = np.array([np.percentile(sh["aniso"][masks[key] & (d["time"] == t)], [16, 50, 84])
                      if np.any(masks[key] & (d["time"] == t)) else [np.nan] * 3 for t in times])
        bx.fill_between(times, q[:, 0], q[:, 2], color=colour, alpha=0.2, lw=0)
        bx.plot(times, q[:, 1], color=colour, label=label)
    bx.set(xlabel=r"$t$ [Myr]", xlim=(times[0], times[-1]))
    bx.legend(loc="lower center", fontsize=7)
    for v, name in ANISO_MARKS:
        for a_ in (ax, bx):
            a_.axhline(v, color=ns.HAIRLINE, lw=0.6)
        bx.text(1.02, v, name, transform=bx.get_yaxis_transform(), ha="left", va="center",
                fontsize=7, color=ns.MUTED)
    ns.save(fig, out)


# ─── 4. Orientation of the principal axis ────────────────────────────────────

def plot_orientation(d, sh, out):
    m = phases(d, sh)["mixing"]
    times = np.unique(d["time"])
    tedges = np.concatenate([[times[0]], centres(times), [times[-1]]])
    thedges = np.linspace(-90, 90, 73)
    H = np.histogram2d(d["time"][m], sh["theta"][m], bins=[tedges, thedges])[0]
    frac = H / np.maximum(H.sum(axis=1, keepdims=True), 1)   # each time sums to 1

    fig, (ax, bx) = plt.subplots(1, 2, figsize=ns.figsize(6.4), sharey=True,
                                 width_ratios=[3, 1], layout="constrained")
    # at t = 0 every cell is the tanh shear alone, e_1 = x: keep it off the colour scale
    im = ax.pcolormesh(tedges, thedges, np.ma.masked_equal(frac, 0).T, cmap=ns.SLATE,
                       vmax=frac[times > 0].max(), rasterized=True)
    ax.set(xlabel=r"$t$ [Myr]", ylabel=r"angle of $\mathbf{e}_1$ from $\hat{\mathbf{x}}$ in the $x$--$y$ plane [deg]",
           yticks=[-90, -45, 0, 45, 90])
    ns.colorbar(fig, im, ax, "fraction of mixing-layer cells at $t$", location="top")

    bx.hist(sh["theta"][m], bins=thedges, orientation="horizontal", histtype="stepfilled",
            color=ns.SERIES[0], alpha=0.8, weights=np.full(m.sum(), 1 / m.sum()))
    bx.set(xlabel="all times")
    for a_ in (ax, bx):
        for y in (-45, 0, 45):
            a_.axhline(y, color=ns.HAIRLINE, lw=0.6, ls=(0, (1, 2)))
    bx.text(0.95, 0.02, rf"median $|e_{{1,z}}| = {np.median(sh['e1z'][m]):.2f}$",
            transform=bx.transAxes, ha="right", va="bottom", fontsize=7, color=ns.MUTED)
    ns.save(fig, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--b", type=int, default=16, help="coarse block size")
    parser.add_argument("--every", type=int, default=4, help="use every n-th frame")
    parser.add_argument("--recompute", action="store_true", help="ignore the .npz cache")
    args = parser.parse_args()

    cache = CACHE / f"momentum_cells_b{args.b}_every{args.every}.npz"
    if cache.exists() and not args.recompute and int(np.load(cache)["version"]) == CACHE_VERSION:
        d = dict(np.load(cache))
    else:
        d = compute(args.b, args.every)
        cache.parent.mkdir(exist_ok=True)
        np.savez(cache, **d)
    print(f"{d['f'].size} cells of {int(d['b'])}^3, {len(d['frames'])} frames, "
          f"t = {d['time'].min():.1f}-{d['time'].max():.1f} Myr")

    plot_fraction(d, ns.image_path(2, "ksg_fraction.pdf"))
    sh = shape(d)
    plot_barycentric(d, sh, ns.image_path(2, "rshape_barycentric.pdf"))
    plot_targets(d, sh, ns.image_path(2, "rshape_targets.pdf"))
    plot_anisotropy(d, sh, ns.image_path(2, "rshape_anisotropy.pdf"))
    plot_orientation(d, sh, ns.image_path(2, "rshape_orientation.pdf"))
