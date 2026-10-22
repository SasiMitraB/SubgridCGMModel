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

Per-cell values are cached one file per frame (cell_cache.py), written as each
frame is computed, so replotting does not reread the run.
Frames before ns.T_MIN = 2.5 Myr, the initial transient, are left out of every
figure (they stay in the cache).

    python theory/notes/plotting_scripts/momentum_tensor.py [--b 16] [--every 1] [--recompute]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter

from vis_athenak.utils.units import Units

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from data.coarse_grain_utils import block_mean  # noqa: E402

import cell_cache  # noqa: E402
import notes_style as ns  # noqa: E402

FIELDS = ["dens", "velx", "vely", "velz", "eint"]
CACHE_VERSION = 3  # bump when the stored per-cell quantities change

KMS2 = 1e10           # erg g^-1 in (km/s)^2
LOGF_EDGES = np.linspace(-9, 0, 121)  # f > 1e-8 in every cell from T_MIN on
T_MARKS = (1e4, 1e6)  # K, temperatures whose e_th is marked on the k_max axis
PAIRS = ["xx", "yy", "zz", "xy", "xz", "yz"]  # independent components of tau
S_MIN = 1e-10         # smallest eigenvalue of S below this: degenerate (z_k infinite)
ROOT2, ROOT6 = np.sqrt(2), np.sqrt(6)


# ─── Per-cell quantities ─────────────────────────────────────────────────────

def blocks(f, b):
    """View of a 3D field as (Z, b, Y, b, X, b) blocks."""
    nz, ny, nx = f.shape
    return f.reshape(nz // b, b, ny // b, b, nx // b, b)


def coarse_cells(q, sim, b):
    """tau_ij = R_ij/rho_c, k_sg, e_th, k_max and f for each coarse cell, the specific
    energies in cgs [erg g^-1].

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
    v2 = Units.from_params(sim.params).velocity_cgs**2
    return {k: (v if k == "f" else v * v2).ravel() for k, v in cells.items()}


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

    fig, ax = plt.subplots(figsize=ns.figsize(8.0, ns.WIDE), layout="constrained")
    # cells below the lowest f bin (f = 0 among them) land in the bottom row
    *_, im = ax.hist2d(logk, np.clip(logf, LOGF_EDGES[0], LOGF_EDGES[-1]),
                       bins=[logk_edges, LOGF_EDGES],
                       norm=LogNorm(), cmap=ns.SEQ, rasterized=True)
    ax.plot(centres(logk_edges), binned_median(logk, logf, logk_edges),
            color=ns.OVERLAY, label=r"median $f$")

    # f = 0 means k_max = e_th: mark e_th of the cold and hot phases
    for T in T_MARKS:
        x = np.log10(d["t_to_eth"] * T / KMS2)
        ax.axvline(x, color=ns.HAIRLINE, ls=(0, (1, 2)), lw=0.8)
        ax.text(x, 0.5, rf" $\eth(10^{{{np.log10(T):.0f}}}\,\mathrm{{K}})$",
                transform=ax.get_xaxis_transform(), rotation=90,
                ha="left", va="center", fontsize=ns.SMALL, color=ns.MUTED)
    if n_low:
        ax.text(0.98, 0.03,
                rf"\itshape bottom row: {n_low / logf.size:.1%} of cells with $f<10^{{{LOGF_EDGES[0]:.0f}}}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=ns.SMALL, color=ns.MUTED)

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


T_COLD, T_HOT = 1.05e4, 0.95e6  # K: cold below T_COLD, hot from T_HOT up


def phases(d, sh):
    """Masks of cold, mixing-layer and hot cells by temperature (all with k_sg > 0)."""
    has = np.isfinite(sh["aniso"])
    T = d["e_th"] / d["t_to_eth"]
    return {"cold": has & (T < T_COLD),
            "mixing": has & (T >= T_COLD) & (T < T_HOT),
            "hot": has & (T >= T_HOT)}


PHASES = [("mixing", "mixing layer", ns.SERIES[2]),
          ("cold", "cold phase", ns.SERIES[0]),
          ("hot", "hot phase", ns.SERIES[1])]


# ─── 1. Anisotropy: barycentric map ──────────────────────────────────────────

CORNERS = np.array([[1, 0], [0, 0], [0.5, np.sqrt(3) / 2]])  # 1-, 2-, 3-component
CORNER_NAMES = ["one-component\n(one axis)",
                "two-component\n(one plane)",
                "isotropic\n(all three equal)"]


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

    fig, axes = plt.subplots(3, 1, figsize=ns.figsize(15.0))
    fig.subplots_adjust(left=0.16, right=0.78, top=0.93, bottom=0.03, hspace=0.35)
    for ax, (key, label, _) in zip(axes, PHASES):
        im = ax.pcolormesh(*edges, np.ma.masked_equal(H[key], 0).T, norm=norm,
                           cmap=ns.SEQ, rasterized=True)
        ax.fill(*CORNERS.T, fill=False, edgecolor=ns.MUTED, lw=0.6)
        for (cx, cy), name, ha, va in zip(CORNERS, CORNER_NAMES,
                                          ["left", "right", "center"], ["top", "top", "bottom"]):
            ax.text(cx, cy + (0.03 if va == "bottom" else -0.03), name, ha=ha, va=va,
                    fontsize=ns.SMALL, color=ns.MUTED, linespacing=1)
        count = rf"{masks[key].sum():,}".replace(",", r"\,")
        ax.set_title(rf"{label} \quad {count} cells", fontsize=8)
        ax.set(xlim=(-0.5, 1.5), ylim=(-0.2, 1.12), aspect="equal")
        ax.set_axis_off()
    ns.colorbar(fig, im, axes, "coarse cells", shrink=0.8, pad=0.16)
    ns.save(fig, out)


# ─── 2. The network targets z_k ──────────────────────────────────────────────

Z_RANGE = 6           # |z_k| shown, k >= 1; beyond it only cold/hot-phase tails
Z0_MIN = -20          # z_0 = logit f > -19 in every cell from T_MIN on (f > 1e-8)
Z_RANGES = [(Z0_MIN, 0)] + [(-Z_RANGE, Z_RANGE)] * 5
Z_LABELS = [rf"$z_{k}$" for k in range(6)]
Z_KEY = "\n".join([r"$z_0 = \ln\,[f/(1-f)]$", r"$z_1$: $xx-yy$", r"$z_2$: $xx+yy-2zz$",
                   r"$z_3$: $xy$", r"$z_4$: $xz$", r"$z_5$: $yz$"])
CORNER_FRACS = (0.9, 0.5)  # fractions of each phase enclosed by the contours


def enclosing_levels(H, fracs):
    """Density values of the contours that enclose the given fractions of H."""
    flat = np.sort(H.ravel())[::-1]
    c = np.cumsum(flat) / flat.sum()
    return np.unique([flat[np.searchsorted(c, q)] for q in fracs])


def plot_targets(d, sh, out):
    masks = phases(d, sh)
    f = d["f"]
    z = np.column_stack([np.log(f) - np.log1p(-f), sh["z"]])   # z_0 ... z_5
    good = np.isfinite(sh["z"][:, 0])
    sel = {key: masks[key] & good for key, *_ in PHASES}
    n = len(Z_RANGES)
    edges = [np.linspace(*r, 61) for r in Z_RANGES]

    fig, axes = plt.subplots(n, n, figsize=ns.figsize(15.5, ns.WIDE), sharex="col")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.14, hspace=0.07, wspace=0.07)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.set_axis_off()
                continue
            ax.set_xlim(Z_RANGES[j])
            ax.tick_params(labelsize=ns.SMALL, length=2)
            ax.set_xticks([-20, -10, 0] if j == 0 else [-4, 0, 4])
            for key, _, colour in PHASES:
                zs = z[sel[key]]
                if i == j:
                    ax.hist(zs[:, i], bins=edges[i], density=True, histtype="step",
                            color=colour, lw=1)
                    continue
                H = np.histogram2d(zs[:, j], zs[:, i], bins=[edges[j], edges[i]])[0]
                H = gaussian_filter(H, 1.0)
                levels = enclosing_levels(H, CORNER_FRACS)
                ax.contour(centres(edges[j]), centres(edges[i]), H.T, levels=levels,
                           colors=colour, linewidths=[0.5, 1.0][-len(levels):])
            if i == j:
                ax.set_yticks([])
                ax.axvline(0, color=ns.HAIRLINE, lw=0.5)
            else:
                ax.set_ylim(Z_RANGES[i])
                ax.set_yticks([-4, 0, 4])
                ax.axvline(0, color=ns.HAIRLINE, lw=0.5)
                ax.axhline(0, color=ns.HAIRLINE, lw=0.5)
            if i < n - 1:
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel(Z_LABELS[j])
            if j > 0 or i == 0:
                ax.tick_params(labelleft=False)
            if j == 0 and i > 0:
                ax.set_ylabel(Z_LABELS[i])

    handles = [Line2D([], [], color=colour, lw=1.4, label=label) for _, label, colour in PHASES]
    n_deg = np.count_nonzero(np.isfinite(sh["aniso"]) & ~good)
    note = rf"contours enclose {CORNER_FRACS[1]:.0%} and {CORNER_FRACS[0]:.0%} of each phase"
    if n_deg:
        note += rf";" "\n" rf"{n_deg:,} cells with degenerate $\mathsf{{S}}$ left out".replace(",", r"\,")
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.99))
    fig.text(0.6, 0.8, Z_KEY, ha="left", va="top", fontsize=ns.SMALL, linespacing=1.5)
    fig.text(0.55, 0.005, note, ha="center", va="bottom", fontsize=ns.SMALL, color=ns.MUTED,
             linespacing=1.3)
    ns.save(fig, out)


# ─── 3. Anisotropy against f and against time ────────────────────────────────

LOGF_MIN = -10        # f axis of the anisotropy plot starts here
ANISO_MARKS = [(np.sqrt(1 / 6), "two-component"), (np.sqrt(2 / 3), "one-component")]


def plot_anisotropy(d, sh, out):
    masks = phases(d, sh)
    has = masks["mixing"] | masks["cold"] | masks["hot"]
    logf, a = np.clip(np.log10(d["f"][has]), LOGF_MIN, 0), sh["aniso"][has]
    fedges = np.linspace(LOGF_MIN, 0, 101)
    aedges = np.linspace(0, np.sqrt(2 / 3), 81)

    fig, (ax, bx) = plt.subplots(1, 2, figsize=ns.figsize(6.8, ns.WIDE), sharey=True,
                                 width_ratios=[1.6, 1], layout="constrained")
    *_, im = ax.hist2d(logf, a, bins=[fedges, aedges], norm=LogNorm(), cmap=ns.SEQ,
                       rasterized=True)
    ax.plot(centres(fedges), binned_median(logf, a, fedges), color=ns.OVERLAY, label="median")
    ax.set(xlabel=rf"$\log_{{10}}\,f$ \quad (\itshape clipped at ${LOGF_MIN}$)",
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
    bx.legend(loc="lower center", fontsize=ns.SMALL)
    for v, name in ANISO_MARKS:
        for a_ in (ax, bx):
            a_.axhline(v, color=ns.HAIRLINE, lw=0.6)
        bx.text(1.02, v, name, transform=bx.get_yaxis_transform(), ha="left", va="center",
                fontsize=ns.SMALL, color=ns.MUTED)
    ns.save(fig, out)


# ─── 4. Orientation of the principal axis ────────────────────────────────────

def plot_orientation(d, sh, out):
    m = phases(d, sh)["mixing"]
    times = np.unique(d["time"])
    tedges = np.concatenate([[times[0]], centres(times), [times[-1]]])
    thedges = np.linspace(-90, 90, 73)
    H = np.histogram2d(d["time"][m], sh["theta"][m], bins=[tedges, thedges])[0]
    frac = H / np.maximum(H.sum(axis=1, keepdims=True), 1)   # each time sums to 1

    fig, (ax, bx) = plt.subplots(1, 2, figsize=ns.figsize(6.8, ns.WIDE), sharey=True,
                                 width_ratios=[3, 1], layout="constrained")
    im = ax.pcolormesh(tedges, thedges, np.ma.masked_equal(frac, 0).T, cmap=ns.SEQ,
                       rasterized=True)
    ax.set(xlabel=r"$t$ [Myr]", ylabel=r"angle of $\mathbf{e}_1$ in $x$--$y$ [deg]",
           yticks=[-90, -45, 0, 45, 90])
    ns.colorbar(fig, im, ax, "fraction of mixing-layer cells at $t$", location="top")

    bx.hist(sh["theta"][m], bins=thedges, orientation="horizontal", histtype="stepfilled",
            color=ns.SERIES[0], alpha=0.8, weights=np.full(m.sum(), 1 / m.sum()))
    bx.set(xlabel="all times")
    for a_ in (ax, bx):
        for y in (-45, 0, 45):
            a_.axhline(y, color=ns.HAIRLINE, lw=0.6, ls=(0, (1, 2)))
    bx.text(0.95, 0.02, rf"median $|e_{{1,z}}| = {np.median(sh['e1z'][m]):.2f}$",
            transform=bx.transAxes, ha="right", va="bottom", fontsize=ns.SMALL, color=ns.MUTED)
    ns.save(fig, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--b", type=int, default=16, help="coarse block size")
    parser.add_argument("--every", type=int, default=1, help="use every n-th frame")
    parser.add_argument("--recompute", action="store_true", help="recompute cached frames")
    args = parser.parse_args()

    d, sim = cell_cache.load("momentum_cells", args.b, args.every, FIELDS, coarse_cells,
                             CACHE_VERSION, args.recompute)
    # e_th [erg g^-1] = t_to_eth * T [K]
    units = Units.from_params(sim.params)
    gamma = float(sim.params["hydro"]["gamma"])
    d["t_to_eth"] = units.velocity_cgs**2 / ((gamma - 1) * units.temperature_cgs)
    d = ns.drop_transient(d)
    print(f"{d['f'].size} cells of {int(d['b'])}^3, {len(d['frames'])} frames, "
          f"t = {d['time'].min():.1f}-{d['time'].max():.1f} Myr")

    plot_fraction(d, ns.image_path(2, "ksg_fraction.pdf"))
    sh = shape(d)
    plot_barycentric(d, sh, ns.image_path(2, "rshape_barycentric.pdf"))
    plot_targets(d, sh, ns.image_path(2, "rshape_targets.pdf"))
    plot_anisotropy(d, sh, ns.image_path(2, "rshape_anisotropy.pdf"))
    plot_orientation(d, sh, ns.image_path(2, "rshape_orientation.pdf"))
