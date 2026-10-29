"""
The Galilean-invariant subgrid energy flux q over every coarse cell of the
256x1024x256 radiative KH run, through the dimensionless vector w of
eq:bound_on_w (sec:Q-decomp):

  psi     = h + |dv|^2/2,  h = gamma e_int/rho,  dv = v - Favre mean of the block
  q_i     = rho_c Fav(dpsi dv_i)                                 (eq-single_covariance)
  w       = R^{-1/2} q / (sqrt(rho_c) psi_c) = tau^{-1/2} Fav(dpsi dv) / psi_c
  |w|^2  <= B^2 = tau(psi, psi) / psi_c^2                         (eq:bound_on_w)

with tau = R/rho_c and R^{1/2} the symmetric root; psi_c = gamma e_th + k_sg.
The ratio c = |w|/B in [0, 1] is how much of the Cauchy-Schwarz bound a cell uses.

Figures (Overleaf/images/chapter2/):
  w_distribution.pdf    distributions of B, c, |w|, w-hat and w_x, w_y, w_z, by phase
  w_vs_f.pdf            B, |w| and c against f
  w_vs_targets.pdf      w_x, w_y, w_z and c against the outputs z_0..z_5 of R
  w_scaling.pdf         |w| against f, and the exponent alpha of |w| ~ f^alpha
  w_targets.pdf         distributions of the outputs z_6, z_7, z_8 of w, by phase

Per-cell values are cached one file per frame (cell_cache.py). Frames before
ns.T_MIN = 2.5 Myr, the initial transient, are left out of every figure.

    python theory/notes/plotting_scripts/energy_flux.py [--b 16] [--every 1] [--recompute]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from vis_athenak.utils.units import Units

import cell_cache  # noqa: E402
import notes_style as ns  # noqa: E402
from momentum_tensor import (FIELDS, PAIRS, PHASES, S_MIN, binned_median, blocks,  # noqa: E402
                             centres, phases, shape)
from data.coarse_grain_utils import block_mean  # noqa: E402  (path set by momentum_tensor)

CACHE_VERSION = 1  # bump when the stored per-cell quantities change


# ─── Per-cell quantities ─────────────────────────────────────────────────────

def coarse_cells(q, sim, b):
    """tau_ij, k_sg, e_th, f (as momentum_tensor.coarse_cells) and w_x, w_y, w_z, B2 = B^2
    and psi_c of each coarse cell; w and B are dimensionless, the energies in cgs.

    All moments are central moments about the Favre mean of the block, so the
    covariances satisfy Cauchy-Schwarz in floating point too. w is nan where S is
    degenerate (smallest eigenvalue below S_MIN), where tau^{-1/2} does not exist.
    """
    gamma = float(sim.params["hydro"]["gamma"])
    rho = q["dens"]
    rho_c = block_mean(rho, b)
    up = (slice(None), None) * 3
    rho_b = blocks(rho, b)
    fav = lambda x: (rho_b * x).mean(axis=(1, 3, 5)) / rho_c

    dv = {c: blocks(q[v], b) - (block_mean(rho * q[v], b) / rho_c)[up]
          for c, v in zip("xyz", ("velx", "vely", "velz"))}
    psi = gamma * blocks(q["eint"], b) / rho_b + 0.5 * (dv["x"]**2 + dv["y"]**2 + dv["z"]**2)
    psi_c = fav(psi)
    dpsi = psi - psi_c[up]
    del psi

    cells = {f"tau_{i}{j}": fav(dv[i] * dv[j]) for i, j in PAIRS}
    cov = np.stack([fav(dpsi * dv[c]).ravel() for c in "xyz"], axis=1)
    var_psi = fav(dpsi**2)
    k_sg = 0.5 * (cells["tau_xx"] + cells["tau_yy"] + cells["tau_zz"])
    e_th = block_mean(q["eint"], b) / rho_c
    k_max = e_th + k_sg
    cells.update(k_sg=k_sg, e_th=e_th, k_max=k_max, f=k_sg / k_max)

    # w = tau^{-1/2} cov / psi_c, with the symmetric inverse root through eigh
    n = cov.shape[0]
    tau = np.empty((n, 3, 3))
    for (i, j), p in zip([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)], PAIRS):
        tau[:, i, j] = tau[:, j, i] = cells[f"tau_{p}"].ravel()
    s, V = np.linalg.eigh(tau)
    good = s[:, 0] > S_MIN * s.sum(axis=1)
    s_inv = np.where(good[:, None], 1 / np.sqrt(np.where(good[:, None], s, 1.0)), np.nan)
    w = np.einsum("nik,nk,njk,nj->ni", V, s_inv, V, cov) / psi_c.ravel()[:, None]

    v2 = Units.from_params(sim.params).velocity_cgs**2
    out = {k: (v if k == "f" else v * v2).ravel() for k, v in cells.items()}
    out.update(w_x=w[:, 0], w_y=w[:, 1], w_z=w[:, 2],
               B2=(var_psi / psi_c**2).ravel(), psi_c=psi_c.ravel() * v2)
    return out


# ─── w split into its size and direction ─────────────────────────────────────

def flux_params(d):
    """B, |w|, c = |w|/B and the unit vector n = w/|w| of every cell."""
    w = np.column_stack([d["w_x"], d["w_y"], d["w_z"]])
    W = np.linalg.norm(w, axis=1)
    B = np.sqrt(d["B2"])
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(B=B, W=W, c=W / B, n=w / W[:, None], w=w)


# (label, range, ticks) of each quantity, and how to get it from flux_params
QUANT = {
    "logB": (r"$\log_{10} B$", (-4.5, 1), [-4, -2, 0], lambda p: np.log10(p["B"])),
    "logW": (r"$\log_{10} |\mathbf{w}|$", (-5, 1), [-4, -2, 0], lambda p: np.log10(p["W"])),
    "c": (r"$c = |\mathbf{w}|/B$", (0, 1), [0, 0.5, 1], lambda p: p["c"]),
    "nx": (r"$\hat{w}_x$", (-1, 1), [-1, 0, 1], lambda p: p["n"][:, 0]),
    "ny": (r"$\hat{w}_y$", (-1, 1), [-1, 0, 1], lambda p: p["n"][:, 1]),
    "nz": (r"$\hat{w}_z$", (-1, 1), [-1, 0, 1], lambda p: p["n"][:, 2]),
}


# w itself spans ~1e-4 to ~4 in |w_i| and changes sign, so it is shown on a
# symmetric-log axis: linear for |w_i| < W_LIN, logarithmic beyond
W_LIN, W_MAX = 1e-4, 5.0
W_TICKS = [-1, -1e-2, 0, 1e-2, 1]


def symlog(x):
    return np.sign(x) * np.log10(1 + np.abs(x) / W_LIN)


for _i, _c in enumerate("xyz"):
    QUANT["w" + _c] = (rf"$w_{_c}$", (-symlog(W_MAX), symlog(W_MAX)), symlog(np.array(W_TICKS)),
                       lambda p, i=_i: symlog(p["w"][:, i]))


def phase_cells(d, sh, p):
    """Phase masks restricted to cells where w is defined."""
    good = np.isfinite(p["c"]) & np.isfinite(sh["z"][:, 0])
    return {k: m & good for k, m in phases(d, sh).items()}


def phase_legend(fig, **kwargs):
    handles = [Line2D([], [], color=colour, lw=1.4, label=label) for _, label, colour in PHASES]
    fig.legend(handles=handles, **kwargs)


# ─── 1. Distributions ────────────────────────────────────────────────────────

def plot_distribution(d, sh, p, out):
    masks = phase_cells(d, sh, p)
    keys = ["logB", "c", "logW", "nx", "ny", "nz", "wx", "wy", "wz"]
    fig, axes = plt.subplots(3, 3, figsize=ns.figsize(15.0, ns.WIDE), layout="constrained")
    fig.get_layout_engine().set(wspace=0.08)   # keep the end ticks of neighbours apart
    for ax, k in zip(axes.flat, keys):
        label, rng, ticks, get = QUANT[k]
        x = get(p)
        edges = np.linspace(*rng, 81)
        for key, _, colour in PHASES:
            ax.hist(np.clip(x[masks[key]], *rng), bins=edges, density=True,
                    histtype="step", color=colour, lw=1)
        ax.set(xlabel=label, xlim=rng, xticks=ticks, ylim=(0, None))
        if k[0] == "w":
            ax.set_xticklabels([
                "0" if t == 0 else rf"${'-' if t < 0 else ''}10^{{{int(np.log10(abs(t)))}}}$"
                for t in W_TICKS])
        if ax in axes[:, 0]:
            ax.set_ylabel("density")
        if k[0] in "nw":
            ax.axvline(0, color=ns.HAIRLINE, lw=0.5)
    phase_legend(fig, loc="outside upper center", ncols=3)
    ns.save(fig, out)


# ─── 2. Against f ────────────────────────────────────────────────────────────

LOGF_RANGE = (-8, 0)


def plot_vs_f(d, sh, p, out):
    has = np.isfinite(p["c"])
    logf = np.clip(np.log10(d["f"][has]), *LOGF_RANGE)
    fedges = np.linspace(*LOGF_RANGE, 101)
    fig, axes = plt.subplots(1, 3, figsize=ns.figsize(6.2, ns.WIDE), sharex=True,
                             layout="constrained")
    H = []
    for ax, k in zip(axes, ["logB", "logW", "c"]):
        label, rng, ticks, get = QUANT[k]
        y = np.clip(get(p)[has], *rng)
        H.append(ax.hist2d(logf, y, bins=[fedges, np.linspace(*rng, 81)], norm=LogNorm(),
                           cmap=ns.SEQ, rasterized=True)[-1])
        ax.plot(centres(fedges), binned_median(logf, y, fedges), color=ns.OVERLAY,
                label="median")
        ax.set(xlabel=r"$\log_{10} f$", ylabel=label, ylim=rng, yticks=ticks)
    axes[0].legend(loc="upper left")
    ns.colorbar(fig, H[0], axes, "coarse cells")
    ns.save(fig, out)


# ─── 3. Against the outputs z_k of R ─────────────────────────────────────────

def plot_vs_targets(d, sh, p, out):
    """Mixing-layer cells only: the pure phases carry almost no flux (fig. w_vs_f).
    Each panel is titled with the Pearson correlation r of its two quantities."""
    from momentum_tensor import Z_LABELS, Z_RANGES
    m = phase_cells(d, sh, p)["mixing"]
    f = d["f"][m]
    z = np.column_stack([np.log(f) - np.log1p(-f), sh["z"][m]])
    z_ranges = [(-10, 0)] + Z_RANGES[1:]
    rows = ["logB", "c", "nx", "ny", "nz"]

    fig, axes = plt.subplots(len(rows), 6, figsize=ns.figsize(13.0, ns.WIDE),
                             sharex="col", sharey="row", layout="constrained")
    fig.get_layout_engine().set(hspace=0.0, wspace=0.0, h_pad=0.02, w_pad=0.02)
    norm = LogNorm(1, 3e3)
    for i, k in enumerate(rows):
        label, rng, ticks, get = QUANT[k]
        if k == "logB":   # the mixing layer only spans the top of the range
            rng, ticks = (-1.5, 1), [-1, 0, 1]
        y = np.clip(get(p)[m], *rng)
        for j in range(6):
            ax = axes[i, j]
            xedges = np.linspace(*z_ranges[j], 61)
            *_, im = ax.hist2d(z[:, j], y, bins=[xedges, np.linspace(*rng, 41)], norm=norm,
                               cmap=ns.SEQ, rasterized=True)
            ax.plot(centres(xedges), binned_median(z[:, j], y, xedges), color=ns.OVERLAY, lw=1)
            r = np.corrcoef(z[:, j], y)[0, 1]
            ax.set_title(rf"$r = {r:+.2f}$", fontsize=ns.SMALL - 1, color=ns.MUTED, pad=2)
            ax.tick_params(labelsize=ns.SMALL, length=2)
            ax.set(xlim=z_ranges[j], ylim=rng, yticks=ticks,
                   xticks=[-10, -5, 0] if j == 0 else [-4, 0, 4])
            if i == len(rows) - 1:
                ax.set_xlabel(Z_LABELS[j])
        axes[i, 0].set_ylabel(label)
    ns.colorbar(fig, im, axes, "mixing-layer cells", shrink=0.6)
    ns.save(fig, out)


# ─── 4. Scaling of |w| with f ────────────────────────────────────────────────

ALPHAS = np.linspace(0, 1, 201)


def plot_w_scaling(d, sh, p, out):
    """Left: |w| against f, with the least-squares line. Right: the spread of
    log|w| - alpha log f against alpha, smallest near alpha = 1/2 (sec:w-model)."""
    has = np.isfinite(p["c"])
    logf, logw = np.log10(d["f"][has]), np.log10(p["W"][has])
    slope, icpt = np.polyfit(logf, logw, 1)
    spread = np.array([np.std(logw - a * logf) for a in ALPHAS])
    best = ALPHAS[np.argmin(spread)]
    print(f"|w| ~ f^alpha: least-squares slope {slope:.3f}, smallest spread at {best:.3f}")

    fig, (ax, bx) = plt.subplots(1, 2, figsize=ns.figsize(5.6, ns.WIDE), layout="constrained")
    fedges = np.linspace(*LOGF_RANGE, 101)
    _, rng, ticks, _ = QUANT["logW"]
    *_, im = ax.hist2d(np.clip(logf, *LOGF_RANGE), np.clip(logw, *rng),
                       bins=[fedges, np.linspace(*rng, 81)], norm=LogNorm(), cmap=ns.SEQ,
                       rasterized=True)
    ax.plot(LOGF_RANGE, icpt + slope * np.array(LOGF_RANGE), color=ns.OVERLAY,
            label=rf"slope ${slope:.2f}$")
    ax.set(xlabel=r"$\log_{10} f$", ylabel=QUANT["logW"][0], xlim=LOGF_RANGE, ylim=rng,
           yticks=ticks)
    ax.legend(loc="upper left")
    ns.colorbar(fig, im, ax, "coarse cells")

    bx.plot(ALPHAS, spread, color=ns.OVERLAY)
    bx.axvline(best, color=ns.HAIRLINE, lw=0.5)
    bx.annotate(rf"$\alpha = {best:.2f}$", (best, spread.min()), xytext=(4, -14),
                textcoords="offset points", fontsize=ns.SMALL, color=ns.MUTED)
    bx.set(xlabel=r"$\alpha$", ylabel="spread (dex)",
           xlim=(0, 1), ylim=(0, None))
    ns.save(fig, out)


# ─── 5. The outputs z_6, z_7, z_8 of w ───────────────────────────────────────

Z_W_RANGE = (-5, 5)


def w_targets(d, p):
    """z_6, z_7, z_8 of every cell (sec:w-model): w mapped radially,
    z = asinh(|w|/sqrt f) w-hat, the inverse of w = sqrt f sinh|z|/|z| z."""
    return p["n"] * np.arcsinh(p["W"] / np.sqrt(d["f"]))[:, None]


def plot_w_targets(d, sh, p, out):
    masks = phase_cells(d, sh, p)
    z = w_targets(d, p)
    edges = np.linspace(*Z_W_RANGE, 121)
    fig, axes = plt.subplots(1, 3, figsize=ns.figsize(5.2, ns.WIDE), layout="constrained")
    for i, ax in enumerate(axes):
        for key, _, colour in PHASES:
            ax.hist(np.clip(z[masks[key], i], *Z_W_RANGE), bins=edges, density=True,
                    histtype="step", color=colour, lw=1)
        ax.axvline(0, color=ns.HAIRLINE, lw=0.5)
        ax.set(xlabel=rf"$z_{6 + i}$", xlim=Z_W_RANGE, xticks=[-4, -2, 0, 2, 4], ylim=(0, None))
    axes[0].set_ylabel("density")
    phase_legend(fig, loc="outside upper center", ncols=3)
    ns.save(fig, out)


def plot_all(d):
    sh = shape(d)
    p = flux_params(d)
    plot_distribution(d, sh, p, ns.image_path(2, "w_distribution.pdf"))
    plot_w_scaling(d, sh, p, ns.image_path(2, "w_scaling.pdf"))
    plot_w_targets(d, sh, p, ns.image_path(2, "w_targets.pdf"))
    plot_vs_f(d, sh, p, ns.image_path(2, "w_vs_f.pdf"))
    plot_vs_targets(d, sh, p, ns.image_path(2, "w_vs_targets.pdf"))
    return sh, p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--b", type=int, default=16, help="coarse block size")
    parser.add_argument("--every", type=int, default=1, help="use every n-th frame")
    parser.add_argument("--recompute", action="store_true", help="recompute cached frames")
    args = parser.parse_args()

    d, sim = cell_cache.load("energyflux_cells", args.b, args.every, FIELDS, coarse_cells,
                             CACHE_VERSION, args.recompute)
    units = Units.from_params(sim.params)
    gamma = float(sim.params["hydro"]["gamma"])
    d["t_to_eth"] = units.velocity_cgs**2 / ((gamma - 1) * units.temperature_cgs)
    d = ns.drop_transient(d)
    print(f"{d['f'].size} cells of {int(d['b'])}^3, {len(d['frames'])} frames, "
          f"t = {d['time'].min():.1f}-{d['time'].max():.1f} Myr")
    plot_all(d)
