"""
Gating R and Q on f (sec:flux-gate): set R = 0 and Q = 0 in every coarse cell with
f < f_cut, and compare the change of their divergences with the resolved flux
divergences of the same equations of eq-final,

  momentum:  F_m = div(<m><m>/<rho>) + grad <P>      against  div R
  energy:    F_E = div[(<E> + <P>) <m>/<rho>]         against  div Q,  Q = v~ . R + q

  eps_R(f_cut) = sum |div(g R) - div R| / sum |F_m|
  eps_Q(f_cut) = sum |div(g Q) - div Q| / sum |F_E|

over all cells and over the mixing-layer cells, with R_ij = <rho> tau_ij and
q = <rho> psi_c tau^{1/2} w (eq:bound_on_w). Everything is in the frame of the run.
The divergences are central differences on the coarse grid (periodic in x and z,
one-sided at the y boundaries), taken after the cut, so the jumps the cut makes at the
edge of the gated region are counted.

Figures and tables (Overleaf/images/chapter2/):
  flux_gate.pdf        eps_R and eps_Q against f_cut, and the share of each phase gated off
  flux_gate_cut.tex    the same at a few f_cut

tau, w, psi_c and f come from the cache of energy_flux.py. The resolved <rho>, <m>, <E>
and <P> are cached here (cell_cache.py), one file per frame, in code units.
Frames before ns.T_MIN = 2.5 Myr, the initial transient, are left out.

    python theory/notes/plotting_scripts/flux_gate.py [--b 16] [--every 1] [--recompute]
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from vis_athenak.utils.units import Units

import cell_cache
import energy_flux as ef
import notes_style as ns
from data.coarse_grain_utils import block_mean  # noqa: E402  (path set by momentum_tensor)
from momentum_tensor import (FIELDS, PAIRS, PHASES, T_COLD, T_HOT, binned_median,
                             centres)

CACHE_VERSION = 1  # bump when the stored per-cell quantities change
F_TABLE = (1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2)
F_CHOSEN = 1e-4                         # the cut proposed in sec:flux-gate, in bold
F_CURVE = np.logspace(-8, 0, 49)
HARM = 0.01                             # a cell is harmed if its divergence changes by more
                                        # than this fraction of its resolved term
IJ = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]   # the order of PAIRS
AXIS = {0: 2, 1: 1, 2: 0}               # component x, y, z -> axis of a (Z, Y, X) array


# ─── Per-cell resolved quantities ────────────────────────────────────────────

def coarse_cells(q, sim, b):
    """<rho>, <m>, <E> and <P> of each coarse cell (code units)."""
    gamma = float(sim.params["hydro"]["gamma"])
    rho = q["dens"]
    cells = {"rho": block_mean(rho, b), "P": (gamma - 1) * block_mean(q["eint"], b)}
    for c, v in zip("xyz", ("velx", "vely", "velz")):
        cells[f"m_{c}"] = block_mean(rho * q[v], b)
    v2 = q["velx"]**2 + q["vely"]**2 + q["velz"]**2
    cells["E"] = block_mean(q["eint"] + 0.5 * rho * v2, b)
    return {k: v.ravel() for k, v in cells.items()}


# ─── Fields on the coarse grid ───────────────────────────────────────────────

def grid_shape(b):
    """(Z, Y, X) coarse cells of the 256x1024x256 run."""
    return (256 // b, 1024 // b, 256 // b)


def fields(e, r, sh, v2):
    """R (3, 3, Z, Y, X), Q (3, ...) and the resolved fluxes of one frame, in code units;
    v2 converts the cgs specific energies of the energy_flux cache to code units."""
    rho = r["rho"]
    tau = np.empty((len(rho), 3, 3))
    for (i, j), p in zip(IJ, PAIRS):
        tau[:, i, j] = tau[:, j, i] = e[f"tau_{p}"] / v2
    s, V = np.linalg.eigh(tau)
    w = np.nan_to_num(np.column_stack([e["w_x"], e["w_y"], e["w_z"]]))
    u = np.einsum("nik,nk,njk,nj->ni", V, np.sqrt(np.clip(s, 0, None)), V, w)
    m = np.column_stack([r["m_x"], r["m_y"], r["m_z"]])
    v = m / rho[:, None]
    R = rho[:, None, None] * tau
    Q = np.einsum("nj,nij->ni", v, R) + (rho * e["psi_c"] / v2)[:, None] * u
    grid = lambda a: np.moveaxis(a, 0, -1).reshape(a.shape[1:] + sh)
    return dict(R=grid(R), Q=grid(Q), mm=grid(m[:, :, None] * v[:, None, :]),
                P=r["P"].reshape(sh), HE=grid((r["E"] + r["P"])[:, None] * v))


def ddx(F, k):
    """Central difference along x, y or z (k = 0, 1, 2), in units of the coarse spacing."""
    a = AXIS[k]
    if k == 1:                          # y is not periodic
        return np.gradient(F, axis=a)
    return 0.5 * (np.roll(F, -1, a) - np.roll(F, 1, a))


def div_tensor(R):
    """(div R)_i = d_j R_ij."""
    return np.stack([sum(ddx(R[i, j], j) for j in range(3)) for i in range(3)])


def div_vector(q):
    return sum(ddx(q[j], j) for j in range(3))


def norm(a):
    return np.linalg.norm(a, axis=0)


def phase_masks(T):
    return {"hot": T >= T_HOT, "mixing": (T >= T_COLD) & (T < T_HOT), "cold": T < T_COLD}


# ─── Errors against f_cut ────────────────────────────────────────────────────

def gate_errors(e_all, r_all, sh, v2, t_to_eth, cuts):
    """Change of div R and div Q under the gate, summed over all frames, as a fraction
    of the summed resolved flux divergences; and the ungated ratios by phase."""
    n = len(e_all["f"])
    keys = ("R", "Q", "R_mix", "Q_mix")
    acc = {k: np.zeros(len(cuts)) for k in keys}
    res = dict.fromkeys(keys, 0.0)
    sub = {k: {key: 0.0 for key, _, _ in PHASES} for k in ("R", "Q")}
    res_ph = {k: {key: 0.0 for key, _, _ in PHASES} for k in ("R", "Q")}
    # |div R|/|F_m| and |div Q|/|F_E| of every cell, and their change under the gate at
    # cuts[0], as a fraction of the same resolved term
    cell = {"R": [], "Q": [], "dR": [], "dQ": [], "mix": []}
    # cells whose divergence changes by more than HARM of the resolved term, per cut
    harm = {k: np.zeros(len(cuts)) for k in keys}
    n_cells = {"all": 0, "mix": 0}
    for fr in np.unique(e_all["frame"]):
        sel = slice(*np.searchsorted(e_all["frame"], [fr, fr + 1]))
        e = {k: v[sel] for k, v in e_all.items() if np.ndim(v) and len(v) == n}
        r = {k: v[sel] for k, v in r_all.items() if np.ndim(v) and len(v) == n}
        F = fields(e, r, sh, v2)
        f = e["f"].reshape(sh)
        ph = phase_masks((e["e_th"] / t_to_eth).reshape(sh))
        dR, dQ = div_tensor(F["R"]), div_vector(F["Q"])
        Fm = norm(div_tensor(F["mm"]) + np.stack([ddx(F["P"], k) for k in range(3)]))
        FE = np.abs(div_vector(F["HE"]))
        cell["R"].append((norm(dR) / Fm).ravel())
        cell["Q"].append((np.abs(dQ) / FE).ravel())
        for key, m in ph.items():
            sub["R"][key] += norm(dR)[m].sum(); res_ph["R"][key] += Fm[m].sum()
            sub["Q"][key] += np.abs(dQ)[m].sum(); res_ph["Q"][key] += FE[m].sum()
        mix = ph["mixing"]
        res["R"] += Fm.sum(); res["Q"] += FE.sum()
        res["R_mix"] += Fm[mix].sum(); res["Q_mix"] += FE[mix].sum()
        for i, fc in enumerate(cuts):
            g = f >= fc
            eR = norm(div_tensor(F["R"] * g) - dR)
            eQ = np.abs(div_vector(F["Q"] * g) - dQ)
            if i == 0:                  # cuts[0] is F_CHOSEN: keep the change of every cell
                cell["dR"].append((eR / Fm).ravel())
                cell["dQ"].append((eQ / FE).ravel())
                cell["mix"].append(mix.ravel())
            acc["R"][i] += eR.sum(); acc["Q"][i] += eQ.sum()
            acc["R_mix"][i] += eR[mix].sum(); acc["Q_mix"][i] += eQ[mix].sum()
            hR, hQ = eR > HARM * Fm, eQ > HARM * FE
            harm["R"][i] += hR.sum(); harm["Q"][i] += hQ.sum()
            harm["R_mix"][i] += hR[mix].sum(); harm["Q_mix"][i] += hQ[mix].sum()
        n_cells["all"] += f.size; n_cells["mix"] += mix.sum()
    err = {k: acc[k] / res[k] for k in keys}
    err.update({"harm_" + k: harm[k] / n_cells["mix" if k.endswith("mix") else "all"]
                for k in keys})
    ratio = {k: {key: sub[k][key] / res_ph[k][key] for key in sub[k]} for k in sub}
    ratio["R"]["all"] = sum(sub["R"].values()) / res["R"]
    ratio["Q"]["all"] = sum(sub["Q"].values()) / res["Q"]
    return err, ratio, {k: np.concatenate(v) for k, v in cell.items()}


def gated_share(f, T, cuts):
    """Share of each phase's cells with f < f_cut."""
    out = {key: np.array([(f[m] < fc).mean() for fc in cuts])
           for key, m in phase_masks(T).items()}
    out["all"] = np.array([(f < fc).mean() for fc in cuts])
    return out


# ─── Outputs ─────────────────────────────────────────────────────────────────

LOGF_RANGE = (-8, 0)
LOGR_RANGE = (-6, 2)


def binned_percentile(x, y, edges, q, min_count=20):
    """q-th percentile of y in each x bin holding at least min_count values (else nan)."""
    idx = np.digitize(x, edges) - 1
    out = np.full(edges.size - 1, np.nan)
    for k in range(edges.size - 1):
        g = y[idx == k]
        if g.size >= min_count:
            out[k] = np.percentile(g, q)
    return out


def plot_gate(f, cell, gated, err, out):
    """A 2x2 grid on one axis, log10 f.
    Top: |div R|/|F_m| and |div Q|/|F_E| of every cell against its f, with the median and
    90th percentile in each f bin. Bottom: share of each phase gated off at f_cut, and
    share of cells whose divergence the gate changes by more than HARM of the resolved term."""
    fig, axes = plt.subplots(2, 2, figsize=ns.figsize(10.0, ns.WIDE), sharex=True,
                             layout="constrained")
    fig.get_layout_engine().set(hspace=0.06, wspace=0.06)
    logf = np.clip(np.log10(f), *LOGF_RANGE)
    fedges = np.linspace(*LOGF_RANGE, 81)
    redges = np.linspace(*LOGR_RANGE, 81)
    x_cut = np.log10(F_CHOSEN)
    norm_c = LogNorm(1, 3e4)
    tops = [(axes[0, 0], "R", r"$\log_{10}\,|\nabla\!\cdot\!\Rvv| \,/\, |\mathbf{F}_\mm|$"),
            (axes[0, 1], "Q", r"$\log_{10}\,|\nabla\!\cdot\!\Qs| \,/\, |F_E|$")]
    for ax, key, label in tops:
        with np.errstate(divide="ignore"):
            y = np.log10(cell[key])
        ok = np.isfinite(y)
        x, y = logf[ok], np.clip(y[ok], *LOGR_RANGE)
        *_, im = ax.hist2d(x, y, bins=[fedges, redges], norm=norm_c, cmap=ns.SEQ,
                           rasterized=True)
        c = centres(fedges)
        ax.plot(c, binned_median(x, y, fedges), color=ns.OVERLAY, lw=1.2, label="median")
        ax.plot(c, binned_percentile(x, y, fedges, 90), color=ns.OVERLAY, lw=1.0, ls="--",
                label="90th percentile")
        ax.axhline(np.log10(HARM), color=ns.MUTED, lw=0.5, ls=":")
        ax.text(LOGF_RANGE[0] + 0.15, np.log10(HARM) + 0.12, r"$1\%$", ha="left",
                va="bottom", fontsize=ns.SMALL, color=ns.MUTED)
        ax.set(ylabel=label, ylim=LOGR_RANGE, yticks=[-6, -4, -2, 0, 2])
    axes[0, 0].legend(loc="upper left", fontsize=7)
    ns.colorbar(fig, im, axes[0, :], "coarse cells")

    # bottom left: the gate as a function of f_cut, direct-labelled by phase
    b = axes[1, 0]
    xc = np.log10(F_CURVE)
    label_at = {"hot": -6.6, "cold": -3.4, "mixing": -1.6}
    for key, label, colour in PHASES:
        b.plot(xc, 100 * gated[key], color=colour, lw=1.4)
        xl = label_at[key]
        yl = 100 * np.interp(xl, xc, gated[key])
        b.annotate(label, (xl, yl), xytext=(5, 0), textcoords="offset points", ha="left",
                   va="center", fontsize=ns.SMALL, color=ns.INK)
    b.plot(xc, 100 * gated["all"], color=ns.INK, lw=1.0, ls=":")
    b.annotate("all cells", (-5.3, 100 * np.interp(-5.3, xc, gated["all"])), xytext=(0, 6),
               textcoords="offset points", ha="center", fontsize=ns.SMALL, color=ns.INK)
    b.set(ylabel=r"cells gated off [\%]", ylim=(0, 102), yticks=[0, 25, 50, 75, 100])

    # bottom right: what the gate costs, cell by cell
    h = axes[1, 1]
    for key, label, colour in [("R", r"$\nabla\!\cdot\!\Rvv$", ns.SERIES[0]),
                               ("Q", r"$\nabla\!\cdot\!\Qs$", ns.SERIES[1])]:
        h.plot(xc, 100 * err["harm_" + key], color=colour, lw=1.4)
        h.plot(xc, 100 * err["harm_" + key + "_mix"], color=colour, lw=1.0, ls="--")
    h.set(yscale="log", ylabel=r"cells changed by $> 1\%$ [\%]", ylim=(1e-2, 2e3),
          yticks=[1e-2, 1e-1, 1, 10, 100])
    handles = [Line2D([], [], color=ns.SERIES[0], lw=1.4, label=r"$\nabla\!\cdot\!\Rvv$"),
               Line2D([], [], color=ns.MUTED, lw=1.2, label="all cells"),
               Line2D([], [], color=ns.SERIES[1], lw=1.4, label=r"$\nabla\!\cdot\!\Qs$"),
               Line2D([], [], color=ns.MUTED, lw=1.0, ls="--", label="mixing layer")]
    h.legend(handles=handles, loc="upper left", ncols=2, fontsize=7, handlelength=1.6,
             columnspacing=1.0)

    for ax in axes.flat:
        ax.axvline(x_cut, color=ns.HAIRLINE, lw=0.6)
        ax.set(xlim=LOGF_RANGE, xticks=[-8, -6, -4, -2, 0])
    for ax in axes[0]:
        ax.tick_params(labelbottom=True)
        ax.set_xlabel(r"$\log_{10} f$")
    for ax in axes[1]:
        ax.set_xlabel(r"$\log_{10} f_\mathrm{cut}$")
    for ax, tag in zip(axes.flat, "abcd"):
        ax.text(0.98, 0.04, f"({tag})", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=ns.SMALL, color=ns.MUTED)
    ns.save(fig, out)


def pct(x):
    """A percentage: three decimals below 1 %, two significant figures above."""
    x = 100 * x
    if x < 1e-3:
        return r"{<}\,0.001"
    return f"{x:.3f}" if x < 1 else f"{x:.2g}" if x < 10 else f"{x:.0f}"


def gate_table(err, gated, out):
    lines = [r"\begin{widetabular}{rrrrrr}", r"\toprule",
             r"$f_\mathrm{cut}$ & cells gated off & \multicolumn{2}{c}{change of $\nabla\cdot\Rvv$}"
             r" & \multicolumn{2}{c}{change of $\nabla\cdot\Qs$} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
             r" & & all cells & mixing layer & all cells & mixing layer \\", r"\midrule"]
    for i, fc in enumerate(F_TABLE):
        p = int(np.floor(np.log10(fc)))
        m = fc / 10**p
        fstr = rf"10^{{{p}}}" if np.isclose(m, 1) else rf"{m:.0f}\times10^{{{p}}}"
        cells = [fstr, rf"{100 * gated['all'][i]:.0f}\,\%"] + [
            rf"{pct(err[k][i])}\,\%" for k in ("R", "R_mix", "Q", "Q_mix")]
        bold = np.isclose(fc, F_CHOSEN)
        lines.append(" & ".join(rf"$\mathbf{{{c}}}$" if bold else f"${c}$" for c in cells)
                     + r" \\")
    lines += [r"\bottomrule", r"\end{widetabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--b", type=int, default=16, help="coarse block size")
    parser.add_argument("--every", type=int, default=1, help="use every n-th frame")
    parser.add_argument("--recompute", action="store_true", help="recompute cached frames")
    args = parser.parse_args()

    e, sim = cell_cache.load("energyflux_cells", args.b, args.every, FIELDS, ef.coarse_cells,
                             ef.CACHE_VERSION)
    r, _ = cell_cache.load("resolved_cells", args.b, args.every, FIELDS, coarse_cells,
                           CACHE_VERSION, args.recompute)
    e, r = ns.drop_transient(e), ns.drop_transient(r)
    assert np.array_equal(e["frame"], r["frame"])
    units = Units.from_params(sim.params)
    gamma = float(sim.params["hydro"]["gamma"])
    v2 = units.velocity_cgs**2
    t_to_eth = v2 / ((gamma - 1) * units.temperature_cgs)
    e_th = r["P"] / ((gamma - 1) * r["rho"]) * v2
    print(f"max relative mismatch of e_th between the caches: "
          f"{np.max(np.abs(e_th - e['e_th']) / e['e_th']):.1e}")
    sh = grid_shape(args.b)

    n = len(F_TABLE)
    err, ratio, cell = gate_errors(e, r, sh, v2, t_to_eth,
                                   np.concatenate([[F_CHOSEN], F_TABLE, F_CURVE]))
    err_t = {k: v[1:1 + n] for k, v in err.items()}
    err_c = {k: v[1 + n:] for k, v in err.items()}
    for key, label in (("dR", "div R"), ("dQ", "div Q")):
        x, mix = cell[key], cell["mix"]
        touched = x > 0
        print(f"gate at {F_CHOSEN:.0e}, per cell change of {label} / resolved: changed in "
              f"{100 * touched.mean():.0f}% of cells; of those median {np.median(x[touched]):.1e}, "
              f"90th {np.percentile(x[touched], 90):.1e}, 99th {np.percentile(x[touched], 99):.1e};"
              f" > 1% in {100 * (x > 0.01).mean():.1f}% of all cells, {100 * (x[mix] > 0.01).mean():.2f}%"
              f" of mixing cells; > 10% in {100 * (x > 0.1).mean():.2f}% of all cells")
        T_all = e["e_th"] / t_to_eth
        big = x > 0.01
        print("   cells > 1% by phase: " + ", ".join(
            f"{k} {100 * (big & m).sum() / big.sum():.0f}%" for k, m in phase_masks(T_all).items())
            + f"; of those, gated off themselves {100 * (big & (e['f'] < F_CHOSEN)).sum() / big.sum():.0f}%")
    for k, label in (("R", "sum|div R| / sum|F_m|"), ("Q", "sum|div Q| / sum|F_E|")):
        print(label + ": " + ", ".join(f"{p} {100 * x:.2f}%" for p, x in ratio[k].items()))
    T = e["e_th"] / t_to_eth
    gated_t = gated_share(e["f"], T, F_TABLE)
    for i, fc in enumerate(F_TABLE):
        print(f"f_cut {fc:.0e}: div R {100 * err_t['R'][i]:.4f}% (mixing "
              f"{100 * err_t['R_mix'][i]:.4f}%), div Q {100 * err_t['Q'][i]:.4f}% (mixing "
              f"{100 * err_t['Q_mix'][i]:.4f}%); gated off " +
              ", ".join(f"{k} {100 * v[i]:.0f}%" for k, v in gated_t.items()))
    for i, fc in enumerate(F_TABLE):
        print(f"f_cut {fc:.0e}: cells changed by > {100 * HARM:.0f}%: div R {100 * err_t['harm_R'][i]:.3f}%"
              f" (mixing {100 * err_t['harm_R_mix'][i]:.3f}%), div Q {100 * err_t['harm_Q'][i]:.3f}%"
              f" (mixing {100 * err_t['harm_Q_mix'][i]:.3f}%)")
    print(f"f_cut = 1: div R {100 * err_c['harm_R'][-1]:.1f}%, div Q {100 * err_c['harm_Q'][-1]:.1f}%")
    gate_table(err_t, gated_t, ns.image_path(2, "flux_gate_cut.tex"))
    for key, label in (("R", "|div R|/|F_m|"), ("Q", "|div Q|/|F_E|")):
        for lo, hi in ((0, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1)):
            m = (e["f"] >= lo) & (e["f"] < hi)
            x = cell[key][m]
            print(f"{label} for {lo:.0e} <= f < {hi:.0e}: median {np.median(x):.1e}, "
                  f"90th pct {np.percentile(x, 90):.1e}, > 1% in {100 * (x > 0.01).mean():.0f}% "
                  f"of {m.sum()} cells")
    plot_gate(e["f"], cell, gated_share(e["f"], T, F_CURVE), err_c, ns.image_path(2, "flux_gate.pdf"))
