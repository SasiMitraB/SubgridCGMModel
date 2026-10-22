"""
Figs. 1-4 of Sharma et al. (2025) recreated from the 256x1024x256 radiative KH
run, in the style of the notes (notes_style.py).

The numbers are those of theory/sharma+2025/recreating_plots.py: this script
reuses its per-snapshot reduction (cached in theory/sharma+2025/cache), TRML
frame, flux decompositions and models, and only redraws the figures. Run
recreating_plots.py first if the run is missing from the cache.

Figures (Overleaf/images/chapter2/):
  trml_slices.pdf     Fig. 1  x-z slices of rho, p, t_cool, |du| at t ~ 350 t0
  trml_profiles.pdf   Fig. 2  mean profiles and fluxes in the TRML frame
  trml_pdfs.pdf       Fig. 3  volume, mass and emissivity PDFs of log10 T
  trml_energy.pdf     Fig. 4  local energy balance

    python theory/notes/plotting_scripts/trml_sharma_figs.py [--run kh3d_256x1024x256]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "theory" / "sharma+2025"))
import recreating_plots as rp  # noqa: E402
from vis_athenak.simulation_data import SimulationData  # noqa: E402

import notes_style as ns  # noqa: E402

TMIN = 146      # t0, the paper's averaging window
TSNAP = 350     # t0, the snapshot of Fig. 1
ZLIM = (-21, 11)
BLUE, RED, GREEN, AMBER = ns.SERIES
GREY = "#8A857C"   # this-snapshot profiles and secondary lines


def band(ax, x, mean, std, color, label=None, **kw):
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2, lw=0)
    ax.plot(x, mean, color=color, label=label, **kw)


# ─── Fig. 1: slices ──────────────────────────────────────────────────────────

def fig1(sim, su, P, out):
    tau_all = np.array(sim.times) / su.t0
    frame = sim[int(np.argmin(abs(tau_all - TSNAP)))]
    q = rp.primitives(frame, su)
    k3 = q["rho"].shape[0] // 2
    x, y = frame["x1v"], frame["x2v"]
    zc = P.fit_line[0] + P.fit_line[1] * frame.time
    X, Z = x / su.L, (y - zc) / su.L
    rows = (Z >= ZLIM[0]) & (Z <= ZLIM[1])
    Z = Z[rows]
    sl = {k: a[k3][rows] for k, a in q.items()}
    hm = {k: a.mean(axis=(0, 2))[rows] for k, a in q.items()}
    dumag = np.sqrt(sum((sl[c] - hm[c][:, None])**2 for c in ("u", "w", "s"))) / su.du
    with np.errstate(divide="ignore", invalid="ignore"):
        tcool = sl["p"] / ((su.gamma - 1) * sl["cool"]) / su.t0
        tc_mean = P.mean["p"] / ((su.gamma - 1) * P.mean["cool"]) / su.t0
        tc_snap = hm["p"] / ((su.gamma - 1) * hm["cool"]) / su.t0
    tcool[~np.isfinite(tcool)] = np.nan

    zP, m = P.z / su.L, P.mean
    rms_t = {c: rp.rms(m, c) / su.du for c in ("u", "s", "w")}
    rms_3d = np.sqrt(sum(r**2 for r in rms_t.values()))
    chi = su.rho_c / su.rho_h

    fig, axs = plt.subplots(1, 4, figsize=ns.figsize(10.5, ns.WIDE), sharey=True,
                            layout="constrained")
    ext = [X[0], X[-1], Z[0], Z[-1]]
    specs = [
        (sl["rho"] / su.rho_h, Normalize(0, 1.1 * chi), r"$\rho/\rho_h$",
         [(m["rho"] / su.rho_h, zP, ns.OVERLAY), (hm["rho"] / su.rho_h, Z, GREY)]),
        (sl["p"] / su.p0, Normalize(0.5, 1.5), r"$p/p_0$",
         [(m["p"] / su.p0, zP, ns.OVERLAY), (hm["p"] / su.p0, Z, GREY)]),
        (tcool, LogNorm(1, 10**2.5), r"$t_{\rm cool}/t_0$",
         [(tc_mean, zP, ns.OVERLAY), (tc_snap, Z, GREY)]),
        (dumag, LogNorm(1e-2, 1.5), r"$|\delta\mathbf{u}|/\Delta u$", []),
    ]
    ovs = []
    for ax, (img, norm, label, overlays) in zip(axs, specs):
        im = ax.imshow(img, origin="lower", extent=ext, aspect="auto",
                       interpolation="nearest", cmap="inferno", norm=norm, rasterized=True)
        cb = ns.colorbar(fig, im, ax, label, location="top", pad=0.01)
        cb.ax.tick_params(labelsize=8)
        ax.set_xlabel(r"$x/\Delta u\,t_0$")
        # Overlaid profiles use the colour bar as their horizontal axis.
        ov = ax.twiny()
        ov.set_xscale("log" if isinstance(norm, LogNorm) else "linear")
        ov.set_xlim(norm.vmin, norm.vmax)
        for prof, zz, col in overlays:
            with np.errstate(divide="ignore", invalid="ignore"):
                ov.plot(prof, zz, color=col, lw=1.2)
        ov.set_xticks([])
        ov.minorticks_off()
        ov.spines[:].set_visible(False)
        ov.set_ylim(Z[0], Z[-1])
        ovs.append(ov)
    ov = ovs[3]
    for c, lab, col in (("u", "$x$", ns.OVERLAY), ("s", "$y$", "#E0E0E0"),
                        ("w", "$z$", AMBER)):
        ov.plot(rms_t[c], zP, ls=(0, (3, 1.5)), color=col, lw=1.0, label=lab)
    ov.plot(rms_3d, zP, color="white", lw=1.2, label="3D")
    leg = ov.legend(loc="lower right", fontsize=8, labelcolor="white",
                    handlelength=1.4, borderaxespad=0.3)
    leg.get_frame().set_alpha(0)

    axs[0].streamplot(X, Z, sl["u"], sl["w"] - P.v, color="#C9C3B6", linewidth=0.4,
                      density=1.0, arrowsize=0.5)
    axs[0].set_xlim(X[0], X[-1])
    axs[0].set_ylim(Z[0], Z[-1])
    axs[0].set_ylabel(r"$z/\Delta u\,t_0$")
    frame.unload()
    ns.save(fig, ns.image_path(2, "trml_slices.pdf"))
    plt.close(fig)
    return frame.time / su.t0


# ─── Fig. 2: mean profiles and fluxes ────────────────────────────────────────

def fig2(su, P, out):
    z = P.z / su.L
    m = P.mean
    fig, axs = plt.subplots(4, 2, figsize=ns.figsize(19, ns.WIDE), sharex=True,
                            layout="constrained")
    wins = P.windows()
    wcol = [BLUE, RED, GREEN]

    def windowed(ax, fn, norm, label):
        for (idx, lo, hi), col in zip(wins, wcol):
            ax.plot(z, P.stat(fn, idx)[0] / norm, color=col, lw=0.9,
                    label=rf"${lo:.0f}$--${hi:.0f}\,t_0$")
        mean, std = P.stat(fn)
        band(ax, z, mean / norm, std / norm, ns.INK, lw=1.4,
             label=rf"${P.tau[0]:.0f}$--${P.tau[-1]:.0f}\,t_0$")
        ax.set_ylabel(label)

    windowed(axs[0, 0], lambda D: D["rho"], su.rho_h, r"$\avg{\rho}/\rho_h$")
    ax = axs[1, 0]
    windowed(ax, lambda D: D["u"], su.du, r"$\avg{u_x}/\Delta u$")
    extra = ax.plot(z, P.stat(lambda D: D["rhou"] / D["rho"])[0] / su.du, color=GREY,
                    ls=(0, (4, 2)), lw=1.0, label="mass-weighted")
    ax.legend(handles=extra, loc="upper left", fontsize=8)
    windowed(axs[2, 0], lambda D: D["w"], su.du, r"$\avg{u_z}/\Delta u$")
    ax = axs[3, 0]
    windowed(ax, lambda D: D["T"], su.Th_K, r"$\avg{T}/T_h$")
    extra = ax.plot(z, P.tanh_T(P.z) / su.Th_K, color=AMBER, ls=(0, (1, 1.5)), lw=1.4,
                    label=rf"$\tanh$ fit, $z_0 = {P.z0 / su.L:.2f}\,\Delta u\,t_0$")
    extra += ax.plot(z, m["B"] / (2.5 * su.Th_K / su.T_unit), color=GREY, ls=(0, (4, 2)),
                     lw=1.0, label=r"$\avg{\mathcal B}\,\mu m_p/(2.5\,k_BT_h)$")
    ax.legend(handles=extra, loc="upper left", fontsize=8)

    def fluxes(ax, items, norm, label, loc="lower left"):
        for fn, col, lab in items:
            mean, std = P.stat(fn)
            band(ax, z, mean / norm, std / norm, col, label=lab, lw=1.2)
        ax.axhline(0, color=ns.HAIRLINE, lw=0.5)
        ax.set_ylabel(label)
        for line in ax.legend(loc=loc, fontsize=8).get_lines():
            line.set_linewidth(1.4)

    fluxes(axs[0, 1], [(rp.mf_mean, RED, r"$\avg{\rho}\avg{u_z}$"),
                       (rp.mf_turb, BLUE, r"$\avg{\delta\rho\,\delta u_z}$"),
                       (rp.mf_tot, ns.INK, r"$\avg{\rho u_z}$")],
           su.rho_h * su.du, r"mass flux $/\rho_h\Delta u$", loc="upper left")
    fluxes(axs[1, 1], [(rp.xm_turb, BLUE, r"$\mathcal R_{zx}$"),
                       (rp.xm_adv, RED, r"$\avg{\rho u_z}\avg{u_x}$"),
                       (rp.xm_tot, ns.INK, r"$\avg{\rho u_xu_z}$")],
           su.rho_h * su.du**2, r"$x$-momentum flux $/\rho_h\Delta u^2$")
    fluxes(axs[2, 1], [(rp.zm_p, BLUE, r"$\avg{p}$"),
                       (rp.zm_adv, GREEN, r"$\avg{\rho u_z}\avg{u_z}$"),
                       (rp.zm_turb, RED, r"$\mathcal R_{zz}$"),
                       (rp.zm_tot, ns.INK, r"total")],
           su.p0, r"$z$-momentum flux $/p_0$", loc="center left")
    fluxes(axs[3, 1], [(P.cumcool, BLUE, r"$\int\avg{n^2\Lambda}\,\dd z$"),
                       (rp.en_enth, GREEN, r"$\avg{\mathcal B}\avg{\rho u_z}$"),
                       (rp.en_turb, RED, r"$\mathcal Q_t$"),
                       (lambda D: D["rhowB"] + P.cumcool(D), ns.INK, r"net")],
           su.p0 * su.du, r"energy flux $/p_0\Delta u$")

    axs[0, 0].legend(loc="center left", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel(r"$z/\Delta u\,t_0$")
    axs[0, 0].set_xlim(-20, 10)
    ns.save(fig, ns.image_path(2, "trml_profiles.pdf"))
    plt.close(fig)


# ─── Fig. 3: temperature PDFs ────────────────────────────────────────────────

def fig3(red, su, P):
    edges = red["edges"]
    lgT = 0.5 * (edges[1:] + edges[:-1])
    dlg = np.diff(edges)
    tau = red["time"] / su.t0
    sel = np.flatnonzero((tau >= P.tau[0]) & (tau <= P.tau[-1]))
    y = red["y"]
    lo, hi = np.log10(1.1 * su.Tc_K), np.log10(0.9 * su.Th_K)
    nb = (lgT > lo) & (lgT < hi)

    def normalise(p):
        return p / (p[..., nb] * dlg[nb]).sum(-1, keepdims=True)

    Tbar = P.mean["T"]
    per_snap = []
    for j, zc in zip(sel, P.zc_line):
        Trow = np.interp(y - zc, P.z, Tbar, left=np.nan, right=np.nan)
        r = (Trow > 1.1 * su.Tc_K) & (Trow < 0.9 * su.Th_K)
        per_snap.append(red["hist"][j][:, r].sum(1) / dlg)
    per_snap = np.array(per_snap, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdf = normalise(per_snap.sum(0))
        spread = normalise(per_snap).std(0)

    zf = np.linspace(P.z[0], P.z[-1], 20 * len(P.z))
    h, _ = np.histogram(np.log10(np.interp(zf, P.z, Tbar)), edges,
                        weights=np.interp(zf, P.z, P.mean["cool"]))
    pe_bar = normalise(h / dlg)

    T = 10**lgT
    inside = (T > su.Tc_K) & (T < su.Th_K)
    dT = su.Th_K - su.Tc_K
    Tp = dT / (2 * P.z0) * (1 - 4 * ((T - 0.5 * (su.Th_K + su.Tc_K)) / dT) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        pv = np.where(inside, T * np.log(10) / Tp, np.nan)
        model = {"V": normalise(pv), "M": normalise(pv / T),
                 "E": normalise(pv * rp.cool_lambda(T) / T**2)}
        flat = normalise(np.where(inside, T, np.nan))

    fig, ax = plt.subplots(figsize=ns.figsize(8.5, ns.WIDE), layout="constrained")
    cols = {"V": AMBER, "M": BLUE, "E": GREEN}
    for w, key in enumerate("VME"):
        band(ax, lgT, pdf[w], spread[w], cols[key], label=rf"$\mathcal P_{key}$")
        ax.plot(lgT, model[key], ls=(0, (4, 2)), color=cols[key], lw=1.0)
    ax.plot(lgT, pe_bar, color=RED, lw=1.0, label=r"$\overline{\mathcal P}_E$ vs $\avg{T}$")
    ax.plot(lgT, flat, color=GREY, ls=(0, (1, 1.5)), lw=1.2,
            label=r"$\mathcal P_E$ of a cooling flow")
    ax.set(yscale="log", ylim=(1e-3, 2e1), xlim=(4.0, 6.0),
           xlabel=r"$\log_{10}T$ [K]", ylabel=r"$\mathcal P(\log_{10}T)$")
    ax.grid(which="major", alpha=0.6)
    ax.legend(loc="lower left", fontsize=8, ncols=2)
    ns.save(fig, ns.image_path(2, "trml_pdfs.pdf"))
    plt.close(fig)


# ─── Fig. 4: energy balance ──────────────────────────────────────────────────

def fig4(su, P, smooth=1.0):
    z = P.z / su.L
    m = P.mean
    unit = su.p0 / su.t0
    Tfit = P.tanh_T(P.z) / su.T_unit
    dTdz = np.gradient(Tfit, P.dz)
    sig = smooth * su.L / P.dz

    def sm(a):
        return gaussian_filter1d(a, sig, mode="nearest")

    cool = sm(m["cool"])
    enth = np.gradient(sm(rp.en_enth(m)), P.dz)
    turb = np.gradient(sm(rp.en_turb(m)), P.dz)
    sigma_cool = cool.sum() * P.dz
    mdot = -m["rhow"][P.slab].mean()
    enth_th = -mdot * su.gamma / (su.gamma - 1) * dTdz
    CE = P.cumcool(m) / sigma_cool
    turb_th = np.gradient(sigma_cool * ((P.tanh_T(P.z) - su.Tc_K) / (su.Th_K - su.Tc_K) - CE),
                          P.dz)
    kappa_unit = su.rho_h * su.du**2 * su.t0
    d2Tdz2 = np.gradient(dTdz, P.dz)

    fig, ax = plt.subplots(figsize=ns.figsize(8, ns.WIDE), layout="constrained")
    ax.plot(z, cool / unit, color=BLUE, label=r"$\avg{n^2\Lambda}$")
    ax.plot(z, enth / unit, color=AMBER, label=r"$\dd(\avg{\mathcal B}\avg{\rho u_z})/\dd z$")
    ax.plot(z, enth_th / unit, color=AMBER, ls=(0, (4, 2)), lw=1.0)
    ax.plot(z, turb / unit, color=GREEN, label=r"$\dd\mathcal Q_t/\dd z$")
    ax.plot(z, turb_th / unit, color=GREEN, ls=(0, (4, 2)), lw=1.0)
    ax.plot(z, -0.1 * kappa_unit * d2Tdz2 / unit, color=RED, ls=(0, (1, 1.5)), lw=1.2,
            label=r"$-\kappa_t\,\dd^2\avg{T}/\dd z^2$, $\kappa_t = 0.1$")
    ax.axhline(0, color=ns.HAIRLINE, lw=0.5)
    ax.set(xlim=(-8, 11), xlabel=r"$z/\Delta u\,t_0$", ylabel=r"rate $/(p_0/t_0)$")
    ax.grid(alpha=0.6)
    ax.legend(loc="lower right", fontsize=8)

    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    ticks = [t for t in ax.get_xticks() if -8 <= t <= 11]
    top.set_xticks(ticks, [f"{np.interp(t, z, m['T']) / 1e5:.2f}" for t in ticks])
    top.tick_params(labelsize=8)
    top.set_xlabel(r"$\avg{T}$ [$10^5$\,K]", fontsize=9, color=ns.MUTED)
    top.spines["top"].set_visible(True)
    ns.save(fig, ns.image_path(2, "trml_energy.pdf"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="kh3d_256x1024x256")
    args = ap.parse_args()
    run, athinput, bindir = rp.run_paths(args.run)
    sim = SimulationData(athinput, bindir)
    su = rp.Setup.from_sim(sim)
    red = rp.reduce_run(args.run, workers=1)
    P = rp.Profiles(red, su, TMIN, np.inf)
    fig2(su, P, None)
    fig3(red, su, P)
    fig4(su, P)
    t = fig1(sim, su, P, None)
    print(f"Fig. 1 snapshot at t = {t:.0f} t0")


if __name__ == "__main__":
    main()
