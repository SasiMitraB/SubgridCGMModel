"""
The cooling correction of eq-cooling, and its multiplicative alternative, over
every coarse cell of the 256x1024x256 radiative KH run:

  T_CG = (mu m_p / k_B) <P>/<rho>                                  (eq-temperature)
  C    = <n^2 Lcut(T)> - (<rho>/mu m_p)^2 Lcut(T_CG)                          (eq-cooling)
  g    = 1 if <n^2 Lcut(T)> > 0 else 0      (gate: subgrid gas in the cooling window)
  r    = <n^2 Lcut(T)> / ((<rho>/mu m_p)^2 Lfull(T_CG)),    z_C = ln r   (g = 1 only)
       = <n^2>/(<rho>/mu m_p)^2  x  Lambda_eff/Lfull(T_CG),   Lambda_eff = <n^2 Lcut>/<n^2>
  C    = g (<rho>/mu m_p)^2 [Lfull(T_CG) e^z_C - Lcut(T_CG)]            (eq-cooling-model)

<P> is the bracket of eq-temperature: <E> - <rho v>^2/2<rho> - Tr(R)/2 equals the
block average of the internal energy, so <P> = (gamma-1) <e_int> exactly, the
same Tr(R) as everywhere else. Lcut is AthenaK's ISM cooling with its cut-offs
(cool_lambda of recreating_plots), on the fine grid and on T_CG alike; Lfull is the
same curve without them (stock_lambda of cooling_curve), positive at every T_CG.

Figures (Overleaf/images/chapter2/):
  cooling_correction.pdf    C against T_CG
  cooling_zc_cases.pdf      the gate g, and where in T_CG the gated cells sit
  cooling_zc.pdf            z_C against T_CG, inside and outside the window
  cooling_gate_cut.tex      cooling lost by gating off outside-window cells with z_C < k
  cooling_zc_factors.pdf    z_C split into clumping and Lambda_eff/Lfull(T_CG)
  cooling_targets.pdf       C against z_C as a target: reconstruction errors
and the table cooling_zc_cases.tex next to them.

Per-cell values are cached one file per frame (cell_cache.py), written as each
frame is computed, so replotting does not reread the run.
Frames before ns.T_MIN = 2.5 Myr, the initial transient, are left out of every
figure (they stay in the cache).

    python theory/notes/plotting_scripts/cooling_comparision.py [--b 16] [--every 1] [--recompute]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from sklearn.ensemble import HistGradientBoostingRegressor

from vis_athenak.utils.units import Units

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "theory" / "sharma+2025"))
from data.coarse_grain_utils import block_mean  # noqa: E402
from recreating_plots import cool_lambda  # noqa: E402

import cell_cache  # noqa: E402
from cooling_curve import stock_lambda as lambda_full  # noqa: E402
import notes_style as ns  # noqa: E402

FIELDS = ["dens", "velx", "vely", "velz", "eint", "s_00"]
CACHE_VERSION = 2  # bump when the stored per-cell quantities change

LINTHRESH = 1e-27     # |C| below this is linear on the symlog axis [erg cm^-3 s^-1]
T_WINDOW = (1.05e4, 0.95e6)  # Lambda = 0 outside this window
LOGT_EDGES = np.linspace(3.8, 6.2, 121)
MISS_TARGET = 0.005   # share of the total cooling we accept losing to the gate cut
K_TABLE = (-4.0, -3.0, -2.0, -1.0, 0.0)

CASES = [  # (key, label, colour) of the rows of the gate table
    ("inside", r"$g=1$, $\Tcg$ in window", ns.SERIES[0]),
    ("outside", r"$g=1$, $\Tcg$ outside", ns.SERIES[1]),
    ("off", r"$g=0$", ns.SERIES[3]),
]


# ─── Per-cell quantities ─────────────────────────────────────────────────────

def block_extreme(f, b, fn):
    """Block max or min (fn = np.max / np.min) of a 3D field."""
    nz, ny, nx = f.shape
    return fn(f.reshape(nz // b, b, ny // b, b, nx // b, b), axis=(1, 3, 5))


def coarse_cells(q, sim, b):
    """Every quantity of the module docstring for each b^3 coarse cell (cgs)."""
    units = Units.from_params(sim.params)
    gamma = float(sim.params["hydro"]["gamma"])
    rho, P = q["dens"], (gamma - 1) * q["eint"]
    n = units.number_density_cgs * rho
    n2L = block_mean(n**2 * cool_lambda(units.temperature_cgs * P / rho), b)
    n2 = block_mean(n**2, b)
    rho_c, P_c = block_mean(rho, b), block_mean(P, b)
    T_cg = units.temperature_cgs * P_c / rho_c
    nbar = units.number_density_cgs * rho_c
    resolved = nbar**2 * cool_lambda(T_cg)

    # f = k_sg / (e_th + k_sg) of eq-f, from the exact Tr(R)
    m2 = sum(block_mean(rho * q[v], b)**2 for v in ("velx", "vely", "velz"))
    rv2 = block_mean(rho * (q["velx"]**2 + q["vely"]**2 + q["velz"]**2), b)
    k_sg = np.maximum(0.5 * (rv2 - m2 / rho_c) / rho_c, 0.0)
    e_th = P_c / ((gamma - 1) * rho_c)
    f = k_sg / (e_th + k_sg)

    s0 = block_mean(rho * q["s_00"], b) / rho_c
    contrast = block_extreme(rho, b, np.max) / block_extreme(rho, b, np.min)
    cells = dict(T_cg=T_cg, num=n2L, den=resolved, n2=n2, nbar=nbar, rho=rho_c,
                 f=f, s0=s0, contrast=contrast)
    return {k: v.ravel() for k, v in cells.items()}


def add_full(d):
    """(<rho>/mu m_p)^2 Lfull(T_CG), from cached T_CG and nbar (no cache change)."""
    d["full"] = d["nbar"]**2 * lambda_full(d["T_cg"])
    return d


def gate(d):
    """g = 1 where some subgrid gas cools, i.e. <n^2 Lcut> > 0."""
    return d["num"] > 0


def cases(d):
    """Masks of the gate rows: g, and whether T_CG lies in the cooling window."""
    g, inside = gate(d), d["den"] > 0
    return {"inside": g & inside, "outside": g & ~inside, "off": ~g}


def binned_stat(x, y, edges, fn=np.median, min_count=20):
    """fn(y) in each x bin holding at least min_count values (else nan)."""
    idx = np.digitize(x, edges) - 1
    out = np.full(edges.size - 1, np.nan)
    order = np.argsort(idx, kind="stable")
    groups = np.split(y[order], np.searchsorted(idx[order], np.arange(edges.size)))[1:-1]
    for k, g in enumerate(groups):
        if g.size >= min_count:
            out[k] = fn(g)
    return out


def binned_median(x, y, edges, min_count=20):
    return binned_stat(x, y, edges, np.median, min_count)


def centres(edges):
    return 0.5 * (edges[1:] + edges[:-1])


def window_lines(ax):
    for T_edge in T_WINDOW:
        ax.axvline(np.log10(T_edge), color=ns.HAIRLINE, ls=(0, (1, 2)), lw=0.8)


# ─── 0. C against T_CG ───────────────────────────────────────────────────────

def symlog_edges(lo, hi, n_log=60, n_lin=10):
    """Bin edges symmetric about 0: log-spaced beyond LINTHRESH, linear inside."""
    pos = np.logspace(np.log10(LINTHRESH), np.log10(hi), n_log + 1)
    return np.concatenate([-pos[::-1], np.linspace(-LINTHRESH, LINTHRESH, n_lin + 1)[1:-1], pos])


def plot_correction(d, out):
    logT, C = np.log10(d["T_cg"]), d["num"] - d["den"]
    ybins = symlog_edges(LINTHRESH, 1e-22)

    fig, ax = plt.subplots(figsize=ns.figsize(7.4), layout="constrained")
    *_, im = ax.hist2d(logT, np.clip(C, ybins[0], ybins[-1]), bins=[LOGT_EDGES, ybins],
                       norm=LogNorm(), cmap=ns.SEQ, rasterized=True)
    ax.plot(centres(LOGT_EDGES), binned_median(logT, C, LOGT_EDGES),
            color=ns.OVERLAY, label=r"median $\Cs$")

    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=0.5)
    ax.axhline(0, color=ns.MUTED, lw=0.5)
    window_lines(ax)
    ax.text(0.98, 0.97, r"\itshape $\Cs>0$: the subgrid gas cools more",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color=ns.MUTED)
    ax.text(0.98, 0.03, r"\itshape $\Cs<0$: $\Lambda(\Tcg)$ overestimates the cooling",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=ns.MUTED)

    ax.set(xlabel=r"$\log_{10}\,\Tcg$ [K]",
           ylabel=r"$\Cs$ [erg\,cm$^{-3}$\,s$^{-1}$]",
           xlim=(LOGT_EDGES[0], LOGT_EDGES[-1]))
    ax.legend(loc="upper left")
    ns.colorbar(fig, im, ax, "coarse cells")
    ns.save(fig, out)


# ─── 1. Where z_C is defined ─────────────────────────────────────────────────

def case_table(d, out):
    """Print the gate rows and write them as a booktabs table."""
    m = cases(d)
    N, total, resolved = d["num"].size, d["num"].sum(), d["den"].sum()
    rows = [
        ("inside", r"$1$", r"in window", "finite"),
        ("outside", r"$1$", r"outside", "finite"),
        ("off", r"$0$", r"any", r"--- ($\Cs = 0$)"),
    ]
    print("\n1. the gate g")
    print(f"   {'case':8s} {'cells':>9s} {'% cells':>8s} {'% <n2L>':>8s} {'% resolved':>10s}")
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"$g$ & $\Tcg$ & $z_{\Cs}$ & cells"
             r" & \% of $\sum\avg{n^2\Lcut}$ & \% of $\sum(\avg{\rho}/\mump)^2\Lcut(\Tcg)$ \\",
             r"\midrule"]
    for key, sn, sd, zc in rows:
        sel = m[key]
        pc, pn, pr = 100 * sel.sum() / N, 100 * d["num"][sel].sum() / total, \
            100 * d["den"][sel].sum() / resolved
        print(f"   {key:8s} {sel.sum():9d} {pc:8.2f} {pn:8.2f} {pr:10.2f}")
        lines.append(rf"{sn} & {sd} & {zc} & {pc:.1f}\,\% & {pn:.1f}\,\% & {pr:.1f}\,\% \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    print(f"saved {out}")


def plot_cases(d, out):
    """Cells and cooling of each case against T_CG."""
    logT, m = np.log10(d["T_cg"]), cases(d)
    fig, (a, b) = plt.subplots(1, 2, figsize=ns.figsize(5.6), layout="constrained",
                               sharex=True)
    for key, label, colour in CASES:
        sel = m[key]
        a.hist(logT[sel], bins=LOGT_EDGES, histtype="step", color=colour, lw=1.2,
               label=label)
        if key != "off":  # it holds no cooling
            b.hist(logT[sel], bins=LOGT_EDGES, weights=d["num"][sel] / d["num"].sum(),
                   histtype="step", color=colour, lw=1.2)
    a.set(yscale="log", xlabel=r"$\log_{10}\,\Tcg$ [K]", ylabel="coarse cells",
          xlim=(LOGT_EDGES[0], LOGT_EDGES[-1]))
    b.set(yscale="log", xlabel=r"$\log_{10}\,\Tcg$ [K]",
          ylabel=r"share of $\sum\avg{n^2\Lcut}$ per bin")
    for ax in (a, b):
        window_lines(ax)
    a.legend(loc="upper center", fontsize=7)
    ns.save(fig, out)


# ─── 2. The gate cut z_C < k outside the window ──────────────────────────────

def z_full(d):
    """z_C of every cell; -inf where g = 0."""
    g = gate(d)
    z = np.full(d["num"].size, -np.inf)
    z[g] = np.log(d["num"][g] / d["full"][g])
    return z


def missed(d, k, sel=None):
    """Share of sum <n^2 Lcut> lost by setting g = 0 on outside-window cells with z_C < k."""
    sel = np.ones(d["num"].size, bool) if sel is None else sel
    cut = sel & cases(d)["outside"] & (z_full(d) < k)
    return d["num"][cut].sum() / d["num"][sel].sum()


def k_for(d, target=MISS_TARGET):
    """The k at which the outside-window cut loses `target` of the cooling."""
    ks = np.linspace(-8, 2, 2001)
    loss = np.array([missed(d, k) for k in ks])
    return float(np.interp(target, loss, ks))


def gate_cut_table(d, k_star, out):
    """Cooling lost and cells cut against k, and the same cut applied to all gated cells."""
    m, z, num = cases(d), z_full(d), d["num"]
    total, n_out = num.sum(), m["outside"].sum()
    g = gate(d)
    print(f"\n2. gate cut outside the window: k = {k_star:.2f} loses {MISS_TARGET:.1%}")
    lines = [r"\begin{widetabular}{rrrr}", r"\toprule",
             r"$k$ & \% of $\sum\avg{n^2\Lcut}$ lost & \% of outside cells cut"
             r" & error if cut everywhere \\", r"\midrule"]
    for k in sorted(K_TABLE + (round(k_star, 1),)):
        lost = 100 * missed(d, k)
        cut_out = 100 * (m["outside"] & (z < k)).sum() / n_out
        every = g & (z < k)   # also in the window, where g = 0 keeps the coarse cooling
        err = 100 * np.abs(num[every] - d["den"][every]).sum() / total
        print(f"   k={k:5.1f}  lost {lost:6.3f}%  outside cells cut {cut_out:5.1f}%  "
              f"|error| if global {err:7.2f}%")
        cells = [f"{k:.1f}", rf"{lost:.2f}\,\%", rf"{cut_out:.0f}\,\%", rf"{err:.1f}\,\%"]
        bold = k == round(k_star, 1)   # the chosen k
        lines.append(" & ".join(rf"$\mathbf{{{c}}}$" if bold else f"${c}$" for c in cells)
                     + r" \\")
    lines += [r"\bottomrule", r"\end{widetabular}"]
    out.write_text("\n".join(lines) + "\n")

    per_frame = np.array([missed(d, k_star, d["frame"] == f) for f in d["frames"]])
    cold = m["outside"] & (d["T_cg"] < T_WINDOW[0])
    print(f"   per frame: {100 * per_frame.mean():.2f} +- {100 * per_frame.std():.2f}% "
          f"(range {100 * per_frame.min():.2f}-{100 * per_frame.max():.2f}%)")
    print(f"   outside cooling: cold side {100 * num[cold].sum() / total:.2f}%, "
          f"hot side {100 * num[m['outside'] & ~cold].sum() / total:.2f}%")
    print(f"   min z_C in the window: {z[m['inside']].min():.2f}")
    print(f"saved {out}")


# ─── 3. z_C against T_CG ─────────────────────────────────────────────────────

def plot_zc(d, out, k_star):
    ok = gate(d)
    logT, z = np.log10(d["T_cg"][ok]), np.log(d["num"][ok] / d["full"][ok])
    zbins = np.linspace(-12, 6, 181)

    fig, ax = plt.subplots(figsize=ns.figsize(7.4), layout="constrained")
    *_, im = ax.hist2d(logT, np.clip(z, zbins[0], zbins[-1]), bins=[LOGT_EDGES, zbins],
                       norm=LogNorm(), cmap=ns.SEQ, rasterized=True)
    xc = centres(LOGT_EDGES)
    ax.plot(xc, binned_median(logT, z, LOGT_EDGES), color=ns.OVERLAY,
            label=r"median $z_{\Cs}$")
    for q, lab in ((16, r"16th and 84th percentiles"), (84, None)):
        pct = binned_stat(logT, z, LOGT_EDGES, lambda v: np.percentile(v, q))
        ax.plot(xc, pct, color=ns.OVERLAY, lw=0.7, ls=(0, (4, 2)), label=lab)
    ax.axhline(0, color=ns.MUTED, lw=0.5)
    for lo, hi in ((LOGT_EDGES[0], np.log10(T_WINDOW[0])), (np.log10(T_WINDOW[1]), LOGT_EDGES[-1])):
        ax.plot([lo, hi], [k_star, k_star], color=ns.ACCENT, lw=1.0, ls=(0, (2, 1.5)),
                label=rf"gate cut $k = {k_star:.1f}$" if lo == LOGT_EDGES[0] else None)
    window_lines(ax)
    ax.set(xlabel=r"$\log_{10}\,\Tcg$ [K]",
           ylabel=r"$z_{\Cs} = \ln\dfrac{\avg{n^2\Lcut}}{(\avg{\rho}/\mump)^2\,\Lfull(\Tcg)}$",
           xlim=(LOGT_EDGES[0], LOGT_EDGES[-1]), ylim=(zbins[0], zbins[-1]))
    ax.legend(loc="upper left")
    ns.colorbar(fig, im, ax, "coarse cells")
    ns.save(fig, out)


# ─── 4. Clumping and Lambda_eff / Lambda(T_CG) ───────────────────────────────

def factors(d):
    ok = gate(d)
    num, den, n2, nbar = (d[k][ok] for k in ("num", "full", "n2", "nbar"))
    z = np.log(num / den)
    ln_clump = np.log(n2 / nbar**2)
    ln_leff = np.log(num / n2) - np.log(den / nbar**2)   # Lambda_eff / Lfull(T_CG)
    assert np.allclose(z, ln_clump + ln_leff)
    return ok, z, ln_clump, ln_leff


def variance_shares(z, a, b, logT):
    """Shares of var(z) from var(a), var(b) and 2 cov(a, b); total and within T_CG bins."""
    def shares(z, a, b):
        v = np.var(z)
        return np.var(a) / v, np.var(b) / v, 2 * np.cov(a, b)[0, 1] / v

    def resid(y):
        idx = np.clip(np.digitize(logT, LOGT_EDGES) - 1, 0, LOGT_EDGES.size - 2)
        med = binned_median(logT, y, LOGT_EDGES, min_count=1)
        return y - np.nan_to_num(med[idx])

    return shares(z, a, b), shares(resid(z), resid(a), resid(b))


def plot_factors(d, out):
    ok, z, ln_clump, ln_leff = factors(d)
    logT = np.log10(d["T_cg"][ok])
    total, within = variance_shares(z, ln_clump, ln_leff, logT)
    print("\n3. var(z_C) = var(ln clumping) + var(ln Leff/L) + 2 cov")
    print(f"   all cells:               {total[0]:.2f} + {total[1]:.2f} + {total[2]:+.2f}")
    print(f"   at fixed T_CG (resid.):  {within[0]:.2f} + {within[1]:.2f} + {within[2]:+.2f}")
    print(f"   corr(ln clumping, f) = {np.corrcoef(ln_clump, d['f'][ok])[0, 1]:.2f}, "
          f"corr(ln clumping, ln contrast) = "
          f"{np.corrcoef(ln_clump, np.log(d['contrast'][ok]))[0, 1]:.2f}")

    lc, ll = ln_clump / np.log(10), ln_leff / np.log(10)
    fig, axes = plt.subplots(2, 2, figsize=ns.figsize(11.0), layout="constrained")
    panels = [
        (axes[0, 0], logT, lc, LOGT_EDGES, np.linspace(0, 2.5, 101),
         r"$\log_{10}\,\Tcg$ [K]", r"$\log_{10}\,\avg{n^2}/(\avg{\rho}/\mump)^2$",
         rf"\itshape {total[0]:.0%} of var\,$z_{{\Cs}}$, {within[0]:.0%} at fixed $\Tcg$"),
        (axes[0, 1], logT, ll, LOGT_EDGES, np.linspace(-3, 3, 101),
         r"$\log_{10}\,\Tcg$ [K]", r"$\log_{10}\,\Lambda_{\mathrm{eff}}/\Lfull(\Tcg)$",
         rf"\itshape {total[1]:.0%} of var\,$z_{{\Cs}}$, {within[1]:.0%} at fixed $\Tcg$"),
        (axes[1, 0], d["f"][ok], lc, np.linspace(0, 0.6, 101), np.linspace(0, 2.5, 101),
         r"$f = \ksg/\kmax$", r"$\log_{10}\,\avg{n^2}/(\avg{\rho}/\mump)^2$", None),
        (axes[1, 1], np.log10(d["contrast"][ok]), lc, np.linspace(0, 2.2, 101),
         np.linspace(0, 2.5, 101), r"$\log_{10}\,\rho_{\max}/\rho_{\min}$ in the cell",
         r"$\log_{10}\,\avg{n^2}/(\avg{\rho}/\mump)^2$", None),
    ]
    for ax, x, y, xb, yb, xl, yl, note in panels:
        *_, im = ax.hist2d(x, np.clip(y, yb[0], yb[-1]), bins=[xb, yb], norm=LogNorm(),
                           cmap=ns.SEQ, rasterized=True)
        ax.plot(centres(xb), binned_median(x, y, xb), color=ns.OVERLAY)
        ax.set(xlabel=xl, ylabel=yl)
        if note:
            ax.text(0.03, 0.97, note, transform=ax.transAxes, ha="left", va="top",
                    fontsize=7.5, color=ns.MUTED)
    for ax in axes[0]:
        window_lines(ax)
    axes[0, 1].axhline(0, color=ns.MUTED, lw=0.5)
    ns.colorbar(fig, im, axes, "coarse cells")
    ns.save(fig, out)


# ─── 5. Which target predicts better ─────────────────────────────────────────

def features(d, sel):
    return np.column_stack([np.log10(d["T_cg"][sel]), np.log10(d["rho"][sel]),
                            d["f"][sel], d["s0"][sel]])


def fit_predict(d, train, test):
    """Reconstructed <n^2 Lambda> on the test cells from four models.

    Additive: predict C, then max((<rho>/mu m_p)^2 Lcut(T_CG) + C, 0).
    Log: predict z_C on the g = 1 cells, then (<rho>/mu m_p)^2 Lfull(T_CG) exp(z);
    g = 0 cells keep the coarse cooling. The gate is taken as known (perfect classifier).
    Each target by the median in T_CG bins and by gradient-boosted trees on
    (log T_CG, log <rho>, f, s0).
    """
    num, den, full = d["num"], d["den"], d["full"]
    C = num - den
    pos = gate(d)
    zt = np.log(num[pos] / full[pos])
    logT = np.log10(d["T_cg"])

    def by_bins(x_tr, y_tr, x_te):
        med = binned_median(x_tr, y_tr, LOGT_EDGES, min_count=1)
        idx = np.clip(np.digitize(x_te, LOGT_EDGES) - 1, 0, LOGT_EDGES.size - 2)
        return np.where(np.isnan(med[idx]), np.median(y_tr), med[idx])

    def trees(X_tr, y_tr, X_te):
        model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1,
                                              early_stopping=True, random_state=0)
        return model.fit(X_tr, y_tr).predict(X_te)

    tr_pos, te_pos = train & pos, test & pos
    preds = {}
    for name, fit in (("binned", lambda tr, y, te: by_bins(logT[tr], y, logT[te])),
                      ("trees", lambda tr, y, te: trees(features(d, tr), y, features(d, te)))):
        preds[("additive", name)] = np.maximum(den[test] + fit(train, C[train], test), 0.0)
        rec = den[test].copy()
        rec[pos[test]] = full[te_pos] * np.exp(fit(tr_pos, zt[train[pos]], te_pos))
        preds[("log", name)] = rec
    return preds


def plot_targets(d, out):
    frames = d["frames"]
    # Alternate frames: train on even, test on odd. With every frame (0.08 Myr apart) each
    # test frame sits between two training frames, so the test is close to in-sample.
    test_frames = frames[1::2]
    test = np.isin(d["frame"], test_frames)
    train = ~test
    preds = fit_predict(d, train, test)

    num = d["num"][test]
    cooling = num > 0
    fr = d["frame"][test]
    t_of = {f: d["time"][d["frame"] == f][0] for f in test_frames}

    style = {("additive", "binned"): (ns.SERIES[0], "-"), ("additive", "trees"): (ns.SERIES[0], "--"),
             ("log", "binned"): (ns.SERIES[1], "-"), ("log", "trees"): (ns.SERIES[1], "--")}
    label = {"additive": r"$\Cs$", "log": r"$z_{\Cs}$",
             "binned": r"median in $\Tcg$", "trees": "trees on coarse fields"}

    print("\n4. reconstructed <n2L> on test frames (cells with <n2L> > 0)")
    print(f"   {'target':9s} {'model':7s} {'med |rel|':>9s} {'p90 |rel|':>9s} {'within x2':>9s} "
          f"{'total ratio':>12s} {'per-frame':>16s}")
    fig, (a, b) = plt.subplots(1, 2, figsize=ns.figsize(6.0), layout="constrained")
    bins = np.linspace(-3, 2, 101)
    for key, rec in preds.items():
        colour, ls = style[key]
        rel = np.abs(rec[cooling] - num[cooling]) / num[cooling]
        ratio = np.log10(np.maximum(rec[cooling], 1e-3 * num[cooling]) / num[cooling])
        per_frame = np.array([rec[fr == f].sum() / num[fr == f].sum() for f in test_frames])
        print(f"   {key[0]:9s} {key[1]:7s} {np.median(rel):9.3f} {np.percentile(rel, 90):9.3f} "
              f"{np.mean(np.abs(ratio) < np.log10(2)):9.1%} {rec.sum() / num.sum():12.3f} "
              f"{per_frame.mean():8.3f}±{per_frame.std():.3f}")
        a.hist(np.clip(ratio, bins[0], bins[-1]), bins=bins, histtype="step", color=colour,
               ls=ls, lw=1.1, density=True)
        b.plot([t_of[f] for f in test_frames], per_frame, color=colour, ls=ls, lw=1.1,
               label=f"{label[key[0]]}, {label[key[1]]}")
    a.axvline(0, color=ns.MUTED, lw=0.5)
    a.set(yscale="log", xlabel=r"$\log_{10}$ reconstructed\,/\,true $\avg{n^2\Lambda}$",
          ylabel="test cells (density)", xlim=(bins[0], bins[-1]))
    a.text(0.03, 0.97, r"\itshape left edge: no cooling predicted", transform=a.transAxes,
           ha="left", va="top", fontsize=7.5, color=ns.MUTED)
    b.axhline(1, color=ns.MUTED, lw=0.5)
    b.set(xlabel=r"$t$ [Myr]", ylabel=r"$\sum$ reconstructed\,/\,$\sum$ true")
    b.legend(loc="best", fontsize=7)
    ns.save(fig, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--b", type=int, default=16, help="coarse block size")
    parser.add_argument("--every", type=int, default=1, help="use every n-th frame")
    parser.add_argument("--recompute", action="store_true", help="recompute cached frames")
    args = parser.parse_args()

    d, _ = cell_cache.load("cooling_cells", args.b, args.every, FIELDS, coarse_cells,
                           CACHE_VERSION, args.recompute)
    d = add_full(ns.drop_transient(d))
    print(f"{d['num'].size} cells of {int(d['b'])}^3, {len(d['frames'])} frames, "
          f"t = {d['time'].min():.1f}-{d['time'].max():.1f} Myr")

    plot_correction(d, ns.image_path(2, "cooling_correction.pdf"))
    case_table(d, ns.image_path(2, "cooling_zc_cases.tex"))
    plot_cases(d, ns.image_path(2, "cooling_zc_cases.pdf"))
    k_star = k_for(d)
    gate_cut_table(d, k_star, ns.image_path(2, "cooling_gate_cut.tex"))
    plot_zc(d, ns.image_path(2, "cooling_zc.pdf"), k_star)
    plot_factors(d, ns.image_path(2, "cooling_zc_factors.pdf"))
    plot_targets(d, ns.image_path(2, "cooling_targets.pdf"))
