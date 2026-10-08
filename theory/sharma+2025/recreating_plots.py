"""
Recreate Figs. 1-4 of Sharma et al. 2025, "Universal Structure of Turbulent
Radiative Mixing Layers" (arXiv:2509.03802), from the 3D KH runs in
/data/sasi/simulation_outputs (deck athinputs/kh_fid3D_64_cool.athinput).

    python recreating_plots.py                         # kh3d_128x512x128
    python recreating_plots.py kh3d_64x256x64 kh3d_32x128x32 --workers 16
    python recreating_plots.py --tmin 146 --tmax 515 --tsnap 350

Geometry. The paper's (x, y, z) = (streamwise, spanwise, shear-normal) are
our (x1, x3, x2): arrays are (nx3, nx2, nx1), the hot phase sits on top
(large x2) and the bottom boundary reflects, as in the paper. Below,
u = v1 - vx_cold (streamwise, in the cold-phase frame), w = v2 (vertical)
and s = v3 (spanwise).

Pipeline.
  1. One pass over every snapshot (in parallel, cached as an .npz next to
     this script) stores horizontal means of the raw moments listed in
     MOMENTS, plus per-row histograms of log10 T weighted by volume, mass
     and n^2 Lambda. Re-running only reduces snapshots not in the cache.
  2. The interface height z_c(t) of each snapshot is the height that a
     sharp step would need to hold the same amount of hot gas. A linear
     fit z_c = a + v t over the averaging window gives the TRML frame: the
     profiles are shifted to zeta = x2 - z_c(t), and w -> w - v is applied
     to the moments algebraically (so no second read is needed). The flux
     estimate v = [(rho w)_h - (rho w)_c]/(rho_h - rho_c) of the paper is
     printed as a check.
  3. <.> is the horizontal and temporal mean over the window, and
     covariances are <ab> - <a><b> over that ensemble. Shaded bands are the
     1-sigma spread over snapshots of the same quantity computed from each
     snapshot's horizontal means. z = 0 is where the time-averaged <T>
     crosses (Th + Tc)/2.

Normalisations follow the paper: t0 = t_cool(T0 = sqrt(Tc Th)) at the
initial pressure (0.0194 Myr here), lengths in du t0, mass flux in
rho_h du, momentum flux in rho_h du^2 or p0, energy flux in p0 du. Cooling
uses AthenaK's ISM curve with its cut-offs (zero for T <= 1.05e4 K and
T > 0.95e6 K), exactly as the simulation did.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from vis_athenak.simulation_data import SimulationData  # noqa: E402
from vis_athenak.utils.ism_cooling import ISMCoolFn  # noqa: E402
from vis_athenak.utils.units import Units  # noqa: E402

DATA = Path("/data/sasi/simulation_outputs")
HERE = Path(__file__).resolve().parent
PLOTS = HERE / "plots"
CACHE = HERE / "cache"
CACHE_VERSION = 1
FIELDS = ["dens", "velx", "vely", "velz", "eint"]
LOGT_EDGES = np.linspace(3.9, 6.3, 121)  # 0.02 dex bins for the PDFs

# Horizontal means stored per snapshot (simulation frame, u in the cold frame).
# K = (u^2 + w^2 + s^2)/2, h = gamma/(gamma-1) p/rho, cool = n^2 Lambda.
MOMENTS = ["rho", "u", "w", "s", "p", "T", "h", "K", "u2", "w2", "s2",
           "rhou", "rhow", "rhouw", "rhow2", "rhoK", "rhowK", "rhoh", "rhowh",
           "cool"]


# ─── Setup ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Setup:
    """Parameters of one run, in code units unless named *_K."""
    gamma: float
    rho_c: float
    rho_h: float
    p0: float
    vx_c: float
    du: float
    T_unit: float      # K per unit p/rho
    cool_unit: float   # cgs Lambda per code Lambda
    Tc_K: float
    Th_K: float
    t0: float

    @classmethod
    def from_sim(cls, sim: SimulationData) -> Setup:
        prob = sim.params["problem"]
        gamma = float(sim.params["hydro"]["gamma"])
        units = Units.from_params(sim.params)
        rho_c, rho_h = float(prob["rho_cold"]), float(prob["rho_hot"])
        p0 = float(prob["press"])
        vx_c, vx_h = float(prob["vx_cold"]), float(prob["vx_hot"])
        T_unit = units.temperature_cgs
        cool_unit = units.pressure_cgs / units.time_cgs / units.number_density_cgs**2
        Tc, Th = T_unit * p0 / rho_c, T_unit * p0 / rho_h
        T0 = np.sqrt(Tc * Th)
        rho0 = p0 / (T0 / T_unit)  # isobaric
        t0 = p0 / ((gamma - 1) * rho0**2 * cool_lambda(np.array([T0]))[0] / cool_unit)
        return cls(gamma, rho_c, rho_h, p0, vx_c, vx_h - vx_c, T_unit, cool_unit,
                   Tc, Th, t0)

    @property
    def L(self) -> float:
        """Length unit du t0."""
        return self.du * self.t0


def cool_lambda(T_K):
    """AthenaK's ISMCoolFn (srcterms/ismcooling.hpp) in erg cm^3/s, cut-offs included."""
    T_K = np.asarray(T_K, dtype=float)
    lam = ISMCoolFn(T_K)  # vis_athenak's copy has no cut-offs and is flat below 10^4.2 K
    ki = (T_K > 1.05e4) & (np.log10(T_K) <= 4.2)  # Koyama & Inutsuka branch
    lam[ki] = (2.0e-19 * np.exp(-1.184e5 / (T_K[ki] + 1.0e3))
               + 2.8e-28 * np.sqrt(T_K[ki]) * np.exp(-92.0 / T_K[ki]))
    lam[(T_K <= 1.05e4) | (T_K > 0.95e6)] = 0.0
    return lam


def primitives(frame, su: Setup) -> dict:
    """Derived cell fields of one frame (float64)."""
    frame.load(FIELDS)
    rho = frame["dens"].astype(np.float64)
    p = (su.gamma - 1) * frame["eint"].astype(np.float64)
    q = {"rho": rho, "p": p,
         "u": frame["velx"].astype(np.float64) - su.vx_c,
         "w": frame["vely"].astype(np.float64),
         "s": frame["velz"].astype(np.float64)}
    q["T"] = su.T_unit * p / rho
    q["cool"] = rho**2 * cool_lambda(q["T"]) / su.cool_unit
    return q


# ─── Pass 1: per-snapshot reduction ──────────────────────────────────────────

_SIM: SimulationData | None = None
_SU: Setup | None = None


def _init_worker(athinput: str, bindir: str):
    global _SIM, _SU
    _SIM = SimulationData(athinput, bindir)
    _SU = Setup.from_sim(_SIM)


def _reduce(i: int):
    frame = _SIM[i]
    q = primitives(frame, _SU)
    rho, u, w, s, p, T = (q[k] for k in ("rho", "u", "w", "s", "p", "T"))
    h = _SU.gamma / (_SU.gamma - 1) * p / rho
    K = 0.5 * (u * u + w * w + s * s)
    rhow = rho * w
    cells = {"rho": rho, "u": u, "w": w, "s": s, "p": p, "T": T, "h": h, "K": K,
             "u2": u * u, "w2": w * w, "s2": s * s, "rhou": rho * u, "rhow": rhow,
             "rhouw": rhow * u, "rhow2": rhow * w, "rhoK": rho * K,
             "rhowK": rhow * K, "rhoh": rho * h, "rhowh": rhow * h,
             "cool": q["cool"]}
    mom = np.stack([cells[k].mean(axis=(0, 2)) for k in MOMENTS])

    # Per-row histograms of log10 T, weighted by volume, mass, emissivity.
    nb = len(LOGT_EDGES) - 1
    nx2 = rho.shape[1]
    b = np.digitize(np.log10(T), LOGT_EDGES) - 1
    ok = (b >= 0) & (b < nb)
    idx = (np.arange(nx2)[None, :, None] * nb + b)[ok]
    hist = np.stack([np.bincount(idx, weights=wt[ok], minlength=nx2 * nb)
                     for wt in (np.ones_like(rho), rho, q["cool"])])
    t = frame.time
    frame.unload()
    return i, t, mom, hist.reshape(3, nx2, nb).astype(np.float32)


def run_paths(run: str) -> tuple[str, Path, Path]:
    """(name, athinput, bin folder) of a run folder name under DATA, or a path.

    The deck is <run>/<name>.athinput, else the PAR_DUMP the original fid3D_*
    runs keep in <run>/bin/header.txt.
    """
    root = (Path(run) if os.sep in run else DATA / run).resolve()
    # Name cache and plot files by the path below DATA, so runs in subfolders that
    # share a folder name (e.g. white_noise/kh3d_32x128x32) don't collide.
    name = (str(root.relative_to(DATA)).replace(os.sep, "__")
            if root.is_relative_to(DATA) else root.name)
    for deck in (root / f"{root.name}.athinput", root / "bin" / "header.txt"):
        if deck.exists():
            return name, deck, root / "bin"
    raise SystemExit(f"{root}: no {root.name}.athinput or bin/header.txt")


def reduce_run(run: str, workers: int) -> dict:
    run, athinput, bindir = run_paths(run)
    sim = SimulationData(athinput, bindir)
    numbers = np.array(sim.frame_numbers)
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{run}_moments.npz"

    old = {}
    if path.exists():
        with np.load(path) as z:
            if int(z["version"]) == CACHE_VERSION:
                old = {k: z[k] for k in z.files}
    have = set(old.get("numbers", []))
    todo = [i for i, n in enumerate(numbers) if n not in have]
    print(f"{run}: {len(sim)} snapshots, {len(have)} cached, reducing {len(todo)}")

    new = {}
    if todo:
        with ProcessPoolExecutor(min(workers, len(todo)), initializer=_init_worker,
                                 initargs=(str(athinput), str(bindir))) as ex:
            for k, (i, t, mom, hist) in enumerate(ex.map(_reduce, todo)):
                new[numbers[i]] = (t, mom, hist)
                if (k + 1) % 10 == 0 or k + 1 == len(todo):
                    print(f"  {k + 1}/{len(todo)}  t = {t:.3f}", flush=True)

    rows = {n: (t, m, hs) for n, t, m, hs in zip(
        old.get("numbers", []), old.get("time", []), old.get("mom", []),
        old.get("hist", []))}
    rows.update(new)
    order = sorted(rows)
    out = {"version": CACHE_VERSION, "numbers": np.array(order),
           "time": np.array([rows[n][0] for n in order]),
           "mom": np.stack([rows[n][1] for n in order]),
           "hist": np.stack([rows[n][2] for n in order]),
           "y": sim[0]["x2v"], "edges": LOGT_EDGES}
    if new:
        np.savez(path, **out)
        print(f"  cached {path}")
    return out


# ─── Pass 2: TRML-frame ensemble statistics ──────────────────────────────────

class Profiles:
    """Shifted per-snapshot moments in the TRML frame and their ensemble means."""

    def __init__(self, red: dict, su: Setup, tmin: float, tmax: float):
        self.su = su
        y, dy = red["y"], red["y"][1] - red["y"][0]
        tau = red["time"] / su.t0
        sel = (tau >= tmin) & (tau <= tmax)
        if sel.sum() < 2:
            raise SystemExit(f"fewer than 2 snapshots in {tmin}-{tmax} t0 "
                             f"(run spans {tau.min():.0f}-{tau.max():.0f} t0)")
        self.tau, t = tau[sel], red["time"][sel]
        mom = red["mom"][sel]                      # (nt, nmom, nx2)
        M = {k: mom[:, j] for j, k in enumerate(MOMENTS)}

        # Interface height of each snapshot and its linear drift.
        dT = su.Th_K - su.Tc_K
        f_hot = np.clip((M["T"] - su.Tc_K) / dT, 0, 1)
        y_top = y[-1] + dy / 2
        self.zc_all = y_top - f_hot.sum(axis=1) * dy
        v, a = np.polyfit(t, self.zc_all, 1)
        self.v = v
        zc = a + v * t

        hot = y > y[0] + 0.85 * (y[-1] - y[0])
        cold = y < y[0] + 0.15 * (y[-1] - y[0])
        rw, r = M["rhow"].mean(0), M["rho"].mean(0)
        self.v_flux = (rw[hot].mean() - rw[cold].mean()) / (r[hot].mean() - r[cold].mean())

        # Common zeta grid on which every snapshot has data.
        zeta = y - zc.mean()
        lo, hi = y[0] - zc.min(), y[-1] - zc.max()
        zeta = zeta[(zeta >= lo) & (zeta <= hi)]
        shifted = {k: np.array([np.interp(zeta + c, y, m) for c, m in zip(zc, M[k])])
                   for k in MOMENTS}

        # w -> w' = w - v
        S = shifted
        F = {k: S[k] for k in ("rho", "u", "s", "p", "T", "h", "u2", "s2", "rhou",
                               "rhoh", "cool")}
        F["w"] = S["w"] - v
        F["w2"] = S["w2"] - 2 * v * S["w"] + v * v
        F["rhow"] = S["rhow"] - v * S["rho"]
        F["rhouw"] = S["rhouw"] - v * S["rhou"]
        F["rhow2"] = S["rhow2"] - 2 * v * S["rhow"] + v * v * S["rho"]
        F["K"] = S["K"] - v * S["w"] + 0.5 * v * v
        F["rhoK"] = S["rhoK"] - v * S["rhow"] + 0.5 * v * v * S["rho"]
        F["rhowK"] = (S["rhowK"] - v * S["rhow2"] + 1.5 * v * v * S["rhow"]
                      - v * S["rhoK"] - 0.5 * v**3 * S["rho"])
        F["rhowh"] = S["rhowh"] - v * S["rhoh"]
        F["B"] = F["K"] + F["h"]
        F["rhowB"] = F["rhowK"] + F["rhowh"]

        # z = 0 where the time-averaged <T> crosses (Th + Tc)/2.
        Tm = F["T"].mean(0)
        mid = 0.5 * (su.Th_K + su.Tc_K)
        k = np.flatnonzero((Tm[:-1] < mid) & (Tm[1:] >= mid))[-1]
        z_mid = zeta[k] + (mid - Tm[k]) / (Tm[k + 1] - Tm[k]) * dy
        self.z = zeta - z_mid                         # code length
        self.zc_line = zc + z_mid                     # x2 of z = 0 at each snapshot
        self.fit_line = (a + z_mid, v)
        self.F, self.dz = F, dy
        self.mean = {k: a.mean(0) for k, a in F.items()}

        # tanh fit of <T> over 1.1 Tc < <T> < 0.9 Th, centre fixed at z = 0.
        Tbar = self.mean["T"]
        rng = (Tbar > 1.1 * su.Tc_K) & (Tbar < 0.9 * su.Th_K)
        (self.z0,), _ = curve_fit(lambda z, z0: self.tanh_T(z, z0), self.z[rng],
                                  Tbar[rng], p0=[su.L])
        self.slab = rng

    def tanh_T(self, z, z0=None):
        su = self.su
        z0 = self.z0 if z0 is None else z0
        return 0.5 * (su.Th_K - su.Tc_K) * np.tanh(z / z0) + 0.5 * (su.Th_K + su.Tc_K)

    def stat(self, fn, idx=None):
        """fn(moments) on the ensemble mean, and its 1-sigma spread over snapshots."""
        F = self.F if idx is None else {k: a[idx] for k, a in self.F.items()}
        mean = fn({k: a.mean(0) for k, a in F.items()})
        return mean, fn(F).std(0)

    def windows(self, n=3):
        edges = np.linspace(self.tau[0], self.tau[-1], n + 1)
        return [((self.tau >= lo) & (self.tau <= hi), lo, hi)
                for lo, hi in zip(edges[:-1], edges[1:])]

    def cumcool(self, D):
        return np.cumsum(D["cool"], axis=-1) * self.dz


# Flux decompositions, each a function of a moment dict (ensemble or per snapshot).
def mf_mean(D): return D["rho"] * D["w"]
def mf_turb(D): return D["rhow"] - D["rho"] * D["w"]
def mf_tot(D): return D["rhow"]
def xm_turb(D): return D["rhouw"] - D["u"] * D["rhow"]
def xm_adv(D): return D["rhow"] * D["u"]
def xm_tot(D): return D["rhouw"]
def zm_p(D): return D["p"]
def zm_adv(D): return D["rhow"] * D["w"]
def zm_turb(D): return D["rhow2"] - D["w"] * D["rhow"]
def zm_tot(D): return D["p"] + D["rhow2"]
def en_enth(D): return D["B"] * D["rhow"]
def en_turb(D): return D["rhowB"] - D["B"] * D["rhow"]


def rms(D, c):
    return np.sqrt(np.maximum(D[c + "2"] - D[c]**2, 0))


# ─── Plot helpers ────────────────────────────────────────────────────────────

def band(ax, x, mean, std, color, label=None, **kw):
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25, lw=0)
    ax.plot(x, mean, color=color, label=label, **kw)


def panel_label(ax, text):
    ax.text(0.03, 0.95, text, transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))


# ─── Fig. 1: slices at t ~ tsnap t0 ──────────────────────────────────────────

def fig1(run, sim, su, P: Profiles, tsnap, zlim, out):
    tau_all = np.array(sim.times) / su.t0
    i = int(np.argmin(abs(tau_all - tsnap)))
    frame = sim[i]
    q = primitives(frame, su)
    k3 = q["rho"].shape[0] // 2
    x, y = frame["x1v"], frame["x2v"]
    zc = P.fit_line[0] + P.fit_line[1] * frame.time
    X, Z = x / su.L, (y - zc) / su.L
    rows = (Z >= zlim[0]) & (Z <= zlim[1])
    Z = Z[rows]
    sl = {key: a[k3][rows] for key, a in q.items()}

    # This snapshot's horizontal means, for fluctuations and the gray overlays.
    hm = {key: a.mean(axis=(0, 2))[rows] for key, a in q.items()}
    du = [sl[c] - hm[c][:, None] for c in ("u", "w", "s")]
    dumag = np.sqrt(sum(d * d for d in du)) / su.du
    with np.errstate(divide="ignore"):
        tcool = sl["p"] / ((su.gamma - 1) * sl["cool"]) / su.t0
    tcool[~np.isfinite(tcool)] = np.nan

    zP = P.z / su.L
    m = P.mean
    D_snap = {key: a.mean(axis=(0, 2))[rows] for key, a in
              {"rho": q["rho"], "p": q["p"], "cool": q["cool"]}.items()}
    rms_t = {c: rms(m, c) / su.du for c in ("u", "s", "w")}
    rms_3d = np.sqrt(sum(r**2 for r in rms_t.values()))

    with np.errstate(divide="ignore"):
        tc_mean = m["p"] / ((su.gamma - 1) * m["cool"]) / su.t0
        tc_snap = D_snap["p"] / ((su.gamma - 1) * D_snap["cool"]) / su.t0

    fig, axs = plt.subplots(2, 2, figsize=(8.5, 13), constrained_layout=True)
    ext = [X[0], X[-1], Z[0], Z[-1]]
    imkw = dict(origin="lower", extent=ext, aspect="auto", interpolation="nearest")
    chi = su.rho_c / su.rho_h

    specs = [
        (axs[0, 0], sl["rho"] / su.rho_h, dict(cmap="inferno", vmin=0, vmax=1.1 * chi),
         r"$\rho/\rho_h$", [(m["rho"] / su.rho_h, zP, "C0"),
                            (D_snap["rho"] / su.rho_h, Z, "0.5")], False),
        (axs[0, 1], sl["p"] / su.p0, dict(cmap="inferno", vmin=0.5, vmax=1.5),
         r"$p/p_0$", [(m["p"] / su.p0, zP, "C0"), (D_snap["p"] / su.p0, Z, "0.5")], False),
        (axs[1, 0], tcool, dict(cmap="inferno", norm=LogNorm(1, 10**2.5)),
         r"$t_{\rm cool}/t_0$",
         [(tc_mean, zP, "C0"), (tc_snap, Z, "0.5")], True),
        (axs[1, 1], dumag, dict(cmap="inferno", norm=LogNorm(1e-2, 1.5)),
         r"$u_{\rm turb}/\Delta u$", [], True),
    ]
    for ax, img, ckw, label, overlays, logx in specs:
        im = ax.imshow(img, **imkw, **ckw)
        cb = fig.colorbar(im, ax=ax, location="bottom", pad=0.01, aspect=30)
        panel_label(ax, label)
        ax.set_xlabel(r"$x/\Delta u t_0$")
        ax.set_ylabel(r"$z/\Delta u t_0$")
        # Overlaid profiles use the colour bar below the panel as their x axis.
        ov = ax.twiny()
        ov.set_xlim(cb.norm.vmin, cb.norm.vmax)
        if logx:
            ov.set_xscale("log")
        for prof, zz, col in overlays:
            with np.errstate(divide="ignore", invalid="ignore"):
                ov.plot(prof, zz, color=col, lw=1.5)
        if ax is axs[1, 1]:
            for c, lab, col in (("u", "x", "C1"), ("s", "y", "y"), ("w", "z", "c")):
                ov.plot(rms_t[c], zP, "--", color=col, lw=1.2, label=lab)
            ov.plot(rms_3d, zP, color="w", lw=1.5, label="3D")
            ov.legend(loc="lower right", fontsize=8, facecolor="0.3", labelcolor="w")
        ov.set_xticks([])
        ov.minorticks_off()
        ov.set_ylim(Z[0], Z[-1])

    # Streamlines in the TRML frame on the density panel.
    ax = axs[0, 0]
    ax.streamplot(X, Z, sl["u"], sl["w"] - P.v, color="0.75", linewidth=0.5,
                  density=1.2, arrowsize=0.6)
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(Z[0], Z[-1])

    fig.suptitle(f"{run}: x-z slice at y = 0, t = {frame.time / su.t0:.0f} $t_0$; "
                 f"blue: <.> over {P.tau[0]:.0f}-{P.tau[-1]:.0f} $t_0$, "
                 "gray: this snapshot", fontsize=10)
    frame.unload()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ─── Fig. 2: mean profiles and conserved fluxes ──────────────────────────────

def fig2(run, su, P: Profiles, out):
    z = P.z / su.L
    m = P.mean
    rh_du, rh_du2, p0du = su.rho_h * su.du, su.rho_h * su.du**2, su.p0 * su.du
    fig, axs = plt.subplots(4, 2, figsize=(10, 13), sharex=True, constrained_layout=True)
    wins = P.windows()
    wcol = ["C0", "C3", "C2"]

    def windowed(ax, fn, norm, label):
        for (idx, lo, hi), col in zip(wins, wcol):
            ax.plot(z, P.stat(fn, idx)[0] / norm, color=col, lw=1,
                    label=rf"$({lo:.0f}-{hi:.0f}\,t_0)$")
        mean, std = P.stat(fn)
        band(ax, z, mean / norm, std / norm, "k", label=rf"$({P.tau[0]:.0f}-{P.tau[-1]:.0f}\,t_0)$")
        panel_label(ax, label)

    ax = axs[0, 0]
    windowed(ax, lambda D: D["rho"], su.rho_h, r"$\langle\rho\rangle/\rho_h$")
    ax = axs[1, 0]
    windowed(ax, lambda D: D["u"], su.du, r"$\langle u_x\rangle/\Delta u$")
    mean, std = P.stat(lambda D: D["rhou"] / D["rho"])
    ax.plot(z, mean / su.du, "k--", lw=1, label="mass-wtd")
    ax.legend(fontsize=7, loc="center right")
    ax = axs[2, 0]
    windowed(ax, lambda D: D["w"], su.du, r"$\langle u_z\rangle/\Delta u$ (TRML frame)")
    ax.legend(fontsize=7, loc="lower right")
    ax = axs[3, 0]
    windowed(ax, lambda D: D["T"], su.Th_K, r"$\langle T\rangle/T_h$ (vol-wtd)")
    ax.plot(z, P.tanh_T(P.z) / su.Th_K, ":", color="C1", lw=1.5,
            label=rf"tanh fit, $z_0/\Delta u t_0 = {P.z0 / su.L:.2f}$")
    ax.plot(z, m["B"] / (2.5 * su.Th_K / su.T_unit), "--", color="c", lw=1,
            label=r"$\langle\mathcal{B}\rangle\mu m_p/(2.5 k_B T_h)$")
    ax.legend(fontsize=7, loc="center right")

    def fluxes(ax, items, norm, label):
        for fn, col, lab in items:
            mean, std = P.stat(fn)
            band(ax, z, mean / norm, std / norm, col, label=lab)
        ax.axhline(0, color="0.6", lw=0.5)
        panel_label(ax, label)
        ax.legend(fontsize=7, loc="lower left")

    fluxes(axs[0, 1], [(mf_mean, "C3", r"$\langle\rho\rangle\langle u_z\rangle$"),
                       (mf_turb, "C0", r"$\langle\delta\rho\,\delta u_z\rangle$"),
                       (mf_tot, "k", r"$\langle\rho u_z\rangle$")],
           rh_du, r"mass flux $/\rho_h\Delta u$")
    fluxes(axs[1, 1], [(xm_turb, "C0", r"$\mathcal{R}_{xz}=\langle\delta u_x\,\delta(\rho u_z)\rangle$"),
                       (xm_adv, "C3", r"$\langle\rho u_z\rangle\langle u_x\rangle$"),
                       (xm_tot, "k", r"$\langle\rho u_x u_z\rangle$")],
           rh_du2, r"$x$-momentum flux $/\rho_h\Delta u^2$")
    fluxes(axs[2, 1], [(zm_p, "C0", r"$\langle p\rangle$"),
                       (zm_adv, "C2", r"$\langle\rho u_z\rangle\langle u_z\rangle$"),
                       (zm_turb, "C3", r"$\mathcal{R}_{zz}=\langle\delta u_z\,\delta(\rho u_z)\rangle$"),
                       (zm_tot, "k", r"$\langle p+\rho u_z^2\rangle$")],
           su.p0, r"$z$-momentum flux $/p_0$")
    fluxes(axs[3, 1], [(P.cumcool, "C0", r"$\int\langle n^2\Lambda\rangle dz$"),
                       (en_enth, "C2", r"$\langle\mathcal{B}\rangle\langle\rho u_z\rangle$"),
                       (en_turb, "C3", r"$\mathcal{Q}_t=\langle\delta\mathcal{B}\,\delta(\rho u_z)\rangle$"),
                       (lambda D: D["rhowB"] + P.cumcool(D), "k",
                        r"$\langle\varepsilon_{\rm net}\rangle$")],
           p0du, r"energy flux $/p_0\Delta u$")

    for ax in axs[-1]:
        ax.set_xlabel(r"$z/\Delta u t_0$")
    axs[0, 0].set_xlim(-20, 10)
    fig.suptitle(f"{run}: TRML-frame profiles, {P.tau[0]:.0f}-{P.tau[-1]:.0f} $t_0$ "
                 f"({len(P.tau)} snapshots)", fontsize=10)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ─── Fig. 3: temperature PDFs ────────────────────────────────────────────────

def fig3(run, red, su, P: Profiles, tmin, tmax, out):
    edges = red["edges"]
    lgT = 0.5 * (edges[1:] + edges[:-1])
    dlg = np.diff(edges)
    tau = red["time"] / su.t0
    sel = np.flatnonzero((tau >= tmin) & (tau <= tmax))
    y = red["y"]
    lo, hi = np.log10(1.1 * su.Tc_K), np.log10(0.9 * su.Th_K)
    norm_bins = (lgT > lo) & (lgT < hi)

    def normalise(p):
        return p / (p[..., norm_bins] * dlg[norm_bins]).sum(-1, keepdims=True) \
            if p.ndim > 1 else p / (p[norm_bins] * dlg[norm_bins]).sum()

    # Rows inside the TRML slab, where 1.1 Tc < <T>(z) < 0.9 Th (time-averaged).
    Tbar = P.mean["T"]
    per_snap = []
    for j, zc in zip(sel, P.zc_line):
        Trow = np.interp(y - zc, P.z, Tbar, left=np.nan, right=np.nan)
        rows = (Trow > 1.1 * su.Tc_K) & (Trow < 0.9 * su.Th_K)
        per_snap.append(red["hist"][j][:, rows].sum(1) / dlg)  # (3, nb)
    per_snap = np.array(per_snap, dtype=float)
    pdf = normalise(per_snap.sum(0))
    spread = normalise(per_snap).std(0)

    # Emissivity PDF across the horizontally averaged temperature.
    def pe_mean_T(T_prof, c_prof, n_up=20):
        zf = np.linspace(P.z[0], P.z[-1], n_up * len(P.z))
        Tf, cf = np.interp(zf, P.z, T_prof), np.interp(zf, P.z, c_prof)
        h, _ = np.histogram(np.log10(Tf), edges, weights=cf)
        return normalise(h / dlg)
    # No band: a single snapshot's <T>(z) is not monotonic, so its version is noise.
    pe_bar = pe_mean_T(P.mean["T"], P.mean["cool"])

    # Analytic PDFs from the tanh fit (Eq. 7).
    T = 10**lgT
    inside = (T > su.Tc_K) & (T < su.Th_K)
    dT = su.Th_K - su.Tc_K
    Tp = dT / (2 * P.z0) * (1 - 4 * ((T - 0.5 * (su.Th_K + su.Tc_K)) / dT) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        pv = np.where(inside, T * np.log(10) / Tp, np.nan)
    model = {"V": normalise(pv), "M": normalise(pv / T),
             "E": normalise(pv * cool_lambda(T) / T**2)}
    flat = normalise(np.where(inside, T, np.nan))

    fig, ax = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    cols = {"V": "orange", "M": "C0", "E": "C2"}
    for w, key in enumerate("VME"):
        band(ax, lgT, pdf[w], spread[w], cols[key], label=rf"$\mathcal{{P}}_{key}$")
        ax.plot(lgT, model[key], "--", color=cols[key], lw=1.2)
    ax.plot(lgT, pe_bar, color="purple", label=r"$\overline{\mathcal{P}}_E$ (vs $\langle T\rangle$)")
    ax.plot(lgT, flat, "--", color="r", lw=1, label=r"$\mathcal{P}_E=$ const")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 2e1)
    ax.set_xlim(4.0, 6.0)
    ax.set_xlabel(r"$\log_{10} T$")
    ax.set_ylabel(r"$\mathcal{P}(\log_{10} T)$")
    ax.grid(which="both", lw=0.3, alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(f"{run}: PDFs in the TRML ({P.tau[0]:.0f}-{P.tau[-1]:.0f} $t_0$); "
                 "dashed: Eq. 7 with the tanh fit", fontsize=10)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")


# ─── Fig. 4: energy balance ──────────────────────────────────────────────────

def fig4(run, su, P: Profiles, out, smooth=1.0):
    z = P.z / su.L
    m = P.mean
    unit = su.p0 / su.t0
    zfit = P.tanh_T(P.z) / su.T_unit             # <T> in code (= kT/mu m_p)
    dTdz = np.gradient(zfit, P.dz)

    # 90 snapshots leave acoustic noise in <rho u_z>(z) that the split between the
    # enthalpy and turbulent fluxes amplifies; smooth by `smooth` du t0 before d/dz.
    sig = smooth * su.L / P.dz

    def sm(a):
        return gaussian_filter1d(a, sig, mode="nearest") if sig > 0 else a

    cool = sm(m["cool"])
    enth = np.gradient(sm(en_enth(m)), P.dz)
    turb = np.gradient(sm(en_turb(m)), P.dz)

    # Theory: B = (5/2) kT/mu m_p with the tanh <T>, Q_t from Eq. 9.
    sigma_cool = cool.sum() * P.dz
    mdot = -m["rhow"][P.slab].mean()
    enth_th = -mdot * su.gamma / (su.gamma - 1) * dTdz
    CE = P.cumcool(m) / sigma_cool
    Qt_th = sigma_cool * ((P.tanh_T(P.z) - su.Tc_K) / (su.Th_K - su.Tc_K) - CE)
    turb_th = np.gradient(Qt_th, P.dz)
    # Closure Q_t = -kappa_t d<T>/dz: the paper's kappa_t, and a fit to Q_t in the slab.
    kappa_unit = su.rho_h * su.du**2 * su.t0
    Qt = sm(en_turb(m))
    kappa_fit = -(Qt * dTdz)[P.slab].sum() / (dTdz**2)[P.slab].sum() / kappa_unit
    d2Tdz2 = np.gradient(dTdz, P.dz)

    fig, ax = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    ax.plot(z, cool / unit, color="C0", label=r"$\langle n^2\Lambda(T)\rangle/(p_0/t_0)$")
    ax.plot(z, enth / unit, color="orange",
            label=r"$\frac{1}{p_0/t_0}\frac{d\langle\mathcal{B}\rangle\langle\rho u_z\rangle}{dz}$")
    ax.plot(z, enth_th / unit, "--", color="orange")
    ax.plot(z, turb / unit, color="C2",
            label=r"$\frac{1}{p_0/t_0}\frac{d\langle\delta\mathcal{B}\,\delta\rho u_z\rangle}{dz}$")
    ax.plot(z, turb_th / unit, "--", color="C2")
    for coef, col in ((0.1, "r"), (kappa_fit, "m")):
        ax.plot(z, -coef * kappa_unit * d2Tdz2 / unit, ":", color=col, lw=1.5,
                label=rf"$-\kappa_t\,d^2\langle T\rangle/dz^2$, $\kappa_t = {coef:.3g}$")
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlim(-8, 11)
    ax.set_xlabel(r"$z/\Delta u t_0$")
    ax.grid(lw=0.3, alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")

    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    ticks = ax.get_xticks()
    ticks = ticks[(ticks >= ax.get_xlim()[0]) & (ticks <= ax.get_xlim()[1])]
    top.set_xticks(ticks)
    top.set_xticklabels([f"{np.interp(t, z, m['T']) / 1e5:.2f}" for t in ticks])
    top.set_xlabel(r"$\langle T\rangle\ (10^5\,{\rm K})$")
    smooth_txt = f", smoothed over {smooth:g} $\\Delta u t_0$" if smooth > 0 else ""
    fig.suptitle(f"{run}: energy balance ({P.tau[0]:.0f}-{P.tau[-1]:.0f} $t_0${smooth_txt})\n"
                 "dashed: model (Eqs. 6, 9); $\\kappa_t$ in $k_B\\rho_h\\Delta u^2 t_0/\\mu m_p$ "
                 "(0.1: paper, other: least-squares fit to $\\mathcal{Q}_t$)", fontsize=9)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  saved {out}")
    return sigma_cool, mdot, kappa_fit


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*", default=["kh3d_128x512x128"],
                    help="run folders under /data/sasi/simulation_outputs, or paths "
                         "(e.g. /data/sasi/fid3D_128_cool)")
    ap.add_argument("--tmin", type=float, default=146, help="window start [t0]")
    ap.add_argument("--tmax", type=float, default=np.inf, help="window end [t0]")
    ap.add_argument("--tsnap", type=float, default=350, help="Fig. 1 snapshot time [t0]")
    ap.add_argument("--zlim", type=float, nargs=2, default=(-21, 11),
                    help="Fig. 1 z range [du t0]")
    ap.add_argument("--smooth", type=float, default=1.0,
                    help="Fig. 4 Gaussian smoothing of the fluxes [du t0], 0 for none")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                    help="processes for the reduction pass")
    args = ap.parse_args()

    PLOTS.mkdir(exist_ok=True)
    for run in args.runs:
        path = run
        run, athinput, bindir = run_paths(path)
        sim = SimulationData(athinput, bindir)
        su = Setup.from_sim(sim)
        print(f"{run}: Tc = {su.Tc_K:.3g} K, Th = {su.Th_K:.3g} K, "
              f"t0 = {su.t0:.4g} Myr, du t0 = {su.L:.4g} pc, "
              f"t_end = {sim.times[-1] / su.t0:.0f} t0")
        red = reduce_run(path, args.workers)
        P = Profiles(red, su, args.tmin, args.tmax)
        print(f"  window {P.tau[0]:.0f}-{P.tau[-1]:.0f} t0 ({len(P.tau)} snapshots)")
        print(f"  v_TRML: fit {P.v / su.du:+.4f} du, flux formula {P.v_flux / su.du:+.4f} du")
        print(f"  tanh fit z0 = {P.z0 / su.L:.3f} du t0 (paper: 2.16)")

        fig1(run, sim, su, P, args.tsnap, args.zlim, PLOTS / f"fig1_slices_{run}.png")
        fig2(run, su, P, PLOTS / f"fig2_profiles_{run}.png")
        fig3(run, red, su, P, P.tau[0], P.tau[-1], PLOTS / f"fig3_pdfs_{run}.png")
        sigma_cool, mdot, kappa_fit = fig4(run, su, P, PLOTS / f"fig4_energy_{run}.png",
                                          args.smooth)
        print(f"  Sigma_cool = {sigma_cool / (su.p0 * su.du):.4g} p0 du, "
              f"Sigma_M = -<rho u_z> = {mdot / (su.rho_h * su.du):.4g} rho_h du, "
              f"fitted kappa_t = {kappa_fit:.3g} k_B rho_h du^2 t0/mu m_p (paper: 0.1)")


if __name__ == "__main__":
    main()
