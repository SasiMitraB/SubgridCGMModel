"""
Resolution convergence of the five 3D radiative KH runs against the resolution
study of Sharma et al. (2025), Table I of their Supplemental Material.

For each run, over the paper's steady-state window (t >= 146 t0), in the TRML
frame of theory/sharma+2025/recreating_plots.py:

  Sigma_cool = int <n^2 Lambda> dz          cooling rate per unit area  [p0 du]
  z0         tanh fit to the <T> profile    TRML thickness              [du t0]
  u_{z,h}    <u_z> in the hot phase         inflow velocity             [du]

and the peak turbulent correlations of Table I within the TRML (where the mean
<T> is between 1.1 Tc and 0.9 Th): max <drho du_z> [rho_h du], min R_zx and
max R_zz [rho_h du^2], min Q_t [p0 du].

Points are the ensemble (time and horizontal) values of recreating_plots.py and
error bars the 1-sigma spread of the per-snapshot values, as in the paper. The
hot phase is where the mean <T> > 0.99 Th; there rho = rho_h, so -u_{z,h} is
also the entrainment rate Sigma_M in rho_h du. The turbulent peaks are the mean
and spread of each snapshot's extreme.

The paper's runs are placed at Delta = 2^alpha Delta_fid (their Fig. S6), with
Delta_fid = 10 pc / 280.

Reads only the per-snapshot reductions cached by recreating_plots.py (run that
first for any run missing from theory/sharma+2025/cache).

Figure (Overleaf/images/chapter2/): trml_convergence.pdf

    python theory/notes/plotting_scripts/trml_convergence.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "theory" / "sharma+2025"))
from recreating_plots import (Profiles, Setup, en_turb, mf_turb, reduce_run,  # noqa: E402
                              run_paths, xm_turb, zm_turb)
from vis_athenak.simulation_data import SimulationData  # noqa: E402

import notes_style as ns  # noqa: E402

RUNS = [16, 32, 64, 128, 256]   # kh3d_{N}x{4N}x{N}
L_X = 10.0                      # pc
TMIN = 146                      # t0, start of the paper's averaging window
DX_FID = L_X / 280              # the paper's fiducial cell

# Sharma et al. (2025), SM Table I [varying resolution]: alpha (Delta = 2^alpha
# Delta_fid), then (value, 1-sigma) of Sigma_cool [p0 du], z0 [du t0], u_{z,h} [du].
PAPER = [
    (-1, (1.39, 0.17), (2.15, 0.29), (-0.56, 0.08)),  # res-2x
    (0,  (1.43, 0.21), (2.17, 0.46), (-0.57, 0.09)),  # fiducial, 280 x 280 x 1040
    (1,  (1.39, 0.18), (2.12, 0.43), (-0.56, 0.07)),  # res-1/2x
    (1,  (1.36, 0.18), (2.10, 0.47), (-0.55, 0.08)),  # res~1/2x, L_perp-1x
    (2,  (1.21, 0.14), (1.81, 0.40), (-0.49, 0.06)),  # res-1/4x
    (3,  (1.25, 0.19), (2.23, 0.82), (-0.51, 0.08)),  # res-1/8x, 35 x 35 x 130
    (np.log2((L_X / 32) / DX_FID),
         (1.18, 0.18), (1.84, 0.87), (-0.48, 0.07)),  # 32 x 32 x 128
    (4,  (0.97, 0.54), (0.84, 0.64), (-0.40, 0.21)),  # res~1/16x
    (5,  (1.09, 0.42), (0.78, 0.73), (-0.45, 0.19)),  # res~1/32x
    (6,  (1.93, 0.89), (3.09, 3.76), (-0.80, 0.38)),  # res~1/64x
]


def measure(N):
    """Ensemble value and 1-sigma spread over snapshots of each quantity."""
    run, athinput, bindir = run_paths(f"kh3d_{N}x{4 * N}x{N}")
    su = Setup.from_sim(SimulationData(athinput, bindir))
    P = Profiles(reduce_run(run, workers=1), su, TMIN, np.inf)
    F, m, s = P.F, P.mean, P.slab
    out = {}

    cool = F["cool"].sum(-1) * P.dz / (su.p0 * su.du)
    out["Sigma_cool"] = (m["cool"].sum() * P.dz / (su.p0 * su.du), cool.std())

    def z0_of(T):  # over the ensemble's fit range: a coarse snapshot can have none of its own
        (z0,), _ = curve_fit(lambda z, z0: P.tanh_T(z, z0), P.z[s], T[s], p0=[P.z0])
        return z0 / su.L
    out["z0"] = (P.z0 / su.L, np.std([z0_of(T) for T in F["T"]]))

    hot = m["T"] > 0.99 * su.Th_K
    out["u_zh"] = (m["w"][hot].mean() / su.du, (F["w"][:, hot].mean(-1) / su.du).std())

    for key, fn, ext, unit in [
            ("max drho du_z", mf_turb, np.max, su.rho_h * su.du),
            ("min R_zx", xm_turb, np.min, su.rho_h * su.du**2),
            ("max R_zz", zm_turb, np.max, su.rho_h * su.du**2),
            ("min Q_t", en_turb, np.min, su.p0 * su.du)]:
        per_snap = ext(fn(F)[:, s], axis=-1) / unit
        out[key] = (per_snap.mean(), per_snap.std())
    return out


def main():
    ours = {N: measure(N) for N in RUNS}
    for N, d in ours.items():
        print(f"{N:4d}: " + ", ".join(f"{k} = {v:+.2f} +- {e:.2f}" for k, (v, e) in d.items()))

    keys = ["Sigma_cool", "z0", "u_zh"]
    labels = [r"$\dot\Sigma_{\rm cool}/(p_0\Delta u)$",
              r"$z_0/(\Delta u\,t_0)$",
              r"$-u_{z,h}/\Delta u$"]
    sign = [1, 1, -1]
    fig, axes = plt.subplots(3, 1, figsize=ns.figsize(12, ns.WIDE), sharex=True, layout="constrained")
    a_ours = np.array([np.log2((L_X / N) / DX_FID) for N in RUNS])
    a_p = np.array([row[0] for row in PAPER])
    for i, (ax, key, label) in enumerate(zip(axes, keys, labels)):
        v_p = np.array([row[1 + i] for row in PAPER])
        ax.errorbar(a_p - 0.06, sign[i] * v_p[:, 0], v_p[:, 1], fmt="o", ms=3.5,
                    color=ns.SERIES[0], mfc="none", elinewidth=0.8, capsize=0,
                    label="Sharma et al. (2025)")
        v_o = np.array([ours[N][key] for N in RUNS])
        ax.errorbar(a_ours + 0.06, sign[i] * v_o[:, 0], v_o[:, 1], fmt="o", ms=3.5,
                    color=ns.ACCENT, elinewidth=0.8, capsize=0, label="these runs")
        ax.axhline(sign[i] * PAPER[1][1 + i][0], color=ns.HAIRLINE, ls=(0, (4, 2)), lw=0.8)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="lower left")
    axes[-1].set_xlabel(r"$\log_2(\Delta/\Delta_{\rm fid})$,  $\Delta_{\rm fid} = 10\,{\rm pc}/280$")
    top = axes[0].secondary_xaxis("top", functions=(lambda a: a, lambda a: a))
    top.set_xticks(a_ours, [rf"${N}$" for N in RUNS])
    top.set_xlabel(r"$N_x$ of these runs", fontsize=9, color=ns.MUTED)
    ns.save(fig, ns.image_path(2, "trml_convergence.pdf"))


if __name__ == "__main__":
    main()
