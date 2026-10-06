#!/usr/bin/env python3
"""
gate_evaluation.py
===============================================================================
How well does the CNN gate identify the coarse cells where cooling happens?

The gate g in (0, 1) switches the predicted temperature PDF between a single
sharp peak (g -> 0) and a broad multiphase distribution (g -> 1). It is trained
(pdf_cnn.train_gate_branch) against the binary target "the cell's HR temperature
PDF has mass in the active cooling window", which with the same window is the
same as <n^2 Lambda(T)> > 0. Here we ignore the CNN's predicted cooling and only
compare the gate against that target and against the subgrid cooling term C.

The two models of shell_scripts/run_variable_box_subgrid.sh are evaluated on
the 512x1024 HR run (10 x 20 pc), coarse-grained to their own cell size:
  0.625 pc : run_random_crop_20260929_111014, factor 32 -> 32x16 cells (1 tile)
  1.25 pc  : run_random_crop_20260924_194824, factor 64 -> 16x8 cells  (1 tile)

It also compares the gate with the subgrid energy flux Q (@eq-energy-flux); in 2D
v_z = 0, so Q_z = 0 and only Q_x, Q_y are used.

Outputs: assets/gate_vs_cooling.png, assets/gate_vs_divQ.png, assets/gate_Q_animation.mp4
===============================================================================
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.colors import LogNorm, SymLogNorm
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "models" / "conv_nn"))

import ergane
from data.coarse_grain_utils import block_mean, cnn_input_fields, compute_subgrid_cooling
from pdf_cnn import (ConvNN, device, in_channels, layer_size1, layer_size2, layer_size3,
                     layer_size4, out_channels)
from cooling_cg_comparison import HR_DIR, LOGT_ON, OUT_DIR, lam

MODELS = [
    {"label": "0.625 pc model", "b": 32,
     "dir": PROJECT_ROOT / "runs/run_random_crop_20260929_111014/model_saves"},
    {"label": "1.25 pc model", "b": 64,
     "dir": PROJECT_ROOT / "runs/run_random_crop_20260924_194824/model_saves"},
]
NORM_PREFIX = "cnn_(512, 256)_32"  # default of run_variable_box_subgrid.sh for both models
GATE_ON = 0.5                      # threshold used for the gate accuracy during training
GAMMA = 5.0 / 3.0
# Code unit of energy flux: pressure unit x velocity unit (1 pc, 1 Myr, rho unit = m_H)
FLUX_UNIT = 1.59916e-14 * (3.08568e18 / 3.15576e13)  # erg cm^-2 s^-1


# Code unit of energy per volume per time (same units as C): pressure unit / time unit
DIV_UNIT = 1.59916e-14 / 3.15576e13  # erg cm^-3 s^-1


def divergence(qx, qy, dx, dy):
    """Centred-difference div Q on the coarse grid (code units).

    x is periodic in the KH runs; in y the edges use one-sided differences.
    """
    dqx = (np.roll(qx, -1, axis=1) - np.roll(qx, 1, axis=1)) / (2.0 * dx)
    dqy = np.gradient(qy, dy, axis=0)
    return dqx + dqy


def subgrid_energy_flux(rho, P, vx, vy, b):
    """Q_i = <(E+P) v_i> - (<E> + <P>) <rho v_i> / <rho>  (@eq-energy-flux), code units.

    In 2D v_z = 0, so Q_z = 0 identically and only Q_x, Q_y are returned.
    """
    E = P / (GAMMA - 1.0) + 0.5 * rho * (vx**2 + vy**2)
    rho_c = block_mean(rho, b)
    H_c = block_mean(E, b) + block_mean(P, b)
    return [block_mean((E + P) * v, b) - H_c * block_mean(rho * v, b) / rho_c for v in (vx, vy)]


def load_model(save_dir):
    """Same loading as explore_data/mock_sg_tiled.py:load_tiled_cnn_model."""
    mean = np.load(save_dir / f"{NORM_PREFIX}_input_mean.npy")
    std = np.load(save_dir / f"{NORM_PREFIX}_input_std.npy")
    state = torch.load(save_dir / f"{NORM_PREFIX}.pth", map_location=device)
    ksize = state["encoder.0.weight"].shape[-1]
    model = ConvNN(in_channels, layer_size1, layer_size2, layer_size3, layer_size4,
                   out_channels, ksize).to(device)
    model.load_state_dict(state)
    model.eval()
    to_t = lambda a: torch.tensor(a, dtype=torch.float32, device=device).view(1, -1, 1, 1)
    return model, to_t(mean), to_t(std)


def evaluate(sim, frames, spec):
    """Gate, <n^2 Lambda> and C for every coarse cell of the given frames."""
    model, mean, std = load_model(spec["dir"])
    b = spec["b"]
    gate, true, C, Tc, Qx, Qy, divQ, maps = [], [], [], [], [], [], [], {}
    for n in frames:
        f = sim.get_frame(n)
        rho = f.density / f.units.density
        P = f.pressure / f.units.pressure
        vx = f.velx / f.units.velocity
        vy = f.vely / f.units.velocity
        s0 = f.scalars["scalar_00"]

        x = cnn_input_fields(rho, vx, vy, P, s0, b)  # (rho, T_CG, ux, uy, s0)
        with torch.no_grad():
            _, g = model((torch.from_numpy(x)[None].to(device) - mean) / std)
        g = g[0, 0].cpu().numpy()

        cool = compute_subgrid_cooling(rho, f.temperature, x[0], x[1], lam, b=b)
        gate.append(g.ravel())
        true.append(cool["cool_true"].ravel())
        C.append(cool["tau_lambda"].ravel())
        Tc.append(x[1].ravel())
        qx, qy = subgrid_energy_flux(rho, P, vx, vy, b)
        dq = divergence(qx, qy, (sim.x1max - sim.x1min) / qx.shape[1], (sim.x2max - sim.x2min) / qx.shape[0])
        Qx.append(qx.ravel())
        Qy.append(qy.ravel())
        divQ.append(dq.ravel())
        maps[n] = {"gate": g, "C": cool["tau_lambda"], "true": cool["cool_true"], "Qx": qx, "Qy": qy,
                   "divQ": dq}
    return {"gate": np.concatenate(gate), "true": np.concatenate(true),
            "C": np.concatenate(C), "T_c": np.concatenate(Tc),
            "Qx": np.concatenate(Qx), "Qy": np.concatenate(Qy), "divQ": np.concatenate(divQ),
            "maps": maps}


def scores(r):
    cooling = r["true"] > 0
    on = r["gate"] > GATE_ON
    tp, fp = (on & cooling).sum(), (on & ~cooling).sum()
    fn, tn = (~on & cooling).sum(), (~on & ~cooling).sum()
    return {
        "accuracy": (tp + tn) / r["gate"].size,
        "recall": tp / max(tp + fn, 1),          # cooling cells the gate turns on
        "precision": tp / max(tp + fp, 1),       # gate-on cells that are cooling
        "missed_cooling": r["true"][~on & cooling].sum() / r["true"].sum(),
        "missed_absC": np.abs(r["C"][~on]).sum() / np.abs(r["C"]).sum(),
        "fp": int(fp), "fn": int(fn), "n_cooling": int(cooling.sum()), "n": r["gate"].size,
        "fn_frac_C_pos": (r["C"][~on & cooling] > 0).mean(),
        "fn_median_logTc": np.median(np.log10(r["T_c"][~on & cooling])),
        "fn_frac_Tc_below": (np.log10(r["T_c"][~on & cooling]) < LOGT_ON).mean(),
        "overcool_gate_on": (on & (r["C"] < 0)).sum() / max((r["C"] < 0).sum(), 1),
        "undercool_gate_on": (on & (r["C"] > 0)).sum() / max((r["C"] > 0).sum(), 1),
    }


def plot(sim, results, snap, t_range, n_frames):
    fig, axes = plt.subplots(len(results), 3, figsize=(13, 4.6 * len(results)),
                             gridspec_kw={"width_ratios": [0.55, 1, 1]}, constrained_layout=True)
    extent = [sim.x1min, sim.x1max, sim.x2min, sim.x2max]

    for row, (spec, r, s) in zip(axes, results):
        ax_map, ax_hist, ax_joint = row

        # (a) gate map at the snapshot, outlined where cooling actually happens
        m = r["maps"][snap]
        im = ax_map.imshow(m["gate"], origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=1)
        up = 16  # upsample the mask so the outline follows the cell edges
        mask = np.kron((m["true"] > 0).astype(float), np.ones((up, up)))
        ny, nx = mask.shape
        xc = sim.x1min + (np.arange(nx) + 0.5) * (sim.x1max - sim.x1min) / nx
        yc = sim.x2min + (np.arange(ny) + 0.5) * (sim.x2max - sim.x2min) / ny
        ax_map.contour(xc, yc, mask, levels=[0.5], colors="#b40426", linewidths=1.5)
        fig.colorbar(im, ax=ax_map, location="bottom", label="gate")
        ax_map.set_title(f"{spec['label']}: gate at t = {sim.times[sim.frame_numbers.index(snap)]:.0f} Myr\n"
                         r"(red outline: $\langle n^2\Lambda\rangle > 0$)", fontsize=10)
        ax_map.set_xlabel("x [pc]")
        ax_map.set_ylabel("y [pc]")

        # (b) gate distribution for cooling vs non-cooling cells
        cooling = r["true"] > 0
        bins = np.linspace(0, 1, 41)
        ax_hist.hist(r["gate"][cooling], bins=bins, histtype="step", lw=2, color="#b40426",
                     label=rf"$\langle n^2\Lambda\rangle > 0$ ({cooling.sum()} cells)")
        ax_hist.hist(r["gate"][~cooling], bins=bins, histtype="step", lw=2, color="#555555",
                     label=rf"$\langle n^2\Lambda\rangle = 0$ ({(~cooling).sum()} cells)")
        ax_hist.axvline(GATE_ON, ymax=0.62, color="#333333", lw=1, ls="--")
        ax_hist.set_yscale("log")
        ax_hist.set_xlabel("gate")
        ax_hist.set_ylabel("coarse cells (all frames)")
        ax_hist.set_title(f"Gate distribution, t = {t_range[0]:.0f}–{t_range[1]:.0f} Myr ({n_frames} frames)",
                          fontsize=10)
        ax_hist.set_ylim(top=ax_hist.get_ylim()[1] * 30)  # room for the legend and scores
        ax_hist.legend(loc="upper center", frameon=False, fontsize=9)
        ax_hist.text(0.5, 0.80,
                     f"gate > {GATE_ON}: recall {100 * s['recall']:.1f}%, precision {100 * s['precision']:.1f}%\n"
                     f"cooling in gate-off cells: {100 * s['missed_cooling']:.2f}% of total",
                     transform=ax_hist.transAxes, ha="center", va="top", fontsize=9, color="#333333")

        # (c) gate against |C|, split by sign
        nz = r["C"] != 0
        logabs = np.log10(np.abs(r["C"][nz]))
        xb = np.linspace(np.floor(logabs.min()), np.ceil(logabs.max()), 50)
        for sign, cmap in [(1, "Reds"), (-1, "Blues")]:  # blue drawn on top
            sel = np.sign(r["C"][nz]) == sign
            h, _, _ = np.histogram2d(logabs[sel], r["gate"][nz][sel], bins=[xb, bins])
            ax_joint.pcolormesh(xb, bins, np.ma.masked_equal(h.T, 0), cmap=cmap, norm=LogNorm(1, None))
        ax_joint.axhline(GATE_ON, color="#333333", lw=1, ls="--")
        ax_joint.set_xlabel(r"$\log_{10}|\mathsf{C}|$ [erg cm$^{-3}$ s$^{-1}$]")
        ax_joint.set_ylabel("gate")
        ax_joint.set_title(r"Gate against $|\mathsf{C}|$ (red: $\mathsf{C} > 0$, blue: $\mathsf{C} < 0$)",
                           fontsize=10)
        ax_joint.set_ylim(0, 1)

    out = OUT_DIR / "gate_vs_cooling.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def next_to_gate_on(gate):
    """Gate-off cells that touch a gate-on cell (8-neighbourhood, periodic in x)."""
    on = gate > GATE_ON
    grown = on.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted = np.roll(on, dx, axis=1)
            if dy == 1:
                shifted = np.vstack([np.zeros_like(shifted[:1]), shifted[:-1]])
            elif dy == -1:
                shifted = np.vstack([shifted[1:], np.zeros_like(shifted[:1])])
            grown |= shifted
    return grown & ~on


def plot_divQ(sim, results, snap, t_range, n_frames):
    """Gate against the divergence of the subgrid energy flux (Q_z = 0 in 2D)."""
    fig, axes = plt.subplots(len(results), 3, figsize=(13, 4.6 * len(results)),
                             gridspec_kw={"width_ratios": [0.55, 1, 1]}, constrained_layout=True)
    extent = [sim.x1min, sim.x1max, sim.x2min, sim.x2max]
    div_all = [r["divQ"] * DIV_UNIT for _, r, _ in results]
    absdiv = np.concatenate([np.abs(d[d != 0]) for d in div_all])
    hi = np.ceil(np.log10(absdiv.max()))
    lo = np.floor(np.log10(np.percentile(absdiv, 0.1)))
    top = 10.0 ** np.floor(np.log10(absdiv.max()))

    for row, (spec, r, _), div in zip(axes, results, div_all):
        ax_map, ax_hist, ax_joint = row
        on = r["gate"] > GATE_ON

        # (a) div Q map at the snapshot, outlined where the gate is on
        m = r["maps"][snap]
        im = ax_map.imshow(m["divQ"] * DIV_UNIT, origin="lower", extent=extent, cmap="coolwarm",
                           norm=SymLogNorm(linthresh=top * 1e-2, vmin=-absdiv.max(), vmax=absdiv.max(), base=10))
        up = 16
        mask = np.kron((m["gate"] > GATE_ON).astype(float), np.ones((up, up)))
        ny, nx = mask.shape
        xc = sim.x1min + (np.arange(nx) + 0.5) * (sim.x1max - sim.x1min) / nx
        yc = sim.x2min + (np.arange(ny) + 0.5) * (sim.x2max - sim.x2min) / ny
        ax_map.contour(xc, yc, mask, levels=[0.5], colors="#222222", linewidths=1.5)
        fig.colorbar(im, ax=ax_map, location="bottom", ticks=[-top, 0, top],
                     label=r"$\nabla\cdot\mathbf{Q}$ [erg cm$^{-3}$ s$^{-1}$]")
        ax_map.set_title(f"{spec['label']}: $\\nabla\\cdot\\mathbf{{Q}}$ at t = "
                         f"{sim.times[sim.frame_numbers.index(snap)]:.0f} Myr\n(black outline: gate > {GATE_ON})",
                         fontsize=10)
        ax_map.set_xlabel("x [pc]")
        ax_map.set_ylabel("y [pc]")

        # (b) distribution of |div Q| for gate-on cells, their gate-off neighbours, and the rest
        adj = np.concatenate([next_to_gate_on(mm["gate"]).ravel() for mm in r["maps"].values()])
        rest = ~on & ~adj
        logd = np.log10(np.maximum(np.abs(div), 10.0 ** lo))
        bins = np.linspace(lo, hi, 51)
        absd = np.abs(div)
        for sel, color, name in [(on, "#1f5fa8", f"gate > {GATE_ON}"),
                                 (adj, "#e08214", "gate-off, next to gate-on"),
                                 (rest, "#555555", "other gate-off")]:
            ax_hist.hist(logd[sel], bins=bins, histtype="step", lw=2, color=color,
                         label=f"{name}: {100 * absd[sel].sum() / absd.sum():.1f}% of "
                               r"$\sum|\nabla\cdot\mathbf{Q}|$")
        ax_hist.set_yscale("log")
        ax_hist.set_xlabel(r"$\log_{10}|\nabla\cdot\mathbf{Q}|$ [erg cm$^{-3}$ s$^{-1}$]")
        ax_hist.set_ylabel("coarse cells (all frames)")
        ax_hist.set_title(f"$|\\nabla\\cdot\\mathbf{{Q}}|$ distribution, t = {t_range[0]:.0f}–{t_range[1]:.0f} Myr "
                          f"({n_frames} frames)", fontsize=10)
        ax_hist.set_ylim(top=ax_hist.get_ylim()[1] * 30)
        ax_hist.legend(loc="upper left", frameon=False, fontsize=9)

        # (c) gate against |div Q|
        gb = np.linspace(0, 1, 41)
        h, _, _ = np.histogram2d(logd, r["gate"], bins=[bins, gb])
        im = ax_joint.pcolormesh(bins, gb, np.ma.masked_equal(h.T, 0), cmap="Blues", norm=LogNorm(1, None))
        fig.colorbar(im, ax=ax_joint, label="coarse cells")
        ax_joint.axhline(GATE_ON, color="#333333", lw=1, ls="--")
        ax_joint.set_xlabel(r"$\log_{10}|\nabla\cdot\mathbf{Q}|$ [erg cm$^{-3}$ s$^{-1}$]")
        ax_joint.set_ylabel("gate")
        ax_joint.set_title(r"Gate against $|\nabla\cdot\mathbf{Q}|$", fontsize=10)

    out = OUT_DIR / "gate_vs_divQ.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def animate_gate_Q(sim, results, frames, fps=25):
    """Animation of both gates with the gate-on cells outlined and the Q vectors on top."""
    Q_DECADES = 4  # arrow length spans the top Q_DECADES decades of |Q|
    q_top = max(np.log10(np.hypot(r["Qx"], r["Qy"]).max() * FLUX_UNIT) for _, r, _ in results)

    fig, axes = plt.subplots(1, len(results), figsize=(4.2 * len(results) + 1.2, 7.4), constrained_layout=True)
    extent = [sim.x1min, sim.x1max, sim.x2min, sim.x2max]
    panels = []
    for ax, (spec, r, _) in zip(axes, results):
        m = r["maps"][frames[0]]
        ny, nx = m["gate"].shape
        dx = (sim.x1max - sim.x1min) / nx
        xc = sim.x1min + (np.arange(nx) + 0.5) * dx
        yc = sim.x2min + (np.arange(ny) + 0.5) * (sim.x2max - sim.x2min) / ny
        up = 16
        xu = sim.x1min + (np.arange(nx * up) + 0.5) * dx / up
        yu = sim.x2min + (np.arange(ny * up) + 0.5) * (sim.x2max - sim.x2min) / (ny * up)

        im = ax.imshow(m["gate"], origin="lower", extent=extent, cmap="Oranges", vmin=0, vmax=1)
        X, Y = np.meshgrid(xc, yc)
        # scale so that a full-length arrow is 0.9 of a coarse cell
        quiv = ax.quiver(X, Y, np.zeros_like(X), np.zeros_like(Y), angles="xy", scale_units="xy",
                         scale=1.0 / (0.9 * dx), width=0.006, color="#222222", pivot="mid")
        ax.set_title(spec["label"], fontsize=11)
        ax.set_xlabel("x [pc]")
        panels.append({"r": r, "im": im, "quiv": quiv, "ax": ax, "xu": xu, "yu": yu, "up": up, "contour": None})
    axes[0].set_ylabel("y [pc]")
    fig.colorbar(panels[-1]["im"], ax=axes, location="right", shrink=0.6, label="gate")
    title = fig.suptitle("", fontsize=11)
    legend = (f"blue outline: gate > {GATE_ON};  arrows: direction of $\\mathbf{{Q}}$, length "
              f"$\\propto \\log_{{10}}|\\mathbf{{Q}}|$ over the top {Q_DECADES} decades "
              f"(max $10^{{{q_top:.1f}}}$ erg cm$^{{-2}}$ s$^{{-1}}$)")

    def update(n):
        for p in panels:
            m = p["r"]["maps"][n]
            p["im"].set_data(m["gate"])
            qx, qy = m["Qx"], m["Qy"]
            qmag = np.hypot(qx, qy)
            length = np.clip((np.log10(np.maximum(qmag * FLUX_UNIT, 1e-300)) - (q_top - Q_DECADES)) / Q_DECADES, 0, 1)
            with np.errstate(invalid="ignore", divide="ignore"):
                ux = np.where(qmag > 0, qx / qmag, 0.0) * length
                uy = np.where(qmag > 0, qy / qmag, 0.0) * length
            p["quiv"].set_UVC(ux, uy)
            if p["contour"] is not None:
                p["contour"].remove()
            mask = np.kron((m["gate"] > GATE_ON).astype(float), np.ones((p["up"], p["up"])))
            p["contour"] = p["ax"].contour(p["xu"], p["yu"], mask, levels=[0.5], colors="#1f5fa8", linewidths=1.8)
        title.set_text(f"Gate and subgrid energy flux, t = {sim.times[sim.frame_numbers.index(n)]:.2f} Myr\n{legend}")
        return []

    out = OUT_DIR / "gate_Q_animation.mp4"
    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    try:
        writer = FFMpegWriter(fps=fps, codec="h264_nvenc", extra_args=["-preset", "p4", "-pix_fmt", "yuv420p"])
        anim.save(str(out), writer=writer, dpi=120)
    except Exception:
        writer = FFMpegWriter(fps=fps, codec="mpeg4", extra_args=["-q:v", "2", "-pix_fmt", "yuv420p"])
        anim.save(str(out), writer=writer, dpi=120)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--t-range", type=float, nargs=2, default=[5.0, 10.0])
    parser.add_argument("--snapshot-time", type=float, default=8.0)
    parser.add_argument("--skip-animation", action="store_true", help="do not write the mp4")
    args = parser.parse_args()

    sim = ergane.SimulationData(datafolder=str(HR_DIR / "bin"),
                                athinp=str(HR_DIR / "kh_radiative_512x1024.athinput"))
    keep = (sim.times >= args.t_range[0]) & (sim.times <= args.t_range[1])
    frames = [n for n, k in zip(sim.frame_numbers, keep) if k]
    snap = frames[int(np.argmin(np.abs(sim.times[keep] - args.snapshot_time)))]

    results = []
    for spec in MODELS:
        r = evaluate(sim, frames, spec)
        s = scores(r)
        results.append((spec, r, s))
        print(spec["label"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})
    print("Saved", plot(sim, results, snap, args.t_range, len(frames)))
    print("Saved", plot_divQ(sim, results, snap, args.t_range, len(frames)))
    if not args.skip_animation:
        print("Saved", animate_gate_Q(sim, results, frames))


if __name__ == "__main__":
    main()
