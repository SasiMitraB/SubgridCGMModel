"""
Predicted against true subgrid momentum flux R over every coarse cell of the held-out
(validation) frames, for a model trained by models/rvv_tensor_model.py.  R is compared in
the two pieces the model predicts and in the dimensionless tensor they build,
  R_ij / (<rho> k_max) = 2 f S_ij                                       (eq-R-split, eq-f)
which is what enters the momentum and energy equations (up to the resolved <rho> k_max).

Writes
  --out          2D histograms of log10 f (all cells), S_ij and ||S - I/3||_F
                 (mixing-layer cells, f > f_shape)
  --tensor-out   2D histograms of the six components of 2 f S_ij over all cells
                 (log axes on the diagonal, symlog off it)
  --z-out        2D histograms of the six network outputs z_0..z_5 (f = sigmoid(z_0),
                 S = e^A / Tr e^A with A = sum_k z_k E^(k)); z_0 over all cells, z_1..z_5
                 over the mixing-layer cells (f > f_shape)
  --temp-out     coarse-grained temperature T_CG = (mu m_u / k_B)(gamma - 1) k_max (1 - f)
                 (eq-temperature, eq-f), predicted against true over all cells, and the
                 error in log10 T_CG of the model and of no subgrid model (f = 0, all the
                 unresolved energy taken as heat) against true T_CG
  --anim-out     mid-z (x-y) slice of log10 f, 2 f S_xy and ||S - I/3||_F: true, predicted
                 and their difference, over the selected frames
  --loss-out     training / validation loss (total, trace, temp, shape) against epoch

  python benchmarking/plot_rvv_true_vs_pred.py --model models/rvv_tensor_model.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarking.common import load_model_and_split, model_input, plot_loss
from models.rvv_tensor_model import (BASIS, PAIRS, apply, channels, load_dataset, log_shape,
                                     predict, unpack_S)
from models.stencil import athinput, run
from vis_athenak import SimulationData
from vis_athenak.utils.units import Units

NAMES = ["xx", "yy", "zz", "xy", "xz", "yz"]
LINTHRESH = 1e-5  # linear range of symlog(2 f S_ij) off the diagonal
LOGF_MIN = -10    # log10 f below this is clipped into the first bin


def predict_frames(m, X):
    """f (frames, nz, ny, nx) and S (frames, nz, ny, nx, 3, 3) for inputs X, a frame at a time."""
    out = [predict(m["params"], model_input(m, x)) for x in X]
    return (np.concatenate([np.asarray(f) for f, _ in out]),
            np.concatenate([np.asarray(S) for _, S in out]))


def predict_z(m, X):
    """Raw outputs z_0..z_5, (frames, nz, ny, nx, 6), for inputs X, a frame at a time."""
    return np.concatenate([np.asarray(apply(m["params"], model_input(m, x))) for x in X])


def true_z(Y):
    """z_0 = logit f and z_1..z_5 = components of traceless log S on BASIS (as in the loss)."""
    f = np.clip(Y[:, 0].astype(np.float64), 1e-10, 1 - 1e-7)
    L, _ = log_shape(jnp.asarray(Y))
    z = np.asarray(jnp.einsum("...ij,kij->...k", unpack_S(L, start=0), BASIS), dtype=np.float64)
    return np.concatenate([np.log(f / (1 - f))[..., None], z], axis=-1)


def plot_z(z_t, z_p, f_t, f_shape, title, out):
    """The six network outputs; z_0 over all cells, z_1..z_5 over the mixing layer."""
    mix = f_t > f_shape
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    for k, ax in enumerate(axes.flat):
        sel = slice(None) if k == 0 else mix
        t, p = z_t[sel, k], z_p[sel, k]
        lo, hi = np.percentile(np.concatenate([t, p]), [0.01, 99.99])
        im = hist_panel(ax, t, p, (lo, hi), rf"$z_{k}$" + (r" (= logit $f$)" if k == 0 else ""))
    fig.colorbar(im, ax=axes, label="coarse cells", shrink=0.6)
    fig.suptitle(title + rf"  (z$_1$-z$_5$: $f > f_{{shape}}$ = {f_shape:g} cells)")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def symlog(x, linthresh=LINTHRESH):
    return np.sign(x) * np.log10(1 + np.abs(x) / linthresh)


def anisotropy(S):
    return np.sqrt(np.sum((S - np.eye(3) / 3)**2, axis=(-2, -1)))


def stats(t, p):
    """RMSE and Pearson r of p against t."""
    return np.sqrt(np.mean((p - t)**2)), np.corrcoef(t, p)[0, 1]


def hist_panel(ax, t, p, lims, label, nbins=150):
    """2D histogram of p against t on a square box lims, with the 1:1 line and RMSE, r."""
    bins = np.linspace(*lims, nbins + 1)
    *_, im = ax.hist2d(np.clip(t, *lims), np.clip(p, *lims), bins=[bins, bins],
                       norm=LogNorm(), cmap="viridis", rasterized=True)
    ax.plot(lims, lims, "k--", lw=0.8)
    rmse, r = stats(t, p)
    ax.set(xlabel=f"true {label}", ylabel=f"predicted {label}", xlim=lims, ylim=lims,
           aspect="equal")
    ax.text(0.03, 0.97, f"RMSE {rmse:.3g}\n$r$ = {r:.3f}", transform=ax.transAxes,
            ha="left", va="top", fontsize=8, bbox=dict(fc="w", ec="none", alpha=0.7))
    return im


def plot_parts(f_t, f_p, S_t, S_p, f_shape, title, out):
    """log10 f over all cells; S_ij and ||S - I/3||_F over the mixing-layer cells."""
    mix = f_t > f_shape
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.5), layout="constrained")
    ims = [hist_panel(axes[0, 0], np.log10(np.maximum(f_t, 10.0**LOGF_MIN)),
                      np.log10(np.maximum(f_p, 10.0**LOGF_MIN)), (LOGF_MIN, 0),
                      r"$\log_{10} f$")]
    axes[0, 0].axvline(np.log10(f_shape), color="r", lw=0.6, ls=":")
    axes[0, 0].set_title(f"all {f_t.size} cells")
    panels = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]
    for (r, c), (i, j), name in zip(panels, PAIRS, NAMES):
        lims = (0, 1) if i == j else (-0.5, 0.5)
        ims.append(hist_panel(axes[r, c], S_t[mix, i, j], S_p[mix, i, j], lims,
                              rf"$\mathsf{{S}}_{{{name}}}$"))
    ims.append(hist_panel(axes[1, 3], anisotropy(S_t[mix]), anisotropy(S_p[mix]),
                          (0, np.sqrt(2 / 3)), r"$\Vert\mathsf{S}-\mathrm{I}/3\Vert_F$"))
    axes[0, 1].set_title(f"{mix.sum()} mixing-layer cells, $f > {f_shape:g}$")
    fig.colorbar(ims[-1], ax=axes, label="coarse cells", shrink=0.6)
    fig.suptitle(title)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  (log10 f RMSE {stats(np.log10(np.maximum(f_t, 1e-10)), np.log10(np.maximum(f_p, 1e-10)))[0]:.3f}, "
          f"mixing-layer S RMSE {np.sqrt(np.mean(np.sum((S_p[mix] - S_t[mix])**2, axis=(-2, -1)))):.3f})")


def plot_tensor(f_t, f_p, S_t, S_p, title, out):
    """2 f S_ij = R_ij / (<rho> k_max), all cells: log10 on the diagonal, symlog off it."""
    T_t, T_p = 2 * f_t[..., None, None] * S_t, 2 * f_p[..., None, None] * S_p
    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout="constrained")
    for ax, (i, j), name in zip(axes.flat, PAIRS, NAMES):
        if i == j:
            fl = 10.0**(LOGF_MIN - 1)
            t, p = np.log10(np.maximum(T_t[..., i, j], fl)), np.log10(np.maximum(T_p[..., i, j], fl))
            im = hist_panel(ax, t, p, (LOGF_MIN - 1, 0), rf"$\log_{{10}}\,2f\mathsf{{S}}_{{{name}}}$")
        else:
            t, p = symlog(T_t[..., i, j]), symlog(T_p[..., i, j])
            lim = np.ceil(max(np.abs(t).max(), np.abs(p).max()))
            im = hist_panel(ax, t, p, (-lim, lim), rf"symlog $2f\mathsf{{S}}_{{{name}}}$")
    fig.colorbar(im, ax=axes, label="coarse cells", shrink=0.6)
    fig.suptitle(rf"{title}" "\n" rf"$R_{{ij}}/(\langle\rho\rangle k_{{max}}) = 2f\mathsf{{S}}_{{ij}}$,"
                 rf" off-diagonal linthresh = {LINTHRESH:g}")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_temperature(f_t, f_p, k_max, title, out, nbins=150):
    """T_CG [K] from <P>/<rho> = (gamma - 1) k_max (1 - f), f true and predicted, all cells:
    predicted against true, and Delta log10 T_CG = log10(T_pred / T_true) against true for
    the model and for f = 0.  Since <rho> and k_max are inputs, Delta log10 T_CG is the
    error in log10 <P> as well."""
    sim = SimulationData(athinput, f"{run}/bin")
    gamma = float(sim.params["hydro"]["gamma"])
    T_unit = Units.from_params(sim.params).temperature_cgs * (gamma - 1) * k_max
    T_t, T_p, T_0 = T_unit * (1 - f_t), T_unit * (1 - f_p), T_unit
    logT = np.log10(T_t)
    d_p, d_0 = np.log10(T_p / T_t), np.log10(T_0 / T_t)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), layout="constrained")
    lims = (np.floor(logT.min() * 4) / 4, np.ceil(logT.max() * 4) / 4)
    im = hist_panel(axes[0], logT, np.log10(T_p), lims, r"$\log_{10} T_{CG}$ [K]", nbins)
    axes[0].set_title("all cells")
    dlim = 1.05 * max(np.abs(d_p).max(), np.abs(d_0).max())
    for ax, d, name in ((axes[1], d_p, "model"), (axes[2], d_0, r"no subgrid model, $f = 0$")):
        *_, im = ax.hist2d(logT, d, bins=[np.linspace(*lims, nbins + 1),
                                          np.linspace(-dlim, dlim, nbins + 1)],
                           norm=LogNorm(), cmap="viridis", rasterized=True)
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set(xlabel=r"true $\log_{10} T_{CG}$ [K]",
               ylabel=r"$\log_{10}(T_{CG}^{pred} / T_{CG}^{true})$", title=name)
        ax.text(0.03, 0.97, f"RMS {np.sqrt(np.mean(d**2)):.3g} dex\n"
                f"max |error| {np.abs(d).max():.3g} dex", transform=ax.transAxes,
                ha="left", va="top", fontsize=8, bbox=dict(fc="w", ec="none", alpha=0.7))
    fig.colorbar(im, ax=axes, label="coarse cells", shrink=0.8)
    fig.suptitle(title)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  (log10 T_CG RMS error: model {np.sqrt(np.mean(d_p**2)):.3g} dex, "
          f"f = 0 {np.sqrt(np.mean(d_0**2)):.3g} dex)")


def animate(f_t, f_p, S_t, S_p, f_shape, t, out, fps=8):
    """Mid-z slice of log10 f, symlog 2 f S_xy and ||S - I/3||_F (mixing-layer cells only):
    true, predicted and predicted - true, one frame per snapshot."""
    k = f_t.shape[1] // 2
    f_t, f_p, S_t, S_p = f_t[:, k], f_p[:, k], S_t[:, k], S_p[:, k]
    mix = f_t > f_shape
    logf = lambda f: np.log10(np.maximum(f, 10.0**LOGF_MIN))
    rows = [(r"$\log_{10} f$", logf(f_t), logf(f_p), "viridis", (LOGF_MIN, 0)),
            (r"symlog $2f\mathsf{S}_{xy}$", symlog(2 * f_t * S_t[..., 0, 1]),
             symlog(2 * f_p * S_p[..., 0, 1]), "RdBu_r", None),
            (r"$\Vert\mathsf{S}-\mathrm{I}/3\Vert_F$, $f > f_{shape}$",
             np.where(mix, anisotropy(S_t), np.nan), np.where(mix, anisotropy(S_p), np.nan),
             "magma", (0, np.sqrt(2 / 3)))]

    fig, axes = plt.subplots(3, 3, figsize=(8, 12.5), layout="constrained", sharex=True, sharey=True)
    ims = []
    for row, (label, tr, pr, cmap, lims) in zip(axes, rows):
        if lims is None:
            v = np.ceil(max(np.abs(tr).max(), np.abs(pr).max()))
            lims = (-v, v)
        d = pr - tr
        dlim = np.nanmax(np.abs(d))
        for ax, data, name, cm, (lo, hi) in zip(row, (tr, pr, d), ("true", "predicted", "predicted $-$ true"),
                                               (cmap, cmap, "RdBu_r"), (lims, lims, (-dlim, dlim))):
            im = ax.imshow(data[0], origin="lower", cmap=cm, vmin=lo, vmax=hi)
            ax.set_title(name)
            ims.append((im, data))
        fig.colorbar(row[1].images[0], ax=row[:2], label=label, shrink=0.8)
        fig.colorbar(row[2].images[0], ax=row[2], label=r"$\Delta$", shrink=0.8)
    for ax in axes[-1]:
        ax.set_xlabel("x [coarse cells]")
    for ax in axes[:, 0]:
        ax.set_ylabel("y [coarse cells]")
    suptitle = fig.suptitle("")

    def update(i):
        for im, data in ims:
            im.set_data(data[i])
        suptitle.set_text(f"z slice {k}, t = {t[i]:.2f}")
        return [im for im, _ in ims] + [suptitle]

    FuncAnimation(fig, update, frames=len(f_t), blit=False).save(out, fps=fps, dpi=100)
    plt.close(fig)
    print(f"saved {out}  ({len(f_t)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/rvv_tensor_model.pkl")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--out", default="benchmarking/rvv_true_vs_pred.png")
    parser.add_argument("--tensor-out", default="benchmarking/rvv_tensor_true_vs_pred.png",
                        help="components of 2 f S_ij; empty string to skip")
    parser.add_argument("--z-out", default="benchmarking/rvv_z_true_vs_pred.png",
                        help="the six outputs z_0..z_5; empty string to skip")
    parser.add_argument("--temp-out", default="benchmarking/rvv_temperature_true_vs_pred.png",
                        help="predicted vs true T_CG; empty string to skip")
    parser.add_argument("--anim-out", default="benchmarking/rvv_true_vs_pred_slice.mp4",
                        help="mid-z slice animation (.mp4 or .gif); empty string to skip")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--loss-out", default="benchmarking/rvv_loss.png",
                        help="training / validation loss vs epoch; empty string to skip")
    args = parser.parse_args()

    m, X, Y, t = load_model_and_split(args.model, load_dataset, args.split)

    f_t, S_t = Y[:, 0].astype(np.float64), np.asarray(unpack_S(jnp.asarray(Y)), dtype=np.float64)
    f_p, S_p = predict_frames(m, X)
    f_p, S_p = f_p.astype(np.float64), S_p.astype(np.float64)
    title = f"{args.split}: {len(X)} frames, {f_t.size} cells"

    plot_parts(f_t.ravel(), f_p.ravel(), S_t.reshape(-1, 3, 3), S_p.reshape(-1, 3, 3),
               m["f_shape"], title, args.out)
    if args.tensor_out:
        plot_tensor(f_t.ravel(), f_p.ravel(), S_t.reshape(-1, 3, 3), S_p.reshape(-1, 3, 3),
                    title, args.tensor_out)
    if args.z_out:
        plot_z(true_z(Y).reshape(-1, 6), predict_z(m, X).reshape(-1, 6), f_t.ravel(),
               m["f_shape"], title, args.z_out)
    if args.temp_out:
        k_max = 10.0**X[:, channels.index("log_kmax")].astype(np.float64)
        plot_temperature(f_t.ravel(), f_p.ravel(), k_max.ravel(), title, args.temp_out)
    if args.loss_out:
        plot_loss(m["history"], [("", "total"), ("_trace", r"trace: MSE of $\ln f$"),
                                 ("_temp", r"temp: MSE of $\ln T_{CG}$"),
                                 ("_shape", r"shape: $\Vert\Delta\mathsf{S}\Vert_F^2$, $f > f_{shape}$")],
                  "loss", args.loss_out)
    if args.anim_out:
        animate(f_t, f_p, S_t, S_p, m["f_shape"], t, args.anim_out, args.fps)
