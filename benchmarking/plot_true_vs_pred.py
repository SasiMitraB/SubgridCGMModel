"""
Benchmark of the cooling-correction model trained by models/cooling_model.py on the coarse
cells of the selected frames (default: the held-out validation frames). The quantity shown
is the net cooling <n^2 Lcut(T)>, the coarse cooling (<rho>/mu m_u)^2 Lcut(T_CG) plus the
predicted correction C (eq-cooling, eq-cooling-model), against the true one:

  --out       four panels: predicted against true z on the gated cells with T_CG inside the
              cooling window and outside it; predicted against true net cooling (log10) over
              every cell; and the net cooling summed over each frame against time, with the
              uncorrected coarse cooling (C = 0) as a reference
  --anim-out  mid-z (x-y) slice of the true net cooling, the predicted one and their difference
  --loss-out  training / validation loss against epoch

  python benchmarking/plot_true_vs_pred.py --model models/cooling_model.pkl
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
from benchmarking.common import load_model_and_split, plot_loss
from models.cooling_model import correction, load_dataset, predict, scores, targets

def log_net(q, floor):
    """log10 of the net cooling q [erg cm^-3 s^-1] >= 0. Cells with no cooling (q = 0, most of
    the box) are put at floor, a decade below the smallest cooling in the data."""
    return np.log10(np.where(q > 0, q, floor))


def hist(ax, x, y, lim, xlabel, ylabel, title):
    bins = np.linspace(*lim, 201)
    *_, im = ax.hist2d(x, np.clip(y, *lim), bins=[bins, bins], norm=LogNorm(), cmap="viridis")
    ax.plot(lim, lim, "k--", lw=0.8, label="1:1")
    ax.set(xlabel=xlabel, ylabel=ylabel, aspect="equal", title=title)
    ax.legend(loc="upper left", fontsize=8)
    plt.colorbar(im, ax=ax, label="coarse cells")


def plot(z_true, z_pred, g, inside, net_true, net_pred, net_coarse, t_cells, s, title, out):
    """net_true, net_pred: net cooling of every cell (flat); net_coarse: the coarse cooling
    alone, i.e. net_pred with C = 0; t_cells: the time of every cell."""
    positive = np.concatenate([net_true, net_pred])
    positive = positive[positive > 0]
    floor = 10**np.floor(np.log10(positive.min())) / 10
    n_zero = int(np.sum(net_true == 0))
    fig, axes = plt.subplots(1, 4, figsize=(25, 5.8), layout="constrained")
    for ax, sel, name, key in ((axes[0], (g > 0.5) & (inside > 0.5), "in the window", "rmse_in"),
                               (axes[1], (g > 0.5) & (inside < 0.5), "outside the window", "rmse_out")):
        lo, hi = np.floor(z_true[sel].min()) - 0.5, np.ceil(z_true[sel].max()) + 0.5
        hist(ax, z_true[sel], z_pred[sel], (lo, hi), "true $z$", "predicted $z$",
             f"$g = 1$, $T_\\mathrm{{CG}}$ {name}: {sel.sum()} cells, RMSE {s[key]:.3f}")

    lt, lp = log_net(net_true, floor), log_net(net_pred, floor)
    lo, hi = np.floor(lt.min()), np.ceil(max(lt.max(), lp.max()))
    hist(axes[2], lt, lp, (lo, hi), r"true $\log_{10}\langle n^2\Lambda\rangle$",
         r"predicted $\log_{10}\langle n^2\Lambda\rangle$",
         f"net cooling, all cells: $\\sum|\\Delta|/\\sum\\langle n^2\\Lambda\\rangle$ = {s['abs_err']:.3f}\n"
         f"{n_zero} cells with zero true cooling drawn at $10^{{{np.log10(floor):.0f}}}$")

    ax = axes[3]
    frames = np.unique(t_cells)
    per_frame = lambda q: np.array([q[t_cells == f].sum() for f in frames])
    ax.plot(frames, per_frame(net_true), "k-", lw=1.4, label="true")
    ax.plot(frames, per_frame(net_pred), "o-", ms=3, lw=1, label="predicted, with C")
    ax.plot(frames, per_frame(net_coarse), ls="--", lw=1, color="0.5",
            label="coarse only (C = 0)")
    ax.set(xlabel="t [Myr]", ylabel=r"$\sum \langle n^2\Lambda\rangle$ per frame [erg cm$^{-3}$ s$^{-1}$]",
           title="net cooling summed over the frame")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle(title)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  (" + ", ".join(f"{k} {v:.4f}" for k, v in s.items()) + ")")


def animate(y_true, y_pred, t, out, fps=8):
    """Mid-z slice of the log10 true and predicted net cooling and their difference in dex,
    one frame per snapshot. y_true, y_pred are (frames, nz, ny, nx); y is the vertical axis
    of each panel."""
    k = y_true.shape[1] // 2
    true, pred = y_true[:, k], y_pred[:, k]
    diff = pred - true
    lim = np.ceil(max(true.max(), pred.max()))
    lo = np.floor(min(true.min(), pred.min()))
    dlim = np.abs(diff).max()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained", sharey=True)
    ims = []
    scales = [("viridis", lo, lim), ("viridis", lo, lim), ("RdBu_r", -dlim, dlim)]
    for ax, data, name, (cmap, vmin, vmax) in zip(axes, (true, pred, diff),
                                                  ("true", "predicted", "predicted $-$ true"), scales):
        im = ax.imshow(data[0], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set(title=name, xlabel="x [coarse cells]", aspect="equal")
        ims.append(im)
    axes[0].set_ylabel("y [coarse cells]")
    fig.colorbar(ims[1], ax=axes[:2], label=r"$\log_{10}\langle n^2\Lambda\rangle$", shrink=0.8)
    fig.colorbar(ims[2], ax=axes[2], label=r"$\Delta\log_{10}\langle n^2\Lambda\rangle$ [dex]",
                 shrink=0.8)
    suptitle = fig.suptitle("")

    def update(i):
        for im, data in zip(ims, (true, pred, diff)):
            im.set_data(data[i])
        rmse = np.sqrt(np.mean(diff[i]**2))
        suptitle.set_text(f"z slice {k}, t = {t[i]:.2f}, slice RMSE = {rmse:.3f}")
        return [*ims, suptitle]

    FuncAnimation(fig, update, frames=len(true), blit=False).save(out, fps=fps, dpi=120)
    plt.close(fig)
    print(f"saved {out}  ({len(true)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/cooling_model.pkl")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--out", default="benchmarking/true_vs_pred.png")
    parser.add_argument("--anim-out", default="benchmarking/true_vs_pred_slice.mp4",
                        help="mid-z slice animation (.mp4 or .gif); empty string to skip")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--loss-out", default="benchmarking/loss.png",
                        help="training / validation loss vs epoch; empty string to skip")
    args = parser.parse_args()

    m, X, Y, t = load_model_and_split(args.model, load_dataset, args.split)
    X, Y = jnp.asarray(X), jnp.asarray(Y)
    z_pred, gate = predict(m, X, Y)
    s = scores(z_pred, gate, Y, m["k_cut"])
    z, g, inside = map(np.asarray, targets(Y, m["k_cut"]))
    C_pred = np.asarray(correction(z_pred, gate, Y))
    # net cooling: the coarse cooling (rho/mu m_u)^2 Lcut(T_CG) = inside * Lfull(T_CG) (in
    # window) plus C; the true net cooling is <n^2 Lcut> itself (true_correction + coarse)
    coarse = np.asarray(Y[:, 2] * 10**Y[:, 1])
    net_true = np.asarray(10**Y[:, 0])
    net_pred = coarse + C_pred
    t_cells = np.broadcast_to(np.asarray(t)[:, None, None, None], net_true.shape)
    title = (f"{args.split}: {len(X)} frames, {net_true.size} cells, gate input: "
             f"{m['gate_model'] or 'true g'}, k = {m['k_cut']:g}")
    plot(z.ravel(), np.asarray(z_pred).ravel(), g.ravel(), inside.ravel(), net_true.ravel(),
         net_pred.ravel(), coarse.ravel(), t_cells.ravel(), s, title, args.out)
    if args.loss_out:
        plot_loss(m["history"], [("", "total"), ("_in", "MSE of z, in the window"),
                                 ("_out", "MSE of z, outside the window")],
                  "loss on z", args.loss_out)
    if args.anim_out:
        pos = np.concatenate([net_true[net_true > 0], net_pred[net_pred > 0]])
        floor = 10**np.floor(np.log10(pos.min())) / 10
        animate(log_net(net_true, floor), log_net(net_pred, floor), t, args.anim_out, args.fps)
