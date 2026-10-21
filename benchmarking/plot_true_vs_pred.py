"""
2D histogram of the predicted against the true cooling correction C over every coarse cell
of the held-out (validation) frames, for a model trained by models/cooling_model.py.
Both axes are symlog(C), the space the model is trained in. Also writes an animation of the
mid-z (x-y) slice of true C, predicted C and their difference over the selected frames, and
the training / validation loss against epoch.

  python benchmarking/plot_true_vs_pred.py --model models/cooling_model.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarking.common import load_model_and_split, model_input, plot_loss
from models.cooling_model import apply, load_dataset, symlog


def predict(m, X):
    """symlog(C) predicted for inputs X (frames, C, nz, ny, nx), one frame at a time."""
    return np.concatenate([np.asarray(apply(m["params"], model_input(m, x))) for x in X])


def plot(y_true, y_pred, linthresh, title, out):
    lim = np.ceil(max(np.abs(y_true).max(), np.abs(y_pred).max()))
    bins = np.linspace(-lim, lim, 201)
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))

    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout="constrained")
    *_, im = ax.hist2d(y_true, np.clip(y_pred, -lim, lim), bins=[bins, bins], norm=LogNorm(),
                       cmap="viridis")
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="1:1")
    ax.set(xlabel=r"true symlog$(C)$", ylabel=r"predicted symlog$(C)$", aspect="equal",
           title=f"{title}\nRMSE = {rmse:.3f}, linthresh = {linthresh:.0e} erg cm$^{{-3}}$ s$^{{-1}}$")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, ax=ax, label="coarse cells")
    fig.savefig(out, dpi=150)
    print(f"saved {out}  (RMSE {rmse:.4f} in symlog space)")


def animate(y_true, y_pred, t, out, fps=8):
    """Mid-z slice of true, predicted and (predicted - true) symlog(C), one frame per snapshot.
    y_true, y_pred are (frames, nz, ny, nx); y is the vertical axis of each panel."""
    k = y_true.shape[1] // 2
    true, pred = y_true[:, k], y_pred[:, k]
    diff = pred - true
    lim = np.ceil(max(np.abs(true).max(), np.abs(pred).max()))
    dlim = np.abs(diff).max()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), layout="constrained", sharey=True)
    ims = []
    for ax, data, name, vmax in zip(axes, (true, pred, diff),
                                    ("true", "predicted", "predicted $-$ true"), (lim, lim, dlim)):
        im = ax.imshow(data[0], origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set(title=name, xlabel="x [coarse cells]", aspect="equal")
        ims.append(im)
    axes[0].set_ylabel("y [coarse cells]")
    fig.colorbar(ims[1], ax=axes[:2], label=r"symlog$(C)$", shrink=0.8)
    fig.colorbar(ims[2], ax=axes[2], label=r"$\Delta$ symlog$(C)$", shrink=0.8)
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

    y_true = np.asarray(symlog(Y, m["linthresh"]))
    y_pred = predict(m, X)
    plot(y_true.ravel(), y_pred.ravel(), m["linthresh"],
         f"{args.split}: {len(X)} frames, {y_true.size} cells", args.out)
    if args.loss_out:
        if "history" in m:
            plot_loss(m["history"], [("", "total"), ("_mse", "MSE, all cells"),
                                     ("_pos", r"MSE, $C > 0$"), ("_neg", r"MSE, $C < 0$")],
                      r"loss on symlog$(C)$", args.loss_out)
        else:
            print(f"no loss history in {args.model} (trained before it was recorded); skipping loss plot")
    if args.anim_out:
        animate(y_true, y_pred, t, args.anim_out, args.fps)
