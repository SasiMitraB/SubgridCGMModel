"""
Shared pieces of the true-vs-predicted benchmarks: loading a trained stencil model with its
train / validation frames, and the loss-against-epoch plot.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.stencil import pad, val_mask


def model_input(m, x):
    """One frame x (C, nz, ny, nx) -> the network input (1, C, nz + 2h, ny + 2h, nx + 2h) of
    model m: normalised and padded with the run's boundaries."""
    return pad(jnp.asarray((x[None] - m["x_mean"]) / m["x_std"]), m["stencil"] // 2)


def load_model_and_split(path, load_dataset, split):
    """The pickled model at path and its X, Y, t restricted to split ("val", "train", "all"),
    split exactly as in training."""
    with open(path, "rb") as f:
        m = pickle.load(f)
    X, Y, t = map(np.asarray, load_dataset(m["every"], Path(m["cache"])))
    val = np.asarray(val_mask(len(X), m["val_every"]))
    sel = {"val": val, "train": ~val, "all": np.ones_like(val)}[split]
    return m, X[sel], Y[sel], t[sel]


def plot_loss(history, terms, ylabel, out):
    """Training (every epoch) and validation (every log_every epochs) loss, one panel per
    (suffix, label) in terms whose history is recorded; suffix "" is the total."""
    terms = [(k, label) for k, label in terms if f"train{k}" in history]
    fig, axes = plt.subplots(1, len(terms), figsize=(4.5 * len(terms), 4), squeeze=False,
                             layout="constrained")
    for ax, (k, label) in zip(axes[0], terms):
        ax.plot(history["epoch"], history[f"train{k}"], lw=1, label="train")
        ax.plot(history["val_epoch"], history[f"val{k}"], "o-", ms=3, lw=1, label="validation")
        ax.set(xlabel="epoch", yscale="log",
               title=f"{label}\ntrain {history[f'train{k}'][-1]:.3e}, val {history[f'val{k}'][-1]:.3e}")
        ax.grid(alpha=0.3, which="both")
    axes[0, 0].set_ylabel(ylabel)
    axes[0, 0].legend()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")
