"""
Benchmark of the mixing-layer gate trained by models/gate_model.py on the coarse cells of the
selected frames (default: the held-out validation frames):

  --out       five panels: the confusion matrix at the decision threshold, the predicted
              probability of the mixing (gate = 1) and non-mixing (gate = 0) cells, the
              precision-recall and ROC curves, and the recall of the mixing cells against their
              mixing mass fraction f_mix (are the misses the cells with a trace of mixing gas?)
  --fmix-out  the number and the share of TN / FP / FN / TP cells against the mixing mass
              fraction f_mix [%] computed from the simulation (the cells with f_mix = 0 in a bin
              of their own, then log-spaced bins), plus a .csv of the counts; written next to
              --out as <out>_fmix.png / .csv unless given (empty string to skip)
  --anim-out  mid-z (x-y) slice of the true gate, the predicted probability and the outcome
              (TP / FP / FN / TN) of every cell, one frame per snapshot
  --loss-out  training / validation loss against epoch

  python benchmarking/plot_gate_true_vs_pred.py --model models/gate_model.pkl
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import jax
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarking.common import load_model_and_split, model_input, plot_loss
from models.gate_model import apply, features, gate, load_dataset

OUTCOMES = ["TN", "FP", "FN", "TP"]  # index 2 * true + predicted
OUTCOME_COLORS = ["#f2f2f2", "#e8a33d", "#c0392b", "#2e86c1"]
LINE_COLORS = ["0.55", "#e8a33d", "#c0392b", "#2e86c1"]  # OUTCOMES, visible on white
FMIX_PER_DECADE = 4


def predict(m, X):
    """Gate probability (frames, nz, ny, nx) for cached inputs X, one frame at a time."""
    return np.concatenate([np.asarray(jax.nn.sigmoid(apply(m["params"],
                                                           model_input(m, features(x[None])[0]))))
                           for x in X])


def sweep(p, y):
    """True and false positives of the mixing class for every distinct threshold, from the
    highest p down (cells with equal p are switched on together)."""
    order = np.argsort(-p, kind="stable")
    p, y = p[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(p)), len(p) - 1]  # last cell of each distinct p
    tp = np.cumsum(y)[last]
    return tp, last + 1 - tp


def plot(p, y, f_mix, threshold, title, out):
    pred = p > threshold
    yb = y > 0.5
    counts = np.array([[np.sum(~yb & ~pred), np.sum(~yb & pred)],
                       [np.sum(yb & ~pred), np.sum(yb & pred)]])
    tn, fp, fn, tp = counts.ravel()
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    fig, axes = plt.subplots(1, 5, figsize=(25, 4.8), layout="constrained")

    ax = axes[0]
    rows = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{counts[i, j]}\n{rows[i, j]:.1%}", ha="center", va="center",
                    color="white" if rows[i, j] > 0.5 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["0", "1"], yticklabels=["0", "1"],
           xlabel="predicted gate", ylabel="true gate",
           title=f"threshold {threshold}: accuracy {(tp + tn) / counts.sum():.4f}\n"
                 f"precision {precision:.3f}, recall {recall:.3f}, F1 {f1:.3f}, "
                 f"IoU {tp / max(tp + fp + fn, 1):.3f}")

    ax = axes[1]
    bins = np.linspace(0, 1, 51)
    ax.hist(p[~yb], bins, histtype="step", lw=1.2, label=f"gate = 0 ({np.sum(~yb)})")
    ax.hist(p[yb], bins, histtype="step", lw=1.2, label=f"gate = 1 ({np.sum(yb)})")
    ax.axvline(threshold, color="0.5", ls=":", lw=1)
    ax.set(xlabel="predicted probability", ylabel="coarse cells", yscale="log",
           title="predicted probability by true class")
    ax.legend(fontsize=8)

    ax = axes[2]
    tp_k, fp_k = sweep(p, yb)
    prec, rec = tp_k / (tp_k + fp_k), tp_k / max(tp + fn, 1)
    ap = np.sum(np.diff(np.concatenate([[0], rec])) * prec)
    ax.plot(rec, prec, lw=1.2)
    ax.plot(recall, precision, "o", ms=5, label=f"threshold {threshold}")
    ax.axhline(yb.mean(), color="0.5", ls=":", lw=1, label=f"base rate {yb.mean():.3f}")
    ax.set(xlabel="recall", ylabel="precision", xlim=(0, 1), ylim=(0, 1.02),
           title=f"precision-recall, AP = {ap:.3f}")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    ax = axes[3]
    fpr = np.r_[0, fp_k / max(fp + tn, 1)]
    tpr = np.r_[0, rec]
    auc = np.trapezoid(tpr, fpr)
    ax.plot(fpr, tpr, lw=1.2)
    ax.plot(fp / max(fp + tn, 1), recall, "o", ms=5, label=f"threshold {threshold}")
    ax.plot([0, 1], [0, 1], color="0.5", ls=":", lw=1, label="chance")
    ax.set(xlabel="false positive rate", ylabel="true positive rate (recall)",
           xlim=(0, 1), ylim=(0, 1.02), title=f"ROC, AUC = {auc:.4f}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[4]
    f = f_mix[yb]
    edges = np.logspace(np.floor(np.log10(f.min())), 0, 25) if f.size else np.logspace(-6, 0, 25)
    idx = np.digitize(f, edges) - 1
    n = np.bincount(idx, minlength=len(edges))[:len(edges) - 1]
    hit = np.bincount(idx, weights=pred[yb], minlength=len(edges))[:len(edges) - 1]
    centres = np.sqrt(edges[:-1] * edges[1:])
    ok = n > 0
    ax.plot(centres[ok], hit[ok] / n[ok], "o-", ms=3, lw=1)
    ax.set(xscale="log", xlabel=r"mixing mass fraction $f_\mathrm{mix}$", ylabel="recall",
           ylim=(0, 1.02), title="recall of the mixing cells vs $f_\\mathrm{mix}$")
    ax.grid(alpha=0.3, which="both")
    twin = ax.twinx()
    twin.bar(centres[ok], n[ok], width=np.diff(edges)[ok], color="0.85", zorder=0)
    twin.set(yscale="log", ylabel="mixing cells")
    ax.set_zorder(twin.get_zorder() + 1)
    ax.patch.set_visible(False)

    fig.suptitle(title)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}  (precision {precision:.4f}, recall {recall:.4f}, F1 {f1:.4f}, AP {ap:.4f}, AUC {auc:.4f})")


def outcomes_by_fmix(pred, yb, f_mix):
    """Cells of each outcome (rows in OUTCOMES order) per bin of f_mix: column 0 holds the
    cells with f_mix = 0, columns 1.. log-spaced bins of f_mix > 0 (FMIX_PER_DECADE per
    decade, from the decade of the smallest f_mix up to 1, the last bin closed).
    Returns the bin edges (of the columns 1..) and the counts (4, len(edges))."""
    positive = f_mix > 0
    lo = min(np.floor(np.log10(f_mix[positive].min())), -1.0) if positive.any() else -6.0
    edges = np.logspace(lo, 0, int(-lo) * FMIX_PER_DECADE + 1)
    idx = np.where(positive, np.clip(np.digitize(f_mix, edges) - 1, 0, len(edges) - 2) + 1, 0)
    outcome = 2 * yb + pred
    counts = np.bincount(outcome * len(edges) + idx, minlength=4 * len(edges))
    return edges, counts.reshape(4, len(edges))


def plot_outcomes_by_fmix(pred, yb, f_mix, threshold, min_frac, title, out):
    """TN / FP / FN / TP against f_mix [%]: the cell counts (top) and the share of the cells
    of each bin in each outcome (bottom).  The narrow left panels are the cells with
    f_mix = 0, which hold every true negative when min_frac = 0.  The counts are also
    written to <out>.csv."""
    edges, counts = outcomes_by_fmix(pred, yb, f_mix)
    pct = 100 * edges
    centres = np.sqrt(pct[:-1] * pct[1:])
    share = counts / np.maximum(counts.sum(axis=0), 1)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), width_ratios=[1, 8], sharey="row",
                             layout="constrained")
    axes[0, 1].sharex(axes[1, 1])
    (zero_n, ax_n), (zero_s, ax_s) = axes

    zero_n.bar(range(4), counts[:, 0], color=LINE_COLORS)
    zero_n.set(xticks=range(4), xticklabels=OUTCOMES, yscale="log", ylabel="coarse cells",
               title=r"$f_\mathrm{mix} = 0$")
    for k, name in enumerate(OUTCOMES):
        ax_n.plot(centres, np.where(counts[k, 1:] > 0, counts[k, 1:], np.nan), "o-", ms=3,
                  lw=1, color=LINE_COLORS[k], label=name)
    ax_n.set(xscale="log", yscale="log", ylim=(0.5, None),
             title=f"outcomes against $f_\\mathrm{{mix}}$, threshold {threshold}")
    ax_n.legend(fontsize=8, ncols=4)

    for k in (3, 2, 1, 0):  # stacked: TP at the bottom, then FN, FP, TN
        below = share[[j for j in (3, 2, 1, 0) if j > k]].sum(axis=0)
        zero_s.bar(0, share[k, 0], bottom=below[0], color=LINE_COLORS[k], width=0.8)
        ax_s.bar(pct[:-1], share[k, 1:], bottom=below[1:], width=np.diff(pct), align="edge",
                 color=LINE_COLORS[k], label=OUTCOMES[k])
    zero_s.set(xticks=[0], xticklabels=["0"], xlim=(-0.6, 0.6), ylim=(0, 1),
               ylabel="share of the cells in the bin")
    ax_s.set(xscale="log", xlabel=r"mixing mass fraction $f_\mathrm{mix}$ [%]")
    for ax in (ax_n, ax_s):
        ax.grid(alpha=0.3, which="both")
        if min_frac > 0:
            ax.axvline(100 * min_frac, color="0.3", ls="--", lw=1)
    ax_s.set_xlim(pct[0], 100)
    ax_n.tick_params(labelleft=False)
    ax_s.tick_params(labelleft=False)

    fig.suptitle(title)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    csv = Path(out).with_suffix(".csv")
    lo = np.r_[0, pct[:-1]]
    hi = np.r_[0, pct[1:]]
    np.savetxt(csv, np.column_stack([lo, hi, counts.T]), delimiter=",", fmt=["%.6g", "%.6g"] + ["%d"] * 4,
               header="f_mix_lo_percent,f_mix_hi_percent," + ",".join(OUTCOMES), comments="")
    print(f"saved {out} and {csv}  (f_mix = 0: {counts[:, 0].sum()} cells, f_mix > 0: "
          f"{counts[:, 1:].sum()} cells)")


def animate(y, p, threshold, t, out, fps=8):
    """Mid-z slice of the true gate, predicted probability and outcome, one frame per snapshot.
    y, p are (frames, nz, ny, nx); y is the vertical axis of each panel."""
    k = y.shape[1] // 2
    true, prob = y[:, k], p[:, k]
    outcome = 2 * (true > 0.5) + (prob > threshold)

    fig, axes = plt.subplots(1, 3, figsize=(10, 6.5), layout="constrained", sharey=True)
    ims = [axes[0].imshow(true[0], origin="lower", cmap="Greys", vmin=0, vmax=1),
           axes[1].imshow(prob[0], origin="lower", cmap="Greys", vmin=0, vmax=1),
           axes[2].imshow(outcome[0], origin="lower", cmap=ListedColormap(OUTCOME_COLORS),
                          vmin=-0.5, vmax=3.5)]
    for ax, name in zip(axes, ("true gate", "predicted probability", "outcome")):
        ax.set(title=name, xlabel="x [coarse cells]", aspect="equal")
    axes[0].set_ylabel("y [coarse cells]")
    fig.colorbar(ims[1], ax=axes[:2], label="gate", shrink=0.6)
    cb = fig.colorbar(ims[2], ax=axes[2], ticks=range(4), shrink=0.6)
    cb.ax.set_yticklabels(OUTCOMES)
    suptitle = fig.suptitle("")

    def update(i):
        for im, data in zip(ims, (true, prob, outcome)):
            im.set_data(data[i])
        n = np.bincount(outcome[i].ravel(), minlength=4)
        suptitle.set_text(f"z slice {k}, t = {t[i]:.2f}: "
                          + ", ".join(f"{o} {c}" for o, c in zip(OUTCOMES, n)))
        return [*ims, suptitle]

    FuncAnimation(fig, update, frames=len(true), blit=False).save(out, fps=fps, dpi=120)
    plt.close(fig)
    print(f"saved {out}  ({len(true)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="models/gate_model.pkl")
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="gate = 1 where the predicted probability exceeds this")
    parser.add_argument("--out", default="benchmarking/gate_true_vs_pred.png")
    parser.add_argument("--fmix-out", default=None,
                        help="TN / FP / FN / TP against f_mix (.png, plus a .csv of the counts); "
                             "default: --out with _fmix appended; empty string to skip")
    parser.add_argument("--anim-out", default="benchmarking/gate_true_vs_pred_slice.mp4",
                        help="mid-z slice animation (.mp4 or .gif); empty string to skip")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--loss-out", default="benchmarking/gate_loss.png",
                        help="training / validation loss vs epoch; empty string to skip")
    args = parser.parse_args()

    m, X, F, t = load_model_and_split(args.model, load_dataset, args.split)
    Y = np.asarray(gate(F, m["min_frac"]))
    P = predict(m, X)
    plot(P.ravel(), Y.ravel(), F.ravel(), args.threshold,
         f"{args.split}: {len(X)} frames, {Y.size} cells, gate = 1 where f_mix > {m['min_frac']:g}",
         args.out)
    fmix_out = (args.fmix_out if args.fmix_out is not None
                else str(Path(args.out).with_name(Path(args.out).stem + "_fmix.png")))
    if fmix_out:
        plot_outcomes_by_fmix(P.ravel() > args.threshold, Y.ravel() > 0.5, F.ravel(),
                              args.threshold, m["min_frac"],
                              f"{args.split}: outcomes of the gate against f_mix, "
                              f"gate = 1 where f_mix > {m['min_frac']:g}", fmix_out)
    if args.loss_out:
        plot_loss(m["history"], [("", "total"), ("_pos", "BCE, gate = 1"),
                                 ("_neg", "BCE, gate = 0"), ("_aux", "MSE, f_mix (auxiliary)")],
                  "loss", args.loss_out)
    if args.anim_out:
        animate(Y, P, args.threshold, t, args.anim_out, args.fps)
