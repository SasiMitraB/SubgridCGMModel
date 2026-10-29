"""
Mixing-layer gate: a stencil classifier for whether a coarse cell of the 16x64x16 grid
holds any mixing-layer gas, from the resolved fields of the 256x1024x256 KH run (block
size b = 16) and their gradients.

The target comes from the subgrid temperature PDF of each coarse cell, the mass-weighted
histogram of the fine-cell temperatures in its block over 60 log bins from 10^3 to 10^7 K
(cached with every frame, see temperature_pdf).  The gate is 1 if any mass lies in the
mixing range 1.05e4 < T <= 0.95e6 K, where ismcooling.hpp cools, and 0 otherwise.

The mixing mass fraction f_mix is taken from the fine cells with these exact limits, not
summed from the histogram: the bins are 1/15 dex wide and the limits fall inside them
(bin 15 spans 1.00-1.17e4 K, bin 44 8.6e5-1.00e6 K), while the cold and hot phases sit at
~1e4 and ~1e6 K, so a bin sum would gate on gas just outside the range.  The cache keeps
f_mix, so the gate threshold (--min-frac, default 0: any mass) is chosen at training time.

Inputs are the coarse fields of cooling_model.py plus their gradients on the coarse grid
(central differences, in units of a coarse cell): grad log rho, grad log T, grad s0, the
strain rate S_ij and the vorticity of the resolved velocity (features).

The network has two outputs per cell: the logit of the gate and, as an auxiliary target, the
mixing mass fraction f_mix itself (through a sigmoid, trained with mean squared error).  The
gate is all that is used at prediction time (apply); f_mix gives the shared layers graded
information on how much mixing gas a cell holds, which the binary label throws away.

  python models/gate_model.py --every 2 --epochs 200
"""

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vis_athenak.utils.units import Units

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.coarse_grain_utils import block_mean
from models import stencil
from models.stencil import b, masked_mean

base_channels = ["log_rho", "log_T", "ux", "uy", "uz", "s0"]  # cached X
channels = [*base_channels,
            "dx_log_rho", "dy_log_rho", "dz_log_rho",
            "dx_log_T", "dy_log_T", "dz_log_T",
            "dx_s0", "dy_s0", "dz_s0",
            "S_xx", "S_yy", "S_zz", "S_xy", "S_xz", "S_yz",
            "w_x", "w_y", "w_z"]
in_types = ["scalar", "scalar", "vector", "scalar",                     # augment.py types
            "vector", "vector", "vector", "sym_tensor", "pseudovector"]
out_types = ["scalar", "scalar"]  # targets: gate, f_mix
fields = ["dens", "velx", "vely", "velz", "eint", "s_00"]
PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

T_MIX = (1.05e4, 0.95e6)  # mixing range [K], the cooling range of ismcooling.hpp
LOG_T_EDGES = np.linspace(3.0, 7.0, 61)  # 60 bins of the subgrid temperature PDF


# ----------------------------------------------------------------------------- data

def temperature_pdf(rho, T):
    """Mass-weighted subgrid temperature PDF (60, nz, ny, nx) of the b^3 blocks of the fine
    fields rho, T: the mass fraction of each block in each bin of LOG_T_EDGES, gas outside
    10^3-10^7 K counted in the end bins."""
    n_bins = len(LOG_T_EDGES) - 1
    width = (LOG_T_EDGES[-1] - LOG_T_EDGES[0]) / n_bins
    bins = jnp.clip(jnp.floor((jnp.log10(T) - LOG_T_EDGES[0]) / width), 0, n_bins - 1)
    shape = tuple(n // b for n in rho.shape)
    block = sum(jnp.arange(n)[(slice(None),) + (None,) * (2 - a)] // b * stride
                for a, (n, stride) in enumerate(zip(rho.shape, (shape[1] * shape[2], shape[2], 1))))
    mass = jnp.bincount((block * n_bins + bins.astype(block.dtype)).ravel(), weights=rho.ravel(),
                        length=np.prod(shape) * n_bins).reshape(*shape, n_bins)
    return jnp.moveaxis(mass / mass.sum(axis=-1, keepdims=True), -1, 0)


def coarse_frame(frame, params):
    """Resolved inputs (len(base_channels), nz, ny, nx), mixing mass fraction f_mix
    (nz, ny, nx) and {"pdf": temperature_pdf} of one frame, with the averaging conventions
    of cooling_model.coarse_frame.  Computed on the device in float64, returned in float32."""
    units = Units.from_params(params)
    gamma = float(params["hydro"]["gamma"])
    with jax.enable_x64(True):
        field = lambda k: jnp.asarray(frame[k]).astype(jnp.float64)
        rho = field("dens")
        P = (gamma - 1) * field("eint")
        T = units.temperature_cgs * P / rho
        rho_c, P_c = block_mean(rho, b), block_mean(P, b)
        u_c = [block_mean(rho * field(v), b) / rho_c for v in ("velx", "vely", "velz")]
        s_c = block_mean(rho * field("s_00"), b) / rho_c
        T_c = units.temperature_cgs * P_c / rho_c

        f_mix = block_mean(rho * ((T > T_MIX[0]) & (T <= T_MIX[1])), b) / rho_c
        x = jnp.stack([jnp.log10(rho_c), jnp.log10(T_c), *u_c, s_c]).astype(jnp.float32)
        return x, f_mix.astype(jnp.float32), {"pdf": temperature_pdf(rho, T).astype(jnp.float32)}


def load_dataset(every, cache):
    """Coarse inputs X (frames, len(base_channels), nz, ny, nx) and f_mix (frames, nz, ny, nx),
    cached; the PDFs stay in the cache (load_pdf)."""
    return stencil.load_dataset(every, cache, fields, coarse_frame)


def load_pdf(cache, i):
    """Subgrid temperature PDF (60, nz, ny, nx) of frame i from the cache."""
    with np.load(stencil.frame_path(stencil.cache_dir(cache), i)) as d:
        return d["pdf"]


def gradient(f):
    """Central differences (d/dx, d/dy, d/dz) of f (..., nz, ny, nx) in units of a coarse
    cell, with the run's boundaries (stencil.pad: periodic in x and z, edge in y)."""
    lead = f.shape[:-3]
    p = stencil.pad(f.reshape(-1, 1, *f.shape[-3:]), 1)[:, 0]
    d = lambda a: (jnp.roll(p, -1, a) - jnp.roll(p, 1, a))[:, 1:-1, 1:-1, 1:-1] / 2
    return [d(a).reshape(*lead, *f.shape[-3:]) for a in (3, 2, 1)]


def features(X):
    """Cached inputs X (frames, len(base_channels), ...) -> network inputs (frames,
    len(channels), ...): the fields, grad log rho, grad log T, grad s0, the strain rate
    S_ij = (d_j u_i + d_i u_j)/2 (PAIRS order) and the vorticity curl u."""
    log_rho, log_T, ux, uy, uz, s0 = (X[:, c] for c in range(len(base_channels)))
    du = [gradient(u) for u in (ux, uy, uz)]  # du[i][j] = d_j u_i
    S = [(du[i][j] + du[j][i]) / 2 for i, j in PAIRS]
    w = [du[2][1] - du[1][2], du[0][2] - du[2][0], du[1][0] - du[0][1]]
    return jnp.concatenate([X, jnp.stack([*gradient(log_rho), *gradient(log_T), *gradient(s0),
                                          *S, *w], axis=1)], axis=1)


def gate(f_mix, min_frac=0.0):
    """1 where the mixing mass fraction exceeds min_frac (0: any mixing gas), else 0."""
    return (f_mix > min_frac).astype(jnp.float32)


# ----------------------------------------------------------------------------- model

def apply(params, x):
    """(B, C, nz + 2h, ny + 2h, nx + 2h) normalised, padded inputs -> (B, nz, ny, nx) logit
    of the gate.  Works on models with or without the auxiliary f_mix output."""
    return stencil.apply(params, x)[..., 0]


def loss_fn(params, x, y, w_pos=1.0, w_neg=1.0, w_aux=1.0):
    """Gate loss plus the auxiliary f_mix loss on padded inputs x and targets y (B, 2, nz,
    ny, nx) = (gate, f_mix).

    The gate term is the binary cross-entropy averaged separately over the mixing (gate = 1)
    and non-mixing (gate = 0) cells, so neither class is swamped by the other whatever their
    ratio; w_neg > w_pos penalises false positives harder.  The auxiliary term is the mean
    squared error of sigmoid(second output) against f_mix over all cells.
    Returns (total, (pos, neg, aux)), the terms unweighted."""
    out = stencil.apply(params, x)
    z, f_hat, gate_true, f_true = out[..., 0], jax.nn.sigmoid(out[..., 1]), y[:, 0], y[:, 1]
    bce = jnp.maximum(z, 0) - z * gate_true + jnp.log1p(jnp.exp(-jnp.abs(z)))
    pos, neg = masked_mean(bce, gate_true > 0.5), masked_mean(bce, gate_true < 0.5)
    aux = jnp.mean((f_hat - f_true) ** 2)
    return w_pos * pos + w_neg * neg + w_aux * aux, (pos, neg, aux)


def scores(params, X, Y, threshold=0.5, batch=8):
    """Classification scores of the gate (probability > threshold) on padded frames X with
    targets Y (frames, 2, ...) (gate first): base rate, accuracy, precision, recall, F1 and
    IoU of the mixing class."""
    p = jnp.concatenate([jax.nn.sigmoid(apply(params, X[i:i + batch]))
                         for i in range(0, len(X), batch)])
    pred, true = p > threshold, Y[:, 0] > 0.5
    tp, fp, fn = (jnp.sum(pred & true), jnp.sum(pred & ~true), jnp.sum(~pred & true))
    precision, recall = tp / jnp.maximum(tp + fp, 1), tp / jnp.maximum(tp + fn, 1)
    return {"base_rate": float(jnp.mean(true)), "accuracy": float(jnp.mean(pred == true)),
            "precision": float(precision), "recall": float(recall),
            "f1": float(2 * precision * recall / jnp.maximum(precision + recall, 1e-12)),
            "iou": float(tp / jnp.maximum(tp + fp + fn, 1))}


# ----------------------------------------------------------------------------- training

def train(args):
    X, F, t = load_dataset(args.every, Path(args.cache))
    print(f"{len(X)} frames of {X.shape[2:]} coarse cells, t = {t.min():.1f}-{t.max():.1f}")
    X, Y = features(X), jnp.stack([gate(F, args.min_frac), F], axis=1)  # (frames, 2, ...)

    val = stencil.val_mask(len(X), args.val_every)
    Xtr, Ytr, Xva, Yva = X[~val], Y[~val], X[val], Y[val]
    print(f"mixing cells (gate = 1): {float(jnp.mean(Ytr[:, 0])):.1%} train, "
          f"{float(jnp.mean(Yva[:, 0])):.1%} validation")

    x_mean, x_std = stencil.standardise(Xtr, in_types)
    norm = lambda X: stencil.pad((X - x_mean) / x_std, args.stencil // 2)
    Xtr, Xva = norm(Xtr), norm(Xva)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = stencil.init_params(init_key, X.shape[1], args.stencil, args.hidden, 2)

    loss = lambda p, x, y: loss_fn(p, x, y, args.w_pos, args.w_neg, args.w_aux)
    params, history = stencil.fit(params, loss, ("pos", "neg", "aux"),
                                  Xtr, Ytr, Xva, Yva, key, args, in_types, out_types)

    metrics = {"train": scores(params, Xtr, Ytr), "val": scores(params, Xva, Yva)}
    for split, m in metrics.items():
        print(f"{split:5s} " + "  ".join(f"{k} {v:.4f}" for k, v in m.items()))

    with open(args.out, "wb") as f:
        pickle.dump({"params": jax.device_get(params),
                     "x_mean": jax.device_get(x_mean), "x_std": jax.device_get(x_std),
                     "channels": channels, "min_frac": args.min_frac, "t_mix": T_MIX,
                     "w_pos": args.w_pos, "w_neg": args.w_neg, "w_aux": args.w_aux,
                     "metrics": metrics,
                     **stencil.common_meta(args), "history": history}, f)
    print(f"saved {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    stencil.add_training_args(parser, "gate_dataset_b16", "gate_model.pkl")
    parser.add_argument("--min-frac", type=float, default=0.0,
                        help="gate = 1 where the mixing mass fraction exceeds this (0: any mass)")
    parser.add_argument("--w-pos", type=float, default=1.0,
                        help="weight of the cross-entropy over mixing cells (gate = 1)")
    parser.add_argument("--w-neg", type=float, default=1.0,
                        help="weight of the cross-entropy over non-mixing cells (gate = 0); "
                             "above --w-pos it penalises false positives harder")
    parser.add_argument("--w-aux", type=float, default=1.0,
                        help="weight of the auxiliary mean squared error on f_mix (0: unused)")
    train(parser.parse_args())
