"""
Stencil model for the subgrid cooling correction (theory/notes/Overleaf, eq-cooling,
eq-cooling-model, eq-gate-cut) on the coarse 16x64x16 grid of the 256x1024x256 KH run
(block size b = 16):

  C   = <n^2 Lcut(T)> - (<rho>/mu m_u)^2 Lcut(T_CG)                       (eq-cooling)
      = g (<rho>/mu m_u)^2 [Lfull(T_CG) e^z - Lcut(T_CG)]              (eq-cooling-model)
  z   = ln <n^2 Lcut(T)> / ((<rho>/mu m_u)^2 Lfull(T_CG))

Lcut is the cooling curve of the run, zero outside the window 1.05e4 < T <= 0.95e6 K, and
Lfull the same curve without the cut-offs.  The gate g is 1 where some subgrid gas cools,
<n^2 Lcut> > 0, except for the cells outside the window with z < k (eq-gate-cut, k = -2.1
loses 0.5% of the cooling).  The network predicts z and is trained on the g = 1 cells
only; C is then built with the known Lfull, Lcut and gate.

The gate is also an input channel, next to the resolved fields.  By default it is the true
g of the targets; with --gate-model it is the mixing-layer gate of models/gate_model.py
(probability > 0.5), which is what is available outside training.

The network, padding and training loop are in stencil.py.

  python models/cooling_model.py --every 2 --epochs 200
  python models/cooling_model.py --gate-model outputs/gate_<timestamp>/gate_model.pkl
"""

import argparse
import math
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from vis_athenak.utils.ism_cooling import ISMCoolFn
from vis_athenak.utils.units import Units

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.coarse_grain_utils import block_mean
from models import gate_model, stencil
from models.stencil import b, masked_mean

base_channels = ["log_rho", "log_T", "ux", "uy", "uz", "s0"]  # cached X
channels = [*base_channels, "gate"]
in_types = ["scalar", "scalar", "vector", "scalar", "scalar"]  # augment.py types
targets = ["z", "g", "inside"]  # training targets built from the cached Y (targets())
out_types = ["scalar"] * len(targets)
fields = ["dens", "velx", "vely", "velz", "eint", "s_00"]
T_WINDOW = (1.05e4, 0.95e6)  # Lcut = 0 outside this window [K]
K_CUT = -2.1  # gate cut on z outside the window (eq-gate-cut)
LN10 = math.log(10.0)


# ----------------------------------------------------------------------------- data

# ISMCoolFn is linear in log T between nodes 0.04 dex apart; its node values let jnp.interp
# reproduce it (to ~1e-14 over 10^4.2 < T <= 10^8.16 K) on the device.
SPEX_LOG_T = [4.12 + 0.04 * i for i in range(102)]
SPEX_LOG_LAMBDA = [math.log10(v) for v in ISMCoolFn([10**l for l in SPEX_LOG_T]).tolist()]


def lambda_full(T):
    """Lfull(T) [erg cm^3 s^-1]: Koyama & Inutsuka (2002) for T <= 10^4.2 K, the SPEX table
    (ISMCoolFn) above, positive at every T (cooling_curve.stock_lambda of the notes)."""
    koyama = 2.0e-19 * jnp.exp(-1.184e5 / (T + 1.0e3)) + 2.8e-28 * jnp.sqrt(T) * jnp.exp(-92.0 / T)
    spex = 10**jnp.interp(jnp.log10(T), jnp.array(SPEX_LOG_T), jnp.array(SPEX_LOG_LAMBDA))
    return jnp.where(T <= 10**4.2, koyama, spex)


def in_window(T):
    return (T > T_WINDOW[0]) & (T <= T_WINDOW[1])


def cooling_function(T):
    """Lcut(T) [erg cm^3 s^-1] as in ismcooling.hpp: Lfull, zero outside the window."""
    return jnp.where(in_window(T), lambda_full(T), 0.0)


def coarse_frame(frame, params):
    """Resolved inputs (len(base_channels), nz, ny, nx) and Y (3, nz, ny, nx) of one frame:
    log10 <n^2 Lcut(T)> (-inf where no subgrid gas cools), log10 (<rho>/mu m_u)^2 Lfull(T_CG)
    [erg cm^-3 s^-1] and whether T_CG lies in the window, with the averaging conventions of
    coarse_grain_utils.coarse_grain (volume-averaged rho, P; mass-weighted velocities and
    scalar; T_CG from <P>/<rho>).  Computed on the device in float64 (Lfull spans many
    decades below 10^4 K), returned in float32."""
    units = Units.from_params(params)
    gamma = float(params["hydro"]["gamma"])
    with jax.enable_x64(True):
        field = lambda k: jnp.asarray(frame[k]).astype(jnp.float64)
        rho = field("dens")
        P = (gamma - 1) * field("eint")
        rho_c, P_c = block_mean(rho, b), block_mean(P, b)
        u_c = [block_mean(rho * field(v), b) / rho_c for v in ("velx", "vely", "velz")]
        s_c = block_mean(rho * field("s_00"), b) / rho_c
        num = block_mean((units.number_density_cgs * rho)**2
                         * cooling_function(units.temperature_cgs * P / rho), b)

        T_c = units.temperature_cgs * P_c / rho_c
        full = (units.number_density_cgs * rho_c)**2 * lambda_full(T_c)
        x = jnp.stack([jnp.log10(rho_c), jnp.log10(T_c), *u_c, s_c]).astype(jnp.float32)
        y = jnp.stack([jnp.log10(num), jnp.log10(full), in_window(T_c)]).astype(jnp.float32)
        return x, y


def load_dataset(every, cache):
    """Coarse inputs X (frames, len(base_channels), nz, ny, nx) and Y (frames, 3, nz, ny, nx)
    of coarse_frame, cached."""
    return stencil.load_dataset(every, cache, fields, coarse_frame)


def targets(Y, k=K_CUT):
    """z (0 where g = 0), the gate g of eq-gate-cut with cut k, and the window flag, each
    (frames, nz, ny, nx) float32, from the cached Y."""
    log_num, log_full, inside = Y[:, 0], Y[:, 1], Y[:, 2]
    cools = jnp.isfinite(log_num)
    z = jnp.where(cools, LN10 * (log_num - log_full), 0.0)
    g = cools & ((inside > 0.5) | (z >= k))
    return jnp.where(g, z, 0.0), g.astype(jnp.float32), inside


def correction(z, g, Y):
    """C [erg cm^-3 s^-1] of eq-cooling-model from z, a gate g and the cached Y."""
    return jnp.where(g > 0.5, 10**Y[:, 1] * (jnp.exp(z) - Y[:, 2]), 0.0)


def true_correction(Y):
    """C = <n^2 Lcut> - (<rho>/mu m_u)^2 Lcut(T_CG) of the cached Y, with no gate cut."""
    return 10**Y[:, 0] - Y[:, 2] * 10**Y[:, 1]


def gate_input(X, Y, k=K_CUT, gate_path=None, batch=8):
    """The gate channel (frames, 1, nz, ny, nx): the true g of targets(Y, k), or with
    gate_path the gate model's prediction (probability > 0.5) from the inputs X."""
    if gate_path is None:
        return targets(Y, k)[1][:, None]
    with open(gate_path, "rb") as f:
        m = pickle.load(f)
    assert m["channels"][:len(base_channels)] == base_channels, m["channels"]
    norm = lambda x: stencil.pad((gate_model.features(x) - m["x_mean"]) / m["x_std"],
                                 m["stencil"] // 2)
    p = jnp.concatenate([jax.nn.sigmoid(gate_model.apply(m["params"], norm(X[i:i + batch])))
                         for i in range(0, len(X), batch)])
    return (p > 0.5).astype(jnp.float32)[:, None]


# ----------------------------------------------------------------------------- model

def apply(params, x):
    """(B, C, nz + 2h, ny + 2h, nx + 2h) normalised, padded inputs -> (B, nz, ny, nx) z."""
    return stencil.apply(params, x)[..., 0]


def loss_fn(params, x, y, w_in=1.0, w_out=1.0):
    """MSE of z over the gated cells, averaged separately over the cells with T_CG inside
    the window and outside it (the steep tails of fig-cooling_zc), so the few outside
    cells are not swamped.  y = (z, g, inside).  Returns (total, (in, out))."""
    z, g, inside = y[:, 0], y[:, 1], y[:, 2]
    se = (apply(params, x) - z)**2
    mse_in, mse_out = masked_mean(se, g * inside), masked_mean(se, g * (1 - inside))
    return w_in * mse_in + w_out * mse_out, (mse_in, mse_out)


def predict(m, X, Y, batch=8):
    """z predicted by the saved model m for cached inputs X and Y, and the gate channel it
    was given (frames, nz, ny, nx)."""
    gate = gate_input(X, Y, m["k_cut"], m["gate_model"])
    x = stencil.pad((jnp.concatenate([X, gate], axis=1) - m["x_mean"]) / m["x_std"],
                    m["stencil"] // 2)
    z = jnp.concatenate([apply(m["params"], x[i:i + batch]) for i in range(0, len(x), batch)])
    return z, gate[:, 0]


def scores(z_pred, gate, Y, k=K_CUT):
    """RMSE of z on the gated cells inside and outside the window, and the error in the
    cooling: sum |C_pred - C_true| / sum <n^2 Lcut> and the relative error of the total
    cooling sum <n^2 Lcut>, with C_pred built with the gate given."""
    z, g, inside = targets(Y, k)
    rmse = lambda mask: float(jnp.sqrt(masked_mean((z_pred - z)**2, mask)))
    C_true, C_pred = true_correction(Y), correction(z_pred, gate, Y)
    total = jnp.sum(10**Y[:, 0])
    return {"rmse_in": rmse(g * inside), "rmse_out": rmse(g * (1 - inside)),
            "abs_err": float(jnp.sum(jnp.abs(C_pred - C_true)) / total),
            "total_err": float(jnp.sum(C_pred - C_true) / total)}


# ----------------------------------------------------------------------------- training

def train(args):
    X, Y, t = load_dataset(args.every, Path(args.cache))
    # the split is made on all frames, so the validation frames are the same whatever t_min
    val = stencil.val_mask(len(X), args.val_every)
    keep = t >= args.t_min
    X, Y, t, val = X[keep], Y[keep], t[keep], val[keep]
    z, g, inside = targets(Y, args.k_cut)
    gate = gate_input(X, Y, args.k_cut, args.gate_model)
    print(f"{len(X)} frames of {X.shape[2:]} coarse cells, t = {t.min():.2f}-{t.max():.2f} "
          f"(t_min = {args.t_min:g}); g = 1 in {float(jnp.mean(g)):.1%} of cells "
          f"({float(jnp.mean(g * inside)):.1%} inside the window), gate input "
          f"{args.gate_model or 'true g'} = 1 in {float(jnp.mean(gate)):.1%}")

    Xin, Yt = jnp.concatenate([X, gate], axis=1), jnp.stack([z, g, inside], axis=1)
    Xtr, Ytr, Xva, Yva = Xin[~val], Yt[~val], Xin[val], Yt[val]
    x_mean, x_std = stencil.standardise(Xtr, in_types)
    norm = lambda X: stencil.pad((X - x_mean) / x_std, args.stencil // 2)
    Xtr, Xva = norm(Xtr), norm(Xva)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = stencil.init_params(init_key, Xin.shape[1], args.stencil, args.hidden, 1)

    loss = lambda p, x, y: loss_fn(p, x, y, args.w_in, args.w_out)
    params, history = stencil.fit(params, loss, ("in", "out"),
                                  Xtr, Ytr, Xva, Yva, key, args, in_types, out_types)

    m = {"params": jax.device_get(params),
         "x_mean": jax.device_get(x_mean), "x_std": jax.device_get(x_std),
         "channels": channels, "k_cut": args.k_cut, "gate_model": args.gate_model,
         "t_min": args.t_min, "w_in": args.w_in, "w_out": args.w_out,
         **stencil.common_meta(args), "history": history}
    z_pred, gate_used = predict(m, X, Y)
    m["metrics"] = {s: scores(z_pred[sel], gate_used[sel], Y[sel], args.k_cut)
                    for s, sel in (("train", ~val), ("val", val))}
    for s, v in m["metrics"].items():
        print(f"{s:5s} " + "  ".join(f"{k} {x:.4f}" for k, x in v.items()))

    with open(args.out, "wb") as f:
        pickle.dump(m, f)
    print(f"saved {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    stencil.add_training_args(parser, "cooling_z_dataset_b16", "cooling_model.pkl")
    parser.add_argument("--k-cut", type=float, default=K_CUT,
                        help="gate cut: g = 0 for the cells outside the window with z < k")
    parser.add_argument("--gate-model", default=None,
                        help="gate_model.pkl whose prediction is the gate input "
                             "(default: the true gate)")
    parser.add_argument("--t-min", type=float, default=2.5,
                        help="drop the frames before this time, the initial transient [Myr]")
    parser.add_argument("--w-in", type=float, default=1.0,
                        help="weight of the z MSE over gated cells inside the window")
    parser.add_argument("--w-out", type=float, default=1.0,
                        help="weight of the z MSE over gated cells outside the window")
    train(parser.parse_args())
