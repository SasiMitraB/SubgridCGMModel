"""
Stencil model for the subgrid cooling correction (theory/notes/Overleaf, eq-cooling):
  C = <n^2 Lambda(T)> - (<rho>/mu m_u)^2 Lambda(T_CG)
predicted on the coarse 16x64x16 grid from the resolved (coarse-grained) fields of the
256x1024x256 KH run (block size b = 16).

The model looks at a stencil x stencil x stencil box of coarse cells and returns one value
for the centre cell (the network, padding and training loop are in stencil.py).

  python models/cooling_model.py --every 2 --epochs 200
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
from models import stencil
from models.stencil import b, masked_mean

channels = ["log_rho", "log_T", "ux", "uy", "uz", "s0"]
in_types = ["scalar", "scalar", "vector", "scalar"]  # augment.py field types of the channels
out_types = None  # C is a bare scalar field
fields = ["dens", "velx", "vely", "velz", "eint", "s_00"]
linthresh = 1e-27  # |C| below this is linear in the symlog target [erg cm^-3 s^-1]


# ----------------------------------------------------------------------------- data

# ISMCoolFn is linear in log T between nodes 0.04 dex apart; its node values let jnp.interp
# reproduce it (to ~1e-14 over 10^4.2 < T <= 0.95e6 K) on the device.
SPEX_LOG_T = [4.12 + 0.04 * i for i in range(102)]
SPEX_LOG_LAMBDA = [math.log10(v) for v in ISMCoolFn([10**l for l in SPEX_LOG_T]).tolist()]


def cooling_function(T):
    """Lambda(T) [erg cm^3 s^-1] as in ismcooling.hpp: zero outside 1.05e4 < T <= 0.95e6 K,
    Koyama & Inutsuka (2002) for T <= 10^4.2 K, the SPEX table (ISMCoolFn) above."""
    koyama = 2.0e-19 * jnp.exp(-1.184e5 / (T + 1.0e3)) + 2.8e-28 * jnp.sqrt(T) * jnp.exp(-92.0 / T)
    spex = 10**jnp.interp(jnp.log10(T), jnp.array(SPEX_LOG_T), jnp.array(SPEX_LOG_LAMBDA))
    Lambda = jnp.where(T <= 10**4.2, koyama, spex)
    return jnp.where((T > 1.05e4) & (T <= 0.95e6), Lambda, 0.0)


def coarse_frame(frame, params):
    """Resolved inputs (len(channels), nz, ny, nx) and correction C (nz, ny, nx) [erg cm^-3 s^-1]
    of one frame, with the averaging conventions of coarse_grain_utils.coarse_grain
    (volume-averaged rho, P; mass-weighted velocities and scalar; T_CG from <P>/<rho>).
    Computed on the device in float64 (C is a small difference of large terms), returned
    in float32."""
    units = Units.from_params(params)
    gamma = float(params["hydro"]["gamma"])
    with jax.enable_x64(True):
        field = lambda k: jnp.asarray(frame[k]).astype(jnp.float64)
        rho = field("dens")
        P = (gamma - 1) * field("eint")
        rho_c, P_c = block_mean(rho, b), block_mean(P, b)
        u_c = [block_mean(rho * field(v), b) / rho_c for v in ("velx", "vely", "velz")]
        s_c = block_mean(rho * field("s_00"), b) / rho_c
        true_c = block_mean((units.number_density_cgs * rho)**2
                            * cooling_function(units.temperature_cgs * P / rho), b)

        T_c = units.temperature_cgs * P_c / rho_c
        resolved = (units.number_density_cgs * rho_c)**2 * cooling_function(T_c)
        x = jnp.stack([jnp.log10(rho_c), jnp.log10(T_c), *u_c, s_c]).astype(jnp.float32)
        return x, (true_c - resolved).astype(jnp.float32)


def load_dataset(every, cache):
    """Coarse inputs X (frames, C, nz, ny, nx) and targets Y (frames, nz, ny, nx), cached."""
    return stencil.load_dataset(every, cache, fields, coarse_frame)


def symlog(C, linthresh=linthresh):
    """sign(C) log10(1 + |C|/linthresh): linear for |C| << linthresh, log above it."""
    return jnp.sign(C) * jnp.log10(1 + jnp.abs(C) / linthresh)


def inv_symlog(y, linthresh=linthresh):
    """C [erg cm^-3 s^-1] from symlog(C)."""
    return jnp.sign(y) * linthresh * (10**jnp.abs(y) - 1)


# ----------------------------------------------------------------------------- model

def apply(params, x):
    """(B, C, nz + 2h, ny + 2h, nx + 2h) normalised, padded inputs -> (B, nz, ny, nx)
    symlog(C)."""
    return stencil.apply(params, x)[..., 0]


def loss_fn(params, x, y, w_pos=1.0, w_neg=1.0):
    """MSE over all cells plus separate MSEs over the heating (C > 0) and cooling (C < 0)
    cells.  ~93% of cells have C = 0, so the full MSE keeps them pinned to zero while the
    two sign terms, each averaged over its own cells, stop the sparse signed cells from
    being swamped.  Returns (total, (all, pos, neg))."""
    se = (apply(params, x) - y)**2
    mse, pos, neg = jnp.mean(se), masked_mean(se, y > 0), masked_mean(se, y < 0)
    return mse + w_pos * pos + w_neg * neg, (mse, pos, neg)


# ----------------------------------------------------------------------------- training

def train(args):
    X, Y, t = load_dataset(args.every, Path(args.cache))
    print(f"{len(X)} frames of {X.shape[2:]} coarse cells, t = {t.min():.1f}-{t.max():.1f}")

    val = stencil.val_mask(len(X), args.val_every)
    Xtr, Ytr, Xva, Yva = X[~val], Y[~val], X[val], Y[val]

    # per-channel input standardisation; MSE on symlog(C).
    x_mean, x_std = stencil.standardise(Xtr, in_types)
    norm = lambda X, Y: (stencil.pad((X - x_mean) / x_std, args.stencil // 2),
                         symlog(Y, args.linthresh).astype(jnp.float32))
    Xtr, Ytr = norm(Xtr, Ytr)
    Xva, Yva = norm(Xva, Yva)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    params = stencil.init_params(init_key, X.shape[1], args.stencil, args.hidden, 1)

    loss = lambda p, x, y: loss_fn(p, x, y, args.w_pos, args.w_neg)
    params, history = stencil.fit(params, loss, ("mse", "pos", "neg"),
                                  Xtr, Ytr, Xva, Yva, key, args, in_types, out_types)

    with open(args.out, "wb") as f:
        pickle.dump({"params": jax.device_get(params),
                     "x_mean": jax.device_get(x_mean), "x_std": jax.device_get(x_std),
                     "linthresh": args.linthresh, "channels": channels,
                     "w_pos": args.w_pos, "w_neg": args.w_neg,
                     **stencil.common_meta(args), "history": history}, f)
    print(f"saved {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    stencil.add_training_args(parser, "cooling_dataset_b16", "cooling_model.pkl")
    parser.add_argument("--linthresh", type=float, default=linthresh,
                        help="linear range of the symlog target [erg cm^-3 s^-1]")
    parser.add_argument("--w-pos", type=float, default=1.0,
                        help="weight of the MSE over heating cells (C > 0)")
    parser.add_argument("--w-neg", type=float, default=1.0,
                        help="weight of the MSE over cooling cells (C < 0)")
    train(parser.parse_args())
