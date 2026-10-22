"""
Stencil model for the subgrid momentum flux R (theory/notes/Overleaf, chapter 2, sec:R-decomp)
  R_ij = <rho> tau(v_i, v_j) = 2 <rho> k_sg S_ij                              (eq-R-split)
predicted on the coarse 16x64x16 grid from the resolved fields of the 256x1024x256 KH run
(block size b = 16).

The network returns six unbounded numbers z_0..z_5 per coarse cell:
  f = sigmoid(z_0) = k_sg / k_max,   k_max = <E>/<rho> - |<rho v>|^2 / 2<rho>^2   (eq-f, eq-kmax)
  A = sum_k z_k E^(k)  (orthonormal symmetric traceless basis),   S = e^A / Tr e^A  (eq-A, eq-S)
so every output gives a valid R (positive definite, k_sg < k_max, <P> > 0) and every valid
R comes from exactly one output.  k_max is known from the coarse conserved variables alone,
so it is an input; T is not, since <P> depends on the predicted f.

The network, padding and training loop are in stencil.py.  The velocity weights of the
stencil convolution are projected to sum to zero over the stencil, so the model only sees
velocity differences and R is Galilean invariant, as it is by definition.

  python models/rvv_tensor_model.py                  # separate 256^3 networks, 3^3 stencil, 2100 epochs
  python models/rvv_tensor_model.py --lbfgs-steps 200 --lbfgs-chunk 20
"""

import argparse
import pickle
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.coarse_grain_utils import block_mean
from models import stencil
from models.stencil import b, masked_mean

channels = ["log_rho", "log_kmax", "ux", "uy", "uz", "s0"]
fields = ["dens", "velx", "vely", "velz", "eint", "s_00"]
velocity_channels = [2, 3, 4]
targets = ["f", "S_xx", "S_yy", "S_zz", "S_xy", "S_xz", "S_yz"]
in_types = ["scalar", "scalar", "vector", "scalar"]  # augment.py field types
out_types = ["scalar", "sym_tensor", "sym_tensor"]  # f, S and log S (added in train), PAIRS order
PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

# orthonormal basis E^(1..5) of the symmetric traceless 3x3 matrices (sec:R-matrix)
r2, r6 = 1 / 2**0.5, 1 / 6**0.5
BASIS = jnp.array([[[r2, 0, 0], [0, -r2, 0], [0, 0, 0]],
                   [[r6, 0, 0], [0, r6, 0], [0, 0, -2 * r6]],
                   [[0, r2, 0], [r2, 0, 0], [0, 0, 0]],
                   [[0, 0, r2], [0, 0, 0], [r2, 0, 0]],
                   [[0, 0, 0], [0, 0, r2], [0, r2, 0]]], dtype=jnp.float32)


# ----------------------------------------------------------------------------- data

def coarse_frame(frame, params):
    """Resolved inputs (len(channels), nz, ny, nx) and targets (len(targets), nz, ny, nx) of
    one frame in code units: volume-averaged rho and E, mass-weighted velocities and scalar
    (coarse_grain_utils conventions), f = k_sg/k_max and S = R/Tr(R).

    tau_ij = R_ij/<rho> is built from central moments <rho dv_i dv_j>/<rho>, dv = v minus
    the Favre mean of the block, which stays positive semi-definite in floating point.
    Computed on the device in float64 (f spans ~15 decades), returned in float32."""
    with jax.enable_x64(True):
        field = lambda k: jnp.asarray(frame[k]).astype(jnp.float64)
        rho = field("dens")
        rho_c = block_mean(rho, b)
        up = (slice(None), None) * 3  # coarse (Z, Y, X) -> broadcast over the blocks
        nz, ny, nx = rho.shape
        blocks = lambda f: f.reshape(nz // b, b, ny // b, b, nx // b, b)

        u_c, dv = [], []
        for v in ("velx", "vely", "velz"):
            vel = field(v)
            u_c.append(block_mean(rho * vel, b) / rho_c)
            dv.append(blocks(vel) - u_c[-1][up])
        tau = [jnp.mean(blocks(rho) * dv[i] * dv[j], axis=(1, 3, 5)) / rho_c for i, j in PAIRS]
        s_c = block_mean(rho * field("s_00"), b) / rho_c

        k_sg = 0.5 * (tau[0] + tau[1] + tau[2])
        e_th = block_mean(field("eint"), b) / rho_c  # <P>/[(gamma-1)<rho>]
        k_max = e_th + k_sg
        tr = jnp.where(k_sg > 0, 2 * k_sg, 1.0)
        S = [jnp.where(k_sg > 0, t / tr, (i == j) / 3) for t, (i, j) in zip(tau, PAIRS)]

        x = jnp.stack([jnp.log10(rho_c), jnp.log10(k_max), *u_c, s_c]).astype(jnp.float32)
        y = jnp.stack([k_sg / k_max, *S]).astype(jnp.float32)
        return x, y


def load_dataset(every, cache):
    """Coarse inputs X (frames, C, nz, ny, nx), targets Y (frames, 7, nz, ny, nx), cached."""
    return stencil.load_dataset(every, cache, fields, coarse_frame)


# ----------------------------------------------------------------------------- R from z

def shape_tensor(z):
    """(..., 5) outputs z_1..z_5 -> (..., 3, 3) shape tensor S = e^A / Tr e^A (eq-S).

    e^A is taken with expm rather than through eigh: the eigh gradient has 1/(l_i - l_j)
    terms and is nan at the isotropic point A = 0.  A is shifted down by a Gershgorin bound
    on its largest eigenvalue so that e^A cannot overflow; the shift cancels in the ratio."""
    A = jnp.einsum("...k,kij->...ij", z, BASIS)
    lmax = jnp.max(jnp.diagonal(A, axis1=-2, axis2=-1)
                   + jnp.sum(jnp.abs(A), axis=-1) - jnp.abs(jnp.diagonal(A, axis1=-2, axis2=-1)),
                   axis=-1)
    A = A - jax.lax.stop_gradient(lmax)[..., None, None] * jnp.eye(3)
    shape = A.shape
    eA = jax.vmap(expm)(A.reshape(-1, 3, 3)).reshape(shape)
    return eA / jnp.trace(eA, axis1=-2, axis2=-1)[..., None, None]


def shape_outputs(S):
    """Inverse of shape_tensor: (..., 3, 3) positive-definite S -> (..., 5) z_k
    = <log S - Tr(log S) I/3, E^(k)>_F.  For diagnostics (eigh: not differentiable at
    degenerate S)."""
    s, V = jnp.linalg.eigh(S)
    logs = jnp.log(s)
    logs = logs - jnp.mean(logs, axis=-1, keepdims=True)
    A = jnp.einsum("...ik,...k,...jk->...ij", V, logs, V)
    return jnp.einsum("...ij,kij->...k", A, BASIS)


def rvv(f, S, rho, k_max):
    """R_ij = 2 <rho> f k_max S_ij (eq-R-split, eq-f); f, rho, k_max (...), S (..., 3, 3)."""
    return (2 * rho * f * k_max)[..., None, None] * S


# ----------------------------------------------------------------------------- model

def init_params(key, n_in, stencil_width, hidden, z0_bias=0.0, separate=False):
    """stencil.init_params with 6 outputs and the last layer at zero, so the model starts
    from S = I/3 and f = sigmoid(z0_bias).  separate: two networks of the same size, one
    for z_0 (f) and one for z_1..z_5 (S), as a dict {"f": ..., "S": ...}, so that the
    two targets do not compete for the hidden units."""
    def net(key, n_out):
        params = stencil.init_params(key, n_in, stencil_width, hidden, n_out)
        params[-1]["w"] = jnp.zeros_like(params[-1]["w"])
        params[-1]["b"] = params[-1]["b"].at[0].set(z0_bias if n_out != 5 else 0.0)
        return params
    if not separate:
        return net(key, 6)
    key_f, key_S = jax.random.split(key)
    return {"f": net(key_f, 1), "S": net(key_S, 5)}


def galilean(w0):
    """Remove the stencil mean of the velocity weights: a uniform boost then drops out."""
    mask = jnp.zeros(w0.shape[3]).at[jnp.array(velocity_channels)].set(1.0)[:, None]
    return w0 - mask * w0.mean(axis=(0, 1, 2), keepdims=True)


def apply(params, x):
    """(B, C, nz + 2h, ny + 2h, nx + 2h) normalised, padded inputs -> (B, nz, ny, nx, 6)
    outputs z_0..z_5, from one network or from the two of init_params(separate=True)."""
    net = lambda params: stencil.apply(params, x, galilean(params[0]["w"]))
    if isinstance(params, dict):
        return jnp.concatenate([net(params["f"]), net(params["S"])], axis=-1)
    return net(params)


def predict(params, x):
    """(B, C, nz + 2h, ny + 2h, nx + 2h) normalised, padded inputs -> f (B, nz, ny, nx),
    S (B, nz, ny, nx, 3, 3)."""
    z = apply(params, x)
    return jax.nn.sigmoid(z[..., 0]), shape_tensor(z[..., 1:])


def unpack_S(y, start=1):
    """(B, C, nz, ny, nx) targets -> the symmetric tensor in channels start..start+5
    (PAIRS order) as (B, nz, ny, nx, 3, 3): S for start = 1, log S for start = 7."""
    S = jnp.zeros(y.shape[:1] + y.shape[2:] + (3, 3), y.dtype)
    for c, (i, j) in enumerate(PAIRS, start=start):
        S = S.at[..., i, j].set(y[:, c]).at[..., j, i].set(y[:, c])
    return S


LOG_FLOOR = 1e-6  # eigenvalue floor of S (unit trace) before the log, ~ float32 resolution

def log_shape(Y, floor=LOG_FLOOR):
    """(frames, 7, nz, ny, nx) targets -> traceless log S (frames, 6, nz, ny, nx) in PAIRS
    order, whose components on BASIS are the z_1..z_5 that give S, and the fraction of
    eigenvalues raised to floor.  Taken in float64 from the cached float32 S, whose entries
    resolve eigenvalues down to ~1e-6; below that they would be rounding noise.  After the
    initial transient (t >~ 2.5) the smallest eigenvalue is ~1e-4, so the floor is a guard."""
    S = np.asarray(unpack_S(jnp.asarray(Y)), dtype=np.float64)
    s, V = np.linalg.eigh(S)
    clipped = float(np.mean(s < floor))
    logs = np.log(np.maximum(s, floor))
    logs -= logs.mean(axis=-1, keepdims=True)
    L = np.einsum("...ik,...k,...jk->...ij", V, logs, V)
    return jnp.asarray(np.stack([L[..., i, j] for i, j in PAIRS], axis=1), jnp.float32), clipped


def loss_fn(params, x, y, f_floor, f_shape, w_trace, w_temp, w_shape):
    """Trace, temperature and shape losses, kept separate since f spans ~15 decades and S is O(1).

    trace: MSE of ln f over all cells, the target floored at f_floor (pure hot or cold
           cells have f -> 0, and ln f = -softplus(-z_0) is stable for any z_0).
    temp:  MSE of ln T over all cells.  With <P> = (gamma-1) <rho> k_max (1 - f) (eq-f) and
           T = (mu m_p / k_B) <P>/<rho> (eq-temperature), and <rho>, k_max known inputs,
             ln T_pred - ln T = ln P_pred - ln P = ln(1 - f_pred) - ln(1 - f),
           so this one term is both the temperature and the pressure loss.  It weighs the
           cells where k_sg takes a real share of the unresolved energy, and ignores the
           ln f error of cells with f -> 0, whose T is right either way.
           ln(1 - f_pred) = -softplus(z_0) is stable for any z_0.
    shape: log-Euclidean distance sum_k (z_k - z_k^true)^2 = ||A - (log S)_traceless||_F^2,
           the outputs z_1..z_5 against the components of log S on BASIS (log_shape), in
           every cell weighted by min(1, f/f_shape).  It measures eigenvalue ratios, so an
           error in a small eigenvalue of S counts as much as one in a large one: the
           error ln(l_pred/l_true) is twice the log of the factor by which R^(1/2), and so
           the energy flux q, is off in that direction.  Rotation invariant (BASIS is
           orthonormal).  Unbounded as l_true -> 0, which only happens in the initial
           transient that train drops (--t-min).  The weight keeps the mixing layer
           in charge while cells with small f still get some gradient.
    The w_* weights only apply with --weighting fixed; the default auto weighting
    (stencil.weighted) balances the three terms by their size.
    Returns (total, (trace, temp, shape))."""
    z = apply(params, x)
    f = y[:, 0]
    trace = jnp.mean((-jax.nn.softplus(-z[..., 0]) - jnp.log(jnp.maximum(f, f_floor)))**2)
    temp = jnp.mean((-jax.nn.softplus(z[..., 0]) - jnp.log1p(-f))**2)
    z_true = jnp.einsum("...ij,kij->...k", unpack_S(y, start=7), BASIS)
    se = jnp.sum((z[..., 1:] - z_true)**2, axis=-1)
    shape = masked_mean(se, jnp.clip(f / f_shape, 0.0, 1.0))
    return w_trace * trace + w_temp * temp + w_shape * shape, (trace, temp, shape)


# ----------------------------------------------------------------------------- training

def train(args):
    X, Y, t = load_dataset(args.every, Path(args.cache))
    # the split is made on all frames, so the validation frames are the same whatever t_min
    val = stencil.val_mask(len(X), args.val_every)
    keep = t >= args.t_min
    X, Y, t, val = X[keep], Y[keep], t[keep], val[keep]
    logS, clipped = log_shape(Y)
    Y = jnp.concatenate([Y, logS], axis=1)
    print(f"{len(X)} frames of {X.shape[2:]} coarse cells, t = {t.min():.2f}-{t.max():.2f} "
          f"(t_min = {args.t_min:g}), {clipped:.2e} of S eigenvalues raised to {LOG_FLOOR:g}")
    print(f"f: median {float(jnp.median(Y[:, 0])):.2e}, "
          f"{float(jnp.mean(Y[:, 0] > args.f_shape)):.1%} of cells "
          f"above f_shape = {args.f_shape:g}")

    Xtr, Ytr, Xva, Yva = X[~val], Y[~val], X[val], Y[val]

    x_mean, x_std = stencil.standardise(Xtr, in_types)
    norm = lambda X, Y: (stencil.pad((X - x_mean) / x_std, args.stencil // 2), Y)
    Xtr, Ytr = norm(Xtr, Ytr)
    Xva, Yva = norm(Xva, Yva)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    f0 = jnp.exp(jnp.mean(jnp.log(jnp.maximum(Ytr[:, 0], args.f_floor))))
    params = init_params(init_key, X.shape[1], args.stencil, args.hidden,
                         z0_bias=float(jnp.log(f0 / (1 - f0))), separate=args.separate)

    loss = lambda p, x, y: loss_fn(p, x, y, args.f_floor, args.f_shape,
                                   args.w_trace, args.w_temp, args.w_shape)
    params, history = stencil.fit(params, loss, ("trace", "temp", "shape"), Xtr, Ytr, Xva, Yva, key, args,
                                  in_types, out_types)

    with open(args.out, "wb") as f:
        pickle.dump({"params": jax.device_get(params),
                     "x_mean": jax.device_get(x_mean), "x_std": jax.device_get(x_std),
                     "channels": channels, "targets": targets,
                     "f_floor": args.f_floor, "f_shape": args.f_shape, "t_min": args.t_min, "log_floor": LOG_FLOOR, "w_trace": args.w_trace,
                     "w_temp": args.w_temp, "w_shape": args.w_shape, "separate": args.separate,
                     **stencil.common_meta(args), "history": history}, f)
    print(f"saved {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    stencil.add_training_args(parser, "rvv_dataset_b16", "rvv_tensor_model.pkl")
    parser.add_argument("--f-floor", type=float, default=1e-8,
                        help="floor on the target f in the ln f loss")
    parser.add_argument("--f-shape", type=float, default=1e-3,
                        help="the shape loss weights a cell by min(1, f / f_shape)")
    parser.add_argument("--t-min", type=float, default=2.5,
                        help="drop frames before this time: the initial transient, before "
                             "the layer is turbulent, where S is near-degenerate")
    parser.set_defaults(weighting="auto", hidden=[256, 256, 256], stencil=3, epochs=2100)
    parser.add_argument("--separate", action=argparse.BooleanOptionalAction, default=True,
                        help="separate networks for f and S (each of size --hidden)")
    parser.add_argument("--w-trace", type=float, default=1.0, help="weight of the ln f loss (--weighting fixed)")
    parser.add_argument("--w-temp", type=float, default=1e4,
                        help="weight of the ln T (= ln <P>) loss (--weighting fixed)")
    parser.add_argument("--w-shape", type=float, default=10.0, help="weight of the shape loss (--weighting fixed)")
    train(parser.parse_args())
