"""
Shared pieces of the stencil models (cooling_model.py, rvv_tensor_model.py): the KH run they
train on, the per-frame cache of the coarse-grained dataset, the stencil network and the training loop.

The network looks at a stencil x stencil x stencil box of coarse cells and returns n_out
values for the centre cell.  It is written as a VALID 3D convolution over the padded grid
followed by a pointwise MLP, which is the same as sliding the stencil cell by cell but
evaluates the whole grid at once.

Padding is part of the data, not the network: frames are padded once with the run's
boundaries (periodic in x and z, edge-copied in y for the reflect / constant boundaries)
and the network maps a padded frame to its interior.  That lets augment.py rotate and
mirror padded frames with their boundaries, so training sees every O_h image of the box
whatever the network is.

A model supplies its coarse_frame (frame -> inputs, targets), the field types of its
inputs and targets (augment.py), its loss and the arguments it adds to add_training_args;
everything else is here.  Arrays are jax arrays throughout; numpy is only used to read and
write the cache files.
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from vis_athenak import SimulationData

from models import augment

run = "/data/sasi/simulation_outputs/kh3d_256x1024x256"
athinput = f"{run}/kh3d_256x1024x256.athinput"
b = 16  # coarse-graining block size: 256x1024x256 -> 16x64x16


# ----------------------------------------------------------------------------- data

def cache_dir(cache):
    """The per-frame cache directory for cache; a legacy single-file cache x.npz (and model
    files that still record it) map to the directory x/."""
    cache = Path(cache)
    return cache.with_suffix("") if cache.suffix == ".npz" else cache


def import_legacy(cache):
    """Split a legacy single-file cache <cache>.npz (X, Y, time, frames) into per-frame files,
    once.  The .npz itself is left in place."""
    legacy = cache.with_suffix(".npz")
    if cache.exists() or not legacy.exists():
        return
    d = np.load(legacy)
    cache.mkdir(parents=True)
    for i, x, y, t in zip(d["frames"].tolist(), d["X"], d["Y"], d["time"].tolist()):
        save_frame(cache, i, x, y, t)
    print(f"split {legacy} into {len(d['frames'])} frames in {cache}/")


def frame_path(cache, i):
    return cache / f"frame_{i:05d}.npz"


def save_frame(cache, i, x, y, t, extra=None):
    """Write frame i atomically (temporary file, then rename), so a build that is killed
    midway never leaves a truncated frame behind.  extra: further named arrays stored with
    the frame but not loaded by load_dataset."""
    tmp = cache / f".frame_{i:05d}.npz.tmp"
    with open(tmp, "wb") as f:
        np.savez(f, X=np.asarray(x), Y=np.asarray(y), time=t,
                 **{k: np.asarray(v) for k, v in (extra or {}).items()})
    tmp.replace(frame_path(cache, i))


def load_dataset(every, cache, fields, coarse_frame):
    """Coarse inputs X (frames, C, nz, ny, nx), targets Y (frames, ...) and times of every
    every-th frame of the run, cached one frame per file in the directory cache_dir(cache).

    Each frame is written as soon as it is coarse-grained, so an interrupted build resumes
    where it stopped, frames the run gains later are the only ones computed, and any
    --every reuses the same files.  coarse_frame(frame, params) -> (x, y) or (x, y, extra),
    extra a dict of arrays saved with the frame (save_frame), is called with
    fields loaded on the frame and the run's parameters.  The next frame is read in the
    background while this one is coarse-grained: the run sits on a spinning disk, so the
    read is the slow part."""
    cache = cache_dir(cache)
    import_legacy(cache)
    cache.mkdir(parents=True, exist_ok=True)
    sim = SimulationData(athinput, f"{run}/bin")
    frames = range(0, len(sim), every)
    todo = [i for i in frames if not frame_path(cache, i).exists()]

    def read(i):
        frame = sim[i]
        frame.load(fields)
        return frame

    if todo:
        print(f"coarse-graining {len(todo)} of {len(frames)} frames into {cache}/")
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(read, todo[0])
            for k in tqdm(range(len(todo)), unit="frame"):
                frame = pending.result()
                if k + 1 < len(todo):
                    pending = pool.submit(read, todo[k + 1])
                out = coarse_frame(frame, sim.params)
                frame.unload()
                save_frame(cache, todo[k], *out[:2], frame.time, *out[2:])

    X, Y, t = [], [], []
    for i in frames:
        with np.load(frame_path(cache, i)) as d:
            X.append(jnp.asarray(d["X"]))
            Y.append(jnp.asarray(d["Y"]))
            t.append(float(d["time"]))
    return jnp.stack(X), jnp.stack(Y), jnp.array(t)


def val_mask(n, val_every):
    """Every val_every-th of n frames is held out for validation (interleaved in time)."""
    return jnp.arange(n) % val_every == val_every - 1


# ----------------------------------------------------------------------------- network

def init_params(key, n_in, stencil, hidden, n_out):
    """Stencil conv (stencil^3 x n_in -> hidden[0]) then pointwise dense layers -> n_out."""
    sizes = [n_in * stencil**3, *hidden, n_out]
    params = []
    for k, (m, n) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        params.append({"w": jax.random.normal(k, (m, n)) * jnp.sqrt(2.0 / m), "b": jnp.zeros(n)})
    params[0]["w"] = params[0]["w"].reshape(stencil, stencil, stencil, n_in, hidden[0])  # DHWIO
    return params


def pad(x, h):
    """Halo of h coarse cells on (B, C, z, y, x): periodic in z and x, edge in y.
    Applied to the normalised frames before training or prediction."""
    x = jnp.pad(x, ((0, 0), (0, 0), (h, h), (0, 0), (h, h)), mode="wrap")
    return jnp.pad(x, ((0, 0), (0, 0), (0, 0), (h, h), (0, 0)), mode="edge")


def apply(params, x, w0=None):
    """(B, C, nz + 2h, ny + 2h, nx + 2h) normalised inputs padded by h = stencil // 2
    -> (B, nz, ny, nx, n_out).
    w0 replaces the stencil weights params[0]["w"], for models that constrain them."""
    w0 = params[0]["w"] if w0 is None else w0
    z = jax.lax.conv_general_dilated(x, w0, (1, 1, 1), "VALID",
                                     dimension_numbers=("NCDHW", "DHWIO", "NDHWC"))
    z = jax.nn.gelu(z + params[0]["b"])
    for layer in params[1:-1]:
        z = jax.nn.gelu(z @ layer["w"] + layer["b"])
    return z @ params[-1]["w"] + params[-1]["b"]


def masked_mean(se, mask):
    """Mean of se over the cells where mask is set (0 if there are none)."""
    return jnp.sum(se * mask) / jnp.maximum(jnp.sum(mask), 1)


# ----------------------------------------------------------------------------- training

def standardise(Xtr, types):
    """Mean and std (C, 1, 1, 1) of the training inputs (frames, C, nz, ny, nx), channels
    tagged with types.  Scalars get their own mean and std.  The components of a vector or
    tensor share one scale and are not shifted, so that rotating the normalised components
    is the same as rotating the physical ones."""
    x_mean = jnp.mean(Xtr, axis=(0, 2, 3, 4), keepdims=True)[0]
    x_std = jnp.std(Xtr, axis=(0, 2, 3, 4), keepdims=True)[0]
    for t, group in zip(types, augment.channel_groups(types)):
        if t != "scalar":
            group = jnp.array(group)
            x_mean = x_mean.at[group].set(0.0)
            x_std = x_std.at[group].set(jnp.sqrt(jnp.mean(Xtr[:, group]**2)))
    return x_mean, x_std + 1e-8


LOG_FLOOR = 1e-12  # added to each loss term before the log of the auto weighting


def weighted(loss, weighting):
    """loss(params, x, y) -> (total, parts) with its total replaced according to weighting:

    fixed: the model's own total, sum_k w_k L_k with the weights it was given.
    auto:  mean_k ln L_k.  This is the uncertainty weighting of Kendall et al. (2018),
           sum_k [L_k / (2 sigma_k^2) + ln sigma_k], with every sigma_k set to its optimum
           sigma_k^2 = L_k: the gradient is mean_k grad L_k / L_k, so each term is weighted
           by the inverse of its current size, whatever its units, and there is nothing
           extra to learn.  The model's weights w_k drop out (ln w_k L_k = ln w_k + ln L_k).
           The total is negative once the terms are below 1."""
    if weighting == "fixed":
        return loss
    def auto(params, x, y):
        _, parts = loss(params, x, y)
        return combine(parts), parts
    return auto


def combine(parts):
    """The auto-weighted total of the loss terms parts (see weighted)."""
    return jnp.mean(jnp.log(jnp.stack(parts) + LOG_FLOOR))


def make_step(optimizer, loss):
    """Jitted optimiser step; loss(params, x, y) -> (total, parts)."""
    @jax.jit
    def step(params, opt_state, x, y):
        (total, parts), grads = jax.value_and_grad(loss, has_aux=True)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, total, parts
    return step


def symmetrised(loss, n_parts, in_types, y_types, weighting, x, y, chunk=None):
    """loss(params, x, y) -> (total, parts) averaged over the 48 O_h images of (x, y).

    The loss terms are averaged over the images first and then combined as in weighted, so
    the result is the objective the Adam stage samples one image at a time.  The images are
    grouped by the orientation of the rotated grid (three groups of 16, each one lax.scan
    over a lax.switch of its 16 elements) and every image is checkpointed, so the backward
    pass holds the activations of one image at a time and nothing is compiled 48 times.

    With chunk, the frames are taken chunk at a time (a lax.scan, plus one call for the
    remainder) and the terms averaged over the chunks weighted by their frame count, so the
    backward pass holds the activations of chunk frames.  A term that is a masked mean is
    then a mean of per-chunk masked means, which differs slightly from the masked mean over
    all frames but is just as deterministic."""
    n = len(x)
    chunk = min(chunk or n, n)
    full, rest = divmod(n, chunk)
    groups = {}
    for g in augment.GROUP:
        shape = jax.eval_shape(lambda g=g: augment.grid(jnp.zeros(x.shape[-3:]), g)).shape
        groups.setdefault(shape, []).append(g)
    zeros = (jnp.zeros(()), (jnp.zeros(()),) * n_parts)
    add = lambda a, b, w=1: jax.tree.map(lambda u, v: u + w * v, a, b)

    def images(params, x, y):
        """Sum of (total, parts) over the 48 images of (x, y)."""
        acc = zeros
        for gs in groups.values():
            branches = [lambda x, y, g=g: (augment.transform(x, in_types, g),
                                           augment.transform(y, y_types, g)) for g in gs]

            @jax.checkpoint
            def image(params, i):
                return loss(params, *jax.lax.switch(i, branches, x, y))

            acc, _ = jax.lax.scan(lambda acc, i: (add(acc, image(params, i)), None),
                                  acc, jnp.arange(len(gs)))
        return acc

    def objective(params):
        split = lambda a: a[:full * chunk].reshape(full, chunk, *a.shape[1:])
        acc, _ = jax.lax.scan(lambda acc, xy: (add(acc, images(params, *xy), chunk), None),
                              zeros, (split(x), split(y)))
        if rest:
            acc = add(acc, images(params, x[full * chunk:], y[full * chunk:]), rest)
        total, parts = jax.tree.map(lambda a: a / (len(augment.GROUP) * n), acc)
        return (total if weighting == "fixed" else combine(parts)), parts
    return objective


def fit(params, loss, parts, Xtr, Ytr, Xva, Yva, key, args, in_types, out_types):
    """Train params on (Xtr, Ytr) with AdamW, a warmup-cosine schedule and global-norm
    clipping, then optionally refine them with full-batch L-BFGS, validating on (Xva, Yva)
    every args.log_every epochs / steps.  X are padded frames.

    With args.augment every Adam step draws one of the 48 elements of O_h and applies it to
    the batch, inputs tagged with in_types and targets with out_types (augment.transform).
    The step is compiled once per grid orientation (y along z, y or x), so three times.

    The L-BFGS stage (args.lbfgs_steps > 0) minimises the loss on all training frames
    averaged over all 48 images (symmetrised), the deterministic objective that the
    augmented Adam steps sample, with optax's zoom line search.  Its steps continue the
    epoch axis of the history after args.epochs; it has no weight decay.

    loss(params, x, y) -> (total, tuple of len(parts)), its total replaced by
    weighted(loss, args.weighting).  Returns params and the history: train / val are the
    total loss, train_<part> / val_<part> its components, val_rot the total validation loss
    averaged over all 48 images of the validation frames (equal to val for a model that has
    learnt the symmetry), lbfgs_start the epoch at which L-BFGS took over."""
    loss = weighted(loss, args.weighting)
    steps_per_epoch = max(1, len(Xtr) // args.batch)
    schedule = optax.warmup_cosine_decay_schedule(0.0, args.lr, 10 * steps_per_epoch,
                                                  args.epochs * steps_per_epoch)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adamw(schedule, weight_decay=args.weight_decay))
    opt_state = optimizer.init(params)
    step = make_step(optimizer, loss)
    val_loss = jax.jit(loss)
    y_types = out_types if Ytr.ndim == 5 else None

    def rotated(g, x, y):
        return augment.transform(x, in_types, g), augment.transform(y, y_types, g)

    history = {"epoch": [], "train": [], "val_epoch": [], "val": [], "val_rot": [], "time": [],
               **{f"{s}_{k}": [] for s in ("train", "val") for k in parts},
               "lbfgs_start": args.epochs}
    t0 = time.perf_counter()

    def record(epoch, losses, validate, stage):
        history["epoch"].append(epoch)
        history["train"].append(float(losses[0]))
        for k, v in zip(parts, losses[1:]):
            history[f"train_{k}"].append(float(v))
        history["time"].append(time.perf_counter() - t0)
        if not validate:
            return
        history["val_epoch"].append(epoch)
        total, comps = val_loss(params, Xva, Yva)
        history["val"].append(float(total))
        for k, v in zip(parts, comps):
            history[f"val_{k}"].append(float(v))
        history["val_rot"].append(float(jnp.mean(jnp.array(
            [val_loss(params, *rotated(g, Xva, Yva))[0] for g in augment.GROUP]))))
        tqdm.write(f"{stage} {epoch:4d}  train {history['train'][-1]:.4e}  "
                   f"val {history['val'][-1]:.4e}  val_rot {history['val_rot'][-1]:.4e}  ("
                   + ", ".join(f"{k} {history[f'val_{k}'][-1]:.3e}" for k in parts) + ")")

    for epoch in tqdm(range(args.epochs), desc="Training Epoch:"):
        key, perm_key, g_key = jax.random.split(key, 3)
        perm = jax.random.permutation(perm_key, len(Xtr))
        gs = jax.random.randint(g_key, (steps_per_epoch,), 0, len(augment.GROUP)).tolist()
        losses = []
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch:(i + 1) * args.batch]
            x, y = Xtr[idx], Ytr[idx]
            if args.augment:
                x, y = rotated(augment.GROUP[gs[i]], x, y)
            params, opt_state, total, comps = step(params, opt_state, x, y)
            losses.append((total, *comps))
        losses = jnp.mean(jnp.array(losses), axis=0).tolist()  # syncs the device
        record(epoch, losses, epoch % args.log_every == 0 or epoch == args.epochs - 1, "epoch")

    if args.lbfgs_steps > 0:
        solver = optax.lbfgs(memory_size=args.lbfgs_memory)
        objective = jax.jit(lambda p, x, y: symmetrised(loss, len(parts), in_types, y_types,
                                                        args.weighting, x, y,
                                                        args.lbfgs_chunk)(p))

        @jax.jit
        def lbfgs_step(params, state, x, y):
            value_fn = lambda p: symmetrised(loss, len(parts), in_types, y_types,
                                             args.weighting, x, y, args.lbfgs_chunk)(p)[0]
            value, grad = optax.value_and_grad_from_state(value_fn)(params, state=state)
            updates, state = solver.update(grad, state, params, value=value, grad=grad,
                                           value_fn=value_fn)
            return optax.apply_updates(params, updates), state

        state = solver.init(params)
        for k in tqdm(range(args.lbfgs_steps), desc="L-BFGS step:"):
            params, state = lbfgs_step(params, state, Xtr, Ytr)
            total, comps = objective(params, Xtr, Ytr)
            if not jnp.isfinite(total):
                tqdm.write(f"L-BFGS step {k}: non-finite loss, stopping")
                break
            record(args.epochs + k, [total, *comps],
                   k % args.log_every == 0 or k == args.lbfgs_steps - 1, "lbfgs")
    return params, history


def add_training_args(parser, cache, out):
    """Arguments shared by the stencil models; --cache (a directory, see load_dataset) and
    --out default to the names cache and out in models/."""
    here = Path(__file__).parent
    parser.add_argument("--every", type=int, default=1, help="use every n-th frame")
    parser.add_argument("--val-every", type=int, default=5, help="every n-th used frame is validation")
    parser.add_argument("--stencil", type=int, default=3, help="stencil width in coarse cells (odd)")
    parser.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--batch", type=int, default=4, help="frames per step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--weighting", choices=("fixed", "auto"), default="fixed",
                        help="fixed: the model's weighted sum of its loss terms; auto: the mean "
                             "of their logs (uncertainty weighting, see stencil.weighted)")
    parser.add_argument("--lbfgs-steps", type=int, default=0,
                        help="full-batch L-BFGS steps after the Adam epochs, on the loss "
                             "averaged over all 48 O_h images (0: none)")
    parser.add_argument("--lbfgs-memory", type=int, default=10,
                        help="number of past updates L-BFGS keeps")
    parser.add_argument("--lbfgs-chunk", type=int, default=None,
                        help="frames per chunk of the L-BFGS objective, to bound its memory "
                             "(default: all training frames at once)")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="train on a random O_h rotation / reflection of every batch")
    parser.add_argument("--cache", default=str(here / cache))
    parser.add_argument("--out", default=str(here / out))


def common_meta(args):
    """The run settings every saved model records."""
    return {"b": b, "stencil": args.stencil, "hidden": args.hidden, "every": args.every,
            "val_every": args.val_every, "cache": args.cache, "augment": args.augment,
            "weighting": args.weighting, "lbfgs_steps": args.lbfgs_steps}
