"""
Data augmentation by the octahedral group O_h, the 48 symmetries of the cube.

The run has cubic cells and cubic coarse blocks and is hydro + cooling (no B, no gravity),
so the local subgrid closure is exactly equivariant under O_h: a rotated or mirrored copy
of a training frame is as valid a sample as the frame itself.  Each element is a signed
permutation matrix R, R[i, p[i]] = s[i], stored as g = (p, s).  A field transforms as

  scalar        phi'(r) = phi(R^T r)
  vector        u'(r)   = R u(R^T r)              (polar: velocity)
  pseudovector  w'(r)   = det(R) R w(R^T r)       (vorticity, B)
  sym_tensor    S'(r)   = R S(R^T r) R^T

so components move as well as cells: mirroring x reverses the cells along x AND flips u_x.

Channels are tagged with these types in the model modules (in_types, out_types); a vector
or pseudovector takes 3 consecutive channels (x, y, z), a sym_tensor 6 (xx, yy, zz, xy, xz,
yz).  Arrays are (..., C, nz, ny, nx), or (..., nz, ny, nx) for a bare scalar (types None):
array axis a holds component 2 - a.

Frames are padded with the run's boundaries BEFORE they are transformed (stencil.pad), so
the reflect/constant y boundaries rotate along with the box and stay physical.  The
transform knows nothing about the model, only about the fields.

  python models/augment.py    # self-test
"""

import itertools
import math

import jax
import jax.numpy as jnp

SIZES = {"scalar": 1, "vector": 3, "pseudovector": 3, "sym_tensor": 6}
PAIRS = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]

# identity first; 24 proper rotations (det +1) and 24 improper ones (det -1)
GROUP = [(p, s) for p in itertools.permutations(range(3))
         for s in itertools.product((1, -1), repeat=3)]


def matrix(g):
    """3x3 matrix R of g = (p, s): R[i, p[i]] = s[i]."""
    p, s = g
    return jnp.zeros((3, 3)).at[jnp.arange(3), jnp.array(p)].set(jnp.array(s, dtype=float))


def det(g):
    """det R = sign of the permutation p times the product of the signs s."""
    p, s = g
    inversions = sum(p[i] > p[j] for i in range(3) for j in range(i + 1, 3))
    return (-1)**inversions * math.prod(s)


def grid(a, g):
    """Move the cells of a (..., nz, ny, nx): a'(r) = a(R^T r).  New axis i is old axis
    p[i], reversed where s[i] = -1 (array axis of component i is 2 - i)."""
    p, s = g
    n = a.ndim - 3
    a = jnp.transpose(a, (*range(n), *(n + 2 - p[2 - k] for k in range(3))))
    flips = tuple(n + 2 - i for i in range(3) if s[i] < 0)
    return jnp.flip(a, flips) if flips else a


def channel_map(types, g):
    """Source channel and sign of every output channel of a stack tagged with types:
    new channel c = sign[c] * old channel index[c] (before the cells are moved)."""
    p, s = g
    index, sign = [], []
    for t in types:
        c0 = len(index)
        if t == "scalar":
            index.append(c0)
            sign.append(1)
        elif t in ("vector", "pseudovector"):
            d = det(g) if t == "pseudovector" else 1
            index += [c0 + p[i] for i in range(3)]
            sign += [d * s[i] for i in range(3)]
        elif t == "sym_tensor":
            for i, j in PAIRS:
                index.append(c0 + PAIRS.index(tuple(sorted((p[i], p[j])))))
                sign.append(s[i] * s[j])
        else:
            raise ValueError(f"unknown field type {t!r}")
    return index, sign


def transform(a, types, g, channel_axis=-4):
    """Apply g to a field stack a: channels tagged with types on channel_axis, cells on the
    last three axes.  types None: a is one bare scalar field."""
    a = jnp.asarray(a)
    if types is not None:
        index, sign = channel_map(types, g)
        if index != list(range(len(index))) or any(v != 1 for v in sign):
            shape = [1] * a.ndim
            shape[channel_axis] = len(sign)
            a = jnp.take(a, jnp.array(index), axis=channel_axis) \
                * jnp.array(sign, dtype=a.dtype).reshape(shape)
    return grid(a, g)


def channel_groups(types):
    """Channel indices of each field in types, e.g. [[0], [1], [2, 3, 4], [5]]."""
    groups, c = [], 0
    for t in types:
        groups.append(list(range(c, c + SIZES[t])))
        c += SIZES[t]
    return groups


# ----------------------------------------------------------------------------- self-test

def _test():
    key = jax.random.PRNGKey(0)
    assert len({tuple(matrix(g).ravel().tolist()) for g in GROUP}) == 48
    assert GROUP[0] == ((0, 1, 2), (1, 1, 1))
    assert all(det(g) == round(float(jnp.linalg.det(matrix(g)))) for g in GROUP)
    assert sorted(det(g) for g in GROUP) == [-1] * 24 + [1] * 24

    # cell-centred coordinates of an n^3 cube centred on 0: reversing an axis negates it
    n = 6
    c = jnp.arange(n) - (n - 1) / 2
    z, y, x = jnp.meshgrid(c, c, c, indexing="ij")
    r = jnp.stack([x, y, z])  # (3, nz, ny, nx), components (x, y, z)

    shear = jnp.zeros((3, 3)).at[0, 1].set(1.0)  # u_x = y, the KH mean flow
    close = lambda a, b: bool(jnp.allclose(a, b, atol=1e-5))
    for A in (shear, jax.random.normal(key, (3, 3))):
        u = jnp.einsum("ij,j...->i...", A, r)
        S = A + A.T  # symmetric tensor field S + phi I, phi = x a scalar ramp
        Sfield = jnp.stack([S[i, j] + (i == j) * x for i, j in PAIRS])
        stack = jnp.concatenate([x[None], u, Sfield, u])
        types = ["scalar", "vector", "sym_tensor", "pseudovector"]
        for g in GROUP:
            R = matrix(g)
            out = transform(stack, types, g)
            # scalar: phi = x  ->  phi'(r) = (R^T r)_x
            ramp = jnp.einsum("j,j...->...", R[:, 0], r)
            assert close(out[0], ramp)
            # vector: u = A r  ->  u' = R A R^T r
            assert close(out[1:4], jnp.einsum("ij,j...->i...", R @ A @ R.T, r))
            # sym tensor: S + phi I -> R S R^T + phi' I
            S2 = R @ S @ R.T
            assert close(out[4:10], jnp.stack([S2[i, j] + (i == j) * ramp for i, j in PAIRS]))
            # pseudovector picks up det(R)
            assert close(out[10:13], det(g) * out[1:4])

    # transforming a padded frame then cropping == transforming the frame
    a = jax.random.normal(key, (2, 4, 3, 7, 5))
    h = 2
    a_pad = jnp.pad(a, ((0, 0), (0, 0), (h, h), (h, h), (h, h)), mode="wrap")
    for g in GROUP:
        assert bool(jnp.array_equal(transform(a_pad, ["scalar"] * 4, g)[..., h:-h, h:-h, h:-h],
                                    transform(a, ["scalar"] * 4, g)))
        assert transform(a[:, 0], None, g).shape == transform(a, ["scalar"] * 4, g)[:, 0].shape
    print("augment: all 48 elements pass")


if __name__ == "__main__":
    _test()
