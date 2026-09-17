import numpy as np, timeit
from numba import njit, prange

@njit(parallel=True)
def sumlog(x):
    s = 0.0
    for i in prange(x.size):
        s += np.log(x[i])
    return s

data = np.random.rand(5_000_000) * 9999 + 1
sumlog(data[:1000])  # compile
print("numba scalar-log:", timeit.timeit(lambda: sumlog(data), number=20) / 20 * 1e3, "ms")
print("numpy SIMD log  :", timeit.timeit(lambda: np.log(data), number=20) / 20 * 1e3, "ms")
