"""Optional Numba local Gaussian evaluation with a global absolute-error bound.

Numba is imported only when the user selects engine='sparse'. Each omitted
point/pixel term is at most exp(-0.5*r**2), so the exact-arithmetic total
error at each pixel is <= n_points * exp(-0.5*r**2). Floating-point roundoff
is separate from this mathematical truncation bound.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit


@njit(cache=True)
def sparse_image_numba(
    points: np.ndarray, xs: np.ndarray, ys: np.ndarray,
    sx: float, sy: float, radius: float,
) -> np.ndarray:
    """Use compact x/y index windows and separable rank-one contributions."""
    nx, ny = xs.size, ys.size
    out = np.zeros((nx, ny), dtype=np.float64)
    gx = np.empty(nx, dtype=np.float64)
    gy = np.empty(ny, dtype=np.float64)
    dx = (xs[-1] - xs[0]) / (nx - 1) if nx > 1 else 0.0
    dy = (ys[-1] - ys[0]) / (ny - 1) if ny > 1 else 0.0

    for k in range(points.shape[0]):
        birth, persistence = points[k, 0], points[k, 1]
        if dx > 0.0:
            i0 = max(0, int(math.ceil((birth - radius * sx - xs[0]) / dx)) - 1)
            i1 = min(nx - 1, int(math.floor((birth + radius * sx - xs[0]) / dx)) + 1)
        else:
            i0 = 0 if abs(xs[0] - birth) <= radius * sx else nx
            i1 = nx - 1
        if dy > 0.0:
            j0 = max(0, int(math.ceil((persistence - radius * sy - ys[0]) / dy)) - 1)
            j1 = min(ny - 1, int(math.floor((persistence + radius * sy - ys[0]) / dy)) + 1)
        else:
            j0 = 0 if abs(ys[0] - persistence) <= radius * sy else ny
            j1 = ny - 1
        if i0 > i1 or j0 > j1:
            continue
        for i in range(i0, i1 + 1):
            d = (xs[i] - birth) / sx
            gx[i] = math.exp(-0.5 * d * d)
        for j in range(j0, j1 + 1):
            d = (ys[j] - persistence) / sy
            gy[j] = math.exp(-0.5 * d * d)
        for i in range(i0, i1 + 1):
            v = gx[i]
            for j in range(j0, j1 + 1):
                out[i, j] += v * gy[j]
    return out.ravel()
