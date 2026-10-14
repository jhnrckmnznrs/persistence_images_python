"""Numerical routines matching the persistence_images Rust implementation.

Coordinates are (birth, persistence), not (birth, death). The output is
flattened in C order from a 2-D (nx, ny) array: index = i * ny + j.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class Bounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        values = np.array([self.x_min, self.x_max, self.y_min, self.y_max])
        if not np.all(np.isfinite(values)):
            raise ValueError("Bounds must be finite")
        if self.x_min > self.x_max or self.y_min > self.y_max:
            raise ValueError("Bounds must satisfy min <= max")


def _points(diagram: ArrayLike) -> NDArray[np.float64]:
    """Convert a birth-persistence diagram to a contiguous (n, 2) array."""
    data = np.asarray(diagram, dtype=np.float64)
    if data.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError("Expected a diagram with shape (n, 2)")
    if not np.all(np.isfinite(data)):
        raise ValueError("Diagram coordinates must be finite")
    return np.ascontiguousarray(data)


def compute_bounds(diagrams: Iterable[ArrayLike]) -> Bounds | None:
    """Compute bounds across all (birth, persistence) points; None if empty."""
    x_min = y_min = float("inf")
    x_max = y_max = float("-inf")
    found = False
    for diagram in diagrams:
        data = _points(diagram)
        if len(data) == 0:
            continue
        found = True
        x_min = min(x_min, float(data[:, 0].min()))
        x_max = max(x_max, float(data[:, 0].max()))
        y_min = min(y_min, float(data[:, 1].min()))
        y_max = max(y_max, float(data[:, 1].max()))
    return Bounds(x_min, x_max, y_min, y_max) if found else None


def _validate_params(nx: int, ny: int, sigma_x: float, sigma_y: float, batch_size: int) -> None:
    if not isinstance(nx, (int, np.integer)) or isinstance(nx, (bool, np.bool_)) or nx < 1:
        raise ValueError("nx must be a positive integer")
    if not isinstance(ny, (int, np.integer)) or isinstance(ny, (bool, np.bool_)) or ny < 1:
        raise ValueError("ny must be a positive integer")
    if not np.isfinite(sigma_x) or sigma_x <= 0 or not np.isfinite(sigma_y) or sigma_y <= 0:
        raise ValueError("Gaussian bandwidths must be finite and positive")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")


def _pixel_centers(bounds: Bounds, nx: int, ny: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    xs = bounds.x_min + (np.arange(nx, dtype=np.float64) + 0.5) * (
        (bounds.x_max - bounds.x_min) / nx
    )
    ys = bounds.y_min + (np.arange(ny, dtype=np.float64) + 0.5) * (
        (bounds.y_max - bounds.y_min) / ny
    )
    return xs, ys


def _image_dense(
    points: NDArray[np.float64], xs: NDArray[np.float64], ys: NDArray[np.float64],
    sigma_x: float, sigma_y: float, batch_size: int,
) -> NDArray[np.float64]:
    """Low-memory separable dense calculation; works for arbitrary real coordinates."""
    out = np.zeros((len(xs), len(ys)), dtype=np.float64)
    for start in range(0, len(points), batch_size):
        chunk = points[start:start + batch_size]
        gx = np.exp(-0.5 * ((xs[:, None] - chunk[None, :, 0]) / sigma_x) ** 2)
        gy = np.exp(-0.5 * ((ys[None, :] - chunk[:, None, 1]) / sigma_y) ** 2)
        out += gx @ gy
    return out.ravel(order="C")


def _image_factorized(
    points: NDArray[np.float64], xs: NDArray[np.float64], ys: NDArray[np.float64],
    sigma_x: float, sigma_y: float, *, max_cells: int = 2_000_000,
) -> NDArray[np.float64]:
    """Exact coordinate grouping I = A @ C @ D in real arithmetic.

    C[u,v] counts diagram points with birth b_u and persistence p_v. The
    dense count matrix is allocated only when it fits the explicit cap.
    """
    births, bi = np.unique(points[:, 0], return_inverse=True)
    persistences, pi = np.unique(points[:, 1], return_inverse=True)
    nb, npers = len(births), len(persistences)
    if nb * npers > max_cells:
        raise ValueError("Too many distinct coordinate combinations for factorization; use engine='dense'")
    # Also bound the sizes of the Gaussian matrices, independent of C.
    if len(xs) * nb + npers * len(ys) > 16_000_000:
        raise ValueError("Gaussian basis matrices too large for factorization; use engine='dense'")
    counts = np.bincount(bi * npers + pi, minlength=nb * npers).reshape(nb, npers)
    a = np.exp(-0.5 * ((xs[:, None] - births[None, :]) / sigma_x) ** 2)
    d = np.exp(-0.5 * ((persistences[:, None] - ys[None, :]) / sigma_y) ** 2)
    # Choose the cheaper matrix-chain multiplication order by flop counts.
    # Multiplicity is not a smoothing weight: it reproduces exactly repeated
    # points and hence preserves the Rust unweighted Gaussian convention.
    if len(xs) * nb * npers + len(xs) * npers * len(ys) < nb * npers * len(ys) + len(xs) * nb * len(ys):
        return ((a @ counts) @ d).ravel(order="C")
    return (a @ (counts @ d)).ravel(order="C")


def _candidate_for_factorization(points: NDArray[np.float64], nx: int, ny: int) -> bool:
    """Cheap, conservative screening before computing full coordinate uniques."""
    n = len(points)
    if n < 512 or nx * ny < 1024:
        return False
    sample = points[:min(1024, n)]
    nb = len(np.unique(sample[:, 0]))
    npers = len(np.unique(sample[:, 1]))
    # Sampled continuous data is rejected immediately. This screening is
    # heuristic; the full-size guards are enforced in _image_factorized.
    return nb * npers <= 16 * len(sample) and (nb < len(sample) or npers < len(sample))




@lru_cache(maxsize=2)
def _cached_moment_fft(bounds: tuple[float, float, float, float], nx: int, ny: int,
                       sigma_x: float, sigma_y: float, degree: int):
    from ._grid_fft import MomentFFT
    return MomentFFT(bounds, nx, ny, sigma_x, sigma_y, degree)


def _grid_fft_image(points, bounds, nx, ny, sigma_x, sigma_y, tolerance, *, max_degree=12):
    if tolerance is None:
        raise ValueError("grid_fft requires an explicit absolute_error tolerance")
    try:
        from ._grid_fft import tight_degree_for_points
    except ImportError as exc:
        raise ImportError("grid_fft requires scipy and numba: pip install 'persistence-images-python[grid]'") from exc
    dx = (bounds.x_max-bounds.x_min)/nx
    dy = (bounds.y_max-bounds.y_min)/ny
    degree,certified_error = tight_degree_for_points(points,
        (bounds.x_min,bounds.x_max,bounds.y_min,bounds.y_max),nx,ny,
        sigma_x,sigma_y,tolerance,max_degree)
    if degree is None:
        raise ValueError("Requested tolerance cannot be certified by grid_fft; choose dense or sparse")
    engine = _cached_moment_fft((bounds.x_min,bounds.x_max,bounds.y_min,bounds.y_max),
                                nx, ny, sigma_x, sigma_y, degree)
    return engine.evaluate(points).ravel(order="C")

@lru_cache(maxsize=4)
def _cached_interp_fft(bounds: tuple[float, float, float, float], nx: int, ny: int,
                       sigma_x: float, sigma_y: float, wx: int, wy: int):
    from ._interp_fft import InterpolationFFT
    return InterpolationFFT(bounds, nx, ny, sigma_x, sigma_y, wx, wy)


def _interp_fft_image(points, bounds, nx, ny, sigma_x, sigma_y, tolerance):
    if tolerance is None:
        raise ValueError("interp_fft requires explicit absolute_error")
    try:
        from ._interp_fft import select_uniform_width
    except ImportError as exc:
        raise ImportError("interp_fft requires scipy and numba; install extra [grid]") from exc
    b = (bounds.x_min,bounds.x_max,bounds.y_min,bounds.y_max)
    if b[0] >= b[1] or b[2] >= b[3]:
        raise ValueError('interp_fft requires nondegenerate bounds')
    if (np.any(points[:,0]<b[0]) or np.any(points[:,0]>b[1]) or
            np.any(points[:,1]<b[2]) or np.any(points[:,1]>b[3])):
        raise ValueError('diagram points must lie inside bounds')
    selected = select_uniform_width(len(points),b,nx,ny,sigma_x,sigma_y,tolerance)
    if selected is None:
        raise ValueError('interp_fft cannot certify this tolerance; use sparse or dense')
    _,wx,wy,_ = selected
    plan = _cached_interp_fft(b,nx,ny,sigma_x,sigma_y,wx,wy)
    return plan.evaluate(points).ravel(order="C")


def persistence_image(
    diagram: ArrayLike,
    bounds: Bounds,
    nx: int = 10,
    ny: int = 8,
    sigma_x: float = 0.86,
    sigma_y: float = 0.63984,
    *,
    batch_size: int = 4096,
    engine: str = "auto",
    absolute_error: float | None = None,
) -> NDArray[np.float64]:
    """Compute a Rust-compatible unnormalized anisotropic persistence image.

    The input is a diagram of (birth, persistence) points. The output shape is
    (nx*ny,) with index i*ny+j. Engines:

    - auto: choose coordinate factorization for quantized data, optionally
      localized evaluation if an explicit absolute_error is supplied;
      otherwise use chunked separable matrix multiplication.
    - dense: use chunked separable matrix multiplication for all diagrams.
    - factorized: factor unique birth and persistence values with multiplicity
      matrix C. Can accelerate discretized diagrams dramatically; has a hard
      limit on temporary matrix allocations.
    - interp_fft: optional grid-aware Lagrange source interpolation and ONE
      Gaussian FFT convolution. Uses a uniform analytic truncation certificate.
      Requires an explicit absolute_error and SciPy+Numba.
    - grid_fft: optional SciPy/Numba Gaussian moment-deposition + FFT convolution;
      certified Taylor truncation bound. Requires absolute_error, and only
      supports strictly nondegenerate bounds containing all diagram points.
    - sparse: optional Numba kernel, evaluating Gaussian contributions only
      in local windows. Mathematical truncation error <= absolute_error per
      pixel, ignoring ordinary floating-point roundoff. Requires numba.

    Engines differ in floating-point summation order, so compare with a
    tolerance, not bitwise. Neither Gaussian normalization nor weighting nor
    pixel-area integration is used.
    """
    _validate_params(nx, ny, sigma_x, sigma_y, batch_size)
    if engine not in ("auto", "dense", "factorized", "sparse", "grid_fft", "interp_fft"):
        raise ValueError("Invalid engine: choose auto, dense, factorized, sparse, grid_fft, or interp_fft")
    if absolute_error is not None and (not np.isfinite(absolute_error) or absolute_error <= 0):
        raise ValueError("absolute_error must be finite and positive")
    points = _points(diagram)
    if len(points) == 0:
        return np.zeros(nx * ny, dtype=np.float64)
    xs, ys = _pixel_centers(bounds, nx, ny)
    if engine == "auto" and absolute_error is not None and len(points) >= 128 and nx * ny >= 1024:
        # Estimate the exact number of candidate point-pixel contributions
        # before paying for Numba JIT/dispatch. Conservative default: exact.
        import math
        r = math.sqrt(2.0 * max(0.0, math.log(len(points)) - math.log(absolute_error)))
        ilo = np.searchsorted(xs, points[:, 0] - r * sigma_x, side="left")
        ihi = np.searchsorted(xs, points[:, 0] + r * sigma_x, side="right")
        jlo = np.searchsorted(ys, points[:, 1] - r * sigma_y, side="left")
        jhi = np.searchsorted(ys, points[:, 1] + r * sigma_y, side="right")
        occupancy = np.sum((ihi - ilo).astype(np.float64) * (jhi - jlo))
        if occupancy < 0.12 * len(points) * nx * ny:
            try:
                import numba  # noqa: F401; optional fast dependency
            except ImportError:
                pass
            else:
                engine = "sparse"
    if engine == "sparse":
        import math
        if absolute_error is None:
            absolute_error = 1e-8
        try:
            from ._sparse_numba import sparse_image_numba
        except ImportError as exc:
            raise ImportError(
                "The sparse engine requires Numba. Install with: pip install 'persistence-images-python[fast]'"
            ) from exc
        # Avoid the overflow in N / absolute_error when error is very small.
        r = math.sqrt(2.0 * max(0.0, math.log(len(points)) - math.log(absolute_error)))
        return sparse_image_numba(points, xs, ys, sigma_x, sigma_y, r)
    if engine == "factorized":
        return _image_factorized(points, xs, ys, sigma_x, sigma_y)
    if engine == "interp_fft":
        return _interp_fft_image(points, bounds, nx, ny, sigma_x, sigma_y, absolute_error)
    if engine == "grid_fft":
        return _grid_fft_image(points, bounds, nx, ny, sigma_x, sigma_y, absolute_error)
    if engine == "auto" and _candidate_for_factorization(points, nx, ny):
        try:
            # Precompute both uniqueness tables only when the sample shows
            # sufficient repetition. A rare large full matrix falls back.
            return _image_factorized(points, xs, ys, sigma_x, sigma_y)
        except ValueError:
            pass
    # Try the one-FFT method before the costlier multi-moment FFT.
    # A closed-form stencil certificate makes this selection cheap.
    if (engine == "auto" and absolute_error is not None
            and len(points) >= 10_000 and nx*ny >= 4096
            and bounds.x_max > bounds.x_min and bounds.y_max > bounds.y_min):
        try:
            return _interp_fft_image(points,bounds,nx,ny,sigma_x,sigma_y,absolute_error)
        except (ImportError, ValueError, MemoryError):
            pass
    # Grid-aware algorithm is worthwhile only for large continuous diagrams
    # with adequately resolved Gaussians and a nonzero error budget.
    # Keep auto's previous exact behavior when no tolerance is requested.
    if (engine == "auto" and absolute_error is not None
            and len(points) >= 40_000 and nx * ny >= 4096
            and bounds.x_max > bounds.x_min and bounds.y_max > bounds.y_min):
        try:
            from ._grid_fft import tight_degree_for_points
            degree,certified_error=tight_degree_for_points(points,
                    (bounds.x_min,bounds.x_max,bounds.y_min,bounds.y_max),
                    nx,ny,sigma_x,sigma_y,absolute_error,8)
            if degree is not None:
                return _grid_fft_image(points,bounds,nx,ny,sigma_x,sigma_y,absolute_error,max_degree=8)
        except (ImportError, ValueError, MemoryError):
            pass
    return _image_dense(points, xs, ys, sigma_x, sigma_y, batch_size)


def persistence_image_reference(
    diagram: ArrayLike,
    bounds: Bounds,
    nx: int = 10,
    ny: int = 8,
    sigma_x: float = 0.86,
    sigma_y: float = 0.63984,
) -> NDArray[np.float64]:
    """Small pure-Python loop matching the Rust loop's operation order.

    Intended for testing and checking conventions, not large datasets.
    """
    import math

    _validate_params(nx, ny, sigma_x, sigma_y, 1)
    points = _points(diagram)
    dx = (bounds.x_max - bounds.x_min) / nx
    dy = (bounds.y_max - bounds.y_min) / ny
    out = np.zeros(nx * ny, dtype=np.float64)
    for i in range(nx):
        x = bounds.x_min + (i + 0.5) * dx
        for j in range(ny):
            y = bounds.y_min + (j + 0.5) * dy
            value = 0.0
            for birth, persistence in points:
                xd = x - birth
                yd = y - persistence
                value += math.exp(-(
                    xd * xd / (2.0 * sigma_x * sigma_x)
                    + yd * yd / (2.0 * sigma_y * sigma_y)
                ))
            out[i * ny + j] = value
    return out
