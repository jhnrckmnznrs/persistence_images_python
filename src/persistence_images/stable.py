"""Stable, pixel-integrated persistence images with normalized Gaussian kernels.

Diagrams contain (birth, persistence) pairs. Normalization applies to each
Gaussian density over R², not to the resulting feature vector. The bounded
Lipschitz persistence weight vanishes on the diagonal.
"""
from __future__ import annotations

from math import erfc, sqrt
from typing import Sequence
import numpy as np
from numpy.typing import ArrayLike, NDArray
from .core import Bounds, _points, _validate_params

try:
    from scipy.special import ndtr as _ndtr
except ImportError:
    _ndtr = None

_ROOT_TWO = sqrt(2.0)


def _cdf(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if _ndtr is not None:
        return _ndtr(values)
    return np.fromiter(
        (0.5 * erfc(-float(t) / _ROOT_TWO) for t in values.flat),
        dtype=np.float64, count=values.size,
    ).reshape(values.shape)


def _cdf_intervals(
    edges: NDArray[np.float64], coordinates: NDArray[np.float64], sigma: float,
) -> NDArray[np.float64]:
    """Gaussian probability mass in consecutive edge intervals per source."""
    lower = (edges[:-1, None] - coordinates[None, :]) / sigma
    upper = (edges[1:, None] - coordinates[None, :]) / sigma
    out = np.empty_like(lower)
    right = lower >= 0.0
    left = upper <= 0.0
    cross = ~(right | left)
    if np.any(right):
        out[right] = _cdf(-lower[right]) - _cdf(-upper[right])
    if np.any(left):
        out[left] = _cdf(upper[left]) - _cdf(lower[left])
    if np.any(cross):
        out[cross] = _cdf(upper[cross]) - _cdf(lower[cross])
    return np.maximum(out, 0.0)


def _axes(bounds: Bounds, nx: int, ny: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if bounds.x_max <= bounds.x_min or bounds.y_max <= bounds.y_min:
        raise ValueError("Stable images require strictly positive, fixed pixel bounds")
    return np.linspace(bounds.x_min, bounds.x_max, nx + 1), np.linspace(
        bounds.y_min, bounds.y_max, ny + 1
    )


def _integrate(
    points: NDArray[np.float64],
    xe: NDArray[np.float64], ye: NDArray[np.float64],
    sigma_x: float, sigma_y: float,
    weight_scale: float, batch_size: int,
    counts: NDArray[np.float64],
) -> NDArray[np.float64]:
    nx, ny = len(xe) - 1, len(ye) - 1
    image = np.zeros((nx, ny), dtype=np.float64)
    for start in range(0, len(points), batch_size):
        chunk = points[start:start + batch_size]
        weights = np.minimum(chunk[:, 1] / weight_scale, 1.0) * counts[start:start + batch_size]
        if not np.any(weights):
            continue
        gx = _cdf_intervals(xe, chunk[:, 0], sigma_x)
        gy = _cdf_intervals(ye, chunk[:, 1], sigma_y)
        image += (gx * weights[None, :]) @ gy.T
    return image.ravel(order="C")


def stable_persistence_image(
    diagram: ArrayLike, bounds: Bounds,
    nx: int = 10, ny: int = 8,
    sigma_x: float = 0.86, sigma_y: float = 0.63984,
    *, weight_scale: float = 1.0, batch_size: int = 4096,
    engine: str = "auto",
) -> NDArray[np.float64]:
    """Evaluate normalized, weighted Gaussian integrals over all pixels.

    Requires finite nonnegative persistence, a fixed nondegenerate grid and
    positive bandwidths. The output is flattened with index i*ny+j.
    The output itself is deliberately not normalized to unit mass.
    """
    _validate_params(nx, ny, sigma_x, sigma_y, batch_size)
    if not np.isfinite(weight_scale) or weight_scale <= 0.0:
        raise ValueError("weight_scale must be finite and positive")
    if engine not in {"auto", "dense", "factorized"}:
        raise ValueError("Stable images support auto, dense or factorized engines")
    points = _points(diagram)
    if np.any(points[:, 1] < 0):
        raise ValueError("Persistence must be nonnegative for stable images")
    xe, ye = _axes(bounds, nx, ny)
    if len(points) == 0:
        return np.zeros(nx * ny, dtype=np.float64)
    if engine == "auto":
        sample = points[:min(1024, len(points))]
        engine = (
            "factorized" if len(points) >= 512 and
            len(np.unique(sample, axis=0)) * 3 <= len(sample) else "dense"
        )
    if engine == "factorized":
        unique, multiplicity = np.unique(points, axis=0, return_counts=True)
        if len(unique) > 2_000_000:
            raise ValueError("Unique coordinate count exceeds factorization cap")
        return _integrate(
            unique, xe, ye, sigma_x, sigma_y, weight_scale,
            batch_size, multiplicity.astype(np.float64)
        )
    return _integrate(
        points, xe, ye, sigma_x, sigma_y, weight_scale,
        batch_size, np.ones(len(points), dtype=np.float64)
    )


def stable_batch_images(
    diagrams: Sequence[ArrayLike], bounds: Bounds,
    nx: int = 10, ny: int = 8,
    sigma_x: float = 0.86, sigma_y: float = 0.63984,
    *, spacings: Sequence[float] | None = None,
    weight_scale: float = 1.0, batch_size: int = 4096,
    engine: str = "auto",
) -> list[NDArray[np.float64]]:
    """Evaluate already spacing-scaled diagrams with scaled bandwidths."""
    if spacings is None:
        spacings = [1.0] * len(diagrams)
    if len(spacings) != len(diagrams):
        raise ValueError("spacings length mismatch")
    if any(not np.isfinite(s) or s <= 0.0 for s in spacings):
        raise ValueError("spacings must be finite and positive")
    return [
        stable_persistence_image(
            d, bounds, nx, ny, sigma_x * s, sigma_y * s,
            weight_scale=weight_scale, batch_size=batch_size, engine=engine
        )
        for d, s in zip(diagrams, spacings)
    ]
