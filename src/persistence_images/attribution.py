"""Exact interval contributions for frozen linear, pixel-integrated persistence-image models.

This is an algebraic prediction decomposition, not a causal bone-removal or
mechanical-load attribution. It follows stable_persistence_image's normalized,
bounded-ramp Gaussian integrals and birth-major C-order pixel layout.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import Bounds, _points, _validate_params
from .stable import _axes, _cdf_intervals, stable_persistence_image


@dataclass(frozen=True)
class LinearPIModel:
    """Frozen model: intercept + coef @ ((image - center) / scale).

    scale can be a single positive block scale or a per-feature vector.
    Inactive coordinates must have zero coefficients in the full image vector.
    """
    coef: ArrayLike
    intercept: float
    center: ArrayLike | None = None
    scale: ArrayLike | float = 1.0

    def original_scale(self, n_features: int) -> tuple[float, NDArray[np.float64]]:
        beta = np.asarray(self.coef, dtype=np.float64)
        if beta.shape != (n_features,) or not np.all(np.isfinite(beta)):
            raise ValueError("coef must be a finite vector of image length")
        center = (np.zeros(n_features) if self.center is None
                  else np.asarray(self.center, dtype=np.float64))
        if center.shape != (n_features,) or not np.all(np.isfinite(center)):
            raise ValueError("center must be a finite image-length vector")
        scale = np.asarray(self.scale, dtype=np.float64)
        if scale.ndim > 1 or (scale.ndim == 1 and scale.shape != (n_features,)):
            raise ValueError("scale must be a scalar or image-length vector")
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("scale must be finite and strictly positive")
        if not np.isfinite(self.intercept):
            raise ValueError("intercept must be finite")
        theta = beta / scale
        return float(self.intercept - theta @ center), theta


@dataclass(frozen=True)
class IntervalExplanation:
    diagram: NDArray[np.float64]
    contributions: NDArray[np.float64]
    image: NDArray[np.float64]
    base: float
    other_blocks: float
    prediction: float
    reconstructed_prediction: float
    reconstruction_error: float


def explain_linear_pi(
    diagram: ArrayLike,
    bounds: Bounds,
    nx: int,
    ny: int,
    sigma_x: float,
    sigma_y: float,
    model: LinearPIModel,
    *,
    weight_scale: float = 1.0,
    other_blocks: float = 0.0,
    batch_size: int = 2048,
    atol: float = 1e-9,
    rtol: float = 1e-10,
) -> IntervalExplanation:
    """Explain a fixed stable persistence-image prediction point by point.

    Returned scores sum to the prediction after adding the fixed base and
    optional contributions from other model blocks. Independently computes
    a dense stable image for a reconstruction check. Does not materialize a
    large (n_points, n_pixels) matrix or renormalize out-of-grid mass.
    """
    _validate_params(nx, ny, sigma_x, sigma_y, batch_size)
    if not np.isfinite(weight_scale) or weight_scale <= 0:
        raise ValueError("weight_scale must be finite and positive")
    if not np.isfinite(other_blocks):
        raise ValueError("other_blocks must be finite")
    if not np.isfinite(atol) or atol < 0 or not np.isfinite(rtol) or rtol < 0:
        raise ValueError("tolerances must be finite and nonnegative")
    points = _points(diagram)
    if np.any(points[:, 1] < 0):
        raise ValueError("persistence must be nonnegative")
    c, theta = model.original_scale(nx * ny)
    xe, ye = _axes(bounds, nx, ny)
    theta_2d = theta.reshape(nx, ny)
    scores = np.empty(len(points), dtype=np.float64)
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        chunk = points[start:stop]
        gx = _cdf_intervals(xe, chunk[:, 0], sigma_x)
        gy = _cdf_intervals(ye, chunk[:, 1], sigma_y)
        weights = np.minimum(chunk[:, 1] / weight_scale, 1.0)
        scores[start:stop] = (
            np.einsum("ib,ij,jb->b", gx, theta_2d, gy, optimize=True) * weights
        )
    image = stable_persistence_image(
        points, bounds, nx, ny, sigma_x, sigma_y,
        weight_scale=weight_scale, engine="dense", batch_size=batch_size,
    )
    prediction = float(c + theta @ image + other_blocks)
    reconstructed = float(c + scores.sum() + other_blocks)
    err = abs(prediction - reconstructed)
    if not np.isclose(prediction, reconstructed, rtol=rtol, atol=atol):
        raise ArithmeticError(
            f"interval contributions do not reconstruct image prediction: {err}"
        )
    return IntervalExplanation(
        points.copy(), scores, image, c, float(other_blocks),
        prediction, reconstructed, err,
    )
